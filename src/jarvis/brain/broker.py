"""The LLM broker: every model call in the system goes through here.

* Priority: voice replies (0) > agent steps (1) > coach (2) > learner (3). When an endpoint's rate
  budget is exhausted, waiting requests are served in priority order.
* Self-imposed rate cap per endpoint (the NVIDIA free trial allows ~40 requests/min; we use 30).
* A role's chain ("nvidia:model", "local:model") is walked on 429s, auth/credit problems, server
  errors, timeouts or no network - each failing endpoint is put on a cooldown (circuit breaker).
* Models without native tool calling get prompted tool calling (jarvis.brain.toolproto).
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from jarvis.brain.providers import Provider, ProviderError
from jarvis.brain.toolproto import parse_prompted_tool_calls, render_tools_prompt
from jarvis.brain.types import ChatDelta, ChatResult, Message
from jarvis.common.config import BrainConfig

log = logging.getLogger(__name__)

COOLDOWN_S = {"rate_limit": 30.0, "network": 20.0, "server": 15.0, "auth": 600.0, "credits": 3600.0, "other": 10.0}


class AllEndpointsFailed(RuntimeError):
    pass


class TokenBucket:
    """Rate limiter that serves waiters strictly by (priority, arrival)."""

    def __init__(self, rpm: float, burst: int = 5, now=time.monotonic) -> None:
        self.rate = rpm / 60.0
        self.capacity = float(max(1, burst))
        self.tokens = self.capacity
        self.now = now
        self.t = now()
        self._waiters: list[tuple[int, int, asyncio.Future]] = []
        self._seq = itertools.count()

    def _refill(self) -> None:
        t = self.now()
        self.tokens = min(self.capacity, self.tokens + (t - self.t) * self.rate)
        self.t = t

    def _serve(self) -> None:
        self._refill()
        while self._waiters and self.tokens >= 1.0:
            _p, _s, fut = heapq.heappop(self._waiters)
            if fut.done():
                continue
            self.tokens -= 1.0
            fut.set_result(True)

    async def acquire(self, priority: int, timeout: float) -> bool:
        self._serve()
        if not self._waiters and self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority, next(self._seq), fut))
        deadline = time.monotonic() + timeout
        while not fut.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                fut.cancel()
                return False
            wait = min(remaining, 0.05)  # poll: cheap, and robust to bursts of arrivals
            try:
                await asyncio.wait_for(asyncio.shield(fut), wait)
            except TimeoutError:
                self._serve()
        return fut.done() and not fut.cancelled()


@dataclass
class EndpointHealth:
    open_until: float = 0.0
    last_error: str = ""
    failures: int = 0
    successes: int = 0

    def available(self, now: float) -> bool:
        return now >= self.open_until


@dataclass
class ProbeInfo:
    tools: bool | None = None
    vision: bool | None = None


@dataclass
class BrokerStats:
    calls: dict[str, int] = field(default_factory=dict)
    fallbacks: int = 0


class LLMBroker:
    def __init__(self, cfg: BrainConfig, providers: dict[str, Provider], probe: dict[str, ProbeInfo] | None = None) -> None:
        self.cfg = cfg
        self.providers = providers
        self.health: dict[str, EndpointHealth] = {n: EndpointHealth() for n in providers}
        self.buckets = {n: TokenBucket(cfg.endpoints[n].max_rpm if n in cfg.endpoints else 600) for n in providers}
        self.probe = probe or {}
        self.stats = BrokerStats()

    def _chain(self, role: str) -> list[tuple[str, str]]:
        out = []
        for item in self.cfg.roles[role].chain:
            ep, _, model = item.partition(":")
            if ep in self.providers:
                out.append((ep, model))
        return out

    def _mark_failure(self, ep: str, err: ProviderError) -> None:
        h = self.health[ep]
        h.failures += 1
        h.last_error = f"{err.kind}: {err}"[:200]
        if err.kind in COOLDOWN_S:
            h.open_until = time.monotonic() + COOLDOWN_S[err.kind]
        log.warning("LLM endpoint %s failed (%s); cooling down", ep, err.kind)

    async def stream(
        self,
        role: str,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        has_images: bool = False,
        cancel: asyncio.Event | None = None,
        wait_budget_s: float | None = None,
    ) -> AsyncIterator[ChatDelta]:
        rc = self.cfg.roles[role]
        chain = self._chain(role)
        if not chain:
            raise AllEndpointsFailed(f"role {role!r} has no usable endpoints")
        errors: list[str] = []
        for i, (ep, model) in enumerate(chain):
            if cancel is not None and cancel.is_set():
                return
            now = time.monotonic()
            if not self.health[ep].available(now):
                errors.append(f"{ep}: cooling down ({self.health[ep].last_error})")
                continue
            info = self.probe.get(f"{ep}:{model}", ProbeInfo())
            if has_images and info.vision is False:
                errors.append(f"{ep}:{model}: no image input")
                continue
            budget = wait_budget_s if wait_budget_s is not None else (2.0 if rc.priority == 0 else 20.0)
            if i == len(chain) - 1:
                budget = max(budget, 30.0)
            if not await self.buckets[ep].acquire(rc.priority, budget):
                errors.append(f"{ep}: rate budget exhausted")
                self.stats.fallbacks += 1
                continue
            prompted = bool(tools) and info.tools is False
            msgs = messages
            send_tools = tools
            if prompted:
                msgs = _with_tool_prompt(messages, tools or [])
                send_tools = None
            emitted = False
            try:
                text_acc = []
                async for d in self.providers[ep].stream(
                    model,
                    msgs,
                    tools=send_tools,
                    max_tokens=rc.max_tokens,
                    temperature=rc.temperature,
                    extra_body=rc.extra_body or None,
                    timeout=rc.timeout_s,
                    strip_reasoning=rc.strip_reasoning,
                ):
                    if cancel is not None and cancel.is_set():
                        return
                    if d.kind == "text":
                        emitted = True
                        text_acc.append(d.text)
                        if not prompted:
                            yield d
                    elif d.kind == "done":
                        if prompted:
                            full = "".join(text_acc)
                            calls, remaining = parse_prompted_tool_calls(full, {t["function"]["name"] for t in tools or []})
                            if remaining:
                                yield ChatDelta("text", remaining, endpoint=ep, model=model)
                            d.tool_calls = calls
                        self.health[ep].successes += 1
                        self.stats.calls[ep] = self.stats.calls.get(ep, 0) + 1
                        yield d
                        return
            except ProviderError as e:
                if e.kind == "unsupported" and tools and not prompted:
                    # The model rejected native tools: remember and retry this endpoint prompted.
                    self.probe[f"{ep}:{model}"] = ProbeInfo(tools=False, vision=info.vision)
                    log.info("%s:%s has no native tool calling; switching to prompted tools", ep, model)
                    async for d in self.stream(role, messages, tools=tools, has_images=has_images, cancel=cancel,
                                               wait_budget_s=wait_budget_s):
                        yield d
                    return
                self._mark_failure(ep, e)
                errors.append(f"{ep}: {e.kind}")
                if emitted:
                    yield ChatDelta("error", error=f"{ep} failed mid-answer: {e.kind}", endpoint=ep, model=model)
                    return
                self.stats.fallbacks += 1
                continue
        raise AllEndpointsFailed("; ".join(errors) or "no endpoint available")

    async def complete(self, role: str, messages: list[Message], **kw) -> ChatResult:
        text = []
        done: ChatDelta | None = None
        async for d in self.stream(role, messages, **kw):
            if d.kind == "text":
                text.append(d.text)
            elif d.kind == "done":
                done = d
            elif d.kind == "error":
                raise AllEndpointsFailed(d.error)
        if done is None:
            raise AllEndpointsFailed("stream ended without completion")
        return ChatResult("".join(text), done.tool_calls, done.finish_reason, done.endpoint, done.model)

    def status(self) -> dict:
        now = time.monotonic()
        return {
            ep: {"available": h.available(now), "last_error": h.last_error, "ok": h.successes, "failed": h.failures}
            for ep, h in self.health.items()
        }


def _with_tool_prompt(messages: list[Message], tools: list[dict]) -> list[Message]:
    prompt = render_tools_prompt(tools)
    out = []
    injected = False
    for m in messages:
        if m.get("role") == "tool":
            # Prompted mode has no tool role: present results as user-visible data.
            out.append({"role": "user", "content": f"[tool result {m.get('tool_call_id', '')}]\n{m.get('content', '')}"})
            continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = "\n".join(
                f'```tool\n{{"name": "{c["function"]["name"]}", "arguments": {c["function"]["arguments"]}}}\n```'
                for c in m["tool_calls"]
            )
            out.append({"role": "assistant", "content": (m.get("content") or "") + "\n" + calls})
            continue
        if m.get("role") == "system" and not injected:
            out.append({"role": "system", "content": f"{m['content']}\n\n{prompt}"})
            injected = True
            continue
        out.append(m)
    if not injected:
        out.insert(0, {"role": "system", "content": prompt})
    return out
