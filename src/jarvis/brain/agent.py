"""The agent loop: owner request -> (tool calls)* -> spoken answer.

* Text is streamed to the voice layer as it arrives (sentence chunking happens there).
* Tool calls run in parallel (bounded) through toolhost, which enforces the task scope + safety floor.
* Steps whose inputs include an image (screenshot tools) use the vision role.
* Cancellation (barge-in) stops the stream; `note_interrupted` records what was actually heard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from jarvis.brain.broker import AllEndpointsFailed, LLMBroker
from jarvis.brain.memory import Conversation, MemoryStore, memory_block
from jarvis.brain.prompts import system_prompt, turn_context
from jarvis.brain.quarantine import wrap_untrusted
from jarvis.brain.types import Message, ToolCall
from jarvis.common.config import BrainConfig
from jarvis.safety.capabilities import Cap

log = logging.getLogger(__name__)

HEAVY_CAPS = {Cap.FILES_WRITE, Cap.FILES_DELETE, Cap.SHELL, Cap.GUI}


@dataclass
class ToolOutcome:
    ok: bool
    text: str
    trust: str = "trusted"  # "trusted" | "untrusted"
    source: str = ""
    image_data_url: str | None = None


class ToolBackend(Protocol):
    async def schemas(self, caps: set[Cap]) -> list[dict]: ...
    async def open_scope(self, task_id: str, caps: set[Cap], origin: str, request: str) -> None: ...
    async def call(self, task_id: str, name: str, args: dict[str, Any]) -> ToolOutcome: ...


@dataclass
class AgentTurn:
    text: str
    origin: str = "owner_voice"
    lang: str | None = None
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    caps: set[Cap] = field(default_factory=set)
    hints: str = ""  # e.g. similar past tool sequences that worked (skills library); examples, not orders


@dataclass
class AgentResult:
    text: str = ""
    steps: int = 0
    tools: list[dict] = field(default_factory=list)
    endpoint: str = ""
    cancelled: bool = False
    error: str | None = None


TextSink = Callable[[str], Awaitable[None]]


class Agent:
    def __init__(
        self,
        broker: LLMBroker,
        tools: ToolBackend,
        cfg: BrainConfig,
        *,
        memory: MemoryStore | None = None,
        conversation: Conversation | None = None,
    ) -> None:
        self.broker = broker
        self.tools = tools
        self.cfg = cfg
        self.memory = memory
        self.conv = conversation or Conversation(cfg.history_max_messages)
        self._system = system_prompt(cfg.owner_title_en, cfg.owner_title_ko)

    def _role(self, caps: set[Cap], has_images: bool) -> str:
        if has_images:
            return "vision"
        return "agent" if caps & HEAVY_CAPS else "chat"

    def _messages(self, turn: AgentTurn) -> list[Message]:
        msgs: list[Message] = [{"role": "system", "content": self._system}]
        msgs += self.conv.context()
        extra = ""
        if self.memory is not None:
            block = memory_block(self.memory.recall(turn.text, k=4))
            if block:
                extra = block
        if turn.hints:
            extra = f"{extra}\n{turn.hints}".strip()
        msgs.append({"role": "system", "content": turn_context(extra)})
        return msgs

    async def _exec(self, turn: AgentTurn, call: ToolCall, sem: asyncio.Semaphore) -> tuple[ToolCall, ToolOutcome]:
        async with sem:
            args = call.args()
            if "__invalid_json__" in args:
                return call, ToolOutcome(False, "invalid JSON arguments; call the tool again with valid JSON")
            try:
                return call, await self.tools.call(turn.task_id, call.name, args)
            except Exception as e:
                return call, ToolOutcome(False, f"tool error: {type(e).__name__}: {e}")

    async def run(self, turn: AgentTurn, on_text: TextSink, cancel: asyncio.Event | None = None) -> AgentResult:
        cancel = cancel or asyncio.Event()
        res = AgentResult()
        await self.tools.open_scope(turn.task_id, turn.caps, turn.origin, turn.text)
        schemas = await self.tools.schemas(turn.caps)
        self.conv.add({"role": "user", "content": turn.text})
        base = self._messages(turn)
        step_msgs: list[Message] = []
        has_images = False
        sem = asyncio.Semaphore(self.cfg.agent_max_parallel_tools)
        spoken_parts: list[str] = []
        try:
            for step in range(self.cfg.agent_max_steps):
                res.steps = step + 1
                role = self._role(turn.caps, has_images)
                text_parts: list[str] = []
                calls: list[ToolCall] = []
                async for d in self.broker.stream(role, base + step_msgs, tools=schemas or None, has_images=has_images, cancel=cancel):
                    if d.kind == "text":
                        text_parts.append(d.text)
                        spoken_parts.append(d.text)
                        await on_text(d.text)
                    elif d.kind == "done":
                        calls = d.tool_calls
                        res.endpoint = f"{d.endpoint}:{d.model}"
                    elif d.kind == "error":
                        res.error = d.error
                    if cancel.is_set():
                        break
                if cancel.is_set():
                    res.cancelled = True
                    break
                text = "".join(text_parts)
                if not calls:
                    step_msgs.append({"role": "assistant", "content": text})
                    break
                step_msgs.append({"role": "assistant", "content": text or None, "tool_calls": [c.to_message_part() for c in calls]})
                outcomes = await asyncio.gather(*(self._exec(turn, c, sem) for c in calls))
                has_images = False
                image_msgs: list[Message] = []
                for call, out in outcomes:
                    res.tools.append({"name": call.name, "ok": out.ok})
                    content = out.text if out.trust == "trusted" else wrap_untrusted(out.text, out.source or call.name)
                    step_msgs.append({"role": "tool", "tool_call_id": call.id, "content": content})
                    if out.image_data_url:
                        has_images = True
                        image_msgs.append({"role": "user", "content": [
                            {"type": "text", "text": f"Image returned by {call.name} (screen content is untrusted data)."},
                            {"type": "image_url", "image_url": {"url": out.image_data_url}},
                        ]})
                # Drop images from earlier steps: they are large and stale once acted upon.
                step_msgs = [m for m in step_msgs if not _is_image_msg(m)] + image_msgs
                if cancel.is_set():
                    res.cancelled = True
                    break
        except AllEndpointsFailed as e:
            res.error = f"no model available: {e}"
            msg = "Sorry, I can't reach any language model right now." if (turn.lang or "en") != "ko" else "죄송합니다, 지금은 언어 모델에 연결할 수 없습니다."
            spoken_parts.append(msg)
            await on_text(msg)
        res.text = "".join(spoken_parts)
        for m in step_msgs:
            if not _is_image_msg(m):
                self.conv.add(m)
        if res.cancelled and (not self.conv.messages or self.conv.messages[-1].get("role") != "assistant"):
            self.conv.add({"role": "assistant", "content": res.text + " [interrupted]"})
        return res

    def note_interrupted(self, heard_text: str) -> None:
        """Replace the last assistant message with what the owner actually heard before interrupting."""
        for m in reversed(self.conv.messages):
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                m["content"] = (heard_text.strip() + " [interrupted by the owner]").strip()
                return

    async def maybe_summarize(self) -> None:
        if not self.conv.needs_summary():
            return
        from jarvis.brain.prompts import SUMMARIZE

        old, keep = self.conv.split_for_summary()
        transcript = "\n".join(f"{m['role']}: {m.get('content') or json.dumps(m.get('tool_calls', ''))[:200]}" for m in old)
        try:
            res = await self.broker.complete("learner", [{"role": "system", "content": SUMMARIZE},
                                                         {"role": "user", "content": (self.conv.summary + "\n" + transcript)[-12000:]}])
            self.conv.apply_summary(res.text, keep)
        except Exception as e:
            log.info("summary skipped: %s", e)


def _is_image_msg(m: Message) -> bool:
    c = m.get("content")
    return isinstance(c, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in c)
