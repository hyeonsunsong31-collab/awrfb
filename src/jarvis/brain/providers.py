"""OpenAI-compatible chat providers: NVIDIA API (primary) and local Ollama (fallback)."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any, Protocol

from jarvis.brain.types import ChatDelta, Message, ThinkFilter, ToolCall
from jarvis.common.config import BrainConfig, EndpointConfig

log = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    def __init__(self, kind: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind  # rate_limit | auth | credits | bad_request | unsupported | network | server | other
        self.status = status


class Provider(Protocol):
    name: str
    local: bool

    def stream(
        self,
        model: str,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 60.0,
        strip_reasoning: bool = True,
    ) -> AsyncIterator[ChatDelta]: ...

    async def list_models(self) -> list[str]: ...


def resolve_api_key(ep: EndpointConfig) -> str | None:
    if ep.api_key_env and os.environ.get(ep.api_key_env):
        return os.environ[ep.api_key_env]
    if ep.keyring_service:
        try:
            import keyring

            v = keyring.get_password(ep.keyring_service, ep.keyring_user or "default")
            if v:
                return v
        except Exception:
            pass
    return None


def classify_error(e: Exception) -> ProviderError:
    import openai

    msg = str(e)[:300]
    status = getattr(e, "status_code", None)
    low = msg.lower()
    if isinstance(e, openai.RateLimitError):
        return ProviderError("rate_limit", msg, status)
    if isinstance(e, openai.AuthenticationError | openai.PermissionDeniedError):
        return ProviderError("auth", msg, status)
    if isinstance(e, openai.APITimeoutError | openai.APIConnectionError):
        return ProviderError("network", msg, status)
    if status == 402 or "credit" in low or "quota" in low:
        return ProviderError("credits", msg, status)
    if isinstance(e, openai.BadRequestError | openai.UnprocessableEntityError):
        if "tool" in low or "function" in low or "image" in low:
            return ProviderError("unsupported", msg, status)
        return ProviderError("bad_request", msg, status)
    if isinstance(e, openai.NotFoundError):
        return ProviderError("unsupported", msg, status)
    if isinstance(e, openai.InternalServerError) or (status is not None and status >= 500):
        return ProviderError("server", msg, status)
    return ProviderError("other", msg, status)


class OpenAICompatProvider:
    def __init__(self, name: str, ep: EndpointConfig, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        self.name = name
        self.ep = ep
        self.local = ep.local
        key = api_key or resolve_api_key(ep) or ("ollama" if ep.kind == "ollama" else None)
        self.has_key = key is not None
        self._client = AsyncOpenAI(api_key=key or "missing", base_url=ep.base_url, timeout=ep.timeout_s, max_retries=0)

    async def list_models(self) -> list[str]:
        try:
            page = await self._client.models.list()
            return [m.id for m in page.data]
        except Exception as e:
            raise classify_error(e) from e

    async def stream(
        self,
        model: str,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 60.0,
        strip_reasoning: bool = True,
    ) -> AsyncIterator[ChatDelta]:
        if not self.has_key:
            raise ProviderError("auth", f"no API key for {self.name} (run `jarvis setup key`)")
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = tools
        if extra_body:
            kwargs["extra_body"] = extra_body
        try:
            stream = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise classify_error(e) from e
        calls: dict[int, dict[str, str]] = {}
        finish = None
        think = ThinkFilter() if strip_reasoning else None
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                ch = chunk.choices[0]
                delta = ch.delta
                if delta is not None:
                    text = delta.content or ""
                    if text:
                        if think is not None:
                            text = think.feed(text)
                        if text:
                            yield ChatDelta("text", text, endpoint=self.name, model=model)
                    for tc in delta.tool_calls or []:
                        slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function is not None:
                            if tc.function.name:
                                slot["name"] += tc.function.name
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments
                if ch.finish_reason:
                    finish = ch.finish_reason
        except ProviderError:
            raise
        except Exception as e:
            raise classify_error(e) from e
        if think is not None:
            rest = think.flush()
            if rest:
                yield ChatDelta("text", rest, endpoint=self.name, model=model)
        tool_calls = [
            ToolCall(v["id"] or f"call_{i}", v["name"], v["arguments"]) for i, v in sorted(calls.items()) if v["name"]
        ]
        yield ChatDelta("done", tool_calls=tool_calls, finish_reason=finish, endpoint=self.name, model=model)


def build_providers(cfg: BrainConfig) -> dict[str, Provider]:
    """One OpenAI-compatible client per endpoint. A cloud endpoint without an API key is left out (and
    logged) so the broker goes straight to the local fallback instead of failing every request."""
    out: dict[str, Provider] = {}
    for name, ep in cfg.endpoints.items():
        p = OpenAICompatProvider(name, ep)
        if not p.has_key:
            log.warning("LLM endpoint %s has no API key (set %s or run `jarvis api-key`) - skipped", name, ep.api_key_env)
            continue
        out[name] = p
    return out
