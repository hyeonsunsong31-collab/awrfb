"""Scripted LLM provider for brain tests (no network)."""

from __future__ import annotations

import asyncio

from jarvis.brain.providers import ProviderError
from jarvis.brain.types import ChatDelta, ToolCall


def text(*parts: str, tool_calls: list[ToolCall] | None = None, delay: float = 0.0):
    return {"parts": list(parts), "calls": tool_calls or [], "delay": delay}


class ScriptedProvider:
    def __init__(self, name: str, scripts: list, local: bool = False) -> None:
        self.name = name
        self.local = local
        self.scripts = list(scripts)
        self.calls: list[dict] = []

    async def list_models(self) -> list[str]:
        return ["m"]

    async def stream(self, model, messages, *, tools=None, max_tokens=1024, temperature=None, extra_body=None,
                     timeout=60.0, strip_reasoning=True):
        self.calls.append({"model": model, "messages": messages, "tools": tools})
        if not self.scripts:
            raise ProviderError("other", "script exhausted")
        s = self.scripts.pop(0)
        if isinstance(s, ProviderError):
            raise s
        for p in s["parts"]:
            if isinstance(p, ProviderError):
                raise p
            if s["delay"]:
                await asyncio.sleep(s["delay"])
            yield ChatDelta("text", p, endpoint=self.name, model=model)
        yield ChatDelta("done", tool_calls=s["calls"], finish_reason="stop", endpoint=self.name, model=model)
