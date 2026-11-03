"""Shared message/stream types for the brain (OpenAI-compatible chat format)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Message = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text from the model

    def args(self) -> dict[str, Any]:
        try:
            v = json.loads(self.arguments or "{}")
        except ValueError:
            return {"__invalid_json__": self.arguments}
        return v if isinstance(v, dict) else {"value": v}

    def to_message_part(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments or "{}"}}


@dataclass
class ChatDelta:
    kind: Literal["text", "done", "error"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    endpoint: str = ""
    model: str = ""
    error: str = ""


@dataclass
class ChatResult:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    endpoint: str
    model: str


class ThinkFilter:
    """Removes <think>...</think> sections from a token stream (they can span chunks)."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.inside = False
        self.buf = ""

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out = []
        while True:
            if self.inside:
                i = self.buf.find(self.CLOSE)
                if i < 0:
                    # keep a tail that might be the start of "</think>"
                    self.buf = self.buf[-(len(self.CLOSE) - 1) :]
                    break
                self.buf = self.buf[i + len(self.CLOSE) :]
                self.inside = False
            else:
                i = self.buf.find(self.OPEN)
                if i < 0:
                    keep = len(self.OPEN) - 1
                    safe = self.buf[:-keep] if len(self.buf) > keep else ""
                    # only hold back if the tail could start a tag
                    tail = self.buf[len(safe) :]
                    if "<" not in tail:
                        safe, tail = self.buf, ""
                    out.append(safe)
                    self.buf = tail
                    break
                out.append(self.buf[:i])
                self.buf = self.buf[i + len(self.OPEN) :]
                self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        self.buf = ""
        return rest
