"""Tool definitions shared by toolhost and core."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from jarvis.safety.capabilities import Cap


class Refused(Exception):
    """A safety rule (not an error): the message is returned to the model so it can explain or adapt."""


@dataclass
class ToolResult:
    ok: bool
    text: str
    trust: str = "trusted"  # "untrusted": web pages, files, OCR, clipboard - quarantined before the model sees it
    source: str = ""
    image_data_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "text": self.text, "trust": self.trust, "source": self.source, "image_data_url": self.image_data_url}


@dataclass
class ToolContext:
    policy: Any  # PolicyChecker
    runner: Any = None  # ProcessRunner
    plat: Any = None  # Platform (Windows desktop services) or None
    task_id: str = ""
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    state: dict[str, Any] = field(default_factory=dict)  # per-task scratch (e.g. set-of-marks)
    extras: dict[str, Any] = field(default_factory=dict)  # local tools: memory store, fightd client, hub


Handler = Callable[[ToolContext, dict[str, Any]], Awaitable[ToolResult]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    cap: Cap
    handler: Handler
    params: dict[str, Any] = field(default_factory=dict)  # JSON-schema properties
    required: tuple[str, ...] = ()
    trust: str = "trusted"
    windows_only: bool = False
    timeout_s: float = 60.0

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": self.params, "required": list(self.required)},
        }}


def s(desc: str, **kw: Any) -> dict[str, Any]:
    return {"type": "string", "description": desc, **kw}


def i(desc: str, **kw: Any) -> dict[str, Any]:
    return {"type": "integer", "description": desc, **kw}


def b(desc: str) -> dict[str, Any]:
    return {"type": "boolean", "description": desc}


def ok(text: str, *, trust: str = "trusted", source: str = "", image: str | None = None) -> ToolResult:
    return ToolResult(True, text, trust, source, image)


def clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"\n... [{len(text) - n} more characters cut]"
