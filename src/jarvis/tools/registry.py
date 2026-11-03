"""The tool catalogue: which tools exist on this machine and which a task's scope may see."""

from __future__ import annotations

import sys

from jarvis.safety.capabilities import Cap
from jarvis.tools.base import Tool


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None, *, windows: bool | None = None) -> None:
        self.windows = sys.platform == "win32" if windows is None else windows
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        if tool.windows_only and not self.windows:
            return
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def allowed(self, caps: set[Cap]) -> list[Tool]:
        return [t for t in self._tools.values() if t.cap in caps]

    def schemas(self, caps: set[Cap]) -> list[dict]:
        return [t.schema() for t in sorted(self.allowed(caps), key=lambda t: t.name)]


def toolhost_tools() -> list[Tool]:
    """Everything toolhost offers (Windows-only tools are dropped elsewhere by the registry)."""
    from jarvis.tools import apps, files, gui, shell, system, web, windows

    return [*files.TOOLS, *shell.TOOLS, *apps.TOOLS, *windows.TOOLS, *gui.TOOLS, *web.TOOLS, *system.TOOLS]
