from __future__ import annotations

import ctypes

from jarvis.platform.win.api import user32
from jarvis.platform.win.structs import CURSOR_SHOWING, CURSORINFO, LASTINPUTINFO


class WinCursorProbe:
    def cursor_visible(self) -> bool:
        ci = CURSORINFO()
        ci.cbSize = ctypes.sizeof(CURSORINFO)
        if not user32().GetCursorInfo(ctypes.byref(ci)):
            return True  # fail safe: an unknown cursor state counts as "visible" (gate closes)
        return bool(ci.flags & CURSOR_SHOWING)


def idle_seconds() -> float:
    """Seconds since the last keyboard/mouse input of any kind (GetLastInputInfo)."""
    from jarvis.platform.win.api import kernel32

    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
    if not user32().GetLastInputInfo(ctypes.byref(lii)):
        return 0.0
    now = kernel32().GetTickCount()
    return ((now - lii.dwTime) & 0xFFFFFFFF) / 1000.0
