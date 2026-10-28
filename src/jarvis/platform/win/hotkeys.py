"""System-wide hotkeys via RegisterHotKey on a message-only window thread."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from jarvis.platform.win.api import last_error, user32
from jarvis.platform.win.msgloop import MessageWindow
from jarvis.platform.win.scancodes import parse_hotkey
from jarvis.platform.win.structs import WM_HOTKEY

log = logging.getLogger(__name__)


class WinHotkeys:
    def __init__(self) -> None:
        self._win = MessageWindow("JarvisHotkeys")
        self._win.on(WM_HOTKEY, self._on_hotkey)
        self._win.on_start(self._register_all)
        self._entries: list[tuple[str, int, int, Callable[[], None]]] = []
        self.failed: list[str] = []

    def register(self, combo: str, callback: Callable[[], None]) -> None:
        mods, vk = parse_hotkey(combo)
        self._entries.append((combo, mods, vk, callback))

    def _register_all(self, hwnd: int) -> None:
        u = user32()
        for idx, (combo, mods, vk, _cb) in enumerate(self._entries, start=1):
            if not u.RegisterHotKey(hwnd, idx, mods, vk):
                log.warning("could not register hotkey %s (error %s) - is another app using it?", combo, last_error())
                self.failed.append(combo)

    def _on_hotkey(self, _msg: int, wparam: int, _lparam: int) -> int | None:
        idx = int(wparam)
        if 1 <= idx <= len(self._entries):
            cb = self._entries[idx - 1][3]
            # Run off the message thread so a slow callback never stalls hotkey delivery.
            threading.Thread(target=cb, name=f"hotkey-{idx}", daemon=True).start()
        return 0

    def start(self) -> None:
        self._win.start()

    def stop(self) -> None:
        if self._win.hwnd:
            u = user32()
            for idx in range(1, len(self._entries) + 1):
                u.UnregisterHotKey(self._win.hwnd, idx)
        self._win.stop()
