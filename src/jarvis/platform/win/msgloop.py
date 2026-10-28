"""A message-only window running its own message loop on a dedicated thread.

Used by Raw Input (WM_INPUT) and global hotkeys (WM_HOTKEY). Handlers run on the loop thread and must
be quick; they should only record timestamps or enqueue work.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from collections.abc import Callable

from jarvis.platform.win.api import HWND_MESSAGE, WNDCLASSEXW, WNDPROC, kernel32, user32
from jarvis.platform.win.structs import MSG, WM_CLOSE, WM_QUIT

log = logging.getLogger(__name__)

Handler = Callable[[int, int, int], int | None]  # (msg, wparam, lparam) -> result or None


class MessageWindow:
    _counter = 0

    def __init__(self, name: str = "JarvisMsgWindow") -> None:
        MessageWindow._counter += 1
        self._class_name = f"{name}_{MessageWindow._counter}"
        self._handlers: dict[int, Handler] = {}
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._tid = 0
        self.hwnd: int | None = None
        self._on_start: list[Callable[[int], None]] = []
        self._wndproc = WNDPROC(self._proc)  # keep a reference: ctypes callbacks must outlive use

    def on(self, msg: int, handler: Handler) -> None:
        self._handlers[msg] = handler

    def on_start(self, fn: Callable[[int], None]) -> None:
        """Run ``fn(hwnd)`` on the loop thread once the window exists (e.g. RegisterHotKey)."""
        self._on_start.append(fn)

    def _proc(self, hwnd, msg, wparam, lparam):
        h = self._handlers.get(msg)
        if h is not None:
            try:
                r = h(msg, wparam, lparam)
                if r is not None:
                    return r
            except Exception:  # never let a handler exception unwind into user32
                log.exception("message handler failed (msg=0x%x)", msg)
        return user32().DefWindowProcW(hwnd, msg, wparam, lparam)

    def _run(self) -> None:
        u = user32()
        k = kernel32()
        self._tid = k.GetCurrentThreadId()
        hinst = k.GetModuleHandleW(None)
        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.lpfnWndProc = self._wndproc
        wc.hInstance = hinst
        wc.lpszClassName = self._class_name
        if not u.RegisterClassExW(ctypes.byref(wc)):
            log.error("RegisterClassExW failed")
            self._ready.set()
            return
        hwnd = u.CreateWindowExW(0, self._class_name, self._class_name, 0, 0, 0, 0, 0, HWND_MESSAGE, None, hinst, None)
        if not hwnd:
            log.error("CreateWindowExW failed")
            self._ready.set()
            return
        self.hwnd = int(hwnd)
        for fn in self._on_start:
            try:
                fn(self.hwnd)
            except Exception:
                log.exception("message window start hook failed")
        self._ready.set()
        msg = MSG()
        while True:
            r = u.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if r == 0 or r == -1:
                break
            u.TranslateMessage(ctypes.byref(msg))
            u.DispatchMessageW(ctypes.byref(msg))
        u.DestroyWindow(hwnd)
        u.UnregisterClassW(self._class_name, hinst)

    def start(self, timeout: float = 5.0) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name=self._class_name, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("message window did not start")
        if self.hwnd is None:
            raise RuntimeError("message window failed to create")

    def stop(self) -> None:
        if self._thread and self._thread.is_alive() and self._tid:
            u = user32()
            if self.hwnd:
                u.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)
            u.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
            self._thread.join(timeout=2.0)
        self._thread = None
        self.hwnd = None
