"""Window discovery, focus and geometry via user32 (ctypes)."""

from __future__ import annotations

import ctypes
import time

import psutil

from jarvis.platform.base import Rect, WindowRef
from jarvis.platform.win.api import WNDENUMPROC, kernel32, user32
from jarvis.platform.win.structs import DWORD, POINT, RECT

SW_RESTORE = 9


def _title(hwnd: int) -> str:
    u = user32()
    n = u.GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    u.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _pid(hwnd: int) -> int:
    pid = DWORD(0)
    user32().GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _process_name(pid: int) -> str | None:
    try:
        return psutil.Process(pid).name()
    except (psutil.Error, OSError):
        return None


class WinWindowManager:
    def find_windows(
        self, title_substring: str | None = None, process_name: str | None = None
    ) -> list[WindowRef]:
        found: list[WindowRef] = []
        u = user32()
        want_title = title_substring.lower() if title_substring else None
        want_proc = process_name.lower() if process_name else None

        @WNDENUMPROC
        def _cb(hwnd, _lparam):
            if not u.IsWindowVisible(hwnd):
                return True
            title = _title(hwnd)
            if want_title and want_title not in title.lower():
                return True
            pid = _pid(hwnd)
            pname = _process_name(pid)
            if want_proc and (pname or "").lower() != want_proc:
                return True
            found.append(WindowRef(hwnd=int(hwnd), title=title, pid=pid, process_name=pname))
            return True

        u.EnumWindows(_cb, 0)
        return found

    def foreground(self) -> WindowRef | None:
        hwnd = user32().GetForegroundWindow()
        if not hwnd:
            return None
        pid = _pid(hwnd)
        return WindowRef(hwnd=int(hwnd), title=_title(hwnd), pid=pid, process_name=_process_name(pid))

    def is_foreground(self, hwnd: int) -> bool:
        fg = user32().GetForegroundWindow()
        return bool(fg) and int(fg) == int(hwnd)

    def focus(self, hwnd: int) -> bool:
        """Bring a window to the foreground without injecting any keys (no Alt trick)."""
        u = user32()
        k = kernel32()
        if u.IsIconic(hwnd):
            u.ShowWindow(hwnd, SW_RESTORE)
        if u.SetForegroundWindow(hwnd) and self.is_foreground(hwnd):
            return True
        # Foreground lock workaround: attach our input queue to the current foreground thread.
        fg = u.GetForegroundWindow()
        cur_tid = k.GetCurrentThreadId()
        fg_tid = u.GetWindowThreadProcessId(fg, None) if fg else 0
        attached = False
        try:
            if fg_tid and fg_tid != cur_tid:
                attached = bool(u.AttachThreadInput(cur_tid, fg_tid, True))
            u.BringWindowToTop(hwnd)
            u.SetForegroundWindow(hwnd)
        finally:
            if attached:
                u.AttachThreadInput(cur_tid, fg_tid, False)
        if self.is_foreground(hwnd):
            return True
        u.SwitchToThisWindow(hwnd, True)
        time.sleep(0.05)
        return self.is_foreground(hwnd)

    def client_rect_screen(self, hwnd: int) -> Rect | None:
        u = user32()
        rc = RECT()
        if not u.GetClientRect(hwnd, ctypes.byref(rc)):
            return None
        pt = POINT(0, 0)
        if not u.ClientToScreen(hwnd, ctypes.byref(pt)):
            return None
        return Rect(pt.x, pt.y, pt.x + rc.right - rc.left, pt.y + rc.bottom - rc.top)

    def is_minimized(self, hwnd: int) -> bool:
        return bool(user32().IsIconic(hwnd))

    def exists(self, hwnd: int) -> bool:
        return bool(user32().IsWindow(hwnd))

    def dpi(self, hwnd: int) -> int:
        u = user32()
        if hasattr(u, "GetDpiForWindow"):
            v = int(u.GetDpiForWindow(hwnd))
            return v or 96
        return 96


def keys_physically_down(names: list[str]) -> list[str]:
    """GetAsyncKeyState sweep: which of these keys does Windows currently consider down?"""
    from jarvis.platform.win.scancodes import VK

    u = user32()
    down = []
    for n in names:
        vk = VK.get(n)
        if vk is None:
            continue
        if u.GetAsyncKeyState(vk) & 0x8000:
            down.append(n)
    return down
