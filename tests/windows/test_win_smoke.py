"""Windows API smoke tests (run on the Windows CI runner and on the owner's PC). No Roblox needed."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.windows,
    pytest.mark.skipif(sys.platform != "win32", reason="Windows only"),
]


def test_api_signatures_load():
    from jarvis.platform.win.api import kernel32, user32

    assert user32().GetForegroundWindow is not None
    assert kernel32().GetCurrentThreadId() > 0


def test_window_enumeration():
    from jarvis.platform.win.window import WinWindowManager

    wm = WinWindowManager()
    wins = wm.find_windows()
    assert isinstance(wins, list)


def test_message_window_and_hotkeys_start_stop():
    from jarvis.platform.win.hotkeys import WinHotkeys

    hk = WinHotkeys()
    hk.register("ctrl+alt+shift+f11", lambda: None)
    hk.start()
    hk.stop()


def test_raw_input_monitor_start_stop():
    from jarvis.platform.win.rawinput import RawInputMonitor

    mon = RawInputMonitor()
    mon.start()
    assert mon.last_physical_input_ns() >= 0
    mon.stop()


def test_job_object_kills_children():
    from jarvis.platform.win.job import JobObject

    job = JobObject()
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    job.assign(p)
    job.terminate()
    assert p.wait(timeout=10) != 0
    job.close()


def test_cursor_probe():
    from jarvis.platform.win.cursor import WinCursorProbe, idle_seconds

    assert isinstance(WinCursorProbe().cursor_visible(), bool)
    assert idle_seconds() >= 0
