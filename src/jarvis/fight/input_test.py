"""`jarvis input-test`: prove real OS input works - and stops - before it ever touches Roblox.

Types a fixed phrase into Notepad one key at a time, re-checking before EVERY key that Notepad is
still the foreground window. Alt-Tab away and it stops at the next key. The panic hotkey stops it too.
Afterwards it sweeps GetAsyncKeyState to prove no key is left held down.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from jarvis.platform.base import Platform

PHRASE = "jarvis input test 123 "
_CHAR_KEYS = {" ": "space", **{c: c for c in "abcdefghijklmnopqrstuvwxyz0123456789"}}


@dataclass
class InputTestResult:
    typed: int = 0
    total: int = 0
    stopped_reason: str = ""
    keys_still_down: list[str] = field(default_factory=list)
    notepad: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_or_launch_notepad(plat: Platform, timeout_s: float = 10.0):
    wins = plat.windows.find_windows(process_name="notepad.exe") or plat.windows.find_windows(title_substring="Notepad")
    if wins:
        return wins[0]
    subprocess.Popen(["notepad.exe"])
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        wins = plat.windows.find_windows(process_name="notepad.exe")
        if wins:
            return wins[0]
        time.sleep(0.2)
    return None


def run_input_test(plat: Platform, panic_hotkey: str, *, repeats: int = 3, key_ms: float = 60) -> InputTestResult:
    res = InputTestResult(total=len(PHRASE) * repeats)
    win = find_or_launch_notepad(plat)
    if win is None:
        res.stopped_reason = "could not find or start Notepad"
        return res
    res.notepad = win.title
    plat.windows.focus(win.hwnd)
    time.sleep(0.3)

    stop = threading.Event()
    hk = plat.new_hotkeys()
    hk.register(panic_hotkey, stop.set)
    hk.start()
    sink = plat.new_input_sink()
    try:
        for ch in PHRASE * repeats:
            if stop.is_set():
                res.stopped_reason = "panic hotkey"
                break
            if not plat.windows.is_foreground(win.hwnd):
                res.stopped_reason = f"Notepad lost focus after {res.typed} keys (as designed)"
                break
            key = _CHAR_KEYS[ch]
            sink.key_down(key)
            time.sleep(key_ms / 2000)
            sink.key_up(key)
            res.typed += 1
            time.sleep(key_ms / 2000)
        else:
            res.stopped_reason = "completed"
    finally:
        sink.release_all()
        hk.stop()
    time.sleep(0.1)
    try:
        from jarvis.platform.win.window import keys_physically_down

        res.keys_still_down = keys_physically_down(sorted(set(_CHAR_KEYS.values())))
    except Exception as e:  # pragma: no cover - Windows only
        res.keys_still_down = [f"sweep failed: {e}"]
    return res
