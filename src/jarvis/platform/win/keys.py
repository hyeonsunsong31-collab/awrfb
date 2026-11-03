"""One virtual-key press (media keys). Callers must make sure Roblox is not the foreground window."""

from __future__ import annotations

import ctypes

from jarvis.platform.win.api import user32
from jarvis.platform.win.structs import INPUT, INPUT_KEYBOARD, KEYEVENTF_KEYUP

JARVIS_TAG = 0x4A415256  # dwExtraInfo marker shared with the game input sink ("JARV")


def send_vk(vk: int) -> bool:
    arr = (INPUT * 2)()
    for k, flags in enumerate((0, KEYEVENTF_KEYUP)):
        arr[k].type = INPUT_KEYBOARD
        arr[k].ki.wVk = vk
        arr[k].ki.dwFlags = flags
        arr[k].ki.dwExtraInfo = JARVIS_TAG
    return user32().SendInput(2, arr, ctypes.sizeof(INPUT)) == 2
