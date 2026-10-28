"""Struct layouts and key tables - these mirror 64-bit Windows exactly, so they are testable here."""

from __future__ import annotations

import ctypes

import pytest

from jarvis.platform.win import structs as s
from jarvis.platform.win.scancodes import (
    MOD_ALT,
    MOD_CONTROL,
    MOD_NOREPEAT,
    NEVER_SEND_GAME,
    SCAN_TO_NAME,
    SCANCODES,
    VK,
    normalize_key,
    parse_hotkey,
)


def test_struct_sizes_match_win64():
    assert ctypes.sizeof(s.MOUSEINPUT) == 32
    assert ctypes.sizeof(s.KEYBDINPUT) == 24
    assert ctypes.sizeof(s.HARDWAREINPUT) == 8
    assert ctypes.sizeof(s.INPUT) == 40
    assert ctypes.sizeof(s.RAWINPUTHEADER) == 24
    assert ctypes.sizeof(s.RAWMOUSE) == 24
    assert ctypes.sizeof(s.RAWKEYBOARD) == 16
    assert ctypes.sizeof(s.RAWINPUT) == 48
    assert ctypes.sizeof(s.RAWINPUTDEVICE) == 16
    assert ctypes.sizeof(s.MSG) == 48
    assert ctypes.sizeof(s.JOBOBJECT_BASIC_LIMIT_INFORMATION) == 64
    assert ctypes.sizeof(s.IO_COUNTERS) == 48
    assert ctypes.sizeof(s.JOBOBJECT_EXTENDED_LIMIT_INFORMATION) == 144


def test_key_input_flags():
    inp = s.key_input(0x21, extended=False, up=False)
    assert inp.type == s.INPUT_KEYBOARD
    assert inp.ki.wScan == 0x21 and inp.ki.wVk == 0
    assert inp.ki.dwFlags == s.KEYEVENTF_SCANCODE
    assert inp.ki.dwExtraInfo == s.JARVIS_EXTRA_INFO
    up = s.key_input(0x4B, extended=True, up=True)
    assert up.ki.dwFlags == s.KEYEVENTF_SCANCODE | s.KEYEVENTF_EXTENDEDKEY | s.KEYEVENTF_KEYUP
    m = s.mouse_input(s.MOUSEEVENTF_MOVE, 5, -3)
    assert m.type == s.INPUT_MOUSE and (m.mi.dx, m.mi.dy) == (5, -3)


def test_scancodes_and_aliases():
    assert SCANCODES["f"].code == 0x21
    assert SCANCODES["space"].code == 0x39
    assert SCANCODES["left"].extended is True
    assert normalize_key("Escape") == "esc"
    assert normalize_key("/") == "slash"
    assert SCAN_TO_NAME[(0x11, False)] == "w"
    with pytest.raises(KeyError):
        normalize_key("hyperkey")


def test_never_send_contains_dangerous_keys():
    for k in ("esc", "enter", "tab", "slash", "backquote", "lwin", "lalt", "f4", "f9", "f11"):
        assert k in NEVER_SEND_GAME
    for k in ("w", "a", "s", "d", "f", "space", "q", "e", "r"):
        assert k not in NEVER_SEND_GAME


def test_parse_hotkey():
    mods, vk = parse_hotkey("ctrl+alt+end")
    assert mods == MOD_CONTROL | MOD_ALT | MOD_NOREPEAT
    assert vk == VK["end"] == 0x23
    assert parse_hotkey("ctrl+shift+f12")[1] == 0x7B
    with pytest.raises(ValueError):
        parse_hotkey("ctrl+alt")
    with pytest.raises(ValueError):
        parse_hotkey("a+b")
