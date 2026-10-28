"""Key name tables: Set-1 scan codes (what SendInput sends with KEYEVENTF_SCANCODE, which DirectX-
style games such as Roblox read) and virtual-key codes (for hotkeys and GetAsyncKeyState sweeps).

Pure data - importable and testable on any OS.
"""

from __future__ import annotations

from typing import NamedTuple


class ScanCode(NamedTuple):
    code: int
    extended: bool = False


SCANCODES: dict[str, ScanCode] = {
    "esc": ScanCode(0x01),
    "1": ScanCode(0x02),
    "2": ScanCode(0x03),
    "3": ScanCode(0x04),
    "4": ScanCode(0x05),
    "5": ScanCode(0x06),
    "6": ScanCode(0x07),
    "7": ScanCode(0x08),
    "8": ScanCode(0x09),
    "9": ScanCode(0x0A),
    "0": ScanCode(0x0B),
    "minus": ScanCode(0x0C),
    "equals": ScanCode(0x0D),
    "backspace": ScanCode(0x0E),
    "tab": ScanCode(0x0F),
    "q": ScanCode(0x10),
    "w": ScanCode(0x11),
    "e": ScanCode(0x12),
    "r": ScanCode(0x13),
    "t": ScanCode(0x14),
    "y": ScanCode(0x15),
    "u": ScanCode(0x16),
    "i": ScanCode(0x17),
    "o": ScanCode(0x18),
    "p": ScanCode(0x19),
    "lbracket": ScanCode(0x1A),
    "rbracket": ScanCode(0x1B),
    "enter": ScanCode(0x1C),
    "lctrl": ScanCode(0x1D),
    "a": ScanCode(0x1E),
    "s": ScanCode(0x1F),
    "d": ScanCode(0x20),
    "f": ScanCode(0x21),
    "g": ScanCode(0x22),
    "h": ScanCode(0x23),
    "j": ScanCode(0x24),
    "k": ScanCode(0x25),
    "l": ScanCode(0x26),
    "semicolon": ScanCode(0x27),
    "quote": ScanCode(0x28),
    "backquote": ScanCode(0x29),
    "lshift": ScanCode(0x2A),
    "backslash": ScanCode(0x2B),
    "z": ScanCode(0x2C),
    "x": ScanCode(0x2D),
    "c": ScanCode(0x2E),
    "v": ScanCode(0x2F),
    "b": ScanCode(0x30),
    "n": ScanCode(0x31),
    "m": ScanCode(0x32),
    "comma": ScanCode(0x33),
    "period": ScanCode(0x34),
    "slash": ScanCode(0x35),
    "rshift": ScanCode(0x36),
    "numpad_multiply": ScanCode(0x37),
    "lalt": ScanCode(0x38),
    "space": ScanCode(0x39),
    "capslock": ScanCode(0x3A),
    "f1": ScanCode(0x3B),
    "f2": ScanCode(0x3C),
    "f3": ScanCode(0x3D),
    "f4": ScanCode(0x3E),
    "f5": ScanCode(0x3F),
    "f6": ScanCode(0x40),
    "f7": ScanCode(0x41),
    "f8": ScanCode(0x42),
    "f9": ScanCode(0x43),
    "f10": ScanCode(0x44),
    "numlock": ScanCode(0x45),
    "scrolllock": ScanCode(0x46),
    "numpad7": ScanCode(0x47),
    "numpad8": ScanCode(0x48),
    "numpad9": ScanCode(0x49),
    "numpad_minus": ScanCode(0x4A),
    "numpad4": ScanCode(0x4B),
    "numpad5": ScanCode(0x4C),
    "numpad6": ScanCode(0x4D),
    "numpad_plus": ScanCode(0x4E),
    "numpad1": ScanCode(0x4F),
    "numpad2": ScanCode(0x50),
    "numpad3": ScanCode(0x51),
    "numpad0": ScanCode(0x52),
    "numpad_decimal": ScanCode(0x53),
    "f11": ScanCode(0x57),
    "f12": ScanCode(0x58),
    # E0-prefixed (extended) keys
    "rctrl": ScanCode(0x1D, True),
    "ralt": ScanCode(0x38, True),
    "numpad_enter": ScanCode(0x1C, True),
    "numpad_divide": ScanCode(0x35, True),
    "home": ScanCode(0x47, True),
    "up": ScanCode(0x48, True),
    "pgup": ScanCode(0x49, True),
    "left": ScanCode(0x4B, True),
    "right": ScanCode(0x4D, True),
    "end": ScanCode(0x4F, True),
    "down": ScanCode(0x50, True),
    "pgdn": ScanCode(0x51, True),
    "insert": ScanCode(0x52, True),
    "delete": ScanCode(0x53, True),
    "lwin": ScanCode(0x5B, True),
    "rwin": ScanCode(0x5C, True),
    "apps": ScanCode(0x5D, True),
    "printscreen": ScanCode(0x37, True),
}

# Reverse lookup for raw-input decoding (scan code + E0 flag -> name).
SCAN_TO_NAME: dict[tuple[int, bool], str] = {(v.code, v.extended): k for k, v in SCANCODES.items()}

ALIASES: dict[str, str] = {
    "escape": "esc",
    "return": "enter",
    "ctrl": "lctrl",
    "control": "lctrl",
    "shift": "lshift",
    "alt": "lalt",
    "win": "lwin",
    "spacebar": "space",
    "/": "slash",
    "`": "backquote",
    "-": "minus",
    "=": "equals",
    "[": "lbracket",
    "]": "rbracket",
    ";": "semicolon",
    "'": "quote",
    ",": "comma",
    ".": "period",
    "\\": "backslash",
    "pageup": "pgup",
    "pagedown": "pgdn",
    "del": "delete",
    "ins": "insert",
}

# Keys the GAME actuator may never send, whatever the keymap says. Esc opens the Roblox menu (which
# holds "Leave"), Enter/Tab/slash/backquote interact with chat and menus, Win/Alt/F-keys leave the
# game or open developer/system UI.
NEVER_SEND_GAME: frozenset[str] = frozenset(
    {
        "esc",
        "enter",
        "numpad_enter",
        "tab",
        "slash",
        "numpad_divide",
        "backquote",
        "lwin",
        "rwin",
        "apps",
        "lalt",
        "ralt",
        "printscreen",
        "delete",
        "backspace",
        *(f"f{i}" for i in range(1, 13)),
    }
)

# Virtual-key codes (hotkeys, GetAsyncKeyState sweeps, media keys).
VK: dict[str, int] = {
    **{chr(c): c for c in range(ord("A"), ord("Z") + 1)},
    **{chr(c).lower(): c for c in range(ord("A"), ord("Z") + 1)},
    **{str(d): 0x30 + d for d in range(10)},
    **{f"f{i}": 0x6F + i for i in range(1, 25)},
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
    "alt": 0x12,
    "pause": 0x13,
    "capslock": 0x14,
    "esc": 0x1B,
    "space": 0x20,
    "pgup": 0x21,
    "pgdn": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "printscreen": 0x2C,
    "insert": 0x2D,
    "delete": 0x2E,
    "lwin": 0x5B,
    "rwin": 0x5C,
    "apps": 0x5D,
    "numpad0": 0x60,
    "numpad1": 0x61,
    "numpad2": 0x62,
    "numpad3": 0x63,
    "numpad4": 0x64,
    "numpad5": 0x65,
    "numpad6": 0x66,
    "numpad7": 0x67,
    "numpad8": 0x68,
    "numpad9": 0x69,
    "numpad_multiply": 0x6A,
    "numpad_plus": 0x6B,
    "numpad_minus": 0x6D,
    "numpad_decimal": 0x6E,
    "numpad_divide": 0x6F,
    "numlock": 0x90,
    "scrolllock": 0x91,
    "lshift": 0xA0,
    "rshift": 0xA1,
    "lctrl": 0xA2,
    "rctrl": 0xA3,
    "lalt": 0xA4,
    "ralt": 0xA5,
    "volume_mute": 0xAD,
    "volume_down": 0xAE,
    "volume_up": 0xAF,
    "media_next": 0xB0,
    "media_prev": 0xB1,
    "media_stop": 0xB2,
    "media_play_pause": 0xB3,
    "semicolon": 0xBA,
    "equals": 0xBB,
    "comma": 0xBC,
    "minus": 0xBD,
    "period": 0xBE,
    "slash": 0xBF,
    "backquote": 0xC0,
    "lbracket": 0xDB,
    "backslash": 0xDC,
    "rbracket": 0xDD,
    "quote": 0xDE,
    "mouse_left": 0x01,
    "mouse_right": 0x02,
    "mouse_middle": 0x04,
}

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

_MODIFIERS = {
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
}


def normalize_key(name: str) -> str:
    """Canonical key name, raising ``KeyError`` for unknown keys."""
    k = name.strip().lower()
    k = ALIASES.get(k, k)
    if k not in SCANCODES:
        raise KeyError(f"unknown key name: {name!r}")
    return k


def parse_hotkey(combo: str) -> tuple[int, int]:
    """``"ctrl+alt+end"`` -> (modifiers, vk). Raises ``ValueError`` on bad combos."""
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    mods = 0
    key: str | None = None
    for p in parts:
        if p in _MODIFIERS:
            mods |= _MODIFIERS[p]
        else:
            if key is not None:
                raise ValueError(f"hotkey {combo!r} has more than one non-modifier key")
            key = ALIASES.get(p, p)
    if key is None or key not in VK:
        raise ValueError(f"hotkey {combo!r} has no valid key")
    return mods | MOD_NOREPEAT, VK[key]
