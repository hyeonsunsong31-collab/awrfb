"""Test doubles for the platform layer. Never imported by runtime code."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from jarvis.platform.base import (
    Frame,
    MouseButton,
    PhysicalInputEvent,
    Platform,
    Rect,
    WindowRef,
)
from jarvis.platform.win.scancodes import normalize_key


class FakeInputSink:
    def __init__(self, forbidden_keys=()) -> None:
        self.events: list[tuple[str, str]] = []
        self._held: list[str] = []
        self._forbidden = set(forbidden_keys)

    def key_down(self, key: str) -> None:
        k = normalize_key(key)
        if k in self._forbidden:
            raise ValueError(f"forbidden {k}")
        self.events.append(("down", f"key:{k}"))
        if f"key:{k}" not in self._held:
            self._held.append(f"key:{k}")

    def key_up(self, key: str) -> None:
        k = normalize_key(key)
        self.events.append(("up", f"key:{k}"))
        if f"key:{k}" in self._held:
            self._held.remove(f"key:{k}")

    def mouse_down(self, button: MouseButton) -> None:
        tag = f"mouse:{MouseButton(button).value}"
        self.events.append(("down", tag))
        if tag not in self._held:
            self._held.append(tag)

    def mouse_up(self, button: MouseButton) -> None:
        tag = f"mouse:{MouseButton(button).value}"
        self.events.append(("up", tag))
        if tag in self._held:
            self._held.remove(tag)

    def mouse_move_rel(self, dx: int, dy: int) -> None:
        self.events.append(("move", f"{dx},{dy}"))

    def release_all(self) -> None:
        for tag in list(reversed(self._held)):
            kind, name = tag.split(":", 1)
            if kind == "key":
                self.key_up(name)
            else:
                self.mouse_up(MouseButton(name))

    def held(self) -> frozenset[str]:
        return frozenset(self._held)


class FakeWindowManager:
    def __init__(self, windows: list[WindowRef] | None = None, foreground: int | None = None) -> None:
        self.windows = windows or []
        self.fg = foreground
        self.focus_calls: list[int] = []

    def find_windows(self, title_substring=None, process_name=None):
        out = []
        for w in self.windows:
            if title_substring and title_substring.lower() not in w.title.lower():
                continue
            if process_name and (w.process_name or "").lower() != process_name.lower():
                continue
            out.append(w)
        return out

    def foreground(self):
        return next((w for w in self.windows if w.hwnd == self.fg), None)

    def is_foreground(self, hwnd: int) -> bool:
        return self.fg == hwnd

    def focus(self, hwnd: int) -> bool:
        self.focus_calls.append(hwnd)
        self.fg = hwnd
        return True

    def client_rect_screen(self, hwnd: int):
        return Rect(0, 0, 1920, 1080)

    def is_minimized(self, hwnd: int) -> bool:
        return False

    def exists(self, hwnd: int) -> bool:
        return any(w.hwnd == hwnd for w in self.windows)

    def dpi(self, hwnd: int) -> int:
        return 96


class FakeCursor:
    def __init__(self, visible: bool = False) -> None:
        self.visible = visible

    def cursor_visible(self) -> bool:
        return self.visible


class FakeInputMonitor:
    def __init__(self) -> None:
        self.last_ns = 0
        self.owner_events: list[PhysicalInputEvent] = []
        self.allowed: set[str] | None = None

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def last_physical_input_ns(self) -> int:
        return self.last_ns

    def set_owner_key_recording(self, allowed_keys, predicate=None) -> None:
        self.allowed = allowed_keys

    def drain_owner_events(self):
        out, self.owner_events = self.owner_events, []
        return out


class FakeHotkeys:
    def __init__(self) -> None:
        self.registered: dict[str, Callable[[], None]] = {}

    def register(self, combo, callback) -> None:
        self.registered[combo] = callback

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def fire(self, combo: str) -> None:
        self.registered[combo]()


@dataclass
class FakeFrameSource:
    frames: list[np.ndarray] = field(default_factory=list)
    started: bool = False
    max_fps: int = 120

    def start(self, window: WindowRef, on_frame: Callable[[Frame], None]) -> None:
        self.started = True
        for i, img in enumerate(self.frames):
            on_frame(Frame(image=img, t_ns=i * 16_666_667, index=i + 1))

    def stop(self) -> None:
        self.started = False

    def running(self) -> bool:
        return self.started


ROBLOX = WindowRef(hwnd=101, title="Roblox", pid=4242, process_name="RobloxPlayerBeta.exe")
NOTEPAD = WindowRef(hwnd=202, title="Untitled - Notepad", pid=777, process_name="notepad.exe")


def fake_platform(**kw) -> Platform:
    wm = kw.get("windows") or FakeWindowManager([ROBLOX, NOTEPAD], foreground=ROBLOX.hwnd)
    return Platform(
        name="fake",
        windows=wm,
        cursor=kw.get("cursor") or FakeCursor(False),
        new_input_sink=kw.get("new_input_sink") or FakeInputSink,
        new_input_monitor=kw.get("new_input_monitor") or FakeInputMonitor,
        new_hotkeys=kw.get("new_hotkeys") or FakeHotkeys,
        new_frame_source=kw.get("new_frame_source") or FakeFrameSource,
    )
