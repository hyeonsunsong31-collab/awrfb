"""Platform abstractions. Windows implementations live in ``jarvis.platform.win``; test doubles live
only in ``tests/fakes`` and are refused at runtime unless ``JARVIS_TEST=1``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np


class PlatformUnavailable(RuntimeError):
    """Raised when a Windows-only capability is requested on another OS (outside tests)."""


@dataclass(frozen=True)
class WindowRef:
    hwnd: int
    title: str
    pid: int
    process_name: str | None = None


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class MouseButton(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class Frame:
    """A captured frame. ``image`` is an owned BGR uint8 copy (safe to keep after the callback)."""

    image: np.ndarray
    t_ns: int
    index: int
    source_t_ns: int | None = None  # the capture API's own timestamp, if any

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@runtime_checkable
class InputSink(Protocol):
    """Low-level OS input. Key names come from ``jarvis.platform.win.scancodes``."""

    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def mouse_down(self, button: MouseButton) -> None: ...
    def mouse_up(self, button: MouseButton) -> None: ...
    def mouse_move_rel(self, dx: int, dy: int) -> None: ...
    def release_all(self) -> None: ...
    def held(self) -> frozenset[str]: ...


@runtime_checkable
class WindowManager(Protocol):
    def find_windows(
        self, title_substring: str | None = None, process_name: str | None = None
    ) -> list[WindowRef]: ...
    def foreground(self) -> WindowRef | None: ...
    def is_foreground(self, hwnd: int) -> bool: ...
    def focus(self, hwnd: int) -> bool: ...
    def client_rect_screen(self, hwnd: int) -> Rect | None: ...
    def is_minimized(self, hwnd: int) -> bool: ...
    def exists(self, hwnd: int) -> bool: ...
    def dpi(self, hwnd: int) -> int: ...


@runtime_checkable
class CursorProbe(Protocol):
    def cursor_visible(self) -> bool: ...


@dataclass
class PhysicalInputEvent:
    t_ns: int
    kind: str  # "key" | "mouse_button" | "mouse_move"
    key: str | None = None  # scancode name for keys, "mouse:left" etc for buttons
    down: bool = True
    dx: int = 0
    dy: int = 0


@runtime_checkable
class InputMonitor(Protocol):
    """Detects *physical* (non-injected) keyboard/mouse input, e.g. the owner taking over."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def last_physical_input_ns(self) -> int: ...
    def set_owner_key_recording(self, allowed_keys: set[str] | None) -> None: ...
    def drain_owner_events(self) -> list[PhysicalInputEvent]: ...


@runtime_checkable
class HotkeyManager(Protocol):
    def register(self, combo: str, callback: Callable[[], None]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


@runtime_checkable
class FrameSource(Protocol):
    def start(self, window: WindowRef, on_frame: Callable[[Frame], None]) -> None: ...
    def stop(self) -> None: ...
    def running(self) -> bool: ...


@dataclass
class Platform:
    """Bundle of platform services. Constructed by ``jarvis.platform.factory.get_platform``."""

    name: str
    windows: WindowManager
    cursor: CursorProbe
    new_input_sink: Callable[[], InputSink]
    new_input_monitor: Callable[[], InputMonitor]
    new_hotkeys: Callable[[], HotkeyManager]
    new_frame_source: Callable[..., FrameSource]
    extras: dict[str, object] = field(default_factory=dict)
