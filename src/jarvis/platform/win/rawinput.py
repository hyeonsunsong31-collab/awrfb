"""Physical-input detection via Raw Input (RIDEV_INPUTSINK on a message-only window).

Raw Input is delivered asynchronously as WM_INPUT messages, so unlike a low-level hook a slow Python
callback can never add system-wide input lag. Injected input (SendInput, ours or anyone's) arrives
with ``hDevice == NULL``. Some touchpads also report a NULL device, so a NULL-device event only counts
as *ours* if our actuator recently sent an event of the same kind (``SyntheticLog``).
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from collections import deque
from collections.abc import Callable

from jarvis.platform.base import PhysicalInputEvent
from jarvis.platform.win.api import user32
from jarvis.platform.win.msgloop import MessageWindow
from jarvis.platform.win.scancodes import SCAN_TO_NAME
from jarvis.platform.win.structs import (
    RAWINPUT,
    RAWINPUTDEVICE,
    RAWINPUTHEADER,
    RI_KEY_BREAK,
    RI_KEY_E0,
    RI_MOUSE_LEFT_BUTTON_DOWN,
    RI_MOUSE_LEFT_BUTTON_UP,
    RI_MOUSE_MIDDLE_BUTTON_DOWN,
    RI_MOUSE_MIDDLE_BUTTON_UP,
    RI_MOUSE_RIGHT_BUTTON_DOWN,
    RI_MOUSE_RIGHT_BUTTON_UP,
    RID_INPUT,
    RIDEV_INPUTSINK,
    RIM_TYPEKEYBOARD,
    RIM_TYPEMOUSE,
    UINT,
    WM_INPUT,
)

log = logging.getLogger(__name__)

_BUTTONS = (
    (RI_MOUSE_LEFT_BUTTON_DOWN, "mouse:left", True),
    (RI_MOUSE_LEFT_BUTTON_UP, "mouse:left", False),
    (RI_MOUSE_RIGHT_BUTTON_DOWN, "mouse:right", True),
    (RI_MOUSE_RIGHT_BUTTON_UP, "mouse:right", False),
    (RI_MOUSE_MIDDLE_BUTTON_DOWN, "mouse:middle", True),
    (RI_MOUSE_MIDDLE_BUTTON_UP, "mouse:middle", False),
)


class SyntheticLog:
    """Timestamps of the last events our own actuator injected, per kind."""

    def __init__(self) -> None:
        self._last: dict[str, int] = {}
        self._lock = threading.Lock()

    def note(self, kind: str) -> None:
        with self._lock:
            self._last[kind] = time.perf_counter_ns()

    def recently(self, kind: str, window_ms: float = 120.0) -> bool:
        with self._lock:
            t = self._last.get(kind)
        return t is not None and (time.perf_counter_ns() - t) <= window_ms * 1e6


class RawInputMonitor:
    MOVE_THRESHOLD_COUNTS = 40  # ignore tiny accidental mouse nudges
    MOVE_WINDOW_MS = 120

    def __init__(self, synthetic: SyntheticLog | None = None) -> None:
        self._win = MessageWindow("JarvisRawInput")
        self._win.on(WM_INPUT, self._on_input)
        self._win.on_start(self._register)
        self._synthetic = synthetic or SyntheticLog()
        self._last_physical_ns = 0
        self._lock = threading.Lock()
        self._owner_allowed: set[str] | None = None
        self._owner_predicate: Callable[[], bool] | None = None
        self._owner_events: deque[PhysicalInputEvent] = deque(maxlen=20000)
        self._key_down: set[str] = set()
        self._move_acc = 0
        self._move_t0 = 0

    @property
    def synthetic(self) -> SyntheticLog:
        return self._synthetic

    # ------------------------------------------------------------------ lifecycle
    def _register(self, hwnd: int) -> None:
        devs = (RAWINPUTDEVICE * 2)(
            RAWINPUTDEVICE(0x01, 0x06, RIDEV_INPUTSINK, hwnd),  # keyboard
            RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, hwnd),  # mouse
        )
        if not user32().RegisterRawInputDevices(devs, 2, ctypes.sizeof(RAWINPUTDEVICE)):
            log.error("RegisterRawInputDevices failed; physical takeover detection disabled")

    def start(self) -> None:
        self._win.start()

    def stop(self) -> None:
        self._win.stop()

    # ------------------------------------------------------------------ API
    def last_physical_input_ns(self) -> int:
        with self._lock:
            return self._last_physical_ns

    def set_owner_key_recording(
        self, allowed_keys: set[str] | None, predicate: Callable[[], bool] | None = None
    ) -> None:
        with self._lock:
            self._owner_allowed = set(allowed_keys) if allowed_keys else None
            self._owner_predicate = predicate

    def drain_owner_events(self) -> list[PhysicalInputEvent]:
        with self._lock:
            out = list(self._owner_events)
            self._owner_events.clear()
        return out

    # ------------------------------------------------------------------ WM_INPUT
    def _read(self, lparam: int) -> RAWINPUT | None:
        u = user32()
        size = UINT(0)
        hdr = ctypes.sizeof(RAWINPUTHEADER)
        u.GetRawInputData(lparam, RID_INPUT, None, ctypes.byref(size), hdr)
        if size.value == 0 or size.value > 4096:
            return None
        buf = ctypes.create_string_buffer(size.value)
        got = u.GetRawInputData(lparam, RID_INPUT, buf, ctypes.byref(size), hdr)
        if got != size.value:
            return None
        # Keyboard packets are smaller than sizeof(RAWINPUT); copy into a zero-padded struct so we
        # never alias a temporary buffer or read past its end.
        ri = RAWINPUT()
        ctypes.memmove(ctypes.byref(ri), buf, min(size.value, ctypes.sizeof(RAWINPUT)))
        return ri

    def _physical(self, kind: str, injected_device: bool) -> bool:
        if not injected_device:
            return True
        # NULL device: ours (or another injector) unless we sent nothing of this kind just now.
        return not self._synthetic.recently(kind)

    def _on_input(self, _msg: int, _wparam: int, lparam: int) -> int | None:
        ri = self._read(lparam)
        if ri is None:
            return None
        now = time.perf_counter_ns()
        null_dev = not ri.header.hDevice
        if ri.header.dwType == RIM_TYPEKEYBOARD:
            kb = ri.data.keyboard
            if not self._physical("key", null_dev):
                return None
            down = not (kb.Flags & RI_KEY_BREAK)
            name = SCAN_TO_NAME.get((int(kb.MakeCode), bool(kb.Flags & RI_KEY_E0)))
            with self._lock:
                self._last_physical_ns = now
                if name and self._should_record(name):
                    transition = (name not in self._key_down) if down else (name in self._key_down)
                    if transition:
                        self._owner_events.append(PhysicalInputEvent(now, "key", name, down))
                if name:
                    if down:
                        self._key_down.add(name)
                    else:
                        self._key_down.discard(name)
        elif ri.header.dwType == RIM_TYPEMOUSE:
            m = ri.data.mouse
            flags = int(m.b.usButtonFlags)
            if flags:
                if not self._physical("mouse_button", null_dev):
                    return None
                with self._lock:
                    self._last_physical_ns = now
                    for bit, name, down in _BUTTONS:
                        if flags & bit and self._should_record(name):
                            self._owner_events.append(PhysicalInputEvent(now, "mouse_button", name, down))
            elif m.lLastX or m.lLastY:
                if not self._physical("mouse_move", null_dev):
                    return None
                with self._lock:
                    if (now - self._move_t0) > self.MOVE_WINDOW_MS * 1e6:
                        self._move_t0 = now
                        self._move_acc = 0
                    self._move_acc += abs(int(m.lLastX)) + abs(int(m.lLastY))
                    if self._move_acc >= self.MOVE_THRESHOLD_COUNTS:
                        self._last_physical_ns = now
        return None

    def _should_record(self, name: str) -> bool:
        if not self._owner_allowed or name not in self._owner_allowed:
            return False
        pred = self._owner_predicate
        try:
            return pred() if pred else True
        except Exception:
            return False
