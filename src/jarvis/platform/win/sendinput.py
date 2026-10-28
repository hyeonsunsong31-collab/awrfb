"""Real OS input through Win32 ``SendInput`` (scan codes, so DirectX-style games read them).

This is the only place in the codebase that emits keyboard/mouse events. It tracks what it holds so
``release_all`` can always leave the keyboard clean, and it refuses forbidden keys at the lowest level.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Iterable

from jarvis.platform.base import MouseButton
from jarvis.platform.win.scancodes import SCANCODES, normalize_key
from jarvis.platform.win.structs import (
    INPUT,
    MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_MIDDLEDOWN,
    MOUSEEVENTF_MIDDLEUP,
    MOUSEEVENTF_MOVE,
    MOUSEEVENTF_RIGHTDOWN,
    MOUSEEVENTF_RIGHTUP,
    key_input,
    mouse_input,
)

_BUTTON_FLAGS = {
    MouseButton.LEFT: (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    MouseButton.RIGHT: (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    MouseButton.MIDDLE: (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}


class InputBlocked(RuntimeError):
    """SendInput inserted nothing (e.g. UIPI: the target window runs elevated)."""


class ForbiddenKey(ValueError):
    pass


class SendInputSink:
    def __init__(self, forbidden_keys: Iterable[str] = (), sender=None) -> None:
        self._forbidden = frozenset(forbidden_keys)
        self._held: list[str] = []
        self._lock = threading.RLock()
        self._sender = sender  # injectable for tests; defaults to user32.SendInput

    # ---------------------------------------------------------------- plumbing
    def _send(self, *inputs: INPUT) -> None:
        arr = (INPUT * len(inputs))(*inputs)
        if self._sender is not None:
            n = self._sender(len(inputs), arr, ctypes.sizeof(INPUT))
        else:
            from jarvis.platform.win.api import last_error, user32

            n = user32().SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
            if n != len(inputs):
                raise InputBlocked(f"SendInput inserted {n}/{len(inputs)} events (error {last_error()})")
        if n != len(inputs):
            raise InputBlocked(f"SendInput inserted {n}/{len(inputs)} events")

    def _check(self, key: str) -> str:
        k = normalize_key(key)
        if k in self._forbidden:
            raise ForbiddenKey(f"key {k!r} is forbidden on this input path")
        return k

    # ---------------------------------------------------------------- keys
    def key_down(self, key: str) -> None:
        k = self._check(key)
        sc = SCANCODES[k]
        with self._lock:
            self._send(key_input(sc.code, extended=sc.extended, up=False))
            tag = f"key:{k}"
            if tag not in self._held:
                self._held.append(tag)

    def key_up(self, key: str) -> None:
        k = normalize_key(key)  # releases are always allowed, even for forbidden keys
        sc = SCANCODES[k]
        with self._lock:
            self._send(key_input(sc.code, extended=sc.extended, up=True))
            tag = f"key:{k}"
            if tag in self._held:
                self._held.remove(tag)

    # ---------------------------------------------------------------- mouse
    def mouse_down(self, button: MouseButton) -> None:
        down, _ = _BUTTON_FLAGS[MouseButton(button)]
        with self._lock:
            self._send(mouse_input(down))
            tag = f"mouse:{MouseButton(button).value}"
            if tag not in self._held:
                self._held.append(tag)

    def mouse_up(self, button: MouseButton) -> None:
        _, up = _BUTTON_FLAGS[MouseButton(button)]
        with self._lock:
            self._send(mouse_input(up))
            tag = f"mouse:{MouseButton(button).value}"
            if tag in self._held:
                self._held.remove(tag)

    def mouse_move_rel(self, dx: int, dy: int) -> None:
        with self._lock:
            self._send(mouse_input(MOUSEEVENTF_MOVE, int(dx), int(dy)))

    # ---------------------------------------------------------------- safety
    def release_all(self) -> None:
        with self._lock:
            for tag in list(reversed(self._held)):
                kind, name = tag.split(":", 1)
                try:
                    if kind == "key":
                        self.key_up(name)
                    else:
                        self.mouse_up(MouseButton(name))
                except Exception:
                    # Never let one failure keep other keys held.
                    if tag in self._held:
                        self._held.remove(tag)
            self._held.clear()

    def held(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._held)


def emergency_release(keys: Iterable[str], *, mouse: bool = True, sender=None) -> int:
    """Send key-up for every listed key and mouse-up for every button, tracked or not.

    Used by the supervisor when fightd dies and by the panic hotkey. Returns events sent.
    """
    sink = SendInputSink(sender=sender)
    sent = 0
    for k in keys:
        try:
            sink.key_up(k)
            sent += 1
        except (KeyError, InputBlocked):
            continue
    if mouse:
        for b in MouseButton:
            try:
                sink.mouse_up(b)
                sent += 1
            except InputBlocked:
                continue
    return sent
