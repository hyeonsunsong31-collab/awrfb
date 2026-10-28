from __future__ import annotations

import pytest

from jarvis.platform.base import MouseButton
from jarvis.platform.win import structs as s
from jarvis.platform.win.scancodes import NEVER_SEND_GAME
from jarvis.platform.win.sendinput import ForbiddenKey, InputBlocked, SendInputSink, emergency_release


class Recorder:
    def __init__(self, ok: bool = True) -> None:
        self.calls: list[list[tuple]] = []
        self.ok = ok

    def __call__(self, n, arr, size):
        assert size == 40
        batch = []
        for i in range(n):
            inp = arr[i]
            if inp.type == s.INPUT_KEYBOARD:
                batch.append(("k", inp.ki.wScan, inp.ki.dwFlags))
            else:
                batch.append(("m", inp.mi.dwFlags, inp.mi.dx, inp.mi.dy))
        self.calls.append(batch)
        return n if self.ok else 0


def test_key_down_up_tracks_held():
    rec = Recorder()
    sink = SendInputSink(sender=rec)
    sink.key_down("f")
    assert sink.held() == {"key:f"}
    sink.key_up("f")
    assert sink.held() == frozenset()
    assert rec.calls[0] == [("k", 0x21, s.KEYEVENTF_SCANCODE)]
    assert rec.calls[1] == [("k", 0x21, s.KEYEVENTF_SCANCODE | s.KEYEVENTF_KEYUP)]


def test_forbidden_keys_refused_but_release_allowed():
    rec = Recorder()
    sink = SendInputSink(forbidden_keys=NEVER_SEND_GAME, sender=rec)
    with pytest.raises(ForbiddenKey):
        sink.key_down("esc")
    with pytest.raises(ForbiddenKey):
        sink.key_down("enter")
    sink.key_up("esc")  # releasing is always allowed
    assert rec.calls == [[("k", 0x01, s.KEYEVENTF_SCANCODE | s.KEYEVENTF_KEYUP)]]


def test_release_all_and_mouse():
    rec = Recorder()
    sink = SendInputSink(sender=rec)
    sink.key_down("a")
    sink.mouse_down(MouseButton.RIGHT)
    sink.mouse_move_rel(10, -4)
    sink.release_all()
    assert sink.held() == frozenset()
    flat = [c[0] for c in rec.calls]
    assert ("m", s.MOUSEEVENTF_RIGHTUP, 0, 0) in flat
    assert ("k", 0x1E, s.KEYEVENTF_SCANCODE | s.KEYEVENTF_KEYUP) in flat
    assert ("m", s.MOUSEEVENTF_MOVE, 10, -4) in flat


def test_blocked_input_raises():
    sink = SendInputSink(sender=Recorder(ok=False))
    with pytest.raises(InputBlocked):
        sink.key_down("w")
    assert sink.held() == frozenset()


def test_emergency_release_sends_everything():
    rec = Recorder()
    n = emergency_release(["w", "a", "f", "space"], sender=rec)
    assert n == 4 + 3  # keys + three mouse buttons
