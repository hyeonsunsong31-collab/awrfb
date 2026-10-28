"""The actuator is the only path from fight decisions to OS input - test every safety property."""

from __future__ import annotations

import pytest

from fakes.platform import ROBLOX, FakeInputSink, FakeWindowManager
from jarvis.common.clock import ManualClock
from jarvis.common.config import ActuatorConfig
from jarvis.fight.actuator import Actuator, Command
from jarvis.fight.gate import GateStatus
from jarvis.fight.keymap import Keymap, load_keymap
from jarvis.fight.states import FightState

MS = 1_000_000


@pytest.fixture
def rig():
    clock = ManualClock(1_000 * MS)
    sink = FakeInputSink()
    wm = FakeWindowManager([ROBLOX], foreground=ROBLOX.hwnd)
    km = load_keymap()
    log: list[dict] = []
    fired: list[str] = []
    act = Actuator(
        sink,
        km,
        wm,
        ActuatorConfig(),
        now_ns=clock.now_ns,
        input_log=log.append,
        on_watchdog=fired.append,
    )
    act.set_target(ROBLOX)
    return act, sink, wm, clock, log, fired


def open_gate(act: Actuator, clock: ManualClock, **kw) -> None:
    g = dict(t_ns=clock.now_ns(), foreground=True, combat_hud=True, hud_conf=0.9, cursor_visible=False)
    g.update(kw)
    act.update_gate(GateStatus(**g))


def run_until(act: Actuator, clock: ManualClock, ms: float) -> None:
    end = clock.now_ns() + int(ms * MS)
    while clock.now_ns() < end:
        clock.advance_ms(1)
        act.pump()


def test_disarmed_sends_nothing(rig):
    act, sink, _wm, clock, log, _ = rig
    open_gate(act, clock)
    act.submit([Command("block")])
    run_until(act, clock, 100)
    assert sink.events == []
    assert log and log[0]["ok"] is False and "not allowed in disarmed" in log[0]["why"]


def test_tap_when_fighting(rig):
    act, sink, _wm, clock, _log, _ = rig
    act.set_state(FightState.FIGHTING)
    open_gate(act, clock)
    act.submit([Command("block", dur_ms=40)])
    act.pump()
    assert sink.events == [("down", "key:f")]
    open_gate(act, clock)
    run_until(act, clock, 45)
    assert sink.events == [("down", "key:f"), ("up", "key:f")]


@pytest.mark.parametrize(
    "gate_kw,why",
    [
        ({"popup": "purchase_prompt"}, "popup"),
        ({"combat_hud": False}, "combat HUD"),
        ({"chat_open": True}, "chat"),
        ({"physical_input_recent": True}, "owner"),
        ({"cursor_visible": True}, "cursor"),
        ({"foreground": False}, "foreground"),
    ],
)
def test_gate_blocks_presses(rig, gate_kw, why):
    act, sink, _wm, clock, log, _ = rig
    act.set_state(FightState.FIGHTING)
    open_gate(act, clock, **gate_kw)
    act.submit([Command("light")])
    act.pump()
    assert sink.events == []
    assert why.lower() in log[-1]["why"].lower()


def test_stale_gate_blocks(rig):
    act, sink, _wm, clock, log, _ = rig
    act.set_state(FightState.FIGHTING)
    open_gate(act, clock)
    clock.advance_ms(400)
    act.submit([Command("light")])
    act.pump()
    assert sink.events == []
    assert "stale" in log[-1]["why"]


def test_direct_foreground_check(rig):
    act, sink, wm, clock, log, _ = rig
    act.set_state(FightState.FIGHTING)
    open_gate(act, clock)
    wm.fg = 999  # user alt-tabbed after the frame was analysed
    act.submit([Command("light")])
    act.pump()
    assert sink.events == []
    assert "lost foreground" in log[-1]["why"]


def test_gate_closing_releases_held_keys(rig):
    act, sink, _wm, clock, _log, _ = rig
    act.set_state(FightState.FIGHTING)
    open_gate(act, clock)
    act.submit([Command("block", mode="hold")])
    act.pump()
    assert sink.held() == {"key:f"}
    open_gate(act, clock, popup="shop")
    assert sink.held() == frozenset()
    assert sink.events[-1] == ("up", "key:f")


def test_state_change_releases_and_clears_queue(rig):
    act, sink, _wm, clock, _log, _ = rig
    act.set_state(FightState.FIGHTING)
    open_gate(act, clock)
    act.submit([Command("block", mode="hold"), Command("light", at_ns=clock.now_ns() + 50 * MS)])
    act.pump()
    act.set_state(FightState.POST_MATCH)
    run_until(act, clock, 100)
    assert sink.events == [("down", "key:f"), ("up", "key:f")]


def test_feint_combo(rig):
    act, sink, _wm, clock, _log, _ = rig
    act.set_state(FightState.FIGHTING)
    for _ in range(1):
        open_gate(act, clock)
    act.submit([Command("feint", dur_ms=30)])
    for _ in range(150):
        clock.advance_ms(1)
        open_gate(act, clock)
        act.pump()
    assert sink.events == [
        ("down", "mouse:right"),
        ("up", "mouse:right"),
        ("down", "mouse:left"),
        ("up", "mouse:left"),
    ]


def test_dash_chord(rig):
    act, sink, _wm, clock, _log, _ = rig
    act.set_state(FightState.FIGHTING)
    act.submit([Command("dash_left", dur_ms=30)])
    for _ in range(120):
        open_gate(act, clock)
        act.pump()
        clock.advance_ms(1)
    assert sink.events == [("down", "key:a"), ("down", "key:space"), ("up", "key:space"), ("up", "key:a")]


def test_navigating_allows_only_movement(rig):
    act, sink, _wm, clock, log, _ = rig
    act.set_state(FightState.NAVIGATING)
    open_gate(act, clock, combat_hud=False)
    act.submit([Command("block"), Command("move_forward", mode="hold")])
    act.pump()
    assert sink.events == [("down", "key:w")]
    assert any("not allowed in navigating" in r["why"] for r in log if not r["ok"])


def test_disabled_and_camera_denied(rig):
    act, sink, _wm, clock, log, _ = rig
    act.set_state(FightState.FIGHTING)
    open_gate(act, clock)
    act.submit([Command("dodge_left"), Command("camera", mode="move", dx=10)])
    act.pump()
    assert sink.events == []
    whys = " ".join(r["why"] for r in log)
    assert "disabled" in whys and "camera" in whys


def test_camera_moves_clamped_when_enabled(rig):
    act, sink, _wm, clock, _log, _ = rig
    km = act._km.model_copy(deep=True)
    km.camera.mouse_relative = True
    act._km = km
    act.set_state(FightState.FIGHTING)
    open_gate(act, clock)
    act.submit([Command("camera", mode="move", dx=500, dy=-7)])
    act.pump()
    assert sink.events == [("move", "40,-7")]


def test_watchdog_releases_and_disarms(rig):
    act, sink, _wm, clock, _log, fired = rig
    act.set_state(FightState.FIGHTING)
    open_gate(act, clock)
    act.heartbeat()
    act.submit([Command("block", mode="hold")])
    act.pump()
    clock.advance_ms(300)
    assert act.check_watchdog() is False
    clock.advance_ms(300)
    assert act.check_watchdog() is True
    assert sink.held() == frozenset()
    assert fired and "watchdog" in fired[0]


def test_never_double_press(rig):
    act, sink, _wm, clock, _log, _ = rig
    act.set_state(FightState.FIGHTING)
    open_gate(act, clock)
    act.submit([Command("block", mode="hold"), Command("block", mode="hold")])
    act.pump()
    assert sink.events == [("down", "key:f")]


def test_keymap_rejects_forbidden_binding():
    with pytest.raises(ValueError):
        Keymap.model_validate({"actions": {"menu": {"key": "esc"}}})
    with pytest.raises(ValueError):
        Keymap.model_validate({"actions": {"x": {"combo": ["nope"]}}})
    with pytest.raises(ValueError):
        Keymap.model_validate({"actions": {"x": {"key": "w", "mouse": "left"}}})
