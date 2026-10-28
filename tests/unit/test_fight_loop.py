"""Fight loop integration against a simulated window (plumbing only - not a claim about real play)."""

from __future__ import annotations

import json
import time

import pytest

from fakes.platform import ROBLOX, FakeWindowManager, fake_platform
from fakes.sim_game import SimFight, SimFrameSource, SimSink
from fixtures.hud_synth import ANCHOR_ROI, DEFAULT_ROIS, draw_anchor, ubg_like
from jarvis.common.config import load_config
from jarvis.fight.events import EventConfig
from jarvis.fight.gameio import GameIO
from jarvis.fight.launcher import GameSession
from jarvis.fight.loop import ArmRefused, FightLoop
from jarvis.fight.perception.layout import (
    AnchorSpec,
    BarSpec,
    ColorRange,
    HudLayout,
    roi_to_pixels,
    save_layout,
)
from jarvis.fight.recorder import read_jsonl
from jarvis.fight.states import FightState

GREEN_R = [ColorRange(h=(45, 75), s=(100, 255), v=(80, 255))]
YELLOW_R = [ColorRange(h=(20, 35), s=(100, 255), v=(80, 255))]
SIZE = (960, 540)


def make_layout(cal_dir):
    import cv2

    clean = ubg_like(size=SIZE)
    draw_anchor(clean)
    x, y, w, h = roi_to_pixels(ANCHOR_ROI, clean.shape)
    (cal_dir / "anchors").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cal_dir / "anchors" / "hud_anchor.png"), cv2.cvtColor(clean[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY))
    lay = HudLayout(
        frame_size=SIZE,
        bars={
            "own_hp": BarSpec(roi=DEFAULT_ROIS["own_hp"], fill=GREEN_R, required=True),
            "opp_hp": BarSpec(roi=DEFAULT_ROIS["opp_hp"], fill=GREEN_R, required=True),
            "own_stamina": BarSpec(roi=DEFAULT_ROIS["own_stamina"], fill=YELLOW_R),
        },
        anchors=[AnchorSpec(name="hud_anchor", roi=ANCHOR_ROI, template_file="anchors/hud_anchor.png")],
        cursor_closes_gate=True,
    )
    save_layout(lay, cal_dir)
    return lay


@pytest.fixture
def rig(data_dir):
    cfg = load_config(environ={"JARVIS__FIGHT__RECORDING__ENCODER": "libx264"})
    cal = cfg.paths().ensure().calibration
    make_layout(cal)
    game = SimFight(size=SIZE)
    plat = fake_platform(
        windows=FakeWindowManager([ROBLOX], foreground=ROBLOX.hwnd),
        new_input_sink=lambda forbidden_keys=(): SimSink(game, forbidden_keys),
        new_frame_source=lambda **kw: SimFrameSource(game, **kw),
    )
    io = GameIO(cfg, plat)
    io.attach()
    io.start()
    loop = FightLoop(cfg, io, event_cfg=EventConfig(start_stable_ms=200, end_absent_ms=600), seed=7, fixed_tactic="baseline")
    yield cfg, io, loop, game
    loop.stop()
    io.stop()


def test_arm_refused_outside_private_server(rig):
    _cfg, io, loop, _game = rig
    assert io.session.server_kind == "unknown"
    with pytest.raises(ArmRefused, match="private server"):
        loop.arm()
    assert io.sm.state == FightState.DISARMED


def test_full_match_flow(rig):
    cfg, io, loop, game = rig
    io.session = GameSession(io.session.pid, io.session.hwnd, "private", True, time.time())
    callouts = []
    loop.on_callout = lambda k, d: callouts.append(k)
    telemetry = []
    loop.on_telemetry = telemetry.append
    loop.start()
    time.sleep(0.2)
    loop.arm(matches=1)
    deadline = time.monotonic() + 40
    states = set()
    while time.monotonic() < deadline and not (io.sm.state == FightState.DISARMED and "match_end" in callouts):
        states.add(io.sm.state)
        time.sleep(0.05)
    assert FightState.FIGHTING in states, callouts
    assert "match_start" in callouts and "match_end" in callouts, callouts
    assert io.sm.state == FightState.DISARMED
    assert game.stocks_won == 3
    assert io.sink.held() == frozenset()
    # Only fight inputs ever reached the "game", never a forbidden key.
    assert set(game.presses) <= {"mouse:left", "mouse:right", "key:f", "key:w", "key:s", "key:a", "key:d", "key:space"}
    sessions = sorted((cfg.paths().recordings).glob("*_fight*"))
    assert sessions
    s = sessions[-1]
    end = time.monotonic() + 30  # the proof bundle finishes encoding after the state flips
    while not (s / "summary.json").exists() and time.monotonic() < end:
        time.sleep(0.05)
    summary = json.loads((s / "summary.json").read_text())
    assert summary["stocks_won"] >= 2 and summary["inputs"]["sent"] > 0
    assert summary["policy"]["decisions"]
    ev_types = {e["type"] for e in read_jsonl(s / "events.jsonl")}
    assert {"match_start", "hit_dealt", "stock_won"} <= ev_types
    inputs = read_jsonl(s / "inputs.jsonl")
    assert inputs and all(r["state"] in ("fighting", "post_match", "disarmed", "armed") for r in inputs if r.get("action"))
    assert telemetry and "threat" in telemetry[-1]


def test_panic_disarms_mid_fight(rig):
    _cfg, io, loop, game = rig
    io.session = GameSession(io.session.pid, io.session.hwnd, "private", True, time.time())
    loop.start()
    time.sleep(0.2)
    loop.arm()
    deadline = time.monotonic() + 15
    while io.sm.state != FightState.FIGHTING and time.monotonic() < deadline:
        time.sleep(0.02)
    assert io.sm.state == FightState.FIGHTING
    time.sleep(1.0)
    n_before = len(game.presses)
    io.panic("test panic")
    assert io.sm.state == FightState.DISARMED and io.sink.held() == frozenset()
    time.sleep(0.5)
    assert len(game.presses) == n_before  # nothing pressed after the panic
