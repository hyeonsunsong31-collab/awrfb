"""M1 pipeline tests against a simulated window: calibration wizard, key verification, fingerprint."""

from __future__ import annotations

import cv2
import pytest

from fakes.platform import ROBLOX, FakeWindowManager, fake_platform
from fakes.sim_game import SimFrameSource, SimGame, SimSink
from fixtures.hud_synth import ANCHOR_ROI, DEFAULT_ROIS
from jarvis.common.config import load_config
from jarvis.fight.calibrate.fingerprint import Fingerprint
from jarvis.fight.calibrate.hud_wizard import calibrate_hud
from jarvis.fight.calibrate.keys import KeyVerifier
from jarvis.fight.calibrate.roi_ui import ScriptedRoiPicker
from jarvis.fight.gameio import GameIO
from jarvis.fight.perception.hud_bars import HudReader
from jarvis.fight.perception.layout import load_layout, roi_to_pixels
from jarvis.fight.states import FightState

SIZE = (960, 540)


def boxes() -> dict:
    shape = (SIZE[1], SIZE[0], 3)
    out = {name: roi_to_pixels(DEFAULT_ROIS[name], shape) for name in ("own_hp", "opp_hp", "own_stamina", "own_guard")}
    out["opp_stamina"] = None  # optional bar the owner skipped
    out["anchor"] = roi_to_pixels(ANCHOR_ROI, shape)
    return out


@pytest.fixture
def rig(data_dir):
    cfg = load_config(environ={})
    game = SimGame(size=SIZE)
    plat = fake_platform(
        windows=FakeWindowManager([ROBLOX], foreground=ROBLOX.hwnd),
        new_input_sink=lambda forbidden_keys=(): SimSink(game, forbidden_keys),
        new_frame_source=lambda **kw: SimFrameSource(game, **kw),
    )
    io = GameIO(cfg, plat)
    io.attach()
    io.start()
    yield cfg, io, game
    io.stop()


def test_hud_wizard_end_to_end(rig):
    cfg, io, _game = rig
    cal = cfg.paths().ensure().calibration
    picker = ScriptedRoiPicker(boxes())
    res = calibrate_hud(lambda: io.frames.get_newer(0, 1.0), picker, cal, cursor_visible_in_fight=False, verify_seconds=0.5)
    assert res.live_combat_fraction > 0.9, res.warnings
    assert set(res.layout.bars) == {"own_hp", "opp_hp", "own_stamina", "own_guard"}
    assert res.layout.cursor_closes_gate is True
    assert res.layout.anchors and (cal / "anchors" / "hud_anchor.png").exists()
    loaded = load_layout(cal)
    assert loaded == res.layout
    reader = HudReader(loaded, calibration_dir=cal)
    rd = reader.read(io.frames.get_newer(0, 1.0).image)
    assert rd.combat_hud and rd.values["own_hp"] == pytest.approx(1.0, abs=0.03)
    assert cv2.imread(str(res.snapshot)) is not None


def test_key_verifier_confirms_real_reactions(rig):
    _cfg, io, _game = rig
    events, frames = [], []
    kv = KeyVerifier(io, None, log_event=events.append, add_frame=lambda img, row: frames.append(row),
                     watch_ms=400, cooldown_ms=350)
    reports = kv.run(["block", "light", "dash"], repeats=3)
    assert io.sm.state == FightState.DISARMED
    for r in reports:
        assert r.verified, r.to_dict()
        assert r.median_latency_ms is not None and r.median_latency_ms < 200
    assert len([e for e in events if e["type"] == "calibration_press"]) == 9
    assert frames and all(f["state"] in ("calibrating", "disarmed") for f in frames)
    assert io.sink.held() == frozenset()


def test_key_verifier_detects_ignored_key(data_dir):
    cfg = load_config(environ={})
    game = SimGame(react={"key:f"}, size=SIZE)  # the "game" ignores everything but block
    plat = fake_platform(
        windows=FakeWindowManager([ROBLOX], foreground=ROBLOX.hwnd),
        new_input_sink=lambda forbidden_keys=(): SimSink(game, forbidden_keys),
        new_frame_source=lambda **kw: SimFrameSource(game, **kw),
    )
    io = GameIO(cfg, plat)
    io.attach()
    io.start()
    try:
        reports = KeyVerifier(io, None, watch_ms=350, cooldown_ms=300).run(["block", "light"], repeats=3)
    finally:
        io.stop()
    by = {r.action: r for r in reports}
    assert by["block"].verified
    assert not by["light"].verified and by["light"].success_rate == 0.0


def test_verifier_stops_when_focus_lost(rig):
    _cfg, io, _game = rig
    kv = KeyVerifier(io, None, watch_ms=300, cooldown_ms=200)
    io.plat.windows.fg = 999  # owner alt-tabs away mid-calibration
    io.focus_game = lambda: True  # pretend focusing "worked" so we observe the gate doing its job
    reports = kv.run(["light"], repeats=2)
    assert not reports[0].verified
    assert all("foreground" in (tr.denied or "") for tr in reports[0].trials)
    assert io.sink.events == []  # nothing was ever pressed into a non-Roblox window


def test_fingerprint_detects_changes(data_dir):
    from jarvis.fight.keymap import load_keymap
    from jarvis.fight.perception.layout import BarSpec, ColorRange, HudLayout

    lay = HudLayout(frame_size=(1920, 1080), bars={"own_hp": BarSpec(roi=(0, 0, 0.1, 0.01), fill=[ColorRange(h=(40, 80))], required=True)})
    km = load_keymap()
    fp = Fingerprint.make(lay, km)
    assert fp.problems(1920, 1080, lay, km) == []
    assert fp.problems(1280, 720, lay, km) == []  # same aspect, scaled: fine
    assert "aspect" in fp.problems(1600, 1200, lay, km)[0]
    km2 = km.model_copy(deep=True)
    km2.actions["block"].verified = True
    assert any("keymap" in p for p in fp.problems(1920, 1080, lay, km2))
