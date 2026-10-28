"""Shadow mode: the OWNER fights; JARVIS watches, records, and logs what it WOULD have done.

No input is ever sent (the fight state stays DISARMED, so the actuator refuses everything anyway).
With the owner's opt-in, the owner's own allowlisted game keys are recorded - only while UBG is the
foreground window and the chat box is closed - giving labelled examples of real play.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from jarvis.common.config import JarvisConfig
from jarvis.fight.gameio import GameIO
from jarvis.fight.perception.layout import load_layout
from jarvis.fight.perception.pipeline import Perception
from jarvis.fight.policy.params import load_params
from jarvis.fight.policy.reflex_v0 import ReflexPolicy
from jarvis.fight.recorder import SessionRecorder
from jarvis.fight.replay import VirtualKeys

log = logging.getLogger(__name__)


def run_shadow(cfg: JarvisConfig, io: GameIO, *, minutes: float, record_owner_keys: bool, detector=None) -> Path:
    paths = cfg.paths().ensure()
    layout = load_layout(paths.calibration)
    if layout is None:
        raise RuntimeError("calibrate the HUD first (`jarvis calibrate hud`)")
    params = load_params(cfg.fight.policy_file, paths.calibration)
    perc = Perception(cfg.fight.perception, layout, calibration_dir=paths.calibration, detector=detector,
                      h_close=params.spacing.h_close, h_far=params.spacing.h_far)
    pol = ReflexPolicy(params, io.keymap, seed=0)
    vk = VirtualKeys(io.keymap)
    rc = cfg.fight.recording
    rec = SessionRecorder(paths.recordings, "shadow", width=rc.width, height=rc.height, ffmpeg=rc.ffmpeg,
                          encoder=rc.encoder, meta={"tier": cfg.tier, "record_owner_keys": record_owner_keys})
    chat_open = {"v": False}
    if record_owner_keys and io._monitor is not None:
        # Names as the raw-input monitor reports them: scan-code names ("f", "space") and "mouse:left".
        allowed = io.keymap.all_physical() | {"mouse:left", "mouse:right"}
        io._monitor.set_owner_key_recording(
            allowed,
            predicate=lambda: io.window is not None and io.plat.windows.is_foreground(io.window.hwnd) and not chat_open["v"],
        )
    last = 0
    started = False
    end = time.monotonic() + minutes * 60
    try:
        while time.monotonic() < end:
            f = io.frames.get_newer(last, 0.5)
            if f is None:
                continue
            last = f.index
            obs = perc.process(f)
            chat_open["v"] = obs.ui.chat_open
            if obs.hud.combat_hud and not started:
                pol.reset(f.t_ns)
                started = True
            would = pol.step(obs, frozenset(vk.held)) if obs.hud.combat_hud else []
            vk.apply(would)
            row = obs.row()
            row["state"] = "shadow"
            if would:
                row["would"] = [[c.action, c.mode, c.reason] for c in would]
            rec.add_frame(f.image, row)
            for e in obs.events:
                rec.log("events", e.to_row())
            if record_owner_keys and io._monitor is not None:
                for ev in io._monitor.drain_owner_events():
                    rec.log("inputs", {"t_ns": ev.t_ns, "src": "owner", "phys": ev.key, "event": "down" if ev.down else "up", "ok": True})
    finally:
        if io._monitor is not None:
            io._monitor.set_owner_key_recording(None)
    return rec.close({"kind": "shadow", "minutes": minutes, "policy": pol.summary()})
