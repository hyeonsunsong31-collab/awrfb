"""Frame -> Observation. Every stage is timed; the heavy detector runs every N frames and the tracker
predicts in between, so the fight loop processes every frame at capture rate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from jarvis.common.config import PerceptionConfig
from jarvis.fight.events import EventConfig, FightEvent, HudEventTracker
from jarvis.fight.perception.detector import NullDetector
from jarvis.fight.perception.hud_bars import HudReader, HudReading
from jarvis.fight.perception.layout import HudLayout
from jarvis.fight.perception.queue_box import QueueStatus, find_boxes
from jarvis.fight.perception.self_id import SelfIdentifier
from jarvis.fight.perception.threat import HeuristicThreat, ThreatModel, gap_from_tracks
from jarvis.fight.perception.tracker import Track, Tracker
from jarvis.fight.perception.ui_guard import UiGuard, UiGuardResult
from jarvis.platform.base import Frame


@dataclass
class Observation:
    t_ns: int
    index: int
    hud: HudReading
    ui: UiGuardResult
    events: list[FightEvent] = field(default_factory=list)
    me: Track | None = None
    opp: Track | None = None
    id_conf: float = 0.0
    gap: float | None = None
    threat: dict[int, float] = field(default_factory=dict)
    threat_source: str = "none"
    opp_motion_z: float = 0.0
    approach: float = 0.0
    queue: QueueStatus | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    def row(self) -> dict:
        """Compact frames.jsonl row (what the recorder stores per frame)."""
        def tr(t: Track | None):
            return None if t is None else [round(t.cx, 4), round(t.cy, 4), round(t.w, 4), round(t.h, 4), t.id]

        return {
            "t_ns": self.t_ns,
            "fi": self.index,
            "hud": {k: (None if v is None else round(v, 4)) for k, v in self.hud.values.items()},
            "hud_conf": round(self.hud.hud_conf, 3),
            "combat_hud": self.hud.combat_hud,
            "popup": self.ui.popup,
            "chat": self.ui.chat_open,
            "me": tr(self.me),
            "opp": tr(self.opp),
            "gap": None if self.gap is None else round(self.gap, 3),
            "threat": {str(k): round(v, 3) for k, v in self.threat.items()},
            "motion_z": round(self.opp_motion_z, 2),
            "ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }


class Perception:
    def __init__(
        self,
        cfg: PerceptionConfig,
        layout: HudLayout,
        *,
        calibration_dir: Path | None = None,
        detector=None,
        threat_model: ThreatModel | None = None,
        h_close: float = 0.55,
        h_far: float = 0.18,
        event_cfg: EventConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.layout = layout
        self.hud = HudReader(layout, calibration_dir=calibration_dir)
        self.ui = UiGuard(layout, calibration_dir / "popups" if calibration_dir else None)
        self.events = HudEventTracker(event_cfg)
        self.detector = detector or NullDetector()
        self.tracker = Tracker()
        self.ident = SelfIdentifier(layout.camera_mode)
        self.threat = threat_model or HeuristicThreat()
        self.h_close = h_close
        self.h_far = h_far
        self._n = 0

    def process(self, frame: Frame, *, want_queue: bool = False) -> Observation:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        img = frame.image
        hud = self.hud.read(img, frame.t_ns)
        t1 = time.perf_counter()
        timings["hud"] = (t1 - t0) * 1e3
        ui = self.ui.check(img, hud.combat_hud)
        t2 = time.perf_counter()
        timings["ui"] = (t2 - t1) * 1e3
        events = self.events.update(hud)
        self._n += 1
        if self._n % max(1, self.cfg.detector_every_n_frames) == 0:
            dets = self.detector.detect(img)
            tracks = self.tracker.update(dets, frame.t_ns)
        else:
            tracks = self.tracker.predict(frame.t_ns)
        t3 = time.perf_counter()
        timings["detect"] = (t3 - t2) * 1e3
        ident = self.ident.update(tracks, frame.t_ns)
        gap = gap_from_tracks(ident.me, ident.opp, self.layout.camera_mode, self.h_close, self.h_far)
        th = self.threat.update(img, ident.opp, gap, frame.t_ns)
        t4 = time.perf_counter()
        timings["threat"] = (t4 - t3) * 1e3
        queue = find_boxes(img, self.layout.queue_box) if want_queue else None
        timings["total"] = (time.perf_counter() - t0) * 1e3
        return Observation(
            t_ns=frame.t_ns,
            index=frame.index,
            hud=hud,
            ui=ui,
            events=events,
            me=ident.me,
            opp=ident.opp,
            id_conf=ident.conf,
            gap=gap,
            threat=th.p,
            threat_source=th.source,
            opp_motion_z=th.motion_z,
            approach=th.approach,
            queue=queue,
            timings_ms=timings,
        )
