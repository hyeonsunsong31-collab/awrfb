"""Threat: P(our HP or guard drops within h ms) for h in {100, 200, 300, 500}.

v0 heuristic (no training data needed): opponent closeness, sudden motion inside the opponent's box
(an attack wind-up moves a lot of pixels) and approach speed. A learned model trained on the owner's
own recordings - labelled for free by HUD events - replaces it later (jarvis.learn.train_threat).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from jarvis.fight.perception.tracker import Track

HORIZONS_MS = (100, 200, 300, 500)


@dataclass
class ThreatEstimate:
    p: dict[int, float] = field(default_factory=dict)
    motion_z: float = 0.0
    approach: float = 0.0
    source: str = "heuristic"


class ThreatModel(Protocol):
    def update(self, frame_bgr: np.ndarray, opp: Track | None, gap: float | None, t_ns: int) -> ThreatEstimate: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class HeuristicThreat:
    SIZE = (192, 108)

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None
        self._mean = 0.0
        self._var = 1.0
        self._n = 0

    def _region_energy(self, g: np.ndarray, opp: Track | None) -> float:
        if self._prev is None:
            return 0.0
        diff = np.abs(g - self._prev)
        if opp is None:
            H, W = g.shape
            return float(diff[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4].mean())
        H, W = g.shape
        x0 = int(max(0, (opp.cx - opp.w / 2) * W))
        x1 = int(min(W, (opp.cx + opp.w / 2) * W))
        y0 = int(max(0, (opp.cy - opp.h / 2) * H))
        y1 = int(min(H, (opp.cy + opp.h / 2) * H))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return 0.0
        return float(diff[y0:y1, x0:x1].mean())

    def update(self, frame_bgr: np.ndarray, opp: Track | None, gap: float | None, t_ns: int) -> ThreatEstimate:
        g = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), self.SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
        e = self._region_energy(g, opp)
        self._prev = g
        # running mean/var (EMA) of the opponent-region motion
        self._n += 1
        a = 0.05 if self._n > 20 else 1.0 / self._n
        d = e - self._mean
        self._mean += a * d
        self._var = (1 - a) * self._var + a * d * d
        z = d / math.sqrt(max(self._var, 1e-3))
        approach = float(opp.vh) if opp is not None else 0.0
        closeness = 1.0 - (gap if gap is not None else 0.6)
        logit = -3.2 + 2.6 * closeness + 0.7 * max(-2.0, min(4.0, z)) + 0.8 * max(-1.0, min(2.0, approach * 6))
        p = {h: _sigmoid(logit + math.log(h / 200.0)) for h in HORIZONS_MS}
        return ThreatEstimate(p, float(z), approach)


def gap_from_tracks(me: Track | None, opp: Track | None, camera_mode: str, h_close: float, h_far: float) -> float | None:
    """0 = touching, 1 = far. Follow cam: from the opponent's on-screen height. Side cam: x distance."""
    if opp is None:
        return None
    if camera_mode == "side" and me is not None:
        dist = abs(opp.cx - me.cx) - (opp.w + me.w) / 2
        return float(np.clip(dist / 0.5, 0.0, 1.0))
    # Follow camera: the opponent looks taller the closer they are.
    closeness = float(np.clip((opp.h - h_far) / max(1e-3, h_close - h_far), 0.0, 1.0))
    return 1.0 - closeness
