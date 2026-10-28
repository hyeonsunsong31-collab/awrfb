"""Multi-object tracking with constant-velocity Kalman filters (state: cx, cy, w, h + velocities).

The detector runs every N frames; between detections tracks are predicted forward, so the fight
loop always has smooth positions and velocities at full frame rate.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from jarvis.fight.perception.detector import Detection, iou


@dataclass
class Track:
    id: int
    cx: float
    cy: float
    w: float
    h: float
    vx: float  # per second, normalized units
    vy: float
    vh: float  # growth rate of height (approaching = positive in third person)
    score: float
    age: int
    misses: int

    @property
    def det(self) -> Detection:
        return Detection(self.cx - self.w / 2, self.cy - self.h / 2, self.w, self.h, self.score)


class _KF:
    def __init__(self, d: Detection, t_ns: int) -> None:
        self.x = np.array([d.cx, d.cy, d.w, d.h, 0, 0, 0, 0], dtype=np.float64)
        self.P = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1, 1e-1])
        self.t = t_ns

    def predict(self, t_ns: int) -> None:
        dt = max(0.0, (t_ns - self.t) / 1e9)
        self.t = t_ns
        F = np.eye(8)
        for i in range(4):
            F[i, i + 4] = dt
        q = 0.5 * dt
        Q = np.diag([q * 1e-3] * 4 + [q * 5e-2] * 4)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, d: Detection) -> None:
        H = np.zeros((4, 8))
        H[:4, :4] = np.eye(4)
        R = np.diag([2e-4, 2e-4, 5e-4, 5e-4])
        z = np.array([d.cx, d.cy, d.w, d.h])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P


class Tracker:
    def __init__(self, iou_threshold: float = 0.25, max_misses: int = 20) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._kf: dict[int, _KF] = {}
        self._meta: dict[int, dict] = {}
        self._ids = itertools.count(1)

    def predict(self, t_ns: int) -> list[Track]:
        for kf in self._kf.values():
            kf.predict(t_ns)
        return self.tracks()

    def update(self, dets: list[Detection], t_ns: int) -> list[Track]:
        self.predict(t_ns)
        ids = list(self._kf)
        pairs = []
        for tid in ids:
            x = self._kf[tid].x
            td = Detection(x[0] - x[2] / 2, x[1] - x[3] / 2, x[2], x[3], 1.0)
            for j, d in enumerate(dets):
                pairs.append((iou(td, d), tid, j))
        pairs.sort(reverse=True)
        used_t: set[int] = set()
        used_d: set[int] = set()
        for score, tid, j in pairs:
            if score < self.iou_threshold or tid in used_t or j in used_d:
                continue
            self._kf[tid].update(dets[j])
            m = self._meta[tid]
            m["misses"] = 0
            m["age"] += 1
            m["score"] = 0.7 * m["score"] + 0.3 * dets[j].score
            used_t.add(tid)
            used_d.add(j)
        for tid in ids:
            if tid not in used_t:
                self._meta[tid]["misses"] += 1
                if self._meta[tid]["misses"] > self.max_misses:
                    del self._kf[tid]
                    del self._meta[tid]
        for j, d in enumerate(dets):
            if j not in used_d:
                tid = next(self._ids)
                self._kf[tid] = _KF(d, t_ns)
                self._meta[tid] = {"age": 1, "misses": 0, "score": d.score}
        return self.tracks()

    def tracks(self) -> list[Track]:
        out = []
        for tid, kf in self._kf.items():
            x = kf.x
            m = self._meta[tid]
            out.append(Track(tid, x[0], x[1], max(1e-4, x[2]), max(1e-4, x[3]), x[4], x[5], x[7], m["score"], m["age"], m["misses"]))
        return out
