"""Which tracked character is us, and which is the opponent?

Follow camera (UBG default): our character stays near a fixed screen spot (learned online, seeded at
bottom-centre) while the world - including the opponent - moves around it.
Side camera: both fighters move on screen; ours is the one whose horizontal motion follows our own
movement commands.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from jarvis.fight.perception.tracker import Track


@dataclass
class Identity:
    me: Track | None
    opp: Track | None
    conf: float


class SelfIdentifier:
    def __init__(self, camera_mode: str = "default", seed_xy: tuple[float, float] = (0.5, 0.62)) -> None:
        self.camera_mode = camera_mode
        self.home = np.array(seed_xy, dtype=np.float64)
        self._intent: deque[tuple[int, float]] = deque(maxlen=120)  # (t_ns, -1 left / +1 right / 0)
        self._vel: dict[int, deque[tuple[int, float]]] = {}
        self._me_id: int | None = None

    def note_intent(self, t_ns: int, horizontal: float) -> None:
        self._intent.append((t_ns, horizontal))

    def _side_score(self, tr: Track) -> float:
        hist = self._vel.get(tr.id)
        if not hist or len(self._intent) < 10:
            return 0.0
        # Correlate the track's horizontal velocity with our recent horizontal intent.
        iv = np.array([i for _, i in list(self._intent)[-len(hist):]])
        tv = np.array([v for _, v in hist])[-len(iv):]
        if len(iv) < 5 or iv.std() == 0 or tv.std() == 0:
            return 0.0
        return float(np.corrcoef(iv[-len(tv):], tv)[0, 1])

    def update(self, tracks: list[Track], t_ns: int) -> Identity:
        live = [t for t in tracks if t.misses <= 5 and t.age >= 2]
        for t in live:
            self._vel.setdefault(t.id, deque(maxlen=120)).append((t_ns, t.vx))
        if not live:
            return Identity(None, None, 0.0)
        if self.camera_mode == "side":
            scores = {t.id: self._side_score(t) for t in live}
            me = max(live, key=lambda t: (scores[t.id], t.h))
            conf = max(0.0, scores[me.id])
        else:
            def s(t: Track) -> float:
                d = np.hypot(t.cx - self.home[0], t.cy - self.home[1])
                sticky = 0.15 if t.id == self._me_id else 0.0
                return -d + 0.5 * t.h + sticky

            me = max(live, key=s)
            d = float(np.hypot(me.cx - self.home[0], me.cy - self.home[1]))
            conf = float(np.clip(1.0 - d / 0.3, 0.0, 1.0))
            if conf > 0.5:  # slowly learn where "home" really is for this camera setup
                self.home = 0.995 * self.home + 0.005 * np.array([me.cx, me.cy])
        if len(live) == 1:
            # One visible character: in follow-cam it is usually the opponent if it is far from home.
            if self.camera_mode != "side" and conf < 0.3:
                return Identity(None, live[0], 0.4)
            return Identity(me, None, conf)
        self._me_id = me.id
        others = [t for t in live if t.id != me.id]
        opp = max(others, key=lambda t: t.h * t.score)
        return Identity(me, opp, conf)
