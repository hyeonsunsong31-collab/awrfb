"""Outcome-based opponent model: built only from things we can measure reliably today (HUD events,
spacing, our own actions). It feeds tactic priors, the coach and the HUD radar. A sequence model of the
opponent's actions is added once the action recognizer is accurate enough (M9)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

NS_PER_MS = 1_000_000


@dataclass
class OpponentProfile:
    aggression: float = 0.5  # how much they press forward and hit us
    defense: float = 0.5  # how often our attacks fail to land (blocking/dodging)
    punish: float = 0.5  # how often our heavies get punished
    range_pref: float = 0.5  # 0 = in your face, 1 = keeps distance
    regen: float = 0.0  # backs off to regenerate when hurt
    samples: int = 0

    def radar(self) -> dict[str, float]:
        return {k: round(getattr(self, k), 3) for k in ("aggression", "defense", "punish", "range_pref", "regen")}


@dataclass
class _Pending:
    t_ns: int
    action: str
    punished: bool = False


@dataclass
class OpponentModel:
    ema: float = 0.08
    profile: OpponentProfile = field(default_factory=OpponentProfile)
    _pending: deque = field(default_factory=lambda: deque(maxlen=64))
    _gap_hist: deque = field(default_factory=lambda: deque(maxlen=600))
    _last_t: int | None = None
    _hp_low_retreats: int = 0
    _hp_low_frames: int = 0

    def _blend(self, attr: str, x: float) -> None:
        cur = getattr(self.profile, attr)
        setattr(self.profile, attr, (1 - self.ema) * cur + self.ema * x)

    def note_our_action(self, t_ns: int, action: str) -> None:
        if action in ("light", "heavy", "feint"):
            self._pending.append(_Pending(t_ns, action))

    def update(self, t_ns: int, events: list, gap: float | None, approach: float, opp_hp: float | None) -> None:
        types = {e.type for e in events}
        # ---- our attacks: landed (hit_dealt soon after) vs failed
        window = 350 * NS_PER_MS
        while self._pending and t_ns - self._pending[0].t_ns > window:
            p = self._pending.popleft()
            self._blend("defense", 1.0)  # no damage registered in time: blocked/dodged/missed
            if p.action == "heavy" and not p.punished:
                self._blend("punish", 0.0)
        if "hit_dealt" in types and self._pending:
            self._pending.popleft()
            self._blend("defense", 0.0)
        # ---- punished heavies: we got hit shortly after throwing a heavy (count each heavy once)
        if "hit_taken" in types:
            for p in self._pending:
                if p.action == "heavy" and not p.punished:
                    p.punished = True
                    self._blend("punish", 1.0)
            self._blend("aggression", 1.0)
        if gap is not None:
            self._gap_hist.append(gap)
            self.profile.range_pref = sum(self._gap_hist) / len(self._gap_hist)
            # pressing forward (closing distance) counts as aggression
            if approach > 0.02:
                self._blend("aggression", 0.8)
            elif approach < -0.02:
                self._blend("aggression", 0.2)
            if opp_hp is not None and opp_hp < 0.35:
                self._hp_low_frames += 1
                if approach < -0.02:
                    self._hp_low_retreats += 1
                self.profile.regen = self._hp_low_retreats / max(1, self._hp_low_frames)
        self.profile.samples += 1

    def tactic_priors(self) -> dict[str, float]:
        """Map the profile to prior expected rewards per tactic (small numbers: one HP bar = 1.0)."""
        p = self.profile
        return {
            "pressure": 0.04 * (1 - p.defense) - 0.03 * p.punish,
            "counter": 0.05 * p.aggression,
            "bait_punish": 0.03 * p.aggression + 0.02 * p.punish,
            "keep_range": 0.03 * p.punish + 0.02 * p.aggression - 0.02,
            "feint_mix": 0.05 * p.defense,
            "regen_retreat": -0.01,
        }
