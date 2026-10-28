"""Fight events from HUD readings: the ground truth of what actually happened in the match.

Uses a median-of-3 filter per bar, then detects:
  hit_taken    own HP dropped              (amount = drop)
  hit_blocked  own guard dropped while HP held (a hit landed on our block)
  hit_dealt    opponent HP dropped
  stock_lost / stock_won   a bar emptied and refilled (UBG: 3 stocks each)
  match_start  combat HUD stable for a moment
  match_end    combat HUD gone for a while after play, or a 3rd stock lost
These events are also the free, after-the-fact labels for learning (see jarvis.learn.hindsight).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from jarvis.fight.perception.hud_bars import HudReading

NS_PER_MS = 1_000_000


@dataclass
class FightEvent:
    t_ns: int
    type: str
    amount: float = 0.0
    data: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {"t_ns": self.t_ns, "type": self.type, "amount": round(self.amount, 4), **self.data}


@dataclass
class EventConfig:
    hit_min_drop: float = 0.012
    guard_min_drop: float = 0.02
    stock_empty: float = 0.04
    stock_refill: float = 0.85
    start_stable_ms: float = 600.0
    end_absent_ms: float = 3500.0
    stocks_per_match: int = 3


class _Median3:
    def __init__(self) -> None:
        self.buf: deque[float] = deque(maxlen=3)

    def push(self, v: float | None) -> float | None:
        if v is None:
            return None
        self.buf.append(v)
        s = sorted(self.buf)
        return s[len(s) // 2]


class HudEventTracker:
    def __init__(self, cfg: EventConfig | None = None) -> None:
        self.cfg = cfg or EventConfig()
        self._filters: dict[str, _Median3] = {}
        self._prev: dict[str, float] = {}
        self._min_since_refill: dict[str, float] = {}
        self.in_match = False
        self._hud_since: int | None = None
        self._hud_absent_since: int | None = None
        self.stocks_lost = 0
        self.stocks_won = 0
        self.match_started_ns: int | None = None
        self.last: dict[str, int] = {}  # event type -> last t_ns
        self._final_reported: set[str] = set()

    def _filtered(self, name: str, v: float | None) -> float | None:
        return self._filters.setdefault(name, _Median3()).push(v)

    def reset_match(self) -> None:
        self.stocks_lost = self.stocks_won = 0
        self._final_reported.clear()
        self.in_match = False
        self._prev.clear()
        self._filters.clear()
        self._min_since_refill.clear()

    def update(self, rd: HudReading) -> list[FightEvent]:
        t = rd.t_ns
        ev: list[FightEvent] = []
        c = self.cfg
        # ---- match start / end from HUD presence
        if rd.combat_hud:
            self._hud_absent_since = None
            if self._hud_since is None:
                self._hud_since = t
            if not self.in_match and (t - self._hud_since) >= c.start_stable_ms * NS_PER_MS:
                self.in_match = True
                self.match_started_ns = t
                self.stocks_lost = self.stocks_won = 0
                self._final_reported.clear()
                ev.append(FightEvent(t, "match_start"))
        else:
            self._hud_since = None
            if self.in_match:
                if self._hud_absent_since is None:
                    self._hud_absent_since = t
                elif (t - self._hud_absent_since) >= c.end_absent_ms * NS_PER_MS:
                    self.in_match = False
                    ev.append(FightEvent(t, "match_end", data={"reason": "hud_gone", "stocks_lost": self.stocks_lost, "stocks_won": self.stocks_won}))
            return self._stamp(ev)
        if not self.in_match:
            # keep filters warm so the first in-match frame has a baseline
            for name in ("own_hp", "opp_hp", "own_guard"):
                fv = self._filtered(name, rd.values.get(name))
                if fv is not None:
                    self._prev[name] = fv
            return self._stamp(ev)

        own = self._filtered("own_hp", rd.values.get("own_hp"))
        opp = self._filtered("opp_hp", rd.values.get("opp_hp"))
        guard = self._filtered("own_guard", rd.values.get("own_guard"))

        def drop(name: str, v: float | None) -> float:
            p = self._prev.get(name)
            if v is None or p is None:
                return 0.0
            return p - v

        d_own = drop("own_hp", own)
        d_opp = drop("opp_hp", opp)
        d_guard = drop("own_guard", guard)
        if d_own >= c.hit_min_drop:
            ev.append(FightEvent(t, "hit_taken", d_own))
        elif d_guard >= c.guard_min_drop:
            ev.append(FightEvent(t, "hit_blocked", d_guard))
        if d_opp >= c.hit_min_drop:
            ev.append(FightEvent(t, "hit_dealt", d_opp))

        # ---- stocks: a bar that emptied and then refilled
        for name, v, kind in (("own_hp", own, "stock_lost"), ("opp_hp", opp, "stock_won")):
            if v is None:
                continue
            m = min(self._min_since_refill.get(name, 1.0), v)
            self._min_since_refill[name] = m
            if m <= c.stock_empty and v >= c.stock_refill:
                self._min_since_refill[name] = v
                if kind == "stock_lost":
                    self.stocks_lost += 1
                    n = self.stocks_lost
                else:
                    self.stocks_won += 1
                    n = self.stocks_won
                ev.append(FightEvent(t, kind, data={"count": n}))
        for name, v in (("own_hp", own), ("opp_hp", opp), ("own_guard", guard)):
            if v is not None:
                self._prev[name] = v
        # The last stock ends at an empty bar with no refill (the match ends instead): report the
        # likely outcome once, when the final bar empties.
        for side, v, lost in (("own", own, self.stocks_lost), ("opp", opp, self.stocks_won)):
            if v is not None and v <= c.stock_empty and lost == c.stocks_per_match - 1 and side not in self._final_reported:
                self._final_reported.add(side)
                ev.append(FightEvent(t, "final_stock_empty", data={"side": side}))
        return self._stamp(ev)

    def _stamp(self, ev: list[FightEvent]) -> list[FightEvent]:
        for e in ev:
            self.last[e.type] = e.t_ns
        return ev

    def ms_since(self, etype: str, now_ns: int) -> float:
        t = self.last.get(etype)
        return float("inf") if t is None else (now_ns - t) / NS_PER_MS
