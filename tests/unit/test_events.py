from __future__ import annotations

from jarvis.fight.events import EventConfig, HudEventTracker
from jarvis.fight.perception.hud_bars import HudReading

MS = 1_000_000


class Clock:
    def __init__(self) -> None:
        self.t = 0

    def __call__(self, ms: float = 16.7) -> int:
        self.t += int(ms * MS)
        return self.t


def reading(t, own=1.0, opp=1.0, guard=1.0, hud=True):
    return HudReading(t_ns=t, values={"own_hp": own, "opp_hp": opp, "own_guard": guard}, combat_hud=hud)


def feed(tr, clock, n, **kw):
    out = []
    for _ in range(n):
        out += tr.update(reading(clock(), **kw))
    return out


def types(evs):
    return [e.type for e in evs]


def test_match_start_needs_stable_hud():
    tr, c = HudEventTracker(), Clock()
    assert types(feed(tr, c, 10)) == []  # 167 ms < 600 ms
    assert "match_start" in types(feed(tr, c, 40))
    assert tr.in_match


def test_hits_blocks_and_noise():
    tr, c = HudEventTracker(), Clock()
    feed(tr, c, 50)
    evs = feed(tr, c, 3, own=0.9)  # median filter: needs 2 frames to move
    assert types(evs) == ["hit_taken"] and abs(evs[0].amount - 0.1) < 1e-6
    evs = feed(tr, c, 3, own=0.9, guard=0.8)
    assert types(evs) == ["hit_blocked"]
    evs = feed(tr, c, 3, own=0.9, guard=0.8, opp=0.85)
    assert types(evs) == ["hit_dealt"]
    # A single-frame glitch is filtered out by the median.
    out = tr.update(reading(c(), own=0.2, guard=0.8, opp=0.85))
    out += feed(tr, c, 3, own=0.9, guard=0.8, opp=0.85)
    assert "hit_taken" not in types(out)


def test_stocks_and_final():
    tr, c = HudEventTracker(EventConfig()), Clock()
    feed(tr, c, 50)
    for k in range(2):
        feed(tr, c, 4, opp=0.0)
        evs = feed(tr, c, 4, opp=1.0)
        assert "stock_won" in types(evs)
        assert tr.stocks_won == k + 1
    evs = feed(tr, c, 10, opp=0.0)
    assert types(evs).count("final_stock_empty") == 1


def test_match_end_after_hud_gone():
    tr, c = HudEventTracker(), Clock()
    feed(tr, c, 50)
    evs = feed(tr, c, 150, hud=False)  # 2.5 s
    assert "match_end" not in types(evs)
    evs = feed(tr, c, 100, hud=False)
    assert types(evs).count("match_end") == 1
    assert not tr.in_match
