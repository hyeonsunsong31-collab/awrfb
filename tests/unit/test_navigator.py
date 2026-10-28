from __future__ import annotations

from jarvis.fight.perception.hud_bars import HudReading
from jarvis.fight.perception.pipeline import Observation
from jarvis.fight.perception.queue_box import Box, QueueStatus
from jarvis.fight.perception.ui_guard import UiGuardResult
from jarvis.fight.policy.navigator import QueueNavigator


def obs(t_ms, boxes, queued=False):
    return Observation(int(t_ms * 1e6), 1, HudReading(0), UiGuardResult(), queue=QueueStatus(boxes, queued))


def acts(cmds):
    return {(c.action, c.mode) for c in cmds}


def test_steers_toward_free_box_and_stops_when_queued():
    nav = QueueNavigator()
    cmds, q = nav.step(obs(0, [Box("free", 0.2, 0.55, 0.01)]), set())
    assert not q and acts(cmds) == {("move_left", "hold"), ("move_forward", "hold")}
    cmds, q = nav.step(obs(100, [Box("free", 0.5, 0.75, 0.02)]), {"move_left", "move_forward"})
    assert acts(cmds) == {("move_left", "release"), ("move_forward", "release")}
    cmds, q = nav.step(obs(200, [Box("queued", 0.5, 0.75, 0.02)], queued=True), set())
    assert q and cmds == []


def test_only_movement_and_timeout():
    nav = QueueNavigator(timeout_s=1.0)
    cmds, _ = nav.step(obs(0, []), set())
    assert all(c.action.startswith("move_") for c in cmds)
    assert nav.timed_out(int(1.5e9))
