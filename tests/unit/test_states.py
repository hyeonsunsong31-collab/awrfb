from __future__ import annotations

import pytest

from jarvis.fight.states import FightState, FightStateMachine, IllegalTransition, action_allowed


def test_happy_path_and_listeners():
    sm = FightStateMachine()
    seen = []
    sm.on_change(lambda c: seen.append((c.old, c.new)))
    for s in (FightState.ARMED, FightState.QUEUED, FightState.FIGHTING, FightState.POST_MATCH):
        sm.transition(s, "test")
    sm.disarm("done")
    assert sm.state == FightState.DISARMED
    assert seen[0] == (FightState.DISARMED, FightState.ARMED)
    assert len(seen) == 5


def test_illegal_transitions():
    sm = FightStateMachine()
    with pytest.raises(IllegalTransition):
        sm.transition(FightState.FIGHTING)  # cannot fight without being armed
    sm.transition(FightState.ARMED)
    sm.transition(FightState.FIGHTING)
    with pytest.raises(IllegalTransition):
        sm.transition(FightState.QUEUED)


def test_disarm_always_allowed():
    for start in FightState:
        sm = FightStateMachine()
        sm._state = start
        sm.disarm("panic")
        assert sm.state == FightState.DISARMED


def test_allowlists():
    assert not action_allowed(FightState.DISARMED, "move_forward")
    assert not action_allowed(FightState.POST_MATCH, "light")
    assert not action_allowed(FightState.QUEUED, "move_forward")
    assert action_allowed(FightState.NAVIGATING, "move_left")
    assert not action_allowed(FightState.NAVIGATING, "light")
    assert action_allowed(FightState.FIGHTING, "heavy")
