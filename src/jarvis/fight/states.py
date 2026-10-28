"""Fight state machine. The input allowlist per state is enforced by the actuator, not the policy."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from jarvis.fight.keymap import MOVEMENT

log = logging.getLogger(__name__)


class FightState(StrEnum):
    DISARMED = "disarmed"
    ARMED = "armed"  # owner said "fight": waiting to be queued / for the match to start
    NAVIGATING = "navigating"  # walking to a free ring box (WASD only)
    QUEUED = "queued"  # standing in a green box, waiting for an opponent
    FIGHTING = "fighting"
    POST_MATCH = "post_match"  # no input at all (the reward wheel is never touched)


# None = every enabled keymap action.
ALLOWED_ACTIONS: dict[FightState, frozenset[str] | None] = {
    FightState.DISARMED: frozenset(),
    FightState.ARMED: frozenset(),
    FightState.QUEUED: frozenset(),
    FightState.POST_MATCH: frozenset(),
    FightState.NAVIGATING: frozenset(MOVEMENT),
    FightState.FIGHTING: None,
}

TRANSITIONS: dict[FightState, frozenset[FightState]] = {
    FightState.DISARMED: frozenset({FightState.ARMED}),
    FightState.ARMED: frozenset({FightState.NAVIGATING, FightState.QUEUED, FightState.FIGHTING}),
    FightState.NAVIGATING: frozenset({FightState.ARMED, FightState.QUEUED}),
    FightState.QUEUED: frozenset({FightState.ARMED, FightState.FIGHTING}),
    FightState.FIGHTING: frozenset({FightState.POST_MATCH}),
    FightState.POST_MATCH: frozenset({FightState.ARMED}),
}


def action_allowed(state: FightState, action: str) -> bool:
    allowed = ALLOWED_ACTIONS[state]
    return True if allowed is None else action in allowed


@dataclass
class StateChange:
    old: FightState
    new: FightState
    reason: str
    t_ns: int


class IllegalTransition(RuntimeError):
    pass


class FightStateMachine:
    def __init__(self) -> None:
        self._state = FightState.DISARMED
        self._lock = threading.Lock()
        self._listeners: list[Callable[[StateChange], None]] = []
        self.history: list[StateChange] = []
        self.entered_ns = time.perf_counter_ns()

    @property
    def state(self) -> FightState:
        return self._state

    def on_change(self, fn: Callable[[StateChange], None]) -> None:
        self._listeners.append(fn)

    def can(self, new: FightState) -> bool:
        return new == FightState.DISARMED or new in TRANSITIONS[self._state]

    def transition(self, new: FightState, reason: str = "") -> StateChange:
        with self._lock:
            old = self._state
            if old == new:
                return StateChange(old, new, reason, time.perf_counter_ns())
            # DISARMED is always reachable: stopping is always allowed.
            if new != FightState.DISARMED and new not in TRANSITIONS[old]:
                raise IllegalTransition(f"{old.value} -> {new.value} not allowed ({reason})")
            self._state = new
            self.entered_ns = time.perf_counter_ns()
            change = StateChange(old, new, reason, self.entered_ns)
            self.history.append(change)
            if len(self.history) > 500:
                del self.history[:100]
        log.info("fight state %s -> %s (%s)", old.value, new.value, reason)
        for fn in list(self._listeners):
            try:
                fn(change)
            except Exception:
                log.exception("state listener failed")
        return change

    def disarm(self, reason: str) -> StateChange:
        return self.transition(FightState.DISARMED, reason)

    def seconds_in_state(self) -> float:
        return (time.perf_counter_ns() - self.entered_ns) / 1e9
