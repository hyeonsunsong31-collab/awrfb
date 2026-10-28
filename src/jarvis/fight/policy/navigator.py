"""Auto-queue (M2b): walk into a free (blue) ring box using WASD only, until it turns green.

Visual servoing on the queue-box colour blobs: strafe (A/D) to centre the nearest free box, walk
forward (W) until the box sits under our character. The actuator only allows movement keys in the
NAVIGATING state, so this can never press anything else.
"""

from __future__ import annotations

from jarvis.fight.actuator import Command
from jarvis.fight.perception.pipeline import Observation

NS_PER_MS = 1_000_000
MOVES = ("move_forward", "move_back", "move_left", "move_right")


class QueueNavigator:
    def __init__(self, timeout_s: float = 60.0, feet_y: float = 0.72, lateral_tol: float = 0.07) -> None:
        self.timeout_s = timeout_s
        self.feet_y = feet_y
        self.lateral_tol = lateral_tol
        self.started_ns: int | None = None
        self._search_toggle_ns = 0
        self._search_forward = True

    def reset(self) -> None:
        self.started_ns = None

    def timed_out(self, t_ns: int) -> bool:
        return self.started_ns is not None and (t_ns - self.started_ns) > self.timeout_s * 1e9

    def step(self, obs: Observation, held: set[str]) -> tuple[list[Command], bool]:
        """Returns (commands, queued)."""
        t = obs.t_ns
        if self.started_ns is None:
            self.started_ns = t
        want: set[str] = set()
        q = obs.queue
        queued = bool(q and q.queued)
        if not queued and q is not None:
            target = q.nearest_free()
            if target is not None:
                dx = target.cx - 0.5
                if dx < -self.lateral_tol:
                    want.add("move_left")
                elif dx > self.lateral_tol:
                    want.add("move_right")
                if target.cy < self.feet_y:
                    want.add("move_forward")
            else:
                # Nothing in view: alternate short forward walks and pauses (no camera turning).
                if t - self._search_toggle_ns > 700 * NS_PER_MS:
                    self._search_forward = not self._search_forward
                    self._search_toggle_ns = t
                if self._search_forward:
                    want.add("move_forward")
        cmds: list[Command] = []
        for a in MOVES:
            if a in want and a not in held:
                cmds.append(Command(a, "hold", reason="navigate"))
            elif a not in want and a in held:
                cmds.append(Command(a, "release", reason="navigate"))
        return cmds, queued
