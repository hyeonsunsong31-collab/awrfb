"""The game actuator: the ONLY path from fight decisions to OS input.

Safety properties (each covered by unit tests):
* Only keymap actions allowed in the current fight state are ever pressed (states.ALLOWED_ACTIONS).
* Every press re-checks the gate: fresh per-frame status (combat HUD confirmed, no popup, chat
  closed, owner not touching input) AND a direct foreground-window check right before SendInput.
* Releases are always allowed; when the gate closes or the state leaves FIGHTING/NAVIGATING, every
  held key is released immediately.
* A watchdog releases everything and disarms if the fight loop stops sending heartbeats.
* Forbidden keys (Esc, Enter, Tab, /, Win, Alt, F-keys...) are refused by the keymap validator and
  by the input sink itself.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from jarvis.common.config import ActuatorConfig
from jarvis.fight.gate import GateStatus
from jarvis.fight.keymap import Keymap
from jarvis.fight.states import FightState, action_allowed
from jarvis.platform.base import InputSink, MouseButton, WindowManager, WindowRef

log = logging.getLogger(__name__)

NS_PER_MS = 1_000_000
CHORD_LEAD_MS = 15


@dataclass
class Command:
    action: str
    mode: Literal["tap", "hold", "release", "move"] = "tap"
    dur_ms: float | None = None
    at_ns: int | None = None
    dx: int = 0
    dy: int = 0
    reason: str = ""


@dataclass(order=True)
class _Event:
    t_ns: int
    seq: int
    kind: str = field(compare=False)  # "down" | "up" | "move"
    phys: str = field(compare=False)  # "key:f" | "mouse:left" | "move"
    action: str = field(compare=False)
    group: int = field(compare=False)
    dx: int = field(default=0, compare=False)
    dy: int = field(default=0, compare=False)
    reason: str = field(default="", compare=False)


InputLogFn = Callable[[dict[str, Any]], None]


class Actuator:
    def __init__(
        self,
        sink: InputSink,
        keymap: Keymap,
        windows: WindowManager,
        cfg: ActuatorConfig,
        *,
        now_ns: Callable[[], int] = time.perf_counter_ns,
        input_log: InputLogFn | None = None,
        audit: Any = None,
        synthetic: Any = None,
        on_watchdog: Callable[[str], None] | None = None,
    ) -> None:
        self._sink = sink
        self._km = keymap
        self._win = windows
        self._cfg = cfg
        self._now = now_ns
        self._input_log = input_log
        self._audit = audit
        self._synthetic = synthetic
        self._on_watchdog = on_watchdog

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._heap: list[_Event] = []
        self._seq = itertools.count()
        self._groups = itertools.count(1)
        self._denied_groups: set[int] = set()

        self._state = FightState.DISARMED
        self._target: WindowRef | None = None
        self._gate = GateStatus()
        self._last_heartbeat = 0

        self._running = False
        self._threads: list[threading.Thread] = []
        self.stats = {"sent": 0, "denied": 0, "released": 0, "watchdog": 0}
        self.last_denial: str = ""

    # ------------------------------------------------------------------ configuration
    def set_target(self, window: WindowRef | None) -> None:
        with self._lock:
            self._target = window

    def set_state(self, state: FightState) -> None:
        with self._lock:
            self._state = state
            if state not in (FightState.FIGHTING, FightState.NAVIGATING):
                self._clear_queue()
                self._release_all_locked(f"state -> {state.value}")

    def update_gate(self, gate: GateStatus) -> None:
        with self._lock:
            self._gate = gate
            ok, why = gate.check(self._state)
            if not ok and self._sink.held():
                self._clear_queue()
                self._release_all_locked(f"gate closed: {why}")

    def heartbeat(self) -> None:
        self._last_heartbeat = self._now()

    # ------------------------------------------------------------------ commands
    def submit(self, cmds: list[Command]) -> None:
        now = self._now()
        with self._cond:
            for c in cmds:
                for ev in self._expand(c, c.at_ns if c.at_ns is not None else now):
                    heapq.heappush(self._heap, ev)
            self._cond.notify()

    def _expand(self, c: Command, t0: int) -> list[_Event]:
        g = next(self._groups)
        dur = int((c.dur_ms if c.dur_ms is not None else self._cfg.default_tap_ms) * NS_PER_MS)
        if c.mode == "move" or c.action == "camera":
            return [_Event(t0, next(self._seq), "move", "move", "camera", g, c.dx, c.dy, c.reason)]
        b = self._km.actions.get(c.action)
        if b is None:
            self._deny(c.action, "unknown action", None)
            return []
        evs: list[_Event] = []

        def prim(action: str, t: int, kind: str) -> None:
            pb = self._km.actions[action]
            phys = f"key:{pb.key}" if pb.key is not None else f"mouse:{pb.mouse}"
            evs.append(_Event(t, next(self._seq), kind, phys, c.action, g, reason=c.reason))

        if b.kind in ("key", "mouse"):
            if c.mode == "tap":
                prim(c.action, t0, "down")
                prim(c.action, t0 + dur, "up")
            elif c.mode == "hold":
                prim(c.action, t0, "down")
            elif c.mode == "release":
                prim(c.action, t0, "up")
        elif b.kind == "combo":
            gap = int(self._cfg.feint_gap_ms * NS_PER_MS)
            for i, ref in enumerate(b.combo or []):
                t = t0 + i * gap
                prim(ref, t, "down")
                prim(ref, t + dur, "up")
        elif b.kind == "chord":
            held, tapped = b.chord or ["", ""]
            lead = CHORD_LEAD_MS * NS_PER_MS
            prim(held, t0, "down")
            prim(tapped, t0 + lead, "down")
            prim(tapped, t0 + lead + dur, "up")
            prim(held, t0 + 2 * lead + dur, "up")
        return evs

    # ------------------------------------------------------------------ execution
    def pump(self, now: int | None = None) -> int:
        """Execute every event that is due. Returns the number of events processed."""
        n = 0
        now = self._now() if now is None else now
        with self._lock:
            while self._heap and self._heap[0].t_ns <= now:
                ev = heapq.heappop(self._heap)
                self._execute(ev)
                n += 1
        return n

    def _gate_open(self) -> tuple[bool, str]:
        now = self._now()
        age_ms = (now - self._gate.t_ns) / NS_PER_MS
        if age_ms > self._cfg.gate_max_age_ms:
            return False, f"gate status stale ({age_ms:.0f} ms)"
        ok, why = self._gate.check(self._state)
        if not ok:
            return False, why
        if self._target is None:
            return False, "no target window"
        if not self._win.is_foreground(self._target.hwnd):
            return False, "Roblox lost foreground"
        return True, "open"

    def _execute(self, ev: _Event) -> None:
        if ev.kind == "up":
            if ev.phys in self._sink.held():
                self._do(ev)
            return
        if ev.group in self._denied_groups:
            return
        if ev.kind == "move":
            if not self._km.camera.mouse_relative:
                self._deny_group(ev, "camera moves disabled by calibration")
                return
            ok, why = self._gate_open()
            if not ok:
                self._deny_group(ev, why)
                return
            lim = self._km.camera.max_step_px
            ev.dx = max(-lim, min(lim, ev.dx))
            ev.dy = max(-lim, min(lim, ev.dy))
            self._do(ev)
            return
        # down
        if not action_allowed(self._state, ev.action):
            self._deny_group(ev, f"action {ev.action!r} not allowed in {self._state.value}")
            return
        if not self._km.usable(ev.action):
            self._deny_group(ev, f"action {ev.action!r} is disabled in the keymap")
            return
        ok, why = self._gate_open()
        if not ok:
            self._deny_group(ev, why)
            return
        if ev.phys in self._sink.held():
            return  # already down (e.g. block being held); never double-press
        self._do(ev)

    def _do(self, ev: _Event) -> None:
        kind, _, name = ev.phys.partition(":")
        try:
            if ev.kind == "move":
                self._sink.mouse_move_rel(ev.dx, ev.dy)
                syn = "mouse_move"
            elif kind == "key":
                (self._sink.key_down if ev.kind == "down" else self._sink.key_up)(name)
                syn = "key"
            else:
                btn = MouseButton(name)
                (self._sink.mouse_down if ev.kind == "down" else self._sink.mouse_up)(btn)
                syn = "mouse_button"
        except Exception as e:  # SendInput failure or forbidden key: release everything
            log.error("input failed (%s): %s", ev.phys, e)
            self._release_all_locked(f"input error: {e}")
            return
        if self._synthetic is not None:
            self._synthetic.note(syn)
        self.stats["sent"] += 1
        self._log(ev, ok=True)

    def _deny(self, action: str, why: str, ev: _Event | None) -> None:
        self.stats["denied"] += 1
        self.last_denial = why
        if ev is not None:
            self._log(ev, ok=False, why=why)

    def _deny_group(self, ev: _Event, why: str) -> None:
        self._denied_groups.add(ev.group)
        if len(self._denied_groups) > 10_000:
            self._denied_groups = set(sorted(self._denied_groups)[-1000:])
        self._deny(ev.action, why, ev)

    def _log(self, ev: _Event, ok: bool, why: str = "") -> None:
        rec = {
            "v": 1,
            "t_ns": self._now(),
            "src": "bot",
            "action": ev.action,
            "event": ev.kind,
            "phys": ev.phys,
            "ok": ok,
            "why": why,
            "reason": ev.reason,
            "state": self._state.value,
        }
        if ev.kind == "move":
            rec["dx"], rec["dy"] = ev.dx, ev.dy
        if self._input_log is not None:
            try:
                self._input_log(rec)
            except Exception:
                log.exception("input log failed")
        if self._audit is not None and ok:
            self._audit.append("game_input", rec)

    # ------------------------------------------------------------------ safety
    def _clear_queue(self) -> None:
        self._heap.clear()
        self._denied_groups.clear()

    def _release_all_locked(self, reason: str) -> None:
        held = self._sink.held()
        if not held:
            return
        try:
            self._sink.release_all()
        finally:
            self.stats["released"] += len(held)
            if self._input_log is not None:
                self._input_log(
                    {"v": 1, "t_ns": self._now(), "src": "bot", "event": "release_all", "phys": sorted(held), "ok": True, "why": reason}
                )
            if self._audit is not None:
                self._audit.append("game_release_all", {"held": sorted(held), "reason": reason})

    def release_all(self, reason: str) -> None:
        with self._lock:
            self._clear_queue()
            self._release_all_locked(reason)

    def check_watchdog(self, now: int | None = None) -> bool:
        """Returns True if the watchdog fired."""
        now = self._now() if now is None else now
        with self._lock:
            active = self._state in (FightState.FIGHTING, FightState.NAVIGATING) or bool(self._sink.held())
            if not active:
                return False
            silent_ms = (now - self._last_heartbeat) / NS_PER_MS
            if silent_ms <= self._cfg.heartbeat_timeout_ms:
                return False
            self.stats["watchdog"] += 1
            reason = f"watchdog: no heartbeat for {silent_ms:.0f} ms"
            self._clear_queue()
            self._release_all_locked(reason)
        log.error(reason)
        if self._on_watchdog is not None:
            self._on_watchdog(reason)
        return True

    # ------------------------------------------------------------------ threads
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._last_heartbeat = self._now()
        t1 = threading.Thread(target=self._scheduler, name="actuator", daemon=True)
        t2 = threading.Thread(target=self._watchdog, name="actuator-watchdog", daemon=True)
        self._threads = [t1, t2]
        t1.start()
        t2.start()

    def _scheduler(self) -> None:
        while self._running:
            with self._cond:
                if not self._heap:
                    self._cond.wait(0.05)
                    continue
                wait_ns = self._heap[0].t_ns - self._now()
                if wait_ns > 2 * NS_PER_MS:
                    self._cond.wait((wait_ns - NS_PER_MS) / 1e9)
                    continue
            if wait_ns > 0:
                time.sleep(0)  # sub-2ms: yield until due
            self.pump()

    def _watchdog(self) -> None:
        while self._running:
            time.sleep(0.05)
            try:
                self.check_watchdog()
            except Exception:
                log.exception("watchdog check failed")

    def stop(self) -> None:
        self.release_all("actuator stopped")
        self._running = False
        with self._cond:
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []
