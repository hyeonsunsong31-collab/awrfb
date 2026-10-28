"""The fight loop: one thread that turns each new Roblox frame into (at most) a few safe inputs.

    frame -> perception -> gate -> state machine -> policy -> actuator -> recorder/telemetry

Arming checks (refuses with a spoken reason): server policy (private server unless the owner opted
into public + accepted the ban risk), HUD calibration present, settings fingerprint unchanged, and the
basic actions usable. The loop only ever sends input in FIGHTING (all fight actions) or NAVIGATING
(movement only); everything else is enforced again by the actuator.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.common.config import JarvisConfig
from jarvis.fight.calibrate.fingerprint import Fingerprint
from jarvis.fight.gameio import GameIO
from jarvis.fight.launcher import fight_allowed_in
from jarvis.fight.perception.detector import NullDetector, OnnxDetector
from jarvis.fight.perception.layout import HudLayout, load_layout
from jarvis.fight.perception.pipeline import Observation, Perception
from jarvis.fight.policy.bandit import DiscountedThompson
from jarvis.fight.policy.navigator import QueueNavigator
from jarvis.fight.policy.opponent import OpponentModel
from jarvis.fight.policy.params import PolicyParams, load_params
from jarvis.fight.policy.reflex_v0 import ReflexPolicy
from jarvis.fight.policy.tactics import TACTICS
from jarvis.fight.recorder import SessionRecorder
from jarvis.fight.states import FightState, IllegalTransition

log = logging.getLogger(__name__)
NS_PER_MS = 1_000_000


class ArmRefused(RuntimeError):
    pass


@dataclass
class MatchStats:
    started_ns: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    hits_dealt: int = 0
    hits_taken: int = 0
    hits_blocked: int = 0
    stocks_won: int = 0
    stocks_lost: int = 0
    final: str | None = None
    perception_ms: list[float] = field(default_factory=list)
    gate_closed_frames: int = 0
    frames: int = 0

    def absorb(self, obs: Observation) -> None:
        self.frames += 1
        if len(self.perception_ms) < 20000:
            self.perception_ms.append(obs.timings_ms.get("total", 0.0))
        for e in obs.events:
            if e.type == "hit_dealt":
                self.hits_dealt += 1
                self.damage_dealt += e.amount
            elif e.type == "hit_taken":
                self.hits_taken += 1
                self.damage_taken += e.amount
            elif e.type == "hit_blocked":
                self.hits_blocked += 1
            elif e.type == "stock_won":
                self.stocks_won += 1
            elif e.type == "stock_lost":
                self.stocks_lost += 1
            elif e.type == "final_stock_empty":
                self.final = "won" if e.data.get("side") == "opp" else "lost"

    def result(self) -> str:
        if self.final:
            return self.final
        if self.stocks_won > self.stocks_lost:
            return "likely_won"
        if self.stocks_lost > self.stocks_won:
            return "likely_lost"
        return "unknown"


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 2)


def load_detector(cfg: JarvisConfig):
    p = cfg.fight.perception
    if not p.detector_model:
        return NullDetector()
    path = cfg.paths().models / p.detector_model
    if not path.exists():
        log.warning("detector model %s missing; running HUD-only", path)
        return NullDetector()
    try:
        return OnnxDetector(path, p.detector_input, p.detector_conf)
    except Exception:
        log.exception("could not load detector; running HUD-only")
        return NullDetector()


def load_threat_model(cfg: JarvisConfig):
    p = cfg.fight.perception
    if not p.threat_model:
        return None
    path = cfg.paths().models / p.threat_model
    if not path.exists():
        return None
    try:
        from jarvis.learn.threat_runtime import OnnxThreat

        return OnnxThreat(path, window=p.threat_window, crop=p.crop_size)
    except Exception:
        log.exception("could not load threat model; using heuristic")
        return None


class FightLoop:
    def __init__(
        self,
        cfg: JarvisConfig,
        io: GameIO,
        *,
        layout: HudLayout | None = None,
        params: PolicyParams | None = None,
        detector=None,
        threat_model=None,
        on_telemetry: Callable[[dict[str, Any]], None] | None = None,
        on_callout: Callable[[str, dict[str, Any]], None] | None = None,
        record: bool = True,
        seed: int | None = None,
        event_cfg=None,
        fixed_tactic: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.io = io
        paths = cfg.paths().ensure()
        self.paths = paths
        self.layout = layout or load_layout(paths.calibration)
        self.params = params or load_params(cfg.fight.policy_file, paths.calibration)
        self._detector = detector
        self._threat_model = threat_model
        self.on_telemetry = on_telemetry or (lambda d: None)
        self.on_callout = on_callout or (lambda k, d: None)
        self.record = record
        self.seed = seed
        self.event_cfg = event_cfg
        self.fixed_tactic = fixed_tactic
        self.perception: Perception | None = None
        self.policy: ReflexPolicy | None = None
        self.navigator = QueueNavigator()
        self.recorder: SessionRecorder | None = None
        self.match = MatchStats()
        self.matches_left = 0
        self.auto_queue = False
        self._post_match_until = 0
        self._last_telemetry = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_index = 0
        self._bandit = DiscountedThompson(list(TACTICS), discount=self.params.tactics.discount, seed=seed)
        self._opponent = OpponentModel()
        self.last_obs: Observation | None = None
        self.last_refusal = ""
        io.sm.on_change(self._on_state_change)

    # ------------------------------------------------------------------ arming
    def _build_perception(self) -> None:
        assert self.layout is not None
        if self._detector is None:
            self._detector = load_detector(self.cfg)
        if self._threat_model is None:
            self._threat_model = load_threat_model(self.cfg)
        self.perception = Perception(
            self.cfg.fight.perception,
            self.layout,
            calibration_dir=self.paths.calibration,
            detector=self._detector,
            threat_model=self._threat_model,
            h_close=self.params.spacing.h_close,
            h_far=self.params.spacing.h_far,
            event_cfg=self.event_cfg,
        )

    def preflight(self) -> list[str]:
        problems = []
        ok, why = fight_allowed_in(self.io.session, self.cfg.fight.game)
        if not ok:
            problems.append(why)
        if self.layout is None:
            problems.append("HUD not calibrated yet - run `jarvis calibrate hud` during a match")
        else:
            fp = Fingerprint.load(self.paths.calibration)
            frame = self.io.frames.peek()
            if fp is not None and frame is not None:
                problems += fp.problems(frame.width, frame.height, self.layout, self.io.keymap)
        for a in ("light", "block"):
            if not self.io.keymap.usable(a):
                problems.append(f"keymap action {a!r} is disabled")
        return problems

    def arm(self, matches: int = 1, auto_queue: bool = False) -> None:
        if self.io.sm.state != FightState.DISARMED:
            raise ArmRefused(f"already {self.io.sm.state.value}")
        problems = self.preflight()
        if problems:
            self.last_refusal = "; ".join(problems)
            raise ArmRefused(self.last_refusal)
        self._build_perception()
        self.matches_left = max(1, int(matches))
        self.auto_queue = auto_queue
        self.io.sm.transition(FightState.ARMED, "owner command")
        self.on_callout("armed", {"matches": self.matches_left})

    def disarm(self, reason: str = "owner command") -> None:
        self.matches_left = 0
        self.io.disarm(reason)

    # ------------------------------------------------------------------ state hooks
    def _on_state_change(self, ch) -> None:
        if ch.new == FightState.ARMED and self.record and self.recorder is None:
            rc = self.cfg.fight.recording
            self.recorder = SessionRecorder(
                self.paths.recordings,
                "fight",
                width=rc.width,
                height=rc.height,
                ffmpeg=rc.ffmpeg,
                encoder=rc.encoder,
                record_video=rc.enabled,
                meta={
                    "tier": self.cfg.tier,
                    "server_kind": getattr(self.io.session, "server_kind", None),
                    "layout_digest": self.layout.digest() if self.layout else None,
                    "keymap_verified": {a: b.verified for a, b in self.io.keymap.actions.items()},
                },
            )
            self.io.set_input_log(lambda r: self.recorder.log("inputs", r) if self.recorder else None)
        if ch.new == FightState.FIGHTING:
            self.match = MatchStats(started_ns=time.perf_counter_ns())
            self.policy = ReflexPolicy(self.params, self.io.keymap, bandit=self._bandit, opponent=self._opponent,
                                       seed=self.seed, camera_relative=self.io.keymap.camera.mouse_relative,
                                       fixed_tactic=self.fixed_tactic)
            self.policy.reset(time.perf_counter_ns())
            gc.collect()
            gc.freeze()
            gc.disable()  # no GC pauses mid-fight
            self.on_callout("match_start", {})
        if ch.old == FightState.FIGHTING:
            gc.enable()
            gc.unfreeze()
        if ch.new == FightState.DISARMED:
            self._close_recording(ch.reason)
        if self.recorder is not None:
            self.recorder.log("events", {"t_ns": ch.t_ns, "type": "state", "old": ch.old.value, "new": ch.new.value, "reason": ch.reason})

    def _close_recording(self, reason: str) -> Path | None:
        if self.recorder is None:
            return None
        rec, self.recorder = self.recorder, None
        self.io.set_input_log(None)
        return rec.close(self.summary(reason))

    def summary(self, reason: str = "") -> dict[str, Any]:
        m = self.match
        a = self.io.actuator.stats
        dur = (time.perf_counter_ns() - m.started_ns) / 1e9 if m.started_ns else 0.0
        return {
            "result": m.result(),
            "reason": reason,
            "duration_s": round(dur, 1),
            "stocks_won": m.stocks_won,
            "stocks_lost": m.stocks_lost,
            "damage_dealt": round(m.damage_dealt, 3),
            "damage_taken": round(m.damage_taken, 3),
            "hits": {"dealt": m.hits_dealt, "taken": m.hits_taken, "blocked": m.hits_blocked},
            "inputs": dict(a),
            "gate_closed_frames": m.gate_closed_frames,
            "frames": m.frames,
            "fps": round(self.io.fps.fps, 1),
            "perception_ms": {"p50": _pct(m.perception_ms, 0.5), "p95": _pct(m.perception_ms, 0.95), "p99": _pct(m.perception_ms, 0.99)},
            "policy": self.policy.summary() if self.policy else None,
            "server_kind": getattr(self.io.session, "server_kind", None),
        }

    # ------------------------------------------------------------------ per-frame
    def tick(self) -> Observation | None:
        f = self.io.frames.get_newer(self._last_index, 0.1)
        if f is None:
            # No heartbeat without a processed frame: a stalled capture must trip the watchdog.
            return None
        self._last_index = f.index
        sm = self.io.sm
        state = sm.state
        if self.perception is None:
            if self.layout is None:
                return None
            self._build_perception()
        assert self.perception is not None
        want_queue = state in (FightState.ARMED, FightState.NAVIGATING, FightState.QUEUED)
        obs = self.perception.process(f, want_queue=want_queue)
        self.last_obs = obs
        gate = self.io.base_gate(
            combat_hud=obs.hud.combat_hud,
            hud_conf=obs.hud.hud_conf,
            popup=obs.ui.popup,
            chat_open=obs.ui.chat_open,
            cursor_closes_gate=self.layout.cursor_closes_gate if self.layout else True,
        )
        self.io.actuator.update_gate(gate)
        self.io.actuator.heartbeat()

        held = self.io.sink.held()
        cmds = []
        try:
            if state == FightState.ARMED:
                if self.perception.events.in_match or any(e.type == "match_start" for e in obs.events):
                    sm.transition(FightState.FIGHTING, "match started")
                elif obs.queue is not None and obs.queue.queued:
                    sm.transition(FightState.QUEUED, "standing in a green ring box")
                elif self.auto_queue and obs.queue is not None and obs.queue.nearest_free() is not None:
                    self.navigator.reset()
                    sm.transition(FightState.NAVIGATING, "walking to a free ring box")
            elif state == FightState.NAVIGATING:
                phys_actions = {a for a in ("move_forward", "move_back", "move_left", "move_right")
                                if self._phys(a) in held}
                cmds, queued = self.navigator.step(obs, phys_actions)
                if queued:
                    sm.transition(FightState.QUEUED, "reached a ring box")
                    cmds = []
                elif self.navigator.timed_out(obs.t_ns):
                    sm.transition(FightState.ARMED, "could not reach a free ring box in time")
                    self.on_callout("queue_failed", {})
                    cmds = []
            elif state == FightState.QUEUED:
                if self.perception.events.in_match or any(e.type == "match_start" for e in obs.events):
                    sm.transition(FightState.FIGHTING, "match started")
            elif state == FightState.FIGHTING:
                self.match.absorb(obs)
                ok, _ = gate.check(state)
                if not ok:
                    self.match.gate_closed_frames += 1
                assert self.policy is not None
                cmds = self.policy.step(obs, held)
                for e in obs.events:
                    if e.type in ("stock_won", "stock_lost", "final_stock_empty"):
                        self.on_callout(e.type, e.to_row())
                if any(e.type == "match_end" for e in obs.events):
                    sm.transition(FightState.POST_MATCH, "match ended")
                    self._post_match_until = obs.t_ns + 2500 * NS_PER_MS
                    self.on_callout("match_end", self.summary("match ended"))
            elif state == FightState.POST_MATCH and obs.t_ns >= self._post_match_until:
                self.matches_left -= 1
                if self.matches_left > 0:
                    self._close_recording("next match")
                    sm.transition(FightState.ARMED, f"{self.matches_left} more match(es) requested")
                else:
                    self.io.disarm("requested matches done")
        except IllegalTransition as e:
            log.error("state machine refused transition: %s", e)
            self.io.disarm(str(e))
            cmds = []
        if cmds and sm.state in (FightState.FIGHTING, FightState.NAVIGATING):
            self.io.actuator.submit(cmds)
        if self.recorder is not None:
            row = obs.row()
            row["state"] = sm.state.value
            row["gate"] = gate.check(sm.state)[1]
            if self.policy is not None and sm.state == FightState.FIGHTING:
                row["tactic"] = self.policy.ctx.tactic
            self.recorder.add_frame(f.image, row)
            for e in obs.events:
                self.recorder.log("events", e.to_row())
        if obs.t_ns - self._last_telemetry > 66 * NS_PER_MS:
            self._last_telemetry = obs.t_ns
            self.on_telemetry(self.telemetry(obs, gate.check(sm.state)))
        return obs

    def _phys(self, action: str) -> str:
        b = self.io.keymap.actions[action]
        return f"key:{b.key}" if b.key else f"mouse:{b.mouse}"

    def telemetry(self, obs: Observation, gate: tuple[bool, str]) -> dict[str, Any]:
        return {
            "state": self.io.sm.state.value,
            "gate_open": gate[0],
            "gate": gate[1],
            "hud": obs.hud.values,
            "combat_hud": obs.hud.combat_hud,
            "threat": obs.threat,
            "gap": obs.gap,
            "opp": None if obs.opp is None else [obs.opp.cx, obs.opp.cy, obs.opp.w, obs.opp.h],
            "me": None if obs.me is None else [obs.me.cx, obs.me.cy, obs.me.w, obs.me.h],
            "fps": round(self.io.fps.fps, 1),
            "perception_ms": round(obs.timings_ms.get("total", 0.0), 2),
            "tactic": self.policy.ctx.tactic if self.policy else None,
            "opponent": self._opponent.profile.radar(),
            "inputs": dict(self.io.actuator.stats),
            "match": {"won": self.match.stocks_won, "lost": self.match.stocks_lost},
        }

    # ------------------------------------------------------------------ thread
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def run() -> None:
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception:
                    log.exception("fight loop error - disarming")
                    self.io.disarm("fight loop error")
                    time.sleep(0.2)

        self._thread = threading.Thread(target=run, name="fight-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.disarm("fight loop stopped")
        self._close_recording("stopped")

