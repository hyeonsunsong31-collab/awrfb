"""`jarvis calibrate keys` / `latency`: prove each keymap action really reaches UBG.

Owner-started only (private server / training spot). For every action JARVIS presses it through the
normal actuator (CALIBRATING state, same gate) and watches the real frames for the game's reaction:
motion around the character and, for actions that cost stamina, a stamina drop. Each action is
verified when it is seen to work on >= 90% of tries. Input-to-visual latency comes from the WGC frame
timestamps. Everything is recorded as a proof bundle.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from statistics import median

import cv2
import numpy as np

from jarvis.fight.actuator import Command
from jarvis.fight.perception.hud_bars import HudReader
from jarvis.fight.states import FightState

log = logging.getLogger(__name__)

# Where our own character usually is in UBG's third-person view (refined after calibration).
SELF_REGION = (0.30, 0.25, 0.40, 0.70)
STAMINA_ACTIONS = {"dash", "dash_forward", "dash_back", "dash_left", "dash_right", "light", "heavy", "feint"}
# Hold-type actions need a longer press to show a visible stance.
HOLD_MS = {"block": 350.0, "move_forward": 250.0, "move_back": 250.0, "move_left": 250.0, "move_right": 250.0}


@dataclass
class Trial:
    t_press_ns: int
    detected: bool
    latency_ms: float | None
    motion_peak: float
    stamina_delta: float | None
    denied: str | None = None  # why the actuator refused the press (focus lost, popup, ...)


@dataclass
class ActionReport:
    action: str
    trials: list[Trial] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return sum(t.detected for t in self.trials) / max(1, len(self.trials))

    @property
    def median_latency_ms(self) -> float | None:
        lat = [t.latency_ms for t in self.trials if t.latency_ms is not None]
        return float(median(lat)) if lat else None

    @property
    def verified(self) -> bool:
        return len(self.trials) > 0 and self.success_rate >= 0.9

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "success_rate": self.success_rate,
            "median_latency_ms": self.median_latency_ms,
            "verified": self.verified,
            "trials": [asdict(t) for t in self.trials],
        }


def small_gray(img: np.ndarray, region=SELF_REGION, size=(160, 120)) -> np.ndarray:
    H, W = img.shape[:2]
    x, y, w, h = int(region[0] * W), int(region[1] * H), int(region[2] * W), int(region[3] * H)
    crop = img[y : y + h, x : x + w]
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32)


def motion(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


class KeyVerifier:
    def __init__(
        self,
        io,  # GameIO
        hud: HudReader | None,
        *,
        log_event: Callable[[dict], None] | None = None,
        add_frame: Callable[[np.ndarray, dict], None] | None = None,
        watch_ms: int = 700,
        cooldown_ms: int = 1100,
    ) -> None:
        self.io = io
        self.hud = hud
        self.log_event = log_event or (lambda e: None)
        self.add_frame = add_frame or (lambda img, row: None)
        self.watch_ms = watch_ms
        self.cooldown_ms = cooldown_ms
        self._last_index = 0
        self.noise_threshold = 2.0

    def _next_frame(self, timeout: float = 0.2):
        f = self.io.frames.get_newer(self._last_index, timeout)
        if f is not None:
            self._last_index = f.index
            self.io.actuator.heartbeat()
            self.io.actuator.update_gate(self.io.base_gate(cursor_closes_gate=False))
            row = {"fi": f.index, "t_ns": f.t_ns, "state": self.io.sm.state.value}
            if self.hud is not None:
                rd = self.hud.read(f.image, f.t_ns)
                row["hud"] = {k: v for k, v in rd.values.items()}
            self.add_frame(f.image, row)
        return f

    def _idle_for(self, ms: float) -> None:
        end = time.monotonic() + ms / 1000
        while time.monotonic() < end:
            self._next_frame(0.05)

    def measure_noise(self, seconds: float = 1.5) -> float:
        vals = []
        prev = None
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            f = self._next_frame()
            if f is None:
                continue
            g = small_gray(f.image)
            if prev is not None:
                vals.append(motion(g, prev))
            prev = g
        if not vals:
            raise RuntimeError("no frames - is UBG visible?")
        arr = np.array(vals)
        self.noise_threshold = max(2.0, float(arr.mean() + 4 * arr.std()))
        return self.noise_threshold

    def _stamina(self, img) -> float | None:
        if self.hud is None:
            return None
        return self.hud.read(img).values.get("own_stamina")

    def try_action(self, action: str) -> Trial:
        base = self._next_frame(0.3)
        if base is None:
            raise RuntimeError("lost frames from Roblox")
        g0 = small_gray(base.image)
        st0 = self._stamina(base.image)
        denied_before = self.io.actuator.stats["denied"]
        t_press = time.perf_counter_ns()
        self.io.actuator.submit([Command(action, dur_ms=HOLD_MS.get(action), reason="calibration")])
        self.log_event({"t_ns": t_press, "type": "calibration_press", "action": action})
        detected = False
        latency = None
        peak = 0.0
        st_min = st0
        end = time.monotonic() + self.watch_ms / 1000
        while time.monotonic() < end:
            f = self._next_frame(0.05)
            if f is None:
                continue
            m = motion(small_gray(f.image), g0)
            peak = max(peak, m)
            st = self._stamina(f.image)
            if st is not None and st_min is not None:
                st_min = min(st_min, st)
            stamina_drop = st0 is not None and st_min is not None and (st0 - st_min) > 0.02
            if not detected and (m > self.noise_threshold or (action in STAMINA_ACTIONS and stamina_drop)):
                detected = True
                latency = max(0.0, (f.t_ns - t_press) / 1e6)
        delta = (st_min - st0) if (st0 is not None and st_min is not None) else None
        denied = self.io.actuator.last_denial if self.io.actuator.stats["denied"] > denied_before else None
        return Trial(t_press, detected, latency, peak, delta, denied)

    def run(self, actions: list[str], repeats: int = 10) -> list[ActionReport]:
        io = self.io
        if not io.focus_game():
            raise RuntimeError("could not bring Roblox to the front")
        io.sm.transition(FightState.CALIBRATING, "owner started key calibration")
        reports: list[ActionReport] = []
        try:
            self.measure_noise()
            for action in actions:
                rep = ActionReport(action)
                for _ in range(repeats):
                    if io.sm.state != FightState.CALIBRATING:
                        raise RuntimeError(f"calibration stopped: {io.actuator.last_denial or 'disarmed'}")
                    rep.trials.append(self.try_action(action))
                    self._idle_for(self.cooldown_ms)
                reports.append(rep)
                log.info("%s: %.0f%% detected, median latency %s ms", action, rep.success_rate * 100, rep.median_latency_ms)
        finally:
            io.disarm("key calibration finished")
        return reports
