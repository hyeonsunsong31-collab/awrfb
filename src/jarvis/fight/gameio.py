"""GameIO: the single owner of the live connection to the real Roblox window.

Bundles window attachment, Windows Graphics Capture (newest-frame slot), the physical-input monitor,
the panic hotkey and the actuator. Calibration and fightd both build on it, so every key that ever
reaches Roblox passes through the same actuator and gate.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from typing import Any

from jarvis.common.audit import AuditLog
from jarvis.common.config import JarvisConfig
from jarvis.fight.actuator import Actuator
from jarvis.fight.frames import FpsMeter, LatestFrame
from jarvis.fight.gate import GateStatus
from jarvis.fight.keymap import Keymap, load_keymap
from jarvis.fight.launcher import GameLauncher, GameSession
from jarvis.fight.states import FightState, FightStateMachine
from jarvis.platform.base import Frame, Platform, WindowRef
from jarvis.platform.win.scancodes import NEVER_SEND_GAME

log = logging.getLogger(__name__)


def _set_timer_resolution(enable: bool) -> None:
    """1 ms Windows timer resolution so key taps and frame waits are precise."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        winmm = ctypes.WinDLL("winmm")
        (winmm.timeBeginPeriod if enable else winmm.timeEndPeriod)(1)
    except OSError:
        pass


class GameIO:
    def __init__(
        self,
        cfg: JarvisConfig,
        plat: Platform,
        *,
        keymap: Keymap | None = None,
        audit: AuditLog | None = None,
        input_log: Callable[[dict[str, Any]], None] | None = None,
        on_panic: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.plat = plat
        paths = cfg.paths().ensure()
        self.keymap = keymap or load_keymap(paths.calibration, cfg.fight.keymap_file)
        self.launcher = GameLauncher(plat, cfg.fight.game, paths.run)
        self.audit = audit
        self.sm = FightStateMachine()
        self.frames = LatestFrame()
        self.fps = FpsMeter()
        self.window: WindowRef | None = None
        self.session: GameSession | None = None
        self._capture = None
        self._monitor = None
        self._hotkeys = None
        self._on_panic = on_panic
        self._input_log_cb = input_log
        sink = plat.new_input_sink(forbidden_keys=NEVER_SEND_GAME)
        self.sink = sink
        self.actuator = Actuator(
            sink,
            self.keymap,
            plat.windows,
            cfg.fight.actuator,
            input_log=self._input_log,
            audit=audit,
            on_watchdog=lambda why: self.disarm(why),
        )
        self.sm.on_change(lambda ch: self.actuator.set_state(ch.new))

    # ------------------------------------------------------------------ plumbing
    def _input_log(self, rec: dict[str, Any]) -> None:
        if self._input_log_cb is not None:
            self._input_log_cb(rec)

    def set_input_log(self, fn: Callable[[dict[str, Any]], None] | None) -> None:
        self._input_log_cb = fn

    def _on_frame(self, frame: Frame) -> None:
        self.fps.tick(frame.t_ns)
        self.frames.put(frame)

    # ------------------------------------------------------------------ lifecycle
    def attach(self) -> WindowRef:
        self.session = self.launcher.attach()
        w = self.launcher.window()
        if w is None:
            raise RuntimeError("Roblox window disappeared")
        self.window = w
        self.actuator.set_target(w)
        return w

    def start(self, *, capture: bool = True) -> None:
        if self.window is None:
            self.attach()
        assert self.window is not None
        _set_timer_resolution(True)
        mon = self.plat.new_input_monitor()
        # Share the actuator's synthetic-event log so our own input is never mistaken for the owner's.
        synthetic = getattr(mon, "synthetic", None)
        if synthetic is not None:
            self.actuator._synthetic = synthetic
        mon.start()
        self._monitor = mon
        hk = self.plat.new_hotkeys()
        for combo in self.cfg.safety.panic_hotkeys:
            hk.register(combo, lambda c=combo: self.panic(f"panic hotkey {c}"))
        hk.start()
        self._hotkeys = hk
        self.actuator.start()
        if capture:
            src = self.plat.new_frame_source(max_fps=self.cfg.fight.perception.max_process_fps)
            src.start(self.window, self._on_frame)
            self._capture = src

    def stop(self) -> None:
        self.disarm("GameIO stopped")
        self.actuator.stop()
        for comp in (self._capture, self._monitor, self._hotkeys):
            if comp is not None:
                try:
                    comp.stop()
                except Exception:
                    log.exception("failed to stop %s", comp)
        self._capture = self._monitor = self._hotkeys = None
        _set_timer_resolution(False)

    # ------------------------------------------------------------------ safety
    def disarm(self, reason: str) -> None:
        if self.sm.state != FightState.DISARMED:
            self.sm.disarm(reason)
        self.actuator.release_all(reason)

    def panic(self, reason: str) -> None:
        log.warning("PANIC: %s", reason)
        self.disarm(reason)
        if self.audit is not None:
            self.audit.append("panic", {"reason": reason})
        if self._on_panic is not None:
            self._on_panic(reason)

    # ------------------------------------------------------------------ gate
    def owner_active(self) -> bool:
        if self._monitor is None:
            return False
        last = self._monitor.last_physical_input_ns()
        return last > 0 and (time.perf_counter_ns() - last) < self.cfg.fight.actuator.physical_input_pause_ms * 1e6

    def base_gate(self, *, combat_hud: bool = False, hud_conf: float = 0.0, popup: str | None = None,
                  chat_open: bool = False, cursor_closes_gate: bool = True) -> GateStatus:
        fg = self.window is not None and self.plat.windows.is_foreground(self.window.hwnd)
        return GateStatus(
            t_ns=time.perf_counter_ns(),
            foreground=fg,
            combat_hud=combat_hud,
            hud_conf=hud_conf,
            popup=popup,
            cursor_visible=self.plat.cursor.cursor_visible(),
            chat_open=chat_open,
            physical_input_recent=self.owner_active(),
            cursor_closes_gate=cursor_closes_gate,
        )

    def focus_game(self) -> bool:
        if self.window is None:
            return False
        return self.plat.windows.focus(self.window.hwnd)
