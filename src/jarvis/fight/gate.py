"""Per-frame input gate. Combat inputs are only allowed while the CURRENT frame positively confirms
the combat HUD (an allowlist), Roblox is in front, no known popup is visible, chat is closed and the
owner is not touching the keyboard/mouse."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.fight.states import FightState

INPUT_STATES = (FightState.FIGHTING, FightState.NAVIGATING, FightState.CALIBRATING)


@dataclass
class GateStatus:
    t_ns: int = 0
    foreground: bool = False
    combat_hud: bool = False
    hud_conf: float = 0.0
    popup: str | None = None
    cursor_visible: bool = True
    chat_open: bool = False
    physical_input_recent: bool = False
    # Whether a visible OS cursor should close the gate (set by calibration: in UBG fights the
    # cursor is locked/hidden; a visible cursor means a menu or dialog is open).
    cursor_closes_gate: bool = True

    def check(self, state: FightState) -> tuple[bool, str]:
        if state not in INPUT_STATES:
            return False, f"no input allowed in state {state.value}"
        if not self.foreground:
            return False, "Roblox is not the foreground window"
        if self.physical_input_recent:
            return False, "owner is using the keyboard/mouse"
        if self.popup:
            return False, f"popup detected: {self.popup}"
        if self.chat_open:
            return False, "chat box is open"
        if self.cursor_closes_gate and self.cursor_visible:
            return False, "OS cursor is visible (menu/dialog open?)"
        if state == FightState.FIGHTING and not self.combat_hud:
            return False, f"combat HUD not confirmed (conf {self.hud_conf:.2f})"
        return True, "open"
