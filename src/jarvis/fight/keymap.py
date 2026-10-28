"""UBG action -> physical input bindings (see configs/ubg/keymap.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from jarvis.common.config import read_yaml
from jarvis.common.paths import config_dir
from jarvis.platform.win.scancodes import NEVER_SEND_GAME, normalize_key

MOVEMENT = ("move_forward", "move_back", "move_left", "move_right")


class Binding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str | None = None
    mouse: Literal["left", "right", "middle"] | None = None
    combo: list[str] | None = None  # tap each action in order (e.g. feint = heavy then light)
    chord: list[str] | None = None  # [held action, tapped action] (e.g. dash_left = hold A + tap Space)
    enabled: bool = True
    verified: bool = False

    @model_validator(mode="after")
    def _one_kind(self) -> Binding:
        kinds = [x is not None for x in (self.key, self.mouse, self.combo, self.chord)]
        if sum(kinds) != 1:
            raise ValueError("a binding needs exactly one of key/mouse/combo/chord")
        if self.key is not None:
            self.key = normalize_key(str(self.key))
        if self.chord is not None and len(self.chord) != 2:
            raise ValueError("chord must be [held_action, tapped_action]")
        return self

    @property
    def kind(self) -> str:
        if self.key is not None:
            return "key"
        if self.mouse is not None:
            return "mouse"
        if self.combo is not None:
            return "combo"
        return "chord"


class CameraBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mouse_relative: bool = False
    max_step_px: int = 40


class Keymap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: dict[str, Binding]
    camera: CameraBinding = Field(default_factory=CameraBinding)

    @model_validator(mode="after")
    def _validate(self) -> Keymap:
        for name, b in self.actions.items():
            if b.key and b.key in NEVER_SEND_GAME:
                raise ValueError(f"action {name!r} is bound to forbidden key {b.key!r}")
            for ref in (b.combo or []) + (b.chord or []):
                if ref not in self.actions:
                    raise ValueError(f"action {name!r} references unknown action {ref!r}")
                if self.actions[ref].kind not in ("key", "mouse"):
                    raise ValueError(f"action {name!r} may only reference key/mouse actions")
        return self

    def usable(self, action: str) -> bool:
        b = self.actions.get(action)
        if b is None or not b.enabled:
            return False
        refs = (b.combo or []) + (b.chord or [])
        return all(self.actions[r].enabled for r in refs)

    def primitive_keys(self, enabled_only: bool = True) -> set[str]:
        return {
            b.key for b in self.actions.values() if b.key is not None and (b.enabled or not enabled_only)
        }

    def mouse_buttons(self, enabled_only: bool = True) -> set[str]:
        return {
            b.mouse for b in self.actions.values() if b.mouse is not None and (b.enabled or not enabled_only)
        }

    def all_physical(self) -> set[str]:
        """Every physical input name this keymap could ever press ("w", "mouse:left", ...)."""
        return self.primitive_keys(False) | {f"mouse:{m}" for m in self.mouse_buttons(False)}


def load_keymap(calibration_dir: Path | None = None, relative: str = "ubg/keymap.yaml") -> Keymap:
    """Calibrated keymap (written by `jarvis calibrate keys`) wins over the repo default."""
    if calibration_dir is not None:
        cal = calibration_dir / "keymap.yaml"
        if cal.exists():
            return Keymap.model_validate(read_yaml(cal))
    return Keymap.model_validate(read_yaml(config_dir() / relative))


def save_keymap(km: Keymap, calibration_dir: Path) -> Path:
    calibration_dir.mkdir(parents=True, exist_ok=True)
    out = calibration_dir / "keymap.yaml"
    out.write_text(yaml.safe_dump(km.model_dump(exclude_none=True), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out
