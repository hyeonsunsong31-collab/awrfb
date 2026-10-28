"""Calibrated screen layout of UBG's HUD on THIS PC (written by `jarvis calibrate hud`).

Regions are normalized (x, y, w, h) fractions of the captured Roblox frame. The layout also stores
the frame size it was calibrated at; fightd refuses to fight if the live frame size differs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.config import read_yaml

Roi = tuple[float, float, float, float]


class ColorRange(BaseModel):
    """OpenCV HSV range (H 0-179, S/V 0-255). h[0] > h[1] means the hue range wraps (reds)."""

    model_config = ConfigDict(extra="forbid")
    h: tuple[int, int]
    s: tuple[int, int] = (60, 255)
    v: tuple[int, int] = (60, 255)


def _default_empty() -> list[ColorRange]:
    # Depleted bar background: dark and not very saturated. Bright unknown pixels (dialogs, damage
    # trails) are deliberately NOT "empty" - they lower confidence instead.
    return [ColorRange(h=(0, 179), s=(0, 110), v=(0, 115))]


class BarSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roi: Roi
    fill: list[ColorRange]
    empty: list[ColorRange] = Field(default_factory=_default_empty)
    direction: Literal["ltr", "rtl", "auto"] = "auto"
    orientation: Literal["h", "v"] = "h"
    min_conf: float = 0.6
    required: bool = False  # must be readable for the combat HUD to count as present


class AnchorSpec(BaseModel):
    """A static piece of the fight HUD (e.g. your name or an icon next to your HP bar), matched by
    normalized cross-correlation every frame. Popups that cover the HUD - even dark ones that look
    like an empty bar - break the match and close the input gate."""

    model_config = ConfigDict(extra="forbid")
    name: str
    roi: Roi
    template_file: str  # PNG relative to the calibration directory
    min_ncc: float = 0.7
    search_frac: float = 0.008  # search margin as a fraction of frame width


class QueueBoxSpec(BaseModel):
    """Colours of the ring queue boxes: blue = free, green = queued (from the community wiki)."""

    model_config = ConfigDict(extra="forbid")
    free: list[ColorRange] = Field(default_factory=lambda: [ColorRange(h=(95, 125), s=(90, 255), v=(70, 255))])
    queued: list[ColorRange] = Field(default_factory=lambda: [ColorRange(h=(45, 85), s=(90, 255), v=(70, 255))])
    search_roi: Roi = (0.0, 0.35, 1.0, 0.65)  # lower part of the screen (the floor)
    min_area_frac: float = 0.002


class HudLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    frame_size: tuple[int, int]
    bars: dict[str, BarSpec]
    anchors: list[AnchorSpec] = Field(default_factory=list)
    queue_box: QueueBoxSpec = Field(default_factory=QueueBoxSpec)
    chat_roi: Roi | None = None  # where the chat input box appears when open (optional)
    cursor_closes_gate: bool = True
    camera_mode: str = "default"
    fov_note: str = ""
    ui_scale_note: str = ""

    def required_bars(self) -> list[str]:
        return [k for k, b in self.bars.items() if b.required]

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


def load_layout(calibration_dir: Path) -> HudLayout | None:
    p = calibration_dir / "hud_layout.yaml"
    if not p.exists():
        return None
    return HudLayout.model_validate(read_yaml(p))


def save_layout(layout: HudLayout, calibration_dir: Path) -> Path:
    calibration_dir.mkdir(parents=True, exist_ok=True)
    p = calibration_dir / "hud_layout.yaml"
    data = layout.model_dump(mode="json")
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def roi_to_pixels(roi: Roi, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[0], shape[1]
    x0 = round(roi[0] * w)
    y0 = round(roi[1] * h)
    x1 = round((roi[0] + roi[2]) * w)
    y1 = round((roi[1] + roi[3]) * h)
    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def pixels_to_roi(x: int, y: int, w: int, h: int, shape: tuple[int, ...]) -> Roi:
    H, W = shape[0], shape[1]
    return (x / W, y / H, w / W, h / H)
