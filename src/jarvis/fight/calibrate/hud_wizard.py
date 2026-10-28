"""`jarvis calibrate hud`: locate UBG's fight HUD on THIS PC.

Run it while a UBG match is on screen (private server against a friend, or any match you are in).
The owner drags a box around each bar; the fill colour is estimated automatically, a static HUD
"anchor" is captured, and the result is verified live for a few seconds before it is saved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from jarvis.fight.calibrate.roi_ui import RoiPicker
from jarvis.fight.perception.hud_bars import HudReader, anchor_texture_ok, estimate_fill_ranges
from jarvis.fight.perception.layout import (
    AnchorSpec,
    BarSpec,
    HudLayout,
    pixels_to_roi,
    roi_to_pixels,
    save_layout,
)
from jarvis.platform.base import Frame

log = logging.getLogger(__name__)

BAR_STEPS: list[tuple[str, bool, str]] = [
    ("own_hp", True, "Drag a tight box around YOUR health bar (just the coloured bar)."),
    ("opp_hp", True, "Drag a tight box around your OPPONENT's health bar."),
    ("own_stamina", False, "Drag a box around YOUR stamina bar (skip if you can't see one)."),
    ("own_guard", False, "Drag a box around YOUR block/guard gauge (skip if not visible)."),
    ("opp_stamina", False, "Drag a box around your OPPONENT's stamina bar (skip if not visible)."),
]
ANCHOR_PROMPT = (
    "Drag a box around something in the fight HUD that NEVER changes during a fight - e.g. your name "
    "or an icon next to your health bar. This lets JARVIS tell the real HUD from popups."
)


@dataclass
class HudCalibrationResult:
    layout: HudLayout
    live_combat_fraction: float
    live_conf: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    snapshot: Path | None = None


def annotate(frame_bgr: np.ndarray, layout: HudLayout, reader: HudReader | None = None) -> np.ndarray:
    img = frame_bgr.copy()
    rd = reader.read(frame_bgr) if reader else None
    for name, spec in layout.bars.items():
        x, y, w, h = roi_to_pixels(spec.roi, img.shape)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 229, 0), 2)
        label = name
        if rd is not None:
            v = rd.values.get(name)
            label += f" {'--' if v is None else f'{v:.2f}'} c{rd.conf.get(name, 0):.2f}"
        cv2.putText(img, label, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 229, 0), 1, cv2.LINE_AA)
    for a in layout.anchors:
        x, y, w, h = roi_to_pixels(a.roi, img.shape)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 200), 2)
    return img


def calibrate_hud(
    get_frame,  # () -> Frame | None   (newest frame from the Roblox window)
    picker: RoiPicker,
    calibration_dir: Path,
    *,
    cursor_visible_in_fight: bool,
    camera_mode: str = "default",
    verify_seconds: float = 3.0,
) -> HudCalibrationResult:
    frame: Frame | None = get_frame()
    if frame is None:
        raise RuntimeError("no frame from the Roblox window - is UBG visible (not minimized)?")
    img = frame.image
    bars: dict[str, BarSpec] = {}
    warnings: list[str] = []
    for name, required, prompt in BAR_STEPS:
        box = picker.select(img, name, prompt, optional=not required)
        if box is None:
            if required:
                raise RuntimeError(f"{name} is required for fighting")
            continue
        x, y, w, h = box
        crop = img[y : y + h, x : x + w]
        if crop.size == 0:
            raise RuntimeError(f"empty selection for {name}")
        bars[name] = BarSpec(roi=pixels_to_roi(x, y, w, h, img.shape), fill=estimate_fill_ranges(crop), required=required)

    anchors: list[AnchorSpec] = []
    abox = picker.select(img, "anchor", ANCHOR_PROMPT, optional=True)
    if abox is not None:
        x, y, w, h = abox
        gray = cv2.cvtColor(img[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        if anchor_texture_ok(gray):
            (calibration_dir / "anchors").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(calibration_dir / "anchors" / "hud_anchor.png"), gray)
            anchors.append(AnchorSpec(name="hud_anchor", roi=pixels_to_roi(x, y, w, h, img.shape), template_file="anchors/hud_anchor.png"))
        else:
            warnings.append("anchor region is too plain to match reliably; pick something with text or an icon")
    else:
        warnings.append("no HUD anchor selected: dark popups over the bars are harder to detect")

    layout = HudLayout(
        frame_size=(img.shape[1], img.shape[0]),
        bars=bars,
        anchors=anchors,
        cursor_closes_gate=not cursor_visible_in_fight,
        camera_mode=camera_mode,
    )
    reader = HudReader(layout, calibration_dir=calibration_dir)

    # Live verification: the HUD must be recognized on (almost) every frame of a real fight.
    combat = 0
    total = 0
    confs: dict[str, list[float]] = {k: [] for k in bars}
    deadline = time.monotonic() + verify_seconds
    last_index = -1
    while time.monotonic() < deadline or total == 0:
        f = get_frame()
        if f is None or f.index == last_index:
            if time.monotonic() > deadline + 2:
                break
            time.sleep(0.01)
            continue
        last_index = f.index
        rd = reader.read(f.image, f.t_ns)
        total += 1
        combat += int(rd.combat_hud)
        for k, c in rd.conf.items():
            confs[k].append(c)
    frac = combat / max(1, total)
    live_conf = {k: float(np.median(v)) if v else 0.0 for k, v in confs.items()}
    if frac < 0.9:
        warnings.append(f"HUD recognized on only {frac:.0%} of frames - re-select the bars more tightly")
    for k, c in live_conf.items():
        if c < bars[k].min_conf:
            warnings.append(f"{k} reads with low confidence ({c:.2f})")

    calibration_dir.mkdir(parents=True, exist_ok=True)
    save_layout(layout, calibration_dir)
    snap = calibration_dir / "hud_calibration.png"
    cv2.imwrite(str(snap), annotate(img, layout, reader))
    return HudCalibrationResult(layout, frac, live_conf, warnings, snap)
