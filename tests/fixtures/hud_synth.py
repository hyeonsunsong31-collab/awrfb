"""Synthetic UBG-like frames for testing HUD parsing. These test the ALGORITHM only; real recorded
frames from the owner's PC become replay fixtures once available."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SynthBar:
    roi: tuple[float, float, float, float]
    value: float
    color_bgr: tuple[int, int, int]
    direction: str = "ltr"
    trail: float = 0.0  # width (fraction of bar) of a bright damage trail after the fill
    gradient: float = 0.25  # brightness variation along the bar


def scene(w: int, h: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (h // 40 + 2, w // 40 + 2, 3), dtype=np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (0, 0), 6)


def draw_bar(img: np.ndarray, bar: SynthBar) -> None:
    H, W = img.shape[:2]
    x0 = int(bar.roi[0] * W)
    y0 = int(bar.roi[1] * H)
    bw = int(bar.roi[2] * W)
    bh = max(2, int(bar.roi[3] * H))
    # border + empty interior
    cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + bw + 1, y0 + bh + 1), (15, 15, 15), -1)
    img[y0 : y0 + bh, x0 : x0 + bw] = (45, 45, 48)
    n_fill = round(bar.value * bw)
    n_trail = round(bar.trail * bw)
    base = np.array(bar.color_bgr, dtype=np.float32)
    for i in range(n_fill):
        f = 1.0 - bar.gradient / 2 + bar.gradient * (i / max(1, bw - 1))
        c = np.clip(base * f, 0, 255).astype(np.uint8)
        col = x0 + i if bar.direction == "ltr" else x0 + bw - 1 - i
        img[y0 : y0 + bh, col] = c
    for i in range(n_trail):
        pos = n_fill + i
        if pos >= bw:
            break
        col = x0 + pos if bar.direction == "ltr" else x0 + bw - 1 - pos
        img[y0 : y0 + bh, col] = (235, 235, 245)  # bright "recent damage" segment


def make_frame(
    bars: dict[str, SynthBar],
    size: tuple[int, int] = (1920, 1080),
    seed: int = 0,
    jpeg_quality: int | None = None,
    blur: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    w, h = size
    img = scene(w, h, seed)
    for b in bars.values():
        draw_bar(img, b)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if jpeg_quality is not None:
        _ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


GREEN = (60, 200, 60)
RED = (40, 40, 220)
YELLOW = (40, 210, 230)
BLUE = (220, 140, 40)

DEFAULT_ROIS = {
    "own_hp": (0.05, 0.92, 0.30, 0.022),
    "own_stamina": (0.05, 0.955, 0.25, 0.012),
    "opp_hp": (0.65, 0.92, 0.30, 0.022),
    "own_guard": (0.40, 0.90, 0.20, 0.012),
}


def ubg_like(own_hp=1.0, opp_hp=1.0, stamina=1.0, guard=1.0, **kw) -> np.ndarray:
    bars = {
        "own_hp": SynthBar(DEFAULT_ROIS["own_hp"], own_hp, GREEN, "ltr"),
        "own_stamina": SynthBar(DEFAULT_ROIS["own_stamina"], stamina, YELLOW, "ltr"),
        "opp_hp": SynthBar(DEFAULT_ROIS["opp_hp"], opp_hp, GREEN, "rtl"),
        "own_guard": SynthBar(DEFAULT_ROIS["own_guard"], guard, BLUE, "ltr"),
    }
    return make_frame(bars, **kw)


ANCHOR_ROI = (0.05, 0.875, 0.12, 0.03)


def draw_anchor(img: np.ndarray, roi=ANCHOR_ROI, text: str = "JarvisBot") -> None:
    H, W = img.shape[:2]
    x0, y0 = int(roi[0] * W), int(roi[1] * H)
    w, h = int(roi[2] * W), int(roi[3] * H)
    img[y0 : y0 + h, x0 : x0 + w] = (25, 20, 30)
    cv2.putText(img, text, (x0 + 4, y0 + int(h * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, h / 38, (240, 240, 240), 2, cv2.LINE_AA)


def ubg_with_anchor(**kw) -> np.ndarray:
    scale = kw.pop("scale", 1.0)
    img = ubg_like(**kw)
    draw_anchor(img)
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img
