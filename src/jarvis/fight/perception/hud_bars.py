"""Read UBG's HUD bars (HP, stamina, guard) from the captured frame.

For each column of the bar region we decide "fill-coloured" (calibrated HSV ranges) or "empty"
(dark/grey). The bar value is the best clean split point along the bar: it maximises the number of
columns consistent with [fill ... fill | empty ... empty]. That makes the reading robust to
gradients, damage trails (bright segments that are neither fill nor empty), JPEG noise and blur.

The HUD counts as present only when every required bar reads with good confidence AND at least one of
them is visibly filled - so a black loading screen cannot masquerade as "both HP bars empty".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from jarvis.fight.perception.layout import AnchorSpec, BarSpec, ColorRange, HudLayout, roi_to_pixels


@dataclass
class BarReading:
    value: float | None
    conf: float
    direction: str = "ltr"


@dataclass
class HudReading:
    t_ns: int
    values: dict[str, float | None] = field(default_factory=dict)
    conf: dict[str, float] = field(default_factory=dict)
    combat_hud: bool = False
    hud_conf: float = 0.0
    size_ok: bool = True
    anchors: dict[str, float] = field(default_factory=dict)

    def get(self, name: str, default: float | None = None) -> float | None:
        v = self.values.get(name)
        return default if v is None else v


def color_mask(hsv: np.ndarray, ranges: list[ColorRange]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for r in ranges:
        s_lo, s_hi = r.s
        v_lo, v_hi = r.v
        h_lo, h_hi = r.h
        if h_lo <= h_hi:
            lo = np.array([h_lo, s_lo, v_lo], np.uint8)
            hi = np.array([h_hi, s_hi, v_hi], np.uint8)
            mask |= cv2.inRange(hsv, lo, hi) > 0
        else:  # wrapping hue range (reds)
            mask |= cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo], np.uint8), np.array([179, s_hi, v_hi], np.uint8)) > 0
            mask |= cv2.inRange(hsv, np.array([0, s_lo, v_lo], np.uint8), np.array([h_hi, s_hi, v_hi], np.uint8)) > 0
    return mask


def _best_split(filled: np.ndarray) -> tuple[int, int]:
    """argmax_k  #filled[:k] + #(~filled)[k:]  -> (k, score)."""
    n = filled.size
    cs = np.concatenate([[0], np.cumsum(filled, dtype=np.int64)])
    total = int(cs[-1])
    k = np.arange(n + 1)
    score = cs + ((n - k) - (total - cs))
    best = int(np.argmax(score))
    # Prefer the largest k among ties so a full bar reads 1.0, not the first tied split.
    ties = np.flatnonzero(score == score[best])
    best = int(ties[-1]) if total > 0 else int(ties[0])
    return best, int(score[best])


def read_bar(frame_bgr: np.ndarray, spec: BarSpec) -> BarReading:
    x, y, w, h = roi_to_pixels(spec.roi, frame_bgr.shape)
    if w < 4 or h < 1:
        return BarReading(None, 0.0)
    roi = frame_bgr[y : y + h, x : x + w]
    if spec.orientation == "v":
        roi = np.ascontiguousarray(np.rot90(roi, k=-1))  # bottom of the bar -> left
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    fill = color_mask(hsv, spec.fill)
    empty = (~fill) & color_mask(hsv, spec.empty)
    col_fill = fill.mean(axis=0)
    col_empty = empty.mean(axis=0)
    filled = col_fill >= 0.5
    n = filled.size

    candidates: list[tuple[int, int, str]] = []
    if spec.direction in ("ltr", "auto"):
        k, s = _best_split(filled)
        candidates.append((s, k, "ltr"))
    if spec.direction in ("rtl", "auto"):
        k, s = _best_split(filled[::-1])
        candidates.append((s, k, "rtl"))
    score, k, direction = max(candidates, key=lambda c: (c[0], c[2] == "ltr"))

    consistency = score / n
    recognized = float(((col_fill >= 0.5) | (col_empty >= 0.5)).mean())
    conf = float(consistency * recognized)
    return BarReading(value=k / n, conf=conf, direction=direction)


class AnchorMatcher:
    def __init__(self, spec: AnchorSpec, template_gray: np.ndarray, calib_width: int) -> None:
        self.spec = spec
        self.template = template_gray
        self.calib_width = calib_width
        self._cache: tuple[int, np.ndarray] | None = None

    def _template_for(self, frame_w: int) -> np.ndarray:
        if self._cache is None or self._cache[0] != frame_w:
            s = frame_w / self.calib_width
            t = self.template
            if abs(s - 1.0) > 1e-3:
                t = cv2.resize(t, (max(4, int(t.shape[1] * s)), max(4, int(t.shape[0] * s))), interpolation=cv2.INTER_AREA)
            self._cache = (frame_w, t)
        return self._cache[1]

    def score(self, frame_bgr: np.ndarray) -> float:
        H, W = frame_bgr.shape[:2]
        x, y, w, h = roi_to_pixels(self.spec.roi, frame_bgr.shape)
        m = round(self.spec.search_frac * W) + 1
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(W, x + w + m), min(H, y + h + m)
        tmpl = self._template_for(W)
        search = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        if search.shape[0] < tmpl.shape[0] or search.shape[1] < tmpl.shape[1]:
            return 0.0
        res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
        v = float(res.max())
        return 0.0 if np.isnan(v) else v


def anchor_texture_ok(template_gray: np.ndarray, min_std: float = 12.0) -> bool:
    """Uniform patches make NCC meaningless; calibration rejects them."""
    return float(template_gray.std()) >= min_std


class HudReader:
    def __init__(
        self,
        layout: HudLayout,
        min_visible_fill: float = 0.04,
        calibration_dir: Path | None = None,
        anchor_templates: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.layout = layout
        self.min_visible_fill = min_visible_fill
        self.anchors: list[AnchorMatcher] = []
        templates = dict(anchor_templates or {})
        for a in layout.anchors:
            t = templates.get(a.name)
            if t is None and calibration_dir is not None:
                p = calibration_dir / a.template_file
                if p.exists():
                    t = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if t is not None:
                self.anchors.append(AnchorMatcher(a, t, layout.frame_size[0]))

    def read(self, frame_bgr: np.ndarray, t_ns: int = 0) -> HudReading:
        out = HudReading(t_ns=t_ns)
        fh, fw = frame_bgr.shape[:2]
        cw, ch = self.layout.frame_size
        # Normalized ROIs tolerate scaling, but a different aspect ratio means a different layout.
        out.size_ok = abs((fw / fh) - (cw / ch)) < 0.02
        anchors_ok = True
        for am in self.anchors:
            s = am.score(frame_bgr)
            out.anchors[am.spec.name] = s
            if s < am.spec.min_ncc:
                anchors_ok = False
        req = self.layout.required_bars()
        confs = []
        any_visible = False
        all_req_ok = out.size_ok
        for name, spec in self.layout.bars.items():
            r = read_bar(frame_bgr, spec)
            out.values[name] = r.value if r.conf >= spec.min_conf else None
            out.conf[name] = r.conf
            if name in req:
                confs.append(r.conf)
                if r.conf < spec.min_conf:
                    all_req_ok = False
                elif (r.value or 0.0) >= self.min_visible_fill:
                    any_visible = True
        out.hud_conf = float(min(confs)) if confs else 0.0
        out.combat_hud = bool(req) and all_req_ok and any_visible and anchors_ok
        return out


def estimate_fill_ranges(roi_bgr: np.ndarray, hue_margin: int = 12) -> list[ColorRange]:
    """Guess the fill colour of a (mostly full) bar from its pixels - used by calibration."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    sat_mask = (hsv[:, 1] >= 80) & (hsv[:, 2] >= 80)
    px = hsv[sat_mask]
    if len(px) < 10:  # unsaturated bar (white/grey): fall back to a brightness range
        v = hsv[:, 2]
        lo = int(max(0, np.percentile(v, 10) - 10))
        return [ColorRange(h=(0, 179), s=(0, 60), v=(max(lo, 150), 255))]
    hist = np.bincount(px[:, 0], minlength=180)
    peak = int(np.argmax(np.convolve(np.concatenate([hist[-5:], hist, hist[:5]]), np.ones(11), "valid")))
    # Circular spread of hues around the peak, covering ~95% of saturated pixels.
    d = ((px[:, 0] - peak + 90) % 180) - 90
    spread = int(min(40, max(hue_margin, np.percentile(np.abs(d), 95) + 4)))
    h_lo = (peak - spread) % 180
    h_hi = (peak + spread) % 180
    s_lo = int(max(40, np.percentile(px[:, 1], 3) - 15))
    v_lo = int(max(50, np.percentile(px[:, 2], 3) - 20))
    return [ColorRange(h=(h_lo, h_hi), s=(s_lo, 255), v=(v_lo, 255))]
