"""Ring queue boxes: stand in a blue box, it turns green when you are queued (UBG community wiki)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from jarvis.fight.perception.hud_bars import color_mask
from jarvis.fight.perception.layout import QueueBoxSpec, roi_to_pixels


@dataclass
class Box:
    kind: str  # "free" | "queued"
    cx: float  # normalized to the full frame
    cy: float
    area: float  # fraction of the frame


@dataclass
class QueueStatus:
    boxes: list[Box] = field(default_factory=list)
    queued: bool = False  # a green box right under/in front of us

    def nearest_free(self) -> Box | None:
        free = [b for b in self.boxes if b.kind == "free"]
        # Closest to the bottom-centre (where our feet are in third person) and largest.
        return max(free, key=lambda b: b.area * 4 + b.cy - abs(b.cx - 0.5), default=None)


def find_boxes(img_bgr: np.ndarray, spec: QueueBoxSpec) -> QueueStatus:
    x, y, w, h = roi_to_pixels(spec.search_roi, img_bgr.shape)
    H, W = img_bgr.shape[:2]
    roi = img_bgr[y : y + h, x : x + w]
    if roi.size == 0:
        return QueueStatus()
    small = cv2.resize(roi, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    out = QueueStatus()
    sx = w / small.shape[1]
    sy = h / small.shape[0]
    for kind, ranges in (("free", spec.free), ("queued", spec.queued)):
        mask = color_mask(hsv, ranges).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            area_px = stats[i, cv2.CC_STAT_AREA] * sx * sy
            frac = area_px / (W * H)
            if frac < spec.min_area_frac:
                continue
            cx = (x + cents[i][0] * sx) / W
            cy = (y + cents[i][1] * sy) / H
            out.boxes.append(Box(kind, float(cx), float(cy), float(frac)))
    # Queued = a green box in the lower-centre of the screen (we are standing in it).
    out.queued = any(b.kind == "queued" and abs(b.cx - 0.5) < 0.2 and b.cy > 0.55 for b in out.boxes)
    return out
