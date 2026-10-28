"""UI guard: is anything other than the fight on screen?

The primary guard is the allowlist (the calibrated combat HUD + anchors must be present). This module
adds three negative checks:
  * modal dim: Roblox dialogs (purchase prompt, menus) dim the whole screen and show a central panel;
  * popup templates: screenshots of UBG UIs the owner captured during calibration (shop, spins,
    reward wheel) are matched each frame at low resolution;
  * chat: if a chat input region was calibrated, a bright text box there means chat is open
    (keys would type into chat).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from jarvis.fight.perception.layout import HudLayout, roi_to_pixels

log = logging.getLogger(__name__)

THUMB = (160, 90)


@dataclass
class UiGuardResult:
    popup: str | None = None
    chat_open: bool = False
    luminance: float = 0.0
    dim_ratio: float = 1.0


def thumb_gray(img: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


class UiGuard:
    def __init__(
        self,
        layout: HudLayout | None,
        popup_dir: Path | None = None,
        *,
        dim_ratio: float = 0.62,
        template_ncc: float = 0.82,
    ) -> None:
        self.layout = layout
        self.dim_ratio_threshold = dim_ratio
        self.template_ncc = template_ncc
        self._baseline: float | None = None
        self.templates: dict[str, np.ndarray] = {}
        if popup_dir is not None and popup_dir.exists():
            for p in sorted(popup_dir.glob("*.png")):
                t = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if t is not None:
                    self.templates[p.stem] = cv2.resize(t, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)

    @staticmethod
    def save_popup_template(img_bgr: np.ndarray, popup_dir: Path, name: str) -> Path:
        popup_dir.mkdir(parents=True, exist_ok=True)
        p = popup_dir / f"{name}.png"
        cv2.imwrite(str(p), cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA))
        return p

    def _ncc(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return 0.0 if den == 0 else float((a * b).sum() / den)

    def check(self, img_bgr: np.ndarray, combat_hud: bool) -> UiGuardResult:
        g = thumb_gray(img_bgr)
        lum = float(g.mean())
        res = UiGuardResult(luminance=lum)
        if combat_hud:
            # learn the normal in-fight brightness (slow EMA) only from frames that look like a fight
            self._baseline = lum if self._baseline is None else 0.98 * self._baseline + 0.02 * lum
        if self._baseline:
            res.dim_ratio = lum / max(1.0, self._baseline)
            if res.dim_ratio < self.dim_ratio_threshold:
                H, W = g.shape
                center = g[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4]
                border = np.concatenate([g[: H // 6].ravel(), g[-H // 6 :].ravel()])
                if center.mean() > border.mean() * 1.25:
                    res.popup = "modal_dialog"
        for name, t in self.templates.items():
            if self._ncc(g, t) >= self.template_ncc:
                res.popup = name
                break
        if self.layout is not None and self.layout.chat_roi is not None:
            x, y, w, h = roi_to_pixels(self.layout.chat_roi, img_bgr.shape)
            if w > 0 and h > 0:
                hsv = cv2.cvtColor(img_bgr[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
                bright = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 40)
                res.chat_open = bool(bright.mean() > 0.5)
        return res
