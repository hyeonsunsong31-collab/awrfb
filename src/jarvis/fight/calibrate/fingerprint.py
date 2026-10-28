"""Settings fingerprint: what the calibration assumed. If the live game no longer matches (different
resolution/aspect, re-calibrated keymap, changed layout), fightd refuses to fight until recalibrated."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from jarvis.fight.keymap import Keymap
from jarvis.fight.perception.layout import HudLayout


def keymap_digest(km: Keymap) -> str:
    return hashlib.sha256(km.model_dump_json().encode()).hexdigest()[:16]


@dataclass
class Fingerprint:
    frame_w: int
    frame_h: int
    layout_digest: str
    keymap_digest: str
    cursor_closes_gate: bool
    camera_mode: str
    created: float

    @classmethod
    def make(cls, layout: HudLayout, keymap: Keymap) -> Fingerprint:
        return cls(
            frame_w=layout.frame_size[0],
            frame_h=layout.frame_size[1],
            layout_digest=layout.digest(),
            keymap_digest=keymap_digest(keymap),
            cursor_closes_gate=layout.cursor_closes_gate,
            camera_mode=layout.camera_mode,
            created=time.time(),
        )

    def save(self, calibration_dir: Path) -> Path:
        p = calibration_dir / "fingerprint.json"
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, calibration_dir: Path) -> Fingerprint | None:
        p = calibration_dir / "fingerprint.json"
        if not p.exists():
            return None
        try:
            return cls(**json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, TypeError):
            return None

    def problems(self, frame_w: int, frame_h: int, layout: HudLayout, keymap: Keymap) -> list[str]:
        out = []
        if (frame_w, frame_h) != (self.frame_w, self.frame_h):
            ar_now = frame_w / max(1, frame_h)
            ar_cal = self.frame_w / max(1, self.frame_h)
            if abs(ar_now - ar_cal) > 0.02:
                out.append(f"window aspect changed ({frame_w}x{frame_h} vs calibrated {self.frame_w}x{self.frame_h})")
            elif abs(frame_w - self.frame_w) / self.frame_w > 0.35:
                out.append(f"window size changed a lot ({frame_w}x{frame_h} vs {self.frame_w}x{self.frame_h})")
        if layout.digest() != self.layout_digest:
            out.append("HUD layout changed since calibration")
        if keymap_digest(keymap) != self.keymap_digest:
            out.append("keymap changed since calibration")
        return out
