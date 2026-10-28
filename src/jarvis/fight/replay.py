"""Deterministic replay of recorded sessions, and overlay "proof" videos.

Replay feeds the recorded frames (with their recorded timestamps) through perception + the policy
against a virtual "held keys" state. Same inputs, same seed -> same outputs, so real recordings from
the owner's PC become regression tests for every later change. `render_overlay` draws what JARVIS
saw and did (HUD readings, opponent box, threat, state, inputs) onto the recorded video.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from jarvis.common.config import PerceptionConfig
from jarvis.fight.actuator import Command
from jarvis.fight.keymap import Keymap
from jarvis.fight.perception.layout import HudLayout, roi_to_pixels
from jarvis.fight.perception.pipeline import Observation, Perception
from jarvis.fight.policy.params import PolicyParams
from jarvis.fight.policy.reflex_v0 import ReflexPolicy
from jarvis.fight.recorder import read_jsonl
from jarvis.platform.base import Frame


def iter_session_frames(session_dir: Path) -> Iterator[tuple[int, int, np.ndarray, dict]]:
    """Yields (video_index, t_ns, image, frames.jsonl row)."""
    rows = read_jsonl(session_dir / "frames.jsonl")
    cap = cv2.VideoCapture(str(session_dir / "video.mkv"))
    try:
        for row in rows:
            ok, img = cap.read()
            if not ok:
                break
            yield int(row.get("vi", 0)), int(row["t_ns"]), img, row
    finally:
        cap.release()


@dataclass
class ReplayResult:
    frames: int = 0
    rows: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)

    def digest(self) -> str:
        blob = json.dumps({"rows": self.rows, "decisions": self.decisions}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


class VirtualKeys:
    """Tracks which physical inputs the policy would be holding (no OS input in replay)."""

    def __init__(self, km: Keymap) -> None:
        self.km = km
        self.held: set[str] = set()

    def apply(self, cmds: list[Command]) -> None:
        for c in cmds:
            b = self.km.actions.get(c.action)
            if b is None or b.kind not in ("key", "mouse"):
                continue
            phys = f"key:{b.key}" if b.key else f"mouse:{b.mouse}"
            if c.mode == "hold":
                self.held.add(phys)
            elif c.mode == "release":
                self.held.discard(phys)


def replay_session(
    session_dir: Path,
    layout: HudLayout,
    keymap: Keymap,
    params: PolicyParams,
    *,
    perception_cfg: PerceptionConfig | None = None,
    calibration_dir: Path | None = None,
    detector=None,
    seed: int = 0,
    fixed_tactic: str | None = "baseline",
    only_fighting: bool = False,
) -> ReplayResult:
    perc = Perception(perception_cfg or PerceptionConfig(), layout, calibration_dir=calibration_dir, detector=detector,
                      h_close=params.spacing.h_close, h_far=params.spacing.h_far)
    pol = ReflexPolicy(params, keymap, seed=seed, fixed_tactic=fixed_tactic)
    vk = VirtualKeys(keymap)
    res = ReplayResult()
    started = False
    for vi, t_ns, img, row in iter_session_frames(session_dir):
        if only_fighting and row.get("state") not in (None, "fighting"):
            continue
        obs = perc.process(Frame(img, t_ns, vi))
        if not started:
            pol.reset(t_ns)
            started = True
        cmds = pol.step(obs, frozenset(vk.held)) if obs.hud.combat_hud else []
        vk.apply(cmds)
        r = obs.row()
        r.pop("ms", None)  # timings are not deterministic
        res.rows.append(r)
        if cmds:
            res.decisions.append({"vi": vi, "t_ns": t_ns, "cmds": [[c.action, c.mode, c.reason] for c in cmds]})
        res.frames += 1
    return res


# ------------------------------------------------------------------------------- overlay video
def _draw(img: np.ndarray, row: dict, inputs: list[dict], layout: HudLayout | None) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    cyan = (255, 229, 0)
    if layout is not None:
        for name, spec in layout.bars.items():
            x, y, w, h = roi_to_pixels(spec.roi, out.shape)
            cv2.rectangle(out, (x, y), (x + w, y + h), cyan, 1)
            v = (row.get("hud") or {}).get(name)
            cv2.putText(out, f"{name} {'--' if v is None else f'{v:.2f}'}", (x, max(10, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cyan, 1, cv2.LINE_AA)
    for key, color in (("opp", (40, 40, 255)), ("me", (40, 220, 40))):
        b = row.get(key)
        if b:
            cx, cy, w, h = b[:4]
            cv2.rectangle(out, (int((cx - w / 2) * W), int((cy - h / 2) * H)), (int((cx + w / 2) * W), int((cy + h / 2) * H)), color, 2)
            cv2.putText(out, key, (int((cx - w / 2) * W), int((cy - h / 2) * H) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    thr = (row.get("threat") or {}).get("200", 0.0)
    cv2.rectangle(out, (10, 10), (10 + int(200 * thr), 24), (0, 0, 255), -1)
    cv2.rectangle(out, (10, 10), (210, 24), (255, 255, 255), 1)
    info = f"{row.get('state', '?')}  gap={row.get('gap')}  gate={row.get('gate', '')}  tactic={row.get('tactic', '')}"
    cv2.putText(out, info, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for k, r in enumerate(inputs[-6:]):
        mark = "OK" if r.get("ok") else "X"
        txt = f"{mark} {r.get('action', r.get('event'))} {r.get('event', '')} {r.get('why', '')}"[:70]
        cv2.putText(out, txt, (10, H - 12 - 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200) if r.get("ok") else (120, 120, 255), 1, cv2.LINE_AA)
    return out


def render_overlay(session_dir: Path, out_path: Path, layout: HudLayout | None = None, ffmpeg: str = "ffmpeg") -> Path:
    inputs = read_jsonl(session_dir / "inputs.jsonl")
    inputs.sort(key=lambda r: r.get("t_ns", 0))
    exe = shutil.which(ffmpeg)
    frames = list(iter_session_frames(session_dir))
    if not frames:
        raise RuntimeError("session has no frames/video")
    H, W = frames[0][2].shape[:2]
    if exe is None:
        raise RuntimeError("ffmpeg is needed to render the overlay video")
    proc = subprocess.Popen(
        [exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(out_path)],
        stdin=subprocess.PIPE,
    )
    j = 0
    shown: list[dict] = []
    assert proc.stdin is not None
    for _vi, t_ns, img, row in frames:
        while j < len(inputs) and inputs[j].get("t_ns", 0) <= t_ns:
            shown.append(inputs[j])
            j += 1
        proc.stdin.write(_draw(img, row, shown, layout).tobytes())
    proc.stdin.close()
    proc.wait(timeout=120)
    return out_path


def shadow_report(replay: ReplayResult, session_dir: Path, lead_ms: float = 60.0) -> dict:
    """What share of the hits we took would the policy have blocked? A hit counts as blocked when the
    policy was holding block at impact AND had started at least ``lead_ms`` earlier (input latency).
    Heavies are reported separately (damage clustering)."""
    from jarvis.learn.hindsight import label_session

    labels = label_session(session_dir)
    block_times = []
    holding = False
    for d in replay.decisions:
        for action, mode, _reason in d["cmds"]:
            if action == "block" and mode == "hold":
                holding = True
                block_times.append((d["t_ns"], None))
            elif action == "block" and mode == "release" and holding:
                holding = False
                if block_times:
                    block_times[-1] = (block_times[-1][0], d["t_ns"])

    lead = int(lead_ms * 1e6)

    def blocked_at(t: int) -> bool:
        return any(s <= t - lead and (e is None or e >= t) for s, e in block_times)

    taken = [h for h in labels.hits if h.kind == "hit_taken"]
    heavy = [h for h in taken if h.severity == "heavy"]
    return {
        "hits_taken": len(taken),
        "would_block": sum(blocked_at(h.t_ns) for h in taken),
        "heavies": len(heavy),
        "would_block_heavies": sum(blocked_at(h.t_ns) for h in heavy),
        "share_heavies_blocked": round(sum(blocked_at(h.t_ns) for h in heavy) / len(heavy), 3) if heavy else None,
        "share_all_blocked": round(sum(blocked_at(h.t_ns) for h in taken) / len(taken), 3) if taken else None,
    }


def observation_rows(obs: list[Observation]) -> list[dict]:
    return [o.row() for o in obs]
