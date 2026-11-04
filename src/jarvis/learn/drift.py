"""Game-update drift: when UBG changes its HUD, the calibrated reader's confidence collapses. Then the
fighter must not fight on stale assumptions - it refuses until the HUD is recalibrated, and says why."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from jarvis.fight.recorder import read_jsonl

DRIFT_FILE = "drift.json"


def session_hud_conf(session_dir: Path) -> float | None:
    """Median HUD confidence over the frames where a match was on (or all frames if none were)."""
    rows = read_jsonl(session_dir / "frames.jsonl")
    confs = [float(r["hud_conf"]) for r in rows if r.get("combat_hud") and r.get("hud_conf") is not None]
    if len(confs) < 30:
        confs = [float(r["hud_conf"]) for r in rows if r.get("hud_conf") is not None]
    return statistics.median(confs) if len(confs) >= 30 else None


def detect_drift(confs: list[float], *, recent: int = 2, floor: float = 0.55, drop: float = 0.3) -> tuple[bool, str]:
    """``confs``: per-session medians, oldest first. Drift = the last ``recent`` sessions are all below
    ``floor`` while the earlier ones were clearly better (or there is no history and they are very low)."""
    if len(confs) < recent:
        return False, "not enough sessions"
    last = confs[-recent:]
    before = confs[:-recent]
    if max(last) >= floor:
        return False, "HUD reads fine"
    if before:
        ref = statistics.median(before)
        if ref - max(last) >= drop:
            return True, f"HUD confidence fell from {ref:.2f} to {max(last):.2f} - did UBG update its interface?"
        return False, "HUD confidence has always been low (check calibration)"
    return (max(last) < floor - drop / 2), f"HUD confidence only {max(last):.2f}"


def write_drift(run_dir: Path, drifting: bool, reason: str) -> None:
    f = run_dir / DRIFT_FILE
    if drifting:
        f.write_text(json.dumps({"drift": True, "reason": reason, "t": time.time()}), encoding="utf-8")
    elif f.exists():
        f.unlink()


def drift_problem(run_dir: Path, calibration_dir: Path) -> str | None:
    """Fight preflight: the refusal text, or None. Recalibrating the HUD after the drift clears it."""
    f = run_dir / DRIFT_FILE
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    layout = calibration_dir / "hud_layout.yaml"
    if layout.exists() and layout.stat().st_mtime > float(d.get("t", 0)):
        return None
    return f"{d.get('reason', 'HUD drift detected')} Recalibrate with `jarvis calibrate hud` before fighting."
