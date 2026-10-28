"""After-the-fact ("hindsight") labels from HUD events - free supervision, no human labelling.

For every recorded frame at time t and horizon h:
    y_h = 1  if our HP or guard dropped within (t, t + h]     (an attack is about to land)
Damage size separates light from heavy hits (1-D 2-means over the session's hit amounts).
HUD events are detected a little after the real impact (median filter + game UI latency), so event
times are shifted earlier by ``event_lag_ms``.

The same labelling works on any video whose HUD we can read - including YouTube clips filmed from a
player's own view - which is the honest version of "learning from infinite videos".
"""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jarvis.fight.recorder import read_jsonl

HORIZONS_MS = (100, 200, 300, 500)
NS_PER_MS = 1_000_000


@dataclass
class HitEvent:
    t_ns: int
    kind: str  # "hit_taken" | "hit_blocked"
    amount: float
    severity: str = "light"


@dataclass
class SessionLabels:
    session: str
    t_ns: list[int] = field(default_factory=list)
    vi: list[int] = field(default_factory=list)
    y: dict[int, list[int]] = field(default_factory=dict)  # horizon -> 0/1 per frame
    severity: list[str] = field(default_factory=list)  # severity of the next hit within 500 ms ("" if none)
    hits: list[HitEvent] = field(default_factory=list)
    heavy_threshold: float | None = None

    def positive_rate(self, h: int = 300) -> float:
        ys = self.y.get(h, [])
        return float(np.mean(ys)) if ys else 0.0

    def stats(self) -> dict:
        return {
            "session": self.session,
            "frames": len(self.t_ns),
            "hits": len(self.hits),
            "heavy_hits": sum(h.severity == "heavy" for h in self.hits),
            "heavy_threshold": self.heavy_threshold,
            **{f"pos_rate_{h}ms": round(self.positive_rate(h), 4) for h in HORIZONS_MS},
        }


def two_means_threshold(values: list[float], iters: int = 50) -> float | None:
    """Split 1-D values into two clusters; return the boundary (None if not separable)."""
    v = np.asarray(sorted(values), dtype=np.float64)
    if len(v) < 4 or v[-1] - v[0] < 1e-3:
        return None
    lo, hi = float(np.percentile(v, 25)), float(np.percentile(v, 90))
    for _ in range(iters):
        thr = (lo + hi) / 2
        a, b = v[v <= thr], v[v > thr]
        if len(a) == 0 or len(b) == 0:
            return None
        nlo, nhi = float(a.mean()), float(b.mean())
        if abs(nlo - lo) < 1e-6 and abs(nhi - hi) < 1e-6:
            break
        lo, hi = nlo, nhi
    # Require a real gap between the clusters (heavy hits are clearly bigger than jabs).
    if hi < lo * 1.5:
        return None
    return (lo + hi) / 2


def label_session(session_dir: Path, *, event_lag_ms: float = 33.0, heavy_threshold: float | None = None) -> SessionLabels:
    frames = read_jsonl(session_dir / "frames.jsonl")
    events = read_jsonl(session_dir / "events.jsonl")
    out = SessionLabels(session=session_dir.name)
    lag = int(event_lag_ms * NS_PER_MS)
    hits = [
        HitEvent(int(e["t_ns"]) - lag, e["type"], float(e.get("amount", 0.0)))
        for e in events
        if e.get("type") in ("hit_taken", "hit_blocked")
    ]
    hits.sort(key=lambda h: h.t_ns)
    thr = heavy_threshold if heavy_threshold is not None else two_means_threshold([h.amount for h in hits if h.kind == "hit_taken"])
    out.heavy_threshold = thr
    for h in hits:
        if thr is not None and h.kind == "hit_taken" and h.amount >= thr:
            h.severity = "heavy"
    out.hits = hits
    times = [h.t_ns for h in hits]
    for hz in HORIZONS_MS:
        out.y[hz] = []
    for fr in frames:
        t = int(fr["t_ns"])
        out.t_ns.append(t)
        out.vi.append(int(fr.get("vi", len(out.vi))))
        i = bisect.bisect_right(times, t)  # first hit strictly after t
        nxt = hits[i] if i < len(hits) else None
        for hz in HORIZONS_MS:
            out.y[hz].append(int(nxt is not None and nxt.t_ns - t <= hz * NS_PER_MS))
        out.severity.append(nxt.severity if (nxt is not None and nxt.t_ns - t <= 500 * NS_PER_MS) else "")
    return out


def attack_outcomes(session_dir: Path, window_ms: float = 350.0) -> list[dict]:
    """Our light/heavy presses -> landed or not (a hit_dealt within the window), with the gap at
    the time. Used by `calibrate range` to fit P(land | gap)."""
    inputs = [r for r in read_jsonl(session_dir / "inputs.jsonl") if r.get("ok") and r.get("event") == "down"
              and r.get("action") in ("light", "heavy")]
    dealt = sorted(int(e["t_ns"]) for e in read_jsonl(session_dir / "events.jsonl") if e.get("type") == "hit_dealt")
    frames = read_jsonl(session_dir / "frames.jsonl")
    ft = [int(f["t_ns"]) for f in frames]
    inputs.sort(key=lambda r: int(r["t_ns"]))
    times = [int(r["t_ns"]) for r in inputs]
    landed = [False] * len(inputs)
    # Credit each hit to the latest press before it (within the window), one-to-one.
    for d in dealt:
        j = bisect.bisect_left(times, d) - 1
        while j >= 0 and d - times[j] <= window_ms * NS_PER_MS:
            if not landed[j]:
                landed[j] = True
                break
            j -= 1
    out = []
    for r, hit in zip(inputs, landed, strict=True):
        t = int(r["t_ns"])
        k = max(0, bisect.bisect_right(ft, t) - 1)
        gap = frames[k].get("gap") if frames else None
        out.append({"t_ns": t, "action": r["action"], "landed": hit, "gap": gap})
    return out


def fit_range(outcomes: list[dict], bins: int = 8) -> dict:
    """Empirical P(land | gap) per gap bin, and the largest gap where it stays >= 0.3."""
    pts = [(o["gap"], o["landed"]) for o in outcomes if o.get("gap") is not None]
    if len(pts) < 10:
        return {"samples": len(pts), "attack_gap": None}
    g = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts], dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rates = []
    for a, b in itertools.pairwise(edges):
        m = (g >= a) & (g < b)
        rates.append((float((a + b) / 2), float(y[m].mean()) if m.sum() >= 3 else None, int(m.sum())))
    good = [c for c, r, _n in rates if r is not None and r >= 0.3]
    return {"samples": len(pts), "bins": rates, "attack_gap": max(good) if good else None}
