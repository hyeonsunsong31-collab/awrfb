"""Promotion gate: a new model replaces the active one only when it measured better on held-out sessions
(split by session hash, so no training session leaks into the test set)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from jarvis.learn.metrics import average_precision, best_f1_threshold, roc_auc


@dataclass
class Gate:
    metric: str = "ap_300"
    min_value: float = 0.75  # M6 acceptance: PR-AUC at 300 ms >= 0.75
    min_gain: float = 0.01  # must also beat the active model by this much
    latency_ms: float | None = None  # measured frame->input latency (calibration)
    warning_margin_ms: float = 50.0  # median warning must exceed latency by this much
    min_positives: int = 30  # too few hits in the test set = no verdict


def threat_metrics(y: np.ndarray, p: np.ndarray, horizons: tuple[int, ...]) -> dict[str, float]:
    """y, p: (N, H) labels and probabilities on the test split."""
    out: dict[str, float] = {"n": len(y)}
    for k, h in enumerate(horizons):
        out[f"ap_{h}"] = round(average_precision(y[:, k], p[:, k]), 4)
        out[f"auc_{h}"] = round(roc_auc(y[:, k], p[:, k]), 4)
        out[f"pos_{h}"] = int(np.asarray(y[:, k]).sum())
        out[f"thr_{h}"] = round(best_f1_threshold(y[:, k], p[:, k]), 3)
    return out


def promotion_decision(candidate: dict, active: dict | None, gate: Gate | None = None) -> tuple[bool, str]:
    g = gate or Gate()
    h = g.metric.split("_")[-1]
    value = candidate.get(g.metric)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False, f"candidate has no {g.metric}"
    if candidate.get(f"pos_{h}", g.min_positives) < g.min_positives:
        return False, f"only {candidate.get(f'pos_{h}')} test positives - record more fights first"
    if value < g.min_value:
        return False, f"{g.metric} {value:.3f} is below the {g.min_value:.2f} bar"
    if g.latency_ms is not None and candidate.get("warning_ms_median") is not None:
        need = g.latency_ms + g.warning_margin_ms
        if candidate["warning_ms_median"] < need:
            return False, f"median warning {candidate['warning_ms_median']:.0f} ms < latency + margin ({need:.0f} ms)"
    if active and active.get(g.metric) is not None and value < active[g.metric] + g.min_gain:
        return False, f"{g.metric} {value:.3f} does not beat the active model ({active[g.metric]:.3f}) by {g.min_gain}"
    return True, f"{g.metric} {value:.3f} passes"
