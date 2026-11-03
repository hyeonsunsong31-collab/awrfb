"""Evaluation metrics in plain numpy (no scikit-learn needed at runtime)."""

from __future__ import annotations

import numpy as np


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, as sklearn's average_precision_score). NaN without positives."""
    y = np.asarray(y).astype(bool).reshape(-1)
    s = np.asarray(score, float).reshape(-1)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.diff(s) != 0, True]  # last index of each distinct threshold
    tp = np.cumsum(y)[last]
    fp = np.cumsum(~y)[last]
    precision = tp / (tp + fp)
    recall = tp / pos
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Probability that a random positive outranks a random negative (ties count half)."""
    y = np.asarray(y).astype(bool).reshape(-1)
    s = np.asarray(score, float).reshape(-1)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(bool).reshape(-1)
    s = np.asarray(score, float).reshape(-1)
    best, thr = -1.0, 0.5
    for t in np.unique(np.round(s, 3)):
        pred = s >= t
        tp = float((pred & y).sum())
        if tp == 0:
            continue
        f1 = 2 * tp / (pred.sum() + y.sum())
        if f1 > best:
            best, thr = f1, float(t)
    return thr


def warning_times(t_ms: np.ndarray, p: np.ndarray, hit_t_ms: np.ndarray, threshold: float, max_lead_ms: float = 1500.0) -> list[float]:
    """For each hit: how long before it the prediction crossed ``threshold`` and stayed above until the hit
    (0 = no usable warning). This is what decides whether a reactive dodge can work at all."""
    t = np.asarray(t_ms, float)
    p = np.asarray(p, float)
    out = []
    for h in np.asarray(hit_t_ms, float):
        idx = np.nonzero((t <= h) & (t >= h - max_lead_ms))[0]
        lead = 0.0
        for k in idx[::-1]:  # walk back from the hit while the alarm is on
            if p[k] < threshold:
                break
            lead = h - t[k]
        out.append(float(lead))
    return out
