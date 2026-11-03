"""Live threat model (onnxruntime, CUDA when available): one encoder pass per frame; a ring buffer of the
previous embeddings feeds the temporal head. Plugs into the fight loop as a ``ThreatModel``."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np

from jarvis.fight.perception.threat import HORIZONS_MS, ThreatEstimate
from jarvis.learn.datasets import crop_box


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class OnnxThreat:
    def __init__(self, path: Path | str, window: int = 8, crop: int = 96, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        avail = ort.get_available_providers()
        prov = providers or [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.session = ort.InferenceSession(str(path), so, providers=prov)
        meta = self.session.get_modelmeta().custom_metadata_map
        self.window = int(meta.get("window", window))
        self.crop = int(meta.get("crop", crop))
        self.horizons = tuple(int(h) for h in meta.get("horizons", ",".join(map(str, HORIZONS_MS))).split(","))
        ins = {i.name: i for i in self.session.get_inputs()}
        self.dim = int(ins["prev"].shape[-1])
        self.provider = self.session.get_providers()[0]
        self.reset()

    def reset(self) -> None:
        self.buf: deque[np.ndarray] = deque(maxlen=self.window - 1)
        self.last_box: list[float] | None = None
        self.misses = 0

    def update(self, frame_bgr: np.ndarray, opp, gap: float | None, t_ns: int) -> ThreatEstimate:
        if opp is not None:
            box = [float(opp.cx), float(opp.cy), float(opp.w), float(opp.h)]
            self.misses = 0
        else:
            box = self.last_box
            self.misses += 1
        if box is None or self.misses > 30:  # nothing to look at: the policy falls back to HUD cues
            self.buf.clear()
            return ThreatEstimate({}, source="model:no-target")
        self.last_box = box
        crop = crop_box(frame_bgr, box, self.crop)
        prev = np.zeros((self.window - 1, self.dim), np.float32)  # zero padding = warm-up, as trained
        if self.buf:
            prev[-len(self.buf):] = np.stack(self.buf)
        emb, logits = self.session.run(None, {"prev": prev[None], "frame": crop[None].astype(np.uint8)})
        self.buf.append(emb[0].astype(np.float32))
        p = _sigmoid(np.asarray(logits[0], np.float64))
        est = {h: float(p[k]) for k, h in enumerate(self.horizons)}
        z = math.log(max(est.get(300, 0.0), 1e-6) / max(1 - est.get(300, 0.0), 1e-6))
        return ThreatEstimate(est, motion_z=z, source="model")
