"""Character detector (ONNX Runtime; CUDA -> DirectML -> CPU).

Supports YOLO-format exports (YOLOv8/YOLO11/YOLO-World via Ultralytics): output (1, 4+nc, N) with
xywh + per-class scores. Without a model the fighter still works from HUD + motion heuristics.
Day one uses a zero-shot YOLO-World export (scripts/export_detector.py); later a fine-tuned model
from the owner's own recordings replaces it through the model registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    x: float  # normalized top-left
    y: float
    w: float
    h: float
    score: float
    cls: int = 0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h


def iou(a: Detection, b: Detection) -> float:
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def ort_providers(prefer_gpu: bool = True) -> list[str]:
    import onnxruntime as ort

    avail = ort.get_available_providers()
    order = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"] if prefer_gpu else ["CPUExecutionProvider"]
    return [p for p in order if p in avail] or ["CPUExecutionProvider"]


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top : top + nh, left : left + nw] = resized
    return out, r, (left, top)


def decode_yolo(
    out: np.ndarray,
    r: float,
    pad: tuple[int, int],
    shape: tuple[int, int],
    conf: float,
    classes: list[int] | None,
    channels_first: bool = True,
) -> list[Detection]:
    """Ultralytics exports are (1, 4+nc, N): channels first. Pass channels_first=False for (1, N, 4+nc)."""
    pred = out[0].T if channels_first else out[0]  # -> (N, 4+nc)
    boxes = pred[:, :4]
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(cls)), cls]
    keep = scores >= conf
    if classes is not None:
        keep &= np.isin(cls, classes)
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    if len(scores) == 0:
        return []
    H, W = shape
    xs = (boxes[:, 0] - boxes[:, 2] / 2 - pad[0]) / r
    ys = (boxes[:, 1] - boxes[:, 3] / 2 - pad[1]) / r
    ws = boxes[:, 2] / r
    hs = boxes[:, 3] / r
    rects = [[float(x), float(y), float(w), float(h)] for x, y, w, h in zip(xs, ys, ws, hs, strict=True)]
    idx = cv2.dnn.NMSBoxes(rects, scores.astype(float).tolist(), conf, 0.5)
    dets = []
    for i in np.array(idx).reshape(-1):
        x, y, w, h = rects[i]
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(W), x + w), min(float(H), y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        dets.append(Detection(x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H, float(scores[i]), int(cls[i])))
    return dets


class OnnxDetector:
    def __init__(self, model_path: Path, input_size: int = 640, conf: float = 0.25, classes: list[int] | None = None,
                 prefer_gpu: bool = True, channels_first: bool = True) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), so, providers=ort_providers(prefer_gpu))
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_size = int(shape[2]) if isinstance(shape[2], int) else input_size
        self.conf = conf
        self.classes = classes
        self.fp16 = "float16" in self.session.get_inputs()[0].type
        self.channels_first = channels_first
        log.info("detector %s on %s", model_path.name, self.session.get_providers()[0])

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        lb, r, pad = letterbox(img_bgr, self.input_size)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float16 if self.fp16 else np.float32) / 255.0
        out = self.session.run(None, {self.input_name: x})[0]
        return decode_yolo(np.asarray(out, dtype=np.float32), r, pad, img_bgr.shape[:2], self.conf, self.classes,
                           self.channels_first)


class NullDetector:
    """No model installed: perception falls back to HUD + motion heuristics."""

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        return []
