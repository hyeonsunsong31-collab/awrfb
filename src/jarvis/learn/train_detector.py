"""Fine-tune a small detector (self / opponent) on your own recordings.

Labels are pseudo-labels from what the fight loop already tracked: the zero-shot detector's boxes, with the
self/opponent identity resolved by matching motion to our own dashes (self-id). Only combat frames where both
fighters were tracked are used; frames are split by session like every other dataset. Training uses
Ultralytics YOLO (install the 'detect' extra; note its AGPL-3.0 licence) and exports ONNX in the layout
``jarvis.fight.perception.detector.OnnxDetector`` decodes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import cv2

from jarvis.fight.recorder import read_jsonl
from jarvis.learn.datasets import VideoFrames, split_of

log = logging.getLogger(__name__)

CLASSES = ("self", "opponent")


def _yolo_line(cls: int, box: list[float]) -> str:
    cx, cy, w, h = (min(1.0, max(0.0, float(v))) for v in box[:4])
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build_yolo_dataset(recordings: Path, out_dir: Path, *, stride: int = 6, min_size: float = 0.02,
                       sessions: list[str] | None = None, progress: Callable[[str], None] | None = None) -> Path:
    """Write images/{train,val}/*.jpg + labels/{train,val}/*.txt + data.yaml. Returns data.yaml."""
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    for sd in sorted(p for p in recordings.iterdir() if p.is_dir()):
        if sessions and sd.name not in sessions:
            continue
        rows = read_jsonl(sd / "frames.jsonl")
        if not rows or not (sd / "video.mkv").exists():
            continue
        split = "val" if split_of(sd.name) != "train" else "train"
        try:
            frames = VideoFrames(sd)
        except Exception as e:
            log.info("skipping %s: %s", sd.name, e)
            continue
        n = 0
        try:
            for k, r in enumerate(rows):
                if k % stride or not r.get("combat_hud") or r.get("me") is None or r.get("opp") is None:
                    continue
                me, opp = r["me"], r["opp"]
                if min(me[2], me[3], opp[2], opp[3]) < min_size:
                    continue
                img = frames.get(int(r.get("vi", k)))
                if img is None:
                    continue
                stem = f"{sd.name}_{k:06d}"
                cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out_dir / "labels" / split / f"{stem}.txt").write_text(
                    _yolo_line(0, me) + "\n" + _yolo_line(1, opp) + "\n", encoding="utf-8")
                n += 1
        finally:
            frames.close()
        counts[split] += n
        if progress:
            progress(f"{sd.name}: {n} frames ({split})")
    data = out_dir / "data.yaml"
    data.write_text(f"path: {out_dir.as_posix()}\ntrain: images/train\nval: images/val\n"
                    + "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)), encoding="utf-8")
    log.info("detector dataset: %s", counts)
    return data


def train_detector(data_yaml: Path, out_dir: Path, *, epochs: int = 60, imgsz: int = 640, base: str = "yolo11n.pt",
                   device: str | None = None) -> tuple[Path, dict]:
    """Fine-tune and export ONNX. Returns (onnx path, metrics)."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("install the 'detect' extra (ultralytics) to train the detector") from e
    model = YOLO(base)
    res = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, project=str(out_dir), name="run", exist_ok=True,
                      device=device, fliplr=0.5, mosaic=0.5, patience=15, verbose=False)
    metrics = {}
    for key, name in (("metrics/mAP50(B)", "map50"), ("metrics/mAP50-95(B)", "map50_95")):
        v = getattr(res, "results_dict", {}).get(key)
        if v is not None:
            metrics[name] = round(float(v), 4)
    onnx = Path(model.export(format="onnx", imgsz=imgsz, opset=17, dynamic=False, simplify=False))
    return onnx, metrics
