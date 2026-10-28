"""Training datasets built from recorded sessions.

A sample is a window of ``window`` consecutive frames ending at frame i, with the opponent's box at
each step and the hindsight labels at frame i. The split is by SESSION (hash of its name), never by
frame, so near-identical neighbouring frames cannot leak from train into test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from jarvis.fight.recorder import read_jsonl
from jarvis.learn.hindsight import HORIZONS_MS, label_session


@dataclass
class Sample:
    session: str
    end_vi: int
    vis: list[int]
    boxes: list[list[float] | None]  # opponent [cx, cy, w, h] per step (None if unseen)
    y: dict[str, int]
    severity: str
    source: str  # "own" | "shadow" | "video"
    split: str


def split_of(session: str, val: float = 0.15, test: float = 0.15) -> str:
    h = int(hashlib.sha256(session.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test:
        return "test"
    if h < test + val:
        return "val"
    return "train"


def source_of(session_dir: Path) -> str:
    try:
        kind = json.loads((session_dir / "meta.json").read_text(encoding="utf-8")).get("kind", "")
    except (OSError, ValueError):
        kind = ""
    return {"shadow": "shadow", "video": "video"}.get(kind, "own")


def build_samples(session_dir: Path, *, window: int = 8, stride: int = 2, require_opp: bool = True) -> list[Sample]:
    frames = read_jsonl(session_dir / "frames.jsonl")
    labels = label_session(session_dir)
    by_vi = {int(f.get("vi", i)): f for i, f in enumerate(frames)}
    split = split_of(session_dir.name)
    src = source_of(session_dir)
    out = []
    for k in range(window - 1, len(labels.vi), stride):
        vis = labels.vi[k - window + 1 : k + 1]
        boxes = []
        for vi in vis:
            f = by_vi.get(vi, {})
            opp = f.get("opp")
            boxes.append(None if opp is None else [float(x) for x in opp[:4]])
        if require_opp and boxes[-1] is None:
            continue
        if not by_vi.get(vis[-1], {}).get("combat_hud", True):
            continue
        y = {str(h): labels.y[h][k] for h in HORIZONS_MS}
        out.append(Sample(session_dir.name, vis[-1], vis, boxes, y, labels.severity[k], src, split))
    return out


def write_manifest(samples: list[Sample], out_dir: Path, version: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(s) for s in samples]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist([{**r, "boxes": json.dumps(r["boxes"]), "y": json.dumps(r["y"]),
                                       "vis": json.dumps(r["vis"])} for r in rows])
        p = out_dir / f"manifest-{version}.parquet"
        pq.write_table(table, p)
    except ImportError:
        p = out_dir / f"manifest-{version}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return p


def read_manifest(p: Path) -> list[Sample]:
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(p).to_pylist()
        return [Sample(**{**r, "boxes": json.loads(r["boxes"]), "y": json.loads(r["y"]), "vis": json.loads(r["vis"])}) for r in rows]
    return [Sample(**json.loads(line)) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


class VideoFrames:
    """Random access to a session's recorded frames (sequential reads are cached)."""

    def __init__(self, session_dir: Path) -> None:
        self.path = session_dir / "video.mkv"
        self.cap = cv2.VideoCapture(str(self.path))
        self.pos = 0
        self.cache: dict[int, np.ndarray] = {}

    def get(self, vi: int) -> np.ndarray | None:
        if vi in self.cache:
            return self.cache[vi]
        if vi < self.pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, vi)
            self.pos = vi
        img = None
        while self.pos <= vi:
            ok, img = self.cap.read()
            if not ok:
                return None
            self.cache[self.pos] = img
            self.pos += 1
            if len(self.cache) > 64:
                self.cache.pop(min(self.cache))
        return img

    def close(self) -> None:
        self.cap.release()


def crop_window(frames: VideoFrames, sample: Sample, size: int = 96) -> np.ndarray | None:
    """(window, size, size, 3) crops around the opponent box (last known box reused when missing)."""
    last = next((b for b in reversed(sample.boxes) if b is not None), None)
    if last is None:
        return None
    out = []
    box = last
    for vi, b in zip(sample.vis, sample.boxes, strict=True):
        box = b or box
        img = frames.get(vi)
        if img is None:
            return None
        out.append(crop_box(img, box, size))
    return np.stack(out)


def crop_box(img: np.ndarray, box: list[float], size: int, pad: float = 0.25) -> np.ndarray:
    H, W = img.shape[:2]
    cx, cy, w, h = box
    s = max(w * W, h * H) * (1 + pad)
    x0, y0 = int(cx * W - s / 2), int(cy * H - s / 2)
    x1, y1 = int(x0 + s), int(y0 + s)
    padded = cv2.copyMakeBorder(img, max(0, -y0), max(0, y1 - H), max(0, -x0), max(0, x1 - W), cv2.BORDER_CONSTANT)
    ox, oy = max(0, -x0), max(0, -y0)
    crop = padded[y0 + oy : y1 + oy, x0 + ox : x1 + ox]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
