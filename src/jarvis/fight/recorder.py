"""Session recorder: the proof bundle and the training data engine.

A session directory contains
  video.mkv      every recorded frame (NVENC H.264 when available, else libx264), downscaled
  frames.jsonl   one row per video frame: video index, capture index, t_ns, HUD readings, state...
  inputs.jsonl   every input the bot sent or was refused (and the owner's keys in shadow mode)
  events.jsonl   hit taken/blocked/dealt, stock lost, match start/end, gate halts...
  meta.json      versions, hashes of layout/keymap, settings fingerprint, tier
Rows carry "v": 1. Writing happens on a background thread; frames are dropped (and counted) rather
than ever blocking the fight loop.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "value"):
        return o.value
    return str(o)


_encoder_cache: dict[str, str] = {}


def pick_encoder(ffmpeg: str, preference: str = "auto") -> str | None:
    """Return a working H.264 encoder name, verifying NVENC with a tiny test encode."""
    exe = shutil.which(ffmpeg)
    if not exe:
        return None
    if preference in ("h264_nvenc", "libx264"):
        return preference
    if exe in _encoder_cache:
        return _encoder_cache[exe]
    choice = "libx264"
    try:
        res = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x144:d=0.2",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True,
            timeout=15,
        )
        if res.returncode == 0:
            choice = "h264_nvenc"
    except (OSError, subprocess.SubprocessError):
        pass
    _encoder_cache[exe] = choice
    return choice


@dataclass
class RecorderStats:
    frames_written: int = 0
    frames_dropped: int = 0
    rows_written: int = 0
    encoder: str | None = None
    errors: list[str] = field(default_factory=list)


class SessionRecorder:
    def __init__(
        self,
        root: Path,
        kind: str,
        *,
        width: int = 960,
        height: int = 540,
        ffmpeg: str = "ffmpeg",
        encoder: str = "auto",
        crf: int = 23,
        meta: dict[str, Any] | None = None,
        record_video: bool = True,
        fps_hint: int = 60,
    ) -> None:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        self.dir = root / f"{stamp}_{kind}"
        n = 1
        while self.dir.exists():
            n += 1
            self.dir = root / f"{stamp}_{kind}_{n}"
        self.dir.mkdir(parents=True)
        self.kind = kind
        self.size = (int(width), int(height))
        self.stats = RecorderStats()
        self._files = {
            name: (self.dir / f"{name}.jsonl").open("a", encoding="utf-8") for name in ("frames", "inputs", "events")
        }
        self._q: queue.Queue = queue.Queue(maxsize=240)
        self._lock = threading.Lock()
        self._video_index = 0
        self._ffmpeg: subprocess.Popen | None = None
        self._closed = False
        self._meta = {"v": 1, "kind": kind, "created": time.time(), "size": list(self.size), **(meta or {})}
        if record_video:
            enc = pick_encoder(ffmpeg, encoder)
            self.stats.encoder = enc
            if enc:
                self._ffmpeg = self._start_ffmpeg(shutil.which(ffmpeg) or ffmpeg, enc, crf, fps_hint)
            else:
                self.stats.errors.append("ffmpeg not found: saving JPEG keyframes instead of video")
        (self.dir / "meta.json").write_text(_dumps(self._meta), encoding="utf-8")
        self._thread = threading.Thread(target=self._writer, name=f"recorder-{kind}", daemon=True)
        self._thread.start()

    def _start_ffmpeg(self, exe: str, enc: str, crf: int, fps: int) -> subprocess.Popen | None:
        w, h = self.size
        args = [exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        if enc == "h264_nvenc":
            args += ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf)]
        else:
            args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]
        args += ["-pix_fmt", "yuv420p", str(self.dir / "video.mkv")]
        try:
            return subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            self.stats.errors.append(f"ffmpeg failed to start: {e}")
            return None

    # ------------------------------------------------------------------ producer API (thread-safe)
    def add_frame(self, image: np.ndarray, row: dict[str, Any]) -> int | None:
        """Queue a frame + its frames.jsonl row. Returns the video index or None if dropped."""
        if self._closed:
            return None
        with self._lock:
            vi = self._video_index
            try:
                self._q.put_nowait(("frame", image, {**row, "v": 1, "vi": vi}))
            except queue.Full:
                self.stats.frames_dropped += 1
                return None
            self._video_index += 1
            return vi

    def log(self, stream: str, row: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._q.put_nowait(("row", stream, {"v": 1, **row}))
        except queue.Full:
            # Rows matter more than frames: block briefly rather than lose an input/event record.
            try:
                self._q.put(("row", stream, {"v": 1, **row}), timeout=0.05)
            except queue.Full:
                self.stats.errors.append(f"dropped {stream} row")

    def update_meta(self, **kw: Any) -> None:
        self._meta.update(kw)
        try:
            (self.dir / "meta.json").write_text(_dumps(self._meta), encoding="utf-8")
        except OSError as e:
            self.stats.errors.append(str(e))

    # ------------------------------------------------------------------ writer thread
    def _writer(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                if item[0] == "frame":
                    _, image, row = item
                    self._write_frame(image, row)
                else:
                    _, stream, row = item
                    self._files[stream].write(_dumps(row) + "\n")
                    self.stats.rows_written += 1
            except Exception as e:  # never kill the writer thread
                self.stats.errors.append(f"{type(e).__name__}: {e}")
                log.exception("recorder write failed")

    def _write_frame(self, image: np.ndarray, row: dict[str, Any]) -> None:
        w, h = self.size
        if image.shape[1] != w or image.shape[0] != h:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        if self._ffmpeg is not None and self._ffmpeg.stdin is not None:
            try:
                self._ffmpeg.stdin.write(np.ascontiguousarray(image).tobytes())
            except (BrokenPipeError, OSError) as e:
                self.stats.errors.append(f"ffmpeg pipe: {e}")
                self._ffmpeg = None
        elif row["vi"] % 30 == 0:
            kf = self.dir / "keyframes"
            kf.mkdir(exist_ok=True)
            cv2.imwrite(str(kf / f"{row['vi']:07d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self._files["frames"].write(_dumps(row) + "\n")
        self.stats.frames_written += 1

    def close(self, summary: dict[str, Any] | None = None) -> Path:
        if self._closed:
            return self.dir
        self._closed = True
        self._q.put(None)
        self._thread.join(timeout=30)
        for f in self._files.values():
            f.close()
        if self._ffmpeg is not None and self._ffmpeg.stdin is not None:
            try:
                self._ffmpeg.stdin.close()
                self._ffmpeg.wait(timeout=60)
                err = self._ffmpeg.stderr.read().decode(errors="replace") if self._ffmpeg.stderr else ""
                if self._ffmpeg.returncode != 0:
                    self.stats.errors.append(f"ffmpeg exit {self._ffmpeg.returncode}: {err[-300:]}")
            except (OSError, subprocess.SubprocessError) as e:
                self.stats.errors.append(str(e))
        self.update_meta(
            closed=time.time(),
            frames_written=self.stats.frames_written,
            frames_dropped=self.stats.frames_dropped,
            encoder=self.stats.encoder,
            errors=self.stats.errors[-20:],
        )
        if summary is not None:
            (self.dir / "summary.json").write_text(_dumps(summary), encoding="utf-8")
        return self.dir


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out
