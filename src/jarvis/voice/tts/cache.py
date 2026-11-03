"""Disk cache of synthesized audio (callouts, acknowledgements and other short repeated phrases)."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path

import numpy as np

from jarvis.voice.tts.base import Audio

log = logging.getLogger(__name__)


class TtsCache:
    def __init__(self, root: Path, max_mb: float = 300.0) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_mb * 1024 * 1024)
        self._puts = 0
        self._lock = threading.Lock()

    @staticmethod
    def make_key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:40]

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> Audio | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with np.load(p, allow_pickle=False) as d:
                return Audio(d["pcm"].astype(np.float32), int(d["rate"]))
        except Exception:
            log.warning("dropping unreadable TTS cache entry %s", p.name)
            p.unlink(missing_ok=True)
            return None

    def put(self, key: str, audio: Audio) -> None:
        p = self._path(key)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "wb") as f:
            np.savez(f, pcm=np.asarray(audio.pcm, np.float32), rate=np.int64(audio.rate))
        os.replace(tmp, p)
        with self._lock:
            self._puts += 1
            if self._puts % 50 == 0:
                self.prune()

    def prune(self) -> None:
        files = sorted(self.root.glob("*.npz"), key=lambda f: f.stat().st_mtime)
        total = sum(f.stat().st_size for f in files)
        while files and total > self.max_bytes:
            f = files.pop(0)
            total -= f.stat().st_size
            f.unlink(missing_ok=True)
