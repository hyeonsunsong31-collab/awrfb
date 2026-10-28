"""Owner speaker verification gate (cosine similarity of speaker embeddings).

The embedding model (ECAPA-TDNN) lives in ``jarvis.voice.speaker_id``; this module only scores, so it
is cheap to test. Trust is asymmetric: "stop"/"mute" are accepted from ANY voice; "fight", "unmute"
and PC commands require the owner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class Trust(StrEnum):
    ANY = "any"
    OWNER = "owner"


class Embedder(Protocol):
    def embed(self, audio_16k: np.ndarray) -> np.ndarray: ...


@dataclass
class SpeakerResult:
    is_owner: bool
    score: float
    reason: str


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SpeakerGate:
    def __init__(
        self,
        embedder: Embedder | None,
        owner: np.ndarray | None,
        threshold: float = 0.45,
        min_audio_s: float = 1.5,
        enabled: bool = True,
    ) -> None:
        self._embedder = embedder
        self._owner = owner  # (n, d) enrollment embeddings
        self.threshold = threshold
        self.min_audio_s = min_audio_s
        self.enabled = enabled

    @property
    def enrolled(self) -> bool:
        return self._owner is not None and len(self._owner) > 0

    def score(self, emb: np.ndarray) -> float:
        assert self._owner is not None
        centroid = self._owner.mean(axis=0)
        return max(cosine(emb, centroid), max(cosine(emb, o) for o in self._owner))

    def verify(self, audio_16k: np.ndarray, sample_rate: int = 16000) -> SpeakerResult:
        if not self.enabled:
            return SpeakerResult(True, 1.0, "speaker verification disabled")
        if not self.enrolled or self._embedder is None:
            # Usable before enrollment, but loudly flagged (HUD + doctor show this).
            return SpeakerResult(True, 0.0, "owner not enrolled - run `jarvis setup voiceprint`")
        dur = len(audio_16k) / float(sample_rate)
        if dur < self.min_audio_s:
            return SpeakerResult(False, 0.0, f"too short to verify ({dur:.2f}s < {self.min_audio_s}s)")
        emb = self._embedder.embed(audio_16k)
        s = self.score(emb)
        return SpeakerResult(s >= self.threshold, s, "ok" if s >= self.threshold else "not the owner")

    # ---------------------------------------------------------------- persistence
    @staticmethod
    def load_enrollment(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        try:
            data = np.load(path)
            arr = data["embeddings"] if hasattr(data, "files") else data
            return np.asarray(arr, dtype=np.float32)
        except Exception:
            log.exception("could not load speaker enrollment %s", path)
            return None

    @staticmethod
    def save_enrollment(path: Path, embeddings: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, embeddings=np.asarray(embeddings, dtype=np.float32))
