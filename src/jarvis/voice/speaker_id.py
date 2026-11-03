"""Owner voiceprint: ECAPA-TDNN speaker embeddings (speechbrain, CPU) and enrollment.

The gate logic (thresholds, short-clip rules) lives in ``jarvis.safety.speaker_gate``; this module only
turns audio into embeddings and builds the enrollment file from ~60 s of the owner speaking Korean and
English.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import numpy as np

from jarvis.voice.resample import rms

log = logging.getLogger(__name__)


class EcapaEmbedder:
    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb", savedir: Path | None = None, device: str = "cpu") -> None:
        self.source = source
        self.savedir = savedir
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _ensure(self):
        with self._lock:
            if self._model is None:
                from speechbrain.inference.classifiers import EncoderClassifier
                from speechbrain.utils.fetching import LocalStrategy

                kw = {}
                if sys.platform == "win32":
                    kw["local_strategy"] = LocalStrategy.COPY  # symlinks need admin/developer mode on Windows
                self._model = EncoderClassifier.from_hparams(
                    source=self.source, savedir=str(self.savedir) if self.savedir else None,
                    run_opts={"device": self.device}, **kw,
                )
            return self._model

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        import torch

        model = self._ensure()
        wav = torch.from_numpy(np.ascontiguousarray(audio_16k, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_batch(wav)
        return emb.reshape(-1).cpu().numpy().astype(np.float32)


def enrollment_windows(audio: np.ndarray, window_s: float = 3.0, hop_s: float = 1.5, min_rms: float = 0.01,
                       rate: int = 16000) -> list[np.ndarray]:
    """Speech-bearing windows of the enrollment recording (silent stretches are skipped)."""
    w, h = int(window_s * rate), int(hop_s * rate)
    out = []
    for start in range(0, max(0, len(audio) - w) + 1, h):
        seg = audio[start : start + w]
        if len(seg) == w and rms(seg) >= min_rms:
            out.append(seg)
    return out


def enroll(embedder, audio: np.ndarray, out_path: Path, min_windows: int = 8) -> np.ndarray:
    from jarvis.safety.speaker_gate import SpeakerGate

    wins = enrollment_windows(np.asarray(audio, np.float32))
    if len(wins) < min_windows:
        raise ValueError(f"only {len(wins)} usable speech windows - speak continuously (Korean and English) and try again")
    embs = np.stack([embedder.embed(w) for w in wins])
    SpeakerGate.save_enrollment(out_path, embs)
    return embs
