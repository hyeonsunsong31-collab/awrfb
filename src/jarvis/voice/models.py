"""Voice model files with pinned URLs and SHA-256 hashes (checked after every download).

Whisper, Supertonic and the ECAPA speaker model are fetched by their own libraries from Hugging Face on
first use (``jarvis voice-models --all`` triggers that ahead of time).
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_OWW = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"


@dataclass(frozen=True)
class ModelFile:
    rel: str  # path under <data>/models
    url: str
    sha256: str
    size: int


VOICE_MODELS: tuple[ModelFile, ...] = (
    ModelFile("silero_vad.onnx",
              "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/src/silero_vad/data/silero_vad.onnx",
              "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3", 2327524),
    ModelFile("openwakeword/hey_jarvis_v0.1.onnx", _OWW + "hey_jarvis_v0.1.onnx",
              "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb", 1271370),
    ModelFile("openwakeword/melspectrogram.onnx", _OWW + "melspectrogram.onnx",
              "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f", 1087958),
    ModelFile("openwakeword/embedding_model.onnx", _OWW + "embedding_model.onnx",
              "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f", 1326578),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_valid(models_dir: Path, m: ModelFile) -> bool:
    p = models_dir / m.rel
    return p.exists() and p.stat().st_size == m.size and sha256_file(p) == m.sha256


def missing(models_dir: Path, files: tuple[ModelFile, ...] = VOICE_MODELS) -> list[ModelFile]:
    return [m for m in files if not is_valid(models_dir, m)]


def download(models_dir: Path, files: tuple[ModelFile, ...] = VOICE_MODELS,
             on_progress: Callable[[ModelFile, str], None] | None = None, timeout_s: float = 120.0) -> list[ModelFile]:
    """Download missing/corrupt files; each one is verified before it replaces anything. Returns the
    files that were fetched. Raises on a hash mismatch (nothing is installed in that case)."""
    import httpx

    fetched = []
    for m in files:
        if is_valid(models_dir, m):
            continue
        dest = models_dir / m.rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        if on_progress:
            on_progress(m, "downloading")
        with httpx.stream("GET", m.url, follow_redirects=True, timeout=timeout_s) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 16):
                    f.write(chunk)
        got = sha256_file(tmp)
        if got != m.sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"{m.rel}: SHA-256 mismatch (expected {m.sha256[:12]}…, got {got[:12]}…) - not installed")
        os.replace(tmp, dest)
        fetched.append(m)
        if on_progress:
            on_progress(m, "ok")
    return fetched
