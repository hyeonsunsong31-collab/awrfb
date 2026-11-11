"""Wake word: openWakeWord "hey jarvis" (ONNX, CPU), always on.

Detection only opens the door. The second check runs on the utterance itself: Whisper must hear
"jarvis/자비스" in it, and owner-only actions still need speaker verification. So a TV saying something
that sounds like the wake word gets at most an ignored utterance, never a command.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from jarvis.voice.resample import float_to_int16

log = logging.getLogger(__name__)

FRAME = 1280  # 80 ms at 16 kHz
WARMUP_FRAMES = 32  # ~2.5 s of silence: pushes openWakeWord's random start-up features out of the 2 s window
OWW_FILES = ("melspectrogram.onnx", "embedding_model.onnx")


class WakeModelMissing(RuntimeError):
    pass


def oww_model_paths(models_dir: Path, name: str) -> tuple[Path, Path, Path]:
    d = models_dir / "openwakeword"
    return d / f"{name}_v0.1.onnx", d / OWW_FILES[0], d / OWW_FILES[1]


def load_oww(models_dir: Path, name: str, extra: list[str] | None = None) -> Any:
    """openWakeWord with the main model plus any personal ones that exist (one feature pass for all)."""
    model_path, mel, emb = oww_model_paths(models_dir, name)
    missing = [p.name for p in (model_path, mel, emb) if not p.exists()]
    if missing:
        raise WakeModelMissing(f"wake word model files missing in {model_path.parent}: {', '.join(missing)} - run `jarvis voice-models`")
    paths = [str(model_path)]
    for x in extra or []:
        p = oww_model_paths(models_dir, x)[0]
        if p.exists():
            paths.append(str(p))
        else:
            log.warning("personal wake word %s not found at %s - train it with `jarvis train wakeword`", x, p)
    from openwakeword.model import Model

    return Model(wakeword_models=paths, inference_framework="onnx", melspec_model_path=str(mel), embedding_model_path=str(emb))


class WakeDetector:
    def __init__(self, model: Any, *, threshold: float = 0.5, patience_frames: int = 2, debounce_s: float = 1.5,
                 clock: Callable[[], float] = time.monotonic, thresholds: dict[str, float] | None = None) -> None:
        self.model = model
        self.name = next(iter(getattr(model, "models", {"wake": None}).keys()))
        self.threshold = threshold
        self.thresholds = dict(thresholds or {})  # by configured name; openWakeWord keys carry "_v0.1"
        self.last_model = ""
        self._warm()
        self.patience = max(1, patience_frames)
        self.debounce_s = debounce_s
        self._clock = clock
        self._acc = np.zeros(0, np.int16)
        self._hits = 0
        self._last_fire = float("-inf")
        self.last_score = 0.0

    @classmethod
    def from_config(cls, cfg, models_dir: Path) -> WakeDetector:
        return cls(load_oww(models_dir, cfg.model, list(getattr(cfg, "extra_models", []) or [])), threshold=cfg.threshold,
                   patience_frames=cfg.patience_frames, debounce_s=cfg.debounce_s,
                   thresholds=dict(getattr(cfg, "model_thresholds", {}) or {}))

    def _warm(self) -> None:
        """openWakeWord starts - and resets - with embeddings of random noise in its 2 s window and only
        silences its first 5 frames; until real audio pushes that noise out (~1.3 s), small personal models
        can fire on anything. Feeding silence first puts a quiet past in the window instead."""
        if getattr(self.model, "preprocessor", None) is None:  # not a real openWakeWord model (tests)
            return
        silence = np.zeros(FRAME, np.int16)
        for _ in range(WARMUP_FRAMES):
            self.model.predict(silence)
        self._hits = 0

    def threshold_for(self, key: str) -> float:
        for name, thr in self.thresholds.items():
            if key == name or key.startswith(f"{name}_v"):
                return thr
        return self.threshold

    def feed(self, pcm_f32: np.ndarray) -> list[float]:
        """Feed 16 kHz float audio; returns the scores of detections that fired."""
        x = float_to_int16(np.asarray(pcm_f32, np.float32).reshape(-1))
        self._acc = np.concatenate([self._acc, x]) if len(self._acc) else x
        fired: list[float] = []
        while len(self._acc) >= FRAME:
            frame, self._acc = self._acc[:FRAME], self._acc[FRAME:]
            scores = self.model.predict(frame)
            # Any of the loaded wake words may fire, each against its own threshold.
            margin, s, who = max(((float(v) - self.threshold_for(k), float(v), k) for k, v in scores.items()),
                                 default=(-1.0, 0.0, ""))
            self.last_score = s
            self._hits = self._hits + 1 if margin >= 0 else 0
            now = self._clock()
            if self._hits >= self.patience and now - self._last_fire >= self.debounce_s:
                fired.append(s)
                self.last_model = who  # which wake word fired (kept until the next one fires)
                self._last_fire = now
                self._hits = 0
                reset = getattr(self.model, "reset", None)
                if callable(reset):
                    reset()  # clear the model's internal buffer so one phrase fires once
                    self._warm()  # ...and refill it with silence, not reset()'s random noise
        return fired


def heard_wake_word(text: str, keywords: list[str], threshold: int = 80) -> bool:
    """Second stage: did the transcript actually contain jarvis/자비스 (fuzzy, first few words)?"""
    words = text.lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split()
    head = words[:4]
    for kw in keywords:
        k = kw.lower()
        if any(k in w for w in head):
            return True
        if any(fuzz.ratio(w, k) >= threshold for w in head):
            return True
    return False
