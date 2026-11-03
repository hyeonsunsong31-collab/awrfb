"""Voice activity detection: Silero VAD (ONNX, CPU) with an energy-based fallback, plus a segmenter
that turns per-chunk speech probabilities into utterances.

Segmenter events (one 32 ms chunk at 16 kHz per step):
  onset  - first speech-like chunk (used to duck JARVIS's voice within ~100 ms)
  abort  - the onset did not become speech (cough, click): restore the volume
  start  - min_speech_ms of voiced audio: an utterance has begun
  tick   - every chunk while in an utterance (carries voiced_ms: only chunks above threshold)
  end    - end-of-speech silence or max length; carries the utterance audio (with pre-roll)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
CHUNK = 512  # 32 ms at 16 kHz (what Silero expects)
CHUNK_MS = 32.0


class SileroVad:
    """Silero VAD ONNX. v5/v6: input [1, 64 context + 512], state [2, 1, 128], sr int64.
    v4 (older files): input [1, 512], h/c [2, 1, 64], sr int64."""

    def __init__(self, model_path: Path | str) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self._v4 = "h" in {i.name for i in self.session.get_inputs()}
        self._sr = np.array(SR, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._h = np.zeros((2, 1, 64), np.float32)
        self._c = np.zeros((2, 1, 64), np.float32)
        self._context = np.zeros((1, 64), np.float32)

    def prob(self, chunk_f32: np.ndarray) -> float:
        x = chunk_f32.reshape(1, -1).astype(np.float32)
        if self._v4:
            out, self._h, self._c = self.session.run(None, {"input": x, "sr": self._sr, "h": self._h, "c": self._c})
        else:
            x = np.concatenate([self._context, x], axis=1)
            out, self._state = self.session.run(None, {"input": x, "state": self._state, "sr": self._sr})
            self._context = x[:, -64:]
        return float(np.asarray(out).reshape(-1)[0])


class EnergyVad:
    """Fallback without the model: RMS over an adaptive noise floor (falls fast, rises slowly)."""

    def __init__(self, ratio: float = 3.0, floor: float = 0.003) -> None:
        self.ratio = ratio
        self.floor = floor
        self.noise = floor

    def reset(self) -> None:
        self.noise = self.floor

    def prob(self, chunk_f32: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(chunk_f32, dtype=np.float64)) + 1e-12))
        if rms < self.noise * 1.5:
            self.noise = 0.95 * self.noise + 0.05 * max(rms, self.floor)
        else:  # a louder room (fan switched on) must not look like endless speech
            self.noise = 0.998 * self.noise + 0.002 * rms
        return float(np.clip((rms / max(self.noise, 1e-6) - 1.0) / (self.ratio - 1.0), 0.0, 1.0))


@dataclass
class VadEvent:
    kind: str  # onset | abort | start | tick | end
    t_ms: float
    speech_ms: float = 0.0  # time since onset
    voiced_ms: float = 0.0  # voiced chunks since onset
    audio: np.ndarray | None = None  # start: audio so far (early speaker score); end: whole utterance
    wall: float = 0.0  # clock time when the front end emitted it


class VadSegmenter:
    def __init__(
        self,
        vad,
        threshold: float = 0.5,
        min_speech_ms: float = 250,
        min_silence_ms: float = 650,
        max_utterance_s: float = 20.0,
        pre_roll_ms: float = 300,
        keep_tail_ms: float = 150,
        max_gap_ms: float = 64,
        short_silence_ms: float | None = None,
        short_utterance_ms: float = 0.0,
    ) -> None:
        self.vad = vad
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.short_silence_ms = short_silence_ms
        self.short_utterance_ms = short_utterance_ms
        self.max_ms = max_utterance_s * 1000
        self.keep_tail_ms = keep_tail_ms
        self.max_gap_ms = max_gap_ms
        self._pre: deque[np.ndarray] = deque(maxlen=int((pre_roll_ms + min_speech_ms + max_gap_ms) / CHUNK_MS) + 1)
        self._buf: list[np.ndarray] = []
        self._pending = np.zeros(0, np.float32)
        self.t_ms = 0.0
        self.last_prob = 0.0
        self.threshold_boost = 0.0  # raised while JARVIS speaks (residual echo after AEC)
        self._reset_state()

    @classmethod
    def from_config(cls, vad, cfg) -> VadSegmenter:
        """Build from a ``VadConfig``."""
        return cls(
            vad,
            threshold=cfg.threshold,
            min_speech_ms=cfg.min_speech_ms,
            min_silence_ms=cfg.min_silence_ms,
            max_utterance_s=cfg.max_utterance_s,
            pre_roll_ms=cfg.pre_roll_ms,
            short_silence_ms=cfg.short_silence_ms,
            short_utterance_ms=cfg.short_utterance_ms,
        )

    def _silence_needed(self) -> float:
        if self.short_silence_ms is not None and self._voiced_ms <= self.short_utterance_ms:
            return self.short_silence_ms
        return self.min_silence_ms

    def _reset_state(self) -> None:
        self.in_speech = False
        self._candidate_ms = 0.0
        self._gap_ms = 0.0
        self._since_onset_ms = 0.0
        self._voiced_ms = 0.0
        self._silence_ms = 0.0

    def reset(self) -> None:
        self.vad.reset()
        self._pre.clear()
        self._buf = []
        self._pending = np.zeros(0, np.float32)
        self._reset_state()

    def current_audio(self) -> np.ndarray:
        """Audio of the utterance in progress (pre-roll included) - for an early speaker score."""
        parts = self._buf if self.in_speech else list(self._pre)
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def feed(self, pcm_f32: np.ndarray) -> list[VadEvent]:
        self._pending = np.concatenate([self._pending, np.asarray(pcm_f32, np.float32).reshape(-1)])
        events: list[VadEvent] = []
        while len(self._pending) >= CHUNK:
            chunk, self._pending = self._pending[:CHUNK], self._pending[CHUNK:]
            self.t_ms += CHUNK_MS
            p = self.last_prob = self.vad.prob(chunk)
            speech = p >= min(0.95, self.threshold + self.threshold_boost)
            if not self.in_speech:
                self._pre.append(chunk)
                if speech:
                    if self._candidate_ms == 0:
                        events.append(VadEvent("onset", self.t_ms))
                    self._candidate_ms += CHUNK_MS
                    self._since_onset_ms += CHUNK_MS + self._gap_ms
                    self._gap_ms = 0.0
                elif self._candidate_ms > 0:
                    self._gap_ms += CHUNK_MS
                    if self._gap_ms > self.max_gap_ms:
                        events.append(VadEvent("abort", self.t_ms, self._since_onset_ms, self._candidate_ms))
                        self._reset_state()
                if self._candidate_ms >= self.min_speech_ms:
                    self.in_speech = True
                    self._buf = list(self._pre)
                    self._voiced_ms = self._candidate_ms
                    self._silence_ms = 0.0
                    events.append(VadEvent("start", self.t_ms, self._since_onset_ms, self._voiced_ms, np.concatenate(self._buf)))
                continue
            self._buf.append(chunk)
            self._since_onset_ms += CHUNK_MS
            if speech:
                self._voiced_ms += CHUNK_MS
                self._silence_ms = 0.0
            else:
                self._silence_ms += CHUNK_MS
            events.append(VadEvent("tick", self.t_ms, self._since_onset_ms, self._voiced_ms))
            if self._silence_ms >= self._silence_needed() or self._since_onset_ms >= self.max_ms:
                audio = np.concatenate(self._buf)
                drop = int(max(0.0, self._silence_ms - self.keep_tail_ms) / 1000 * SR)
                if drop:
                    audio = audio[:-drop]
                events.append(VadEvent("end", self.t_ms, self._since_onset_ms - self._silence_ms, self._voiced_ms, audio))
                self._buf = []
                self._pre.clear()
                self._reset_state()
        return events
