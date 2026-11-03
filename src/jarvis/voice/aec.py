"""Acoustic echo cancellation with the WebRTC audio processing module (via livekit).

Near end = microphone, far end = the reference (WASAPI loopback of everything the speakers play, or
the TTS output when no loopback exists). Both run at 16 kHz in 10 ms int16 frames. All APM calls happen
on the audio worker thread; ``push_reference`` (called from the loopback thread) only enqueues.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from jarvis.voice.resample import float_to_int16, int16_to_float

log = logging.getLogger(__name__)

RATE = 16000
FRAME = 160  # 10 ms


class NullAec:
    enabled = False

    def push_reference(self, frame: np.ndarray) -> None:
        pass

    def process(self, near: np.ndarray) -> np.ndarray:
        return near


class EchoCanceller:
    enabled = True

    def __init__(
        self,
        *,
        delay_ms: int = 60,
        noise_suppression: bool = True,
        high_pass_filter: bool = True,
        auto_gain_control: bool = False,
        apm: Any | None = None,
        frame_cls: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_ref_frames: int = 50,
        silence_after_s: float = 0.2,
    ) -> None:
        if apm is None or frame_cls is None:
            from livekit import rtc

            apm = apm or rtc.AudioProcessingModule(
                echo_cancellation=True,
                noise_suppression=noise_suppression,
                high_pass_filter=high_pass_filter,
                auto_gain_control=auto_gain_control,
            )
            frame_cls = frame_cls or rtc.AudioFrame
        self._apm = apm
        self._frame_cls = frame_cls
        self._clock = clock
        self._ref: deque[np.ndarray] = deque(maxlen=max_ref_frames)  # bounded: ~500 ms
        self._lock = threading.Lock()
        self._last_ref = float("-inf")
        self._silence_after_s = silence_after_s
        self._zero = np.zeros(FRAME, np.float32)
        self.delay_ms = delay_ms
        self.ref_frames = 0
        self.near_frames = 0

    @classmethod
    def from_config(cls, cfg) -> EchoCanceller:
        return cls(delay_ms=cfg.delay_ms, noise_suppression=cfg.noise_suppression,
                   high_pass_filter=cfg.high_pass_filter, auto_gain_control=cfg.auto_gain_control)

    def push_reference(self, frame: np.ndarray) -> None:
        with self._lock:
            self._ref.append(np.asarray(frame, np.float32))
            self._last_ref = self._clock()

    def _frame(self, x: np.ndarray):
        return self._frame_cls(bytearray(float_to_int16(x).tobytes()), RATE, 1, FRAME)

    def process(self, near: np.ndarray) -> np.ndarray:
        with self._lock:
            refs = list(self._ref)
            self._ref.clear()
            quiet = self._clock() - self._last_ref > self._silence_after_s
        if not refs and quiet:
            refs = [self._zero]  # speakers silent: keep the far-end timeline moving
        for r in refs:
            self._apm.process_reverse_stream(self._frame(r))
            self.ref_frames += 1
        self._apm.set_stream_delay_ms(int(self.delay_ms))
        f = self._frame(near)
        self._apm.process_stream(f)
        self.near_frames += 1
        return int16_to_float(np.frombuffer(f.data, dtype=np.int16))


def build_aec(cfg) -> EchoCanceller | NullAec:
    """EchoCanceller when enabled and livekit is installed, otherwise a pass-through."""
    if not cfg.enabled or cfg.reference == "none":
        return NullAec()
    try:
        return EchoCanceller.from_config(cfg)
    except Exception as e:
        log.warning("echo cancellation unavailable (%s) - use headphones to avoid self-interruption", e)
        return NullAec()
