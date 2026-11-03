"""Audio front end: one worker thread turning microphone frames into voice events.

mic (PortAudio thread) -> queue -> [worker] echo cancellation -> VAD segmenter (32 ms) + wake word
(80 ms) -> events handed to the asyncio side through ``emit`` (which must be thread-safe, e.g.
``loop.call_soon_threadsafe``). Keeping DSP off the event loop keeps barge-in timing steady.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from jarvis.voice.resample import rms
from jarvis.voice.vad import VadSegmenter

log = logging.getLogger(__name__)


@dataclass
class WakeEvent:
    score: float
    wall: float


@dataclass
class LevelEvent:
    mic: float
    wall: float


class AudioFrontEnd:
    def __init__(self, *, aec, segmenter: VadSegmenter, wake=None, emit: Callable[[object], None],
                 clock: Callable[[], float] = time.monotonic, level_every_s: float = 0.05, max_frames: int = 400) -> None:
        self.aec = aec
        self.seg = segmenter
        self.wake = wake
        self.emit = emit
        self._clock = clock
        self._level_every = level_every_s
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max_frames)  # ~4 s of 10 ms frames
        self._thread: threading.Thread | None = None
        self._last_level = 0.0
        self._peak = 0.0
        self.dropped = 0
        self.frames = 0

    def push_mic(self, frame: np.ndarray) -> None:
        """Called from the audio driver thread: never blocks."""
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="voice-frontend", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        t, self._thread = self._thread, None
        if t is not None:
            try:
                self._q.put(None, timeout=timeout)
            except queue.Full:
                pass
            t.join(timeout)

    def flush(self) -> None:
        """Drop queued audio and partial utterances (after unmute: never act on stale audio)."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self.seg.reset()

    def process(self, frame: np.ndarray) -> None:
        try:
            clean = self.aec.process(frame)
        except Exception:
            log.exception("echo cancellation failed on a frame; using raw audio")
            clean = frame
        now = self._clock()
        self.frames += 1
        for ev in self.seg.feed(clean):
            ev.wall = now
            self.emit(ev)
        if self.wake is not None:
            for score in self.wake.feed(clean):
                self.emit(WakeEvent(score, now))
        self._peak = max(self._peak, rms(clean))
        if now - self._last_level >= self._level_every:
            self.emit(LevelEvent(self._peak, now))
            self._peak = 0.0
            self._last_level = now

    def _run(self) -> None:
        while True:
            f = self._q.get()
            if f is None:
                return
            try:
                self.process(f)
            except Exception:
                log.exception("voice front end failed on a frame")
