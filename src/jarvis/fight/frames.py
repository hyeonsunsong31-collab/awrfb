"""Latest-frame hand-off between the capture thread and the fight loop.

The fight loop must always act on the NEWEST frame, never a backlog: the capture callback overwrites
a single slot and the consumer waits for an index newer than the one it last processed.
"""

from __future__ import annotations

import threading
import time

from jarvis.platform.base import Frame


class LatestFrame:
    def __init__(self) -> None:
        self._frame: Frame | None = None
        self._cond = threading.Condition()
        self.dropped = 0
        self._last_taken = 0

    def put(self, frame: Frame) -> None:
        with self._cond:
            if self._frame is not None and self._frame.index > self._last_taken:
                self.dropped += 1  # consumer never saw the previous frame
            self._frame = frame
            self._cond.notify_all()

    def get_newer(self, than_index: int, timeout: float = 0.1) -> Frame | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._frame is None or self._frame.index <= than_index:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            self._last_taken = self._frame.index
            return self._frame

    def peek(self) -> Frame | None:
        with self._cond:
            return self._frame


class FpsMeter:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.fps = 0.0
        self._last: int | None = None

    def tick(self, t_ns: int) -> float:
        if self._last is not None and t_ns > self._last:
            inst = 1e9 / (t_ns - self._last)
            self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = t_ns
        return self.fps
