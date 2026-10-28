"""One monotonic, high-resolution time base for every timestamp in the system.

On Windows ``time.perf_counter_ns`` is backed by QueryPerformanceCounter, which is also the time
base of Windows Graphics Capture frame timestamps (``SystemRelativeTime``, 100 ns units). Keeping
everything on that clock lets us measure capture-to-input latency directly.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


class Clock(Protocol):
    def now_ns(self) -> int: ...


class MonotonicClock:
    """The real clock. ``now_ns`` is monotonic; ``wall_ns`` is only for human-readable logs."""

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    @staticmethod
    def wall_ns() -> int:
        return time.time_ns()


class ManualClock:
    """A clock that only moves when told to. Used by deterministic replay of recorded sessions."""

    def __init__(self, start_ns: int = 0) -> None:
        self._t = start_ns
        self._lock = threading.Lock()

    def now_ns(self) -> int:
        with self._lock:
            return self._t

    def set(self, t_ns: int) -> None:
        with self._lock:
            if t_ns < self._t:
                raise ValueError("ManualClock cannot go backwards")
            self._t = t_ns

    def advance_ms(self, ms: float) -> int:
        with self._lock:
            self._t += int(ms * NS_PER_MS)
            return self._t


SYSTEM_CLOCK = MonotonicClock()


def now_ns() -> int:
    return time.perf_counter_ns()


def ms_since(t_ns: int, now: int | None = None) -> float:
    return ((now if now is not None else time.perf_counter_ns()) - t_ns) / NS_PER_MS


def wgc_timespan_to_ns(timespan_100ns: int) -> int:
    """Convert a Windows Graphics Capture ``SystemRelativeTime`` (100 ns ticks) to ns."""
    return int(timespan_100ns) * 100
