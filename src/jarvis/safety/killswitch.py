"""Process-local kill switch. Stopping is always safe, so anyone (any voice, any hotkey) may trigger it.

Each process registers what "stop" means locally (fightd: release keys + disarm; core: cancel the
agent + stop speech; toolhost: terminate job objects). The supervisor broadcasts ``kill`` on the bus.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

log = logging.getLogger(__name__)


class KillSwitch:
    def __init__(self, on_broadcast: Callable[[dict], None] | None = None) -> None:
        self._callbacks: list[Callable[[str], None]] = []
        self._lock = threading.Lock()
        self._broadcast = on_broadcast
        self.last_kill_ns: int = 0
        self.last_reason: str = ""

    def on_kill(self, fn: Callable[[str], None]) -> None:
        with self._lock:
            self._callbacks.append(fn)

    def trigger(self, reason: str, *, broadcast: bool = True) -> None:
        self.last_kill_ns = time.perf_counter_ns()
        self.last_reason = reason
        log.warning("KILL: %s", reason)
        with self._lock:
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(reason)
            except Exception:  # one failing callback must not stop the others
                log.exception("kill callback failed")
        if broadcast and self._broadcast is not None:
            try:
                self._broadcast({"reason": reason, "t_ns": self.last_kill_ns})
            except Exception:
                log.exception("kill broadcast failed")
