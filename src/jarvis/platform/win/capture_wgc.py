"""Roblox window capture via Windows Graphics Capture (``windows-capture`` 2.x).

WGC captures the target window's own surface, so overlays drawn above it (our HUD) never appear in
the frames. The library hands us a zero-copy view that is only valid inside the callback, so we copy
the BGR pixels before returning.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from jarvis.common.clock import wgc_timespan_to_ns
from jarvis.platform.base import Frame, WindowRef

log = logging.getLogger(__name__)


class WgcFrameSource:
    def __init__(self, max_fps: int = 120, draw_border: bool | None = False) -> None:
        self._min_interval_ms = max(1, int(1000 / max(1, max_fps)))
        self._draw_border = draw_border
        self._control = None
        self._running = threading.Event()
        self._index = 0
        self.last_error: str | None = None

    def _make_capture(self, hwnd: int):
        from windows_capture import WindowsCapture

        try:
            return WindowsCapture(
                cursor_capture=False,
                draw_border=self._draw_border,
                minimum_update_interval=self._min_interval_ms,
                window_hwnd=hwnd,
            )
        except Exception as e:  # older Windows builds cannot toggle the yellow border
            log.info("WGC border toggle unsupported (%s); retrying with defaults", e)
            return WindowsCapture(cursor_capture=False, draw_border=None, window_hwnd=hwnd)

    def start(self, window: WindowRef, on_frame: Callable[[Frame], None]) -> None:
        if self._running.is_set():
            raise RuntimeError("capture already running")
        cap = self._make_capture(window.hwnd)
        source = self

        @cap.event
        def on_frame_arrived(frame, capture_control):
            if not source._running.is_set():
                capture_control.stop()
                return
            t = time.perf_counter_ns()
            buf = frame.frame_buffer
            img = np.ascontiguousarray(buf[:, :, :3])  # copy out of the native mapped frame
            source._index += 1
            try:
                on_frame(Frame(image=img, t_ns=t, index=source._index, source_t_ns=wgc_timespan_to_ns(frame.timespan)))
            except Exception:
                log.exception("frame consumer failed")

        @cap.event
        def on_closed():
            source._running.clear()
            log.info("WGC capture session closed")

        self._running.set()
        self._control = cap.start_free_threaded()

    def stop(self) -> None:
        self._running.clear()
        if self._control is not None:
            try:
                self._control.stop()
                self._control.wait()
            except Exception as e:
                self.last_error = str(e)
            self._control = None

    def running(self) -> bool:
        return self._running.is_set()
