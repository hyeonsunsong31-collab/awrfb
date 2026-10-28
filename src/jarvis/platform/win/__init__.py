"""Windows platform implementation (ctypes + windows-capture). Importable only on Windows."""

from __future__ import annotations


def build_platform():
    from jarvis.platform.base import Platform
    from jarvis.platform.win.api import make_dpi_aware
    from jarvis.platform.win.capture_wgc import WgcFrameSource
    from jarvis.platform.win.cursor import WinCursorProbe
    from jarvis.platform.win.hotkeys import WinHotkeys
    from jarvis.platform.win.rawinput import RawInputMonitor
    from jarvis.platform.win.sendinput import SendInputSink
    from jarvis.platform.win.window import WinWindowManager

    make_dpi_aware()
    return Platform(
        name="windows",
        windows=WinWindowManager(),
        cursor=WinCursorProbe(),
        new_input_sink=SendInputSink,
        new_input_monitor=RawInputMonitor,
        new_hotkeys=WinHotkeys,
        new_frame_source=WgcFrameSource,
    )
