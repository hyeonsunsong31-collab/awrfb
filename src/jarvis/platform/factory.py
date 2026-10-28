"""Chooses the platform implementation.

On Windows this returns the real implementation. Anywhere else it raises ``PlatformUnavailable``
unless ``JARVIS_TEST=1`` *and* a test platform was registered - so a fake can never silently stand in
for the real game connection at runtime.
"""

from __future__ import annotations

import os
import sys

from jarvis.platform.base import Platform, PlatformUnavailable

_test_platform: Platform | None = None
_cached: Platform | None = None


def test_mode() -> bool:
    return os.environ.get("JARVIS_TEST") == "1"


def register_test_platform(platform: Platform | None) -> None:
    """Only honoured when ``JARVIS_TEST=1``. Tests call this from fixtures."""
    global _test_platform, _cached
    if platform is not None and not test_mode():
        raise PlatformUnavailable("refusing to register a test platform outside JARVIS_TEST=1")
    _test_platform = platform
    _cached = None


def get_platform() -> Platform:
    global _cached
    if _cached is not None:
        return _cached
    if test_mode() and _test_platform is not None:
        _cached = _test_platform
        return _cached
    if sys.platform == "win32":
        from jarvis.platform.win import build_platform

        _cached = build_platform()
        return _cached
    raise PlatformUnavailable(
        "This feature needs Windows (real Roblox window capture and SendInput). "
        f"Current platform: {sys.platform}."
    )
