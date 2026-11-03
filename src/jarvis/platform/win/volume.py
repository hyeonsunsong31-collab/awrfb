"""Speaker volume through the Windows endpoint-volume COM API (pycaw). No key events are sent, so this
can never reach the Roblox window."""

from __future__ import annotations


def _endpoint():
    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    dev = AudioUtilities.GetSpeakers()
    ev = getattr(dev, "EndpointVolume", None)  # newer pycaw wraps the device
    if ev is not None:
        return ev
    return cast(dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None), POINTER(IAudioEndpointVolume))


def change_volume(delta: float) -> float:
    """Add ``delta`` (-1..1) to the master volume; returns the new level 0..1. Call from any thread."""
    import comtypes

    comtypes.CoInitialize()
    try:
        vol = _endpoint()
        new = min(1.0, max(0.0, float(vol.GetMasterVolumeLevelScalar()) + delta))
        vol.SetMasterVolumeLevelScalar(new, None)
        return new
    finally:
        comtypes.CoUninitialize()


def get_volume() -> float:
    """Current master volume 0..1."""
    import comtypes

    comtypes.CoInitialize()
    try:
        return float(_endpoint().GetMasterVolumeLevelScalar())
    finally:
        comtypes.CoUninitialize()
