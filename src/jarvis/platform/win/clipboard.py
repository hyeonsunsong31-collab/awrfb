"""Clipboard text through the Win32 API (no extra packages)."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

_u = ctypes.WinDLL("user32", use_last_error=True)
_k = ctypes.WinDLL("kernel32", use_last_error=True)
_u.OpenClipboard.argtypes, _u.OpenClipboard.restype = [wintypes.HWND], wintypes.BOOL
_u.CloseClipboard.restype = wintypes.BOOL
_u.EmptyClipboard.restype = wintypes.BOOL
_u.GetClipboardData.argtypes, _u.GetClipboardData.restype = [wintypes.UINT], wintypes.HANDLE
_u.SetClipboardData.argtypes, _u.SetClipboardData.restype = [wintypes.UINT, wintypes.HANDLE], wintypes.HANDLE
_k.GlobalAlloc.argtypes, _k.GlobalAlloc.restype = [wintypes.UINT, ctypes.c_size_t], wintypes.HGLOBAL
_k.GlobalLock.argtypes, _k.GlobalLock.restype = [wintypes.HGLOBAL], wintypes.LPVOID
_k.GlobalUnlock.argtypes, _k.GlobalUnlock.restype = [wintypes.HGLOBAL], wintypes.BOOL
_k.GlobalFree.argtypes, _k.GlobalFree.restype = [wintypes.HGLOBAL], wintypes.HGLOBAL


def _open() -> None:
    for _ in range(20):  # another app may hold the clipboard for a moment
        if _u.OpenClipboard(None):
            return
        time.sleep(0.025)
    raise OSError("clipboard is busy")


def get_text() -> str:
    _open()
    try:
        h = _u.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return ""
        p = _k.GlobalLock(h)
        try:
            return ctypes.wstring_at(p) if p else ""
        finally:
            _k.GlobalUnlock(h)
    finally:
        _u.CloseClipboard()


def set_text(text: str) -> None:
    data = ctypes.create_unicode_buffer(text)
    size = ctypes.sizeof(data)
    h = _k.GlobalAlloc(GMEM_MOVEABLE, size)
    if not h:
        raise OSError("GlobalAlloc failed")
    p = _k.GlobalLock(h)
    ctypes.memmove(p, data, size)
    _k.GlobalUnlock(h)
    _open()
    try:
        _u.EmptyClipboard()
        if not _u.SetClipboardData(CF_UNICODETEXT, h):
            _k.GlobalFree(h)
            raise OSError("SetClipboardData failed")
    finally:
        _u.CloseClipboard()
