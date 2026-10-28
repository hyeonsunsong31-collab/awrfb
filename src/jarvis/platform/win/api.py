"""Lazily loaded Win32 functions with explicit argtypes/restype.

Every function gets precise signatures: without them ctypes truncates 64-bit HWND/HANDLE values to
32-bit ints, which "works" until it suddenly doesn't. Import this module only on Windows.
"""

from __future__ import annotations

import ctypes
import sys
from functools import cache

from jarvis.platform.win.structs import (
    BOOL,
    CURSORINFO,
    DWORD,
    HANDLE,
    HWND,
    INPUT,
    INT,
    LASTINPUTINFO,
    LPARAM,
    LRESULT,
    MSG,
    POINT,
    RAWINPUTDEVICE,
    RECT,
    UINT,
    WORD,
    WPARAM,
)

if sys.platform != "win32":  # pragma: no cover - guarded import
    raise ImportError("jarvis.platform.win.api is Windows-only")

WNDPROC = ctypes.WINFUNCTYPE(LRESULT, HWND, UINT, WPARAM, LPARAM)
WNDENUMPROC = ctypes.WINFUNCTYPE(BOOL, HWND, LPARAM)
HWND_MESSAGE = HWND(-3)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = (
        ("cbSize", UINT),
        ("style", UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", INT),
        ("cbWndExtra", INT),
        ("hInstance", HANDLE),
        ("hIcon", HANDLE),
        ("hCursor", HANDLE),
        ("hbrBackground", HANDLE),
        ("lpszMenuName", ctypes.c_wchar_p),
        ("lpszClassName", ctypes.c_wchar_p),
        ("hIconSm", HANDLE),
    )


def _sig(fn, argtypes, restype):
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


@cache
def user32():
    d = ctypes.WinDLL("user32", use_last_error=True)
    _sig(d.SendInput, [UINT, ctypes.POINTER(INPUT), INT], UINT)
    _sig(d.GetForegroundWindow, [], HWND)
    _sig(d.SetForegroundWindow, [HWND], BOOL)
    _sig(d.BringWindowToTop, [HWND], BOOL)
    _sig(d.SwitchToThisWindow, [HWND, BOOL], None)
    _sig(d.ShowWindow, [HWND, INT], BOOL)
    _sig(d.IsIconic, [HWND], BOOL)
    _sig(d.IsWindow, [HWND], BOOL)
    _sig(d.IsWindowVisible, [HWND], BOOL)
    _sig(d.GetWindowTextLengthW, [HWND], INT)
    _sig(d.GetWindowTextW, [HWND, ctypes.c_wchar_p, INT], INT)
    _sig(d.GetClassNameW, [HWND, ctypes.c_wchar_p, INT], INT)
    _sig(d.GetWindowThreadProcessId, [HWND, ctypes.POINTER(DWORD)], DWORD)
    _sig(d.EnumWindows, [WNDENUMPROC, LPARAM], BOOL)
    _sig(d.GetClientRect, [HWND, ctypes.POINTER(RECT)], BOOL)
    _sig(d.GetWindowRect, [HWND, ctypes.POINTER(RECT)], BOOL)
    _sig(d.ClientToScreen, [HWND, ctypes.POINTER(POINT)], BOOL)
    _sig(d.AttachThreadInput, [DWORD, DWORD, BOOL], BOOL)
    _sig(d.GetCursorInfo, [ctypes.POINTER(CURSORINFO)], BOOL)
    _sig(d.GetAsyncKeyState, [INT], ctypes.c_int16)
    _sig(d.MapVirtualKeyW, [UINT, UINT], UINT)
    _sig(d.RegisterHotKey, [HWND, INT, UINT, UINT], BOOL)
    _sig(d.UnregisterHotKey, [HWND, INT], BOOL)
    _sig(d.GetMessageW, [ctypes.POINTER(MSG), HWND, UINT, UINT], BOOL)
    _sig(d.TranslateMessage, [ctypes.POINTER(MSG)], BOOL)
    _sig(d.DispatchMessageW, [ctypes.POINTER(MSG)], LRESULT)
    _sig(d.PostThreadMessageW, [DWORD, UINT, WPARAM, LPARAM], BOOL)
    _sig(d.PostMessageW, [HWND, UINT, WPARAM, LPARAM], BOOL)
    _sig(d.RegisterRawInputDevices, [ctypes.POINTER(RAWINPUTDEVICE), UINT, UINT], BOOL)
    _sig(d.GetRawInputData, [HANDLE, UINT, ctypes.c_void_p, ctypes.POINTER(UINT), UINT], UINT)
    _sig(d.DefWindowProcW, [HWND, UINT, WPARAM, LPARAM], LRESULT)
    _sig(d.RegisterClassExW, [ctypes.POINTER(WNDCLASSEXW)], WORD)
    _sig(d.UnregisterClassW, [ctypes.c_wchar_p, HANDLE], BOOL)
    _sig(
        d.CreateWindowExW,
        [
            DWORD,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            DWORD,
            INT,
            INT,
            INT,
            INT,
            HWND,
            HANDLE,
            HANDLE,
            ctypes.c_void_p,
        ],
        HWND,
    )
    _sig(d.DestroyWindow, [HWND], BOOL)
    _sig(d.GetLastInputInfo, [ctypes.POINTER(LASTINPUTINFO)], BOOL)
    try:
        _sig(d.GetDpiForWindow, [HWND], UINT)
    except AttributeError:  # pre Windows 10 1607
        pass
    try:
        _sig(d.SetProcessDpiAwarenessContext, [HANDLE], BOOL)
    except AttributeError:
        pass
    return d


@cache
def kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    _sig(k.GetCurrentThreadId, [], DWORD)
    _sig(k.GetModuleHandleW, [ctypes.c_wchar_p], HANDLE)
    _sig(k.CreateJobObjectW, [ctypes.c_void_p, ctypes.c_wchar_p], HANDLE)
    _sig(k.SetInformationJobObject, [HANDLE, INT, ctypes.c_void_p, DWORD], BOOL)
    _sig(k.AssignProcessToJobObject, [HANDLE, HANDLE], BOOL)
    _sig(k.TerminateJobObject, [HANDLE, UINT], BOOL)
    _sig(k.CloseHandle, [HANDLE], BOOL)
    _sig(k.OpenProcess, [DWORD, BOOL, DWORD], HANDLE)
    _sig(k.GetTickCount, [], DWORD)
    _sig(k.SetPriorityClass, [HANDLE, DWORD], BOOL)
    _sig(k.GetCurrentProcess, [], HANDLE)
    return k


def last_error() -> int:
    return ctypes.get_last_error()


def make_dpi_aware() -> None:
    """Per-monitor-v2 DPI awareness so window rects/capture coordinates are physical pixels."""
    u = user32()
    if hasattr(u, "SetProcessDpiAwarenessContext"):
        u.SetProcessDpiAwarenessContext(HANDLE(-4))  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
