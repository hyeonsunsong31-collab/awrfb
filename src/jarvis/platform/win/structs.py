"""Win32 structures with explicit fixed-width fields.

We deliberately avoid ``ctypes.wintypes`` here so the layouts are identical (and unit-testable) on
Linux x86-64: ``sizeof(INPUT) == 40`` on 64-bit Windows, and the same on any LP64 host when built from
fixed-width types. Only 64-bit Python is supported.
"""

from __future__ import annotations

import ctypes

BYTE = ctypes.c_uint8
WORD = ctypes.c_uint16
DWORD = ctypes.c_uint32
LONG = ctypes.c_int32
ULONG = ctypes.c_uint32
UINT = ctypes.c_uint32
INT = ctypes.c_int32
BOOL = ctypes.c_int32
ULONG_PTR = ctypes.c_uint64
LONG_PTR = ctypes.c_int64
UINT_PTR = ctypes.c_uint64
SIZE_T = ctypes.c_uint64
LARGE_INTEGER = ctypes.c_int64
ULONGLONG = ctypes.c_uint64
HANDLE = ctypes.c_void_p
HWND = ctypes.c_void_p
WPARAM = UINT_PTR
LPARAM = LONG_PTR
LRESULT = LONG_PTR

# --- SendInput -------------------------------------------------------------------------------
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_ABSOLUTE = 0x8000

# Tag placed in dwExtraInfo of every event we inject ("JARV"), so our own input is identifiable.
JARVIS_EXTRA_INFO = 0x4A415256


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", LONG),
        ("dy", LONG),
        ("mouseData", DWORD),
        ("dwFlags", DWORD),
        ("time", DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", WORD),
        ("wScan", WORD),
        ("dwFlags", DWORD),
        ("time", DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", DWORD), ("wParamL", WORD), ("wParamH", WORD))


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", DWORD), ("u", _INPUTUNION))


def key_input(scan: int, *, extended: bool, up: bool) -> INPUT:
    flags = KEYEVENTF_SCANCODE
    if extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    if up:
        flags |= KEYEVENTF_KEYUP
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.ki = KEYBDINPUT(0, scan, flags, 0, JARVIS_EXTRA_INFO)
    return inp


def vk_input(vk: int, *, up: bool) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.ki = KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP if up else 0, 0, JARVIS_EXTRA_INFO)
    return inp


def mouse_input(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi = MOUSEINPUT(dx, dy, data, flags, 0, JARVIS_EXTRA_INFO)
    return inp


# --- geometry / windows ------------------------------------------------------------------------
class POINT(ctypes.Structure):
    _fields_ = (("x", LONG), ("y", LONG))


class RECT(ctypes.Structure):
    _fields_ = (("left", LONG), ("top", LONG), ("right", LONG), ("bottom", LONG))


class CURSORINFO(ctypes.Structure):
    _fields_ = (("cbSize", DWORD), ("flags", DWORD), ("hCursor", HANDLE), ("ptScreenPos", POINT))


CURSOR_SHOWING = 0x00000001


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = (("cbSize", UINT), ("dwTime", DWORD))


class MSG(ctypes.Structure):
    _fields_ = (
        ("hwnd", HWND),
        ("message", UINT),
        ("wParam", WPARAM),
        ("lParam", LPARAM),
        ("time", DWORD),
        ("pt", POINT),
        ("lPrivate", DWORD),
    )


# --- raw input ---------------------------------------------------------------------------------
RIM_TYPEMOUSE = 0
RIM_TYPEKEYBOARD = 1
RID_INPUT = 0x10000003
RIDEV_INPUTSINK = 0x00000100
RIDEV_REMOVE = 0x00000001
WM_INPUT = 0x00FF
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
WM_USER = 0x0400
WM_CLOSE = 0x0010
RI_KEY_BREAK = 0x01
RI_KEY_E0 = 0x02
RI_KEY_E1 = 0x04
RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
RI_MOUSE_LEFT_BUTTON_UP = 0x0002
RI_MOUSE_RIGHT_BUTTON_DOWN = 0x0004
RI_MOUSE_RIGHT_BUTTON_UP = 0x0008
RI_MOUSE_MIDDLE_BUTTON_DOWN = 0x0010
RI_MOUSE_MIDDLE_BUTTON_UP = 0x0020


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = (("usUsagePage", WORD), ("usUsage", WORD), ("dwFlags", DWORD), ("hwndTarget", HWND))


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = (("dwType", DWORD), ("dwSize", DWORD), ("hDevice", HANDLE), ("wParam", WPARAM))


class _RAWMOUSE_BUTTONS(ctypes.Structure):
    _fields_ = (("usButtonFlags", WORD), ("usButtonData", WORD))


class _RAWMOUSE_U(ctypes.Union):
    _fields_ = (("ulButtons", ULONG), ("b", _RAWMOUSE_BUTTONS))


class RAWMOUSE(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (
        ("usFlags", WORD),
        ("u", _RAWMOUSE_U),
        ("ulRawButtons", ULONG),
        ("lLastX", LONG),
        ("lLastY", LONG),
        ("ulExtraInformation", ULONG),
    )


class RAWKEYBOARD(ctypes.Structure):
    _fields_ = (
        ("MakeCode", WORD),
        ("Flags", WORD),
        ("Reserved", WORD),
        ("VKey", WORD),
        ("Message", UINT),
        ("ExtraInformation", ULONG),
    )


class _RAWINPUT_DATA(ctypes.Union):
    _fields_ = (("mouse", RAWMOUSE), ("keyboard", RAWKEYBOARD))


class RAWINPUT(ctypes.Structure):
    _fields_ = (("header", RAWINPUTHEADER), ("data", _RAWINPUT_DATA))


# --- job objects -------------------------------------------------------------------------------
JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", LARGE_INTEGER),
        ("PerJobUserTimeLimit", LARGE_INTEGER),
        ("LimitFlags", DWORD),
        ("MinimumWorkingSetSize", SIZE_T),
        ("MaximumWorkingSetSize", SIZE_T),
        ("ActiveProcessLimit", DWORD),
        ("Affinity", ULONG_PTR),
        ("PriorityClass", DWORD),
        ("SchedulingClass", DWORD),
    )


class IO_COUNTERS(ctypes.Structure):
    _fields_ = (
        ("ReadOperationCount", ULONGLONG),
        ("WriteOperationCount", ULONGLONG),
        ("OtherOperationCount", ULONGLONG),
        ("ReadTransferCount", ULONGLONG),
        ("WriteTransferCount", ULONGLONG),
        ("OtherTransferCount", ULONGLONG),
    )


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = (
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", SIZE_T),
        ("JobMemoryLimit", SIZE_T),
        ("PeakProcessMemoryUsed", SIZE_T),
        ("PeakJobMemoryUsed", SIZE_T),
    )
