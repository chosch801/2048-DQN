"""Windows screen capture and guarded keyboard input for the desktop player."""

import ctypes
from ctypes import wintypes
import os
import time

from PIL import ImageGrab


ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG), ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", wintypes.DWORD), ("data", _INPUTUNION)]


def _user32():
    if os.name != "nt":
        raise OSError("Desktop input requires Windows")
    api = ctypes.WinDLL("user32", use_last_error=True)
    api.WindowFromPoint.argtypes = [wintypes.POINT]
    api.WindowFromPoint.restype = wintypes.HWND
    api.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    api.GetAncestor.restype = wintypes.HWND
    api.GetForegroundWindow.argtypes = []
    api.GetForegroundWindow.restype = wintypes.HWND
    api.IsWindow.argtypes = [wintypes.HWND]
    api.IsWindow.restype = wintypes.BOOL
    api.IsIconic.argtypes = [wintypes.HWND]
    api.IsIconic.restype = wintypes.BOOL
    api.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    api.ShowWindow.restype = wintypes.BOOL
    api.SetForegroundWindow.argtypes = [wintypes.HWND]
    api.SetForegroundWindow.restype = wintypes.BOOL
    api.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    api.SendInput.restype = wintypes.UINT
    api.GetAsyncKeyState.argtypes = [ctypes.c_int]
    api.GetAsyncKeyState.restype = ctypes.c_short
    return api


def set_dpi_awareness():
    """Call before creating the UI so screen coordinates use physical pixels."""
    api = _user32()
    try:
        aware = api.SetProcessDpiAwarenessContext
    except AttributeError:
        aware = None
    if aware is not None:
        aware.argtypes = [ctypes.c_void_p]
        aware.restype = wintypes.BOOL
        if aware(ctypes.c_void_p(-4)):
            return
        if ctypes.get_last_error() == 5:  # Awareness was already configured.
            return
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
        shcore.SetProcessDpiAwareness.restype = ctypes.c_long
        shcore.SetProcessDpiAwareness(2)
    except OSError:
        api.SetProcessDPIAware.argtypes = []
        api.SetProcessDPIAware.restype = wintypes.BOOL
        api.SetProcessDPIAware()


def capture_region(bbox):
    """Capture (left, top, right, bottom), including negative monitor positions."""
    if len(bbox) != 4:
        raise ValueError("bbox must contain left, top, right, bottom")
    left, top, right, bottom = map(int, bbox)
    if right <= left or bottom <= top:
        raise ValueError("Capture region must have positive width and height")
    return ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True).convert("RGB")


def window_at_point(x, y):
    """Return the root top-level HWND at an absolute screen position, or 0."""
    api = _user32()
    hwnd = api.WindowFromPoint(wintypes.POINT(int(x), int(y)))
    return int(api.GetAncestor(hwnd, 2) or hwnd or 0) if hwnd else 0


def is_foreground(hwnd):
    api = _user32()
    return bool(hwnd and api.IsWindow(hwnd) and api.GetForegroundWindow() == hwnd)


def focus_window(hwnd):
    """Request focus; return False if Windows refuses instead of forcing input."""
    api = _user32()
    if not hwnd or not api.IsWindow(hwnd):
        return False
    if api.IsIconic(hwnd):
        api.ShowWindow(hwnd, 9)  # SW_RESTORE
    api.SetForegroundWindow(hwnd)
    return is_foreground(hwnd)


def tap_action(hwnd, action, key_mode="wasd", hold_seconds=0.04):
    """Tap 0=up, 1=down, 2=left, 3=right only while the target has focus.

    SendInput is foreground input, not HWND-addressed input. A focus change
    during the hold is reported after releasing the key; the caller must pause.
    """
    keys = {"wasd": (0x57, 0x53, 0x41, 0x44), "arrows": (0x26, 0x28, 0x25, 0x27)}
    if key_mode not in keys:
        raise ValueError("key_mode must be 'wasd' or 'arrows'")
    if action not in range(4):
        raise ValueError("action must be an integer from 0 through 3")
    if not 0 <= hold_seconds <= 1:
        raise ValueError("hold_seconds must be between 0 and 1")
    api = _user32()
    if not is_foreground(hwnd):
        raise RuntimeError("Target window is not foreground; input cancelled")
    flags = 0x0001 if key_mode == "arrows" else 0  # KEYEVENTF_EXTENDEDKEY
    down = INPUT(type=1, ki=KEYBDINPUT(wVk=keys[key_mode][action], dwFlags=flags))
    up = INPUT(type=1, ki=KEYBDINPUT(wVk=keys[key_mode][action], dwFlags=flags | 0x0002))
    if not is_foreground(hwnd):
        raise RuntimeError("Target window lost focus; input cancelled")
    try:
        if api.SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT)) != 1:
            raise RuntimeError("Windows rejected key press (check target permissions)")
        time.sleep(hold_seconds)
    finally:
        # Release even on interruption or a focus change to prevent a stuck key.
        if api.SendInput(1, ctypes.byref(up), ctypes.sizeof(INPUT)) != 1:
            raise RuntimeError("Windows rejected key release; pause the player")
    if not is_foreground(hwnd):
        raise RuntimeError("Target window lost focus during input; pause the player")


def escape_pressed():
    return bool(_user32().GetAsyncKeyState(0x1B) & 0x8000)
