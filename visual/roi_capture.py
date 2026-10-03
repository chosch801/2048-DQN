"""Reusable Windows GDI capture of only the selected screen rectangle."""

import ctypes
from ctypes import wintypes
import operator
import os
import threading

from PIL import Image


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BitmapInfo(ctypes.Structure):
    _fields_ = [("header", _BitmapInfoHeader), ("colors", wintypes.DWORD * 1)]


def _load_apis():
    if os.name != "nt":
        raise OSError("Region capture requires Windows")
    user = ctypes.WinDLL("user32", use_last_error=True)
    gdi = ctypes.WinDLL("gdi32", use_last_error=True)
    signatures = [
        (user.GetDC, [wintypes.HWND], wintypes.HDC),
        (user.ReleaseDC, [wintypes.HWND, wintypes.HDC], ctypes.c_int),
        (gdi.CreateCompatibleDC, [wintypes.HDC], wintypes.HDC),
        (gdi.CreateDIBSection, [wintypes.HDC, ctypes.POINTER(_BitmapInfo),
                              wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                              wintypes.HANDLE, wintypes.DWORD], wintypes.HBITMAP),
        (gdi.SelectObject, [wintypes.HDC, wintypes.HANDLE], wintypes.HANDLE),
        (gdi.BitBlt, [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                     ctypes.c_int, wintypes.HDC, ctypes.c_int, ctypes.c_int,
                     wintypes.DWORD], wintypes.BOOL),
        (gdi.GdiFlush, [], wintypes.BOOL),
        (gdi.DeleteObject, [wintypes.HANDLE], wintypes.BOOL),
        (gdi.DeleteDC, [wintypes.HDC], wintypes.BOOL),
    ]
    for function, arguments, result in signatures:
        function.argtypes = arguments
        function.restype = result
    return user, gdi


class RegionCapture:
    """Create, grab and close on one thread; returned RGB images own their pixels."""

    def __init__(self, bbox):
        try:
            self.bbox = tuple(operator.index(value) for value in bbox)
        except (TypeError, ValueError) as exc:
            raise ValueError("bbox must contain four integer screen coordinates") from exc
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain four integer screen coordinates")
        left, top, right, bottom = self.bbox
        self.width, self.height = right - left, bottom - top
        if self.width <= 0 or self.height <= 0:
            raise ValueError("bbox must have positive width and height")
        self._owner = None
        self._screen = self._memory = self._bitmap = self._previous = None
        self._bits = ctypes.c_void_p()

    def _check_thread(self):
        if self._owner is not None and self._owner != threading.get_ident():
            raise RuntimeError("RegionCapture must be used and closed on its owner thread")

    def __enter__(self):
        self._check_thread()
        if self._bitmap:
            return self
        self._user, self._gdi = _load_apis()
        self._owner = threading.get_ident()
        try:
            self._screen = self._user.GetDC(None)
            if not self._screen:
                raise OSError("GetDC failed for screen capture")
            self._memory = self._gdi.CreateCompatibleDC(self._screen)
            if not self._memory:
                raise OSError("CreateCompatibleDC failed for screen capture")
            info = _BitmapInfo()
            info.header.biSize = ctypes.sizeof(_BitmapInfoHeader)
            info.header.biWidth = self.width
            info.header.biHeight = -self.height  # Top-down rows, no flip needed.
            info.header.biPlanes = 1
            info.header.biBitCount = 32
            self._bitmap = self._gdi.CreateDIBSection(
                self._screen, ctypes.byref(info), 0, ctypes.byref(self._bits), None, 0)
            if not self._bitmap or not self._bits.value:
                raise OSError("CreateDIBSection failed for screen capture")
            previous = self._gdi.SelectObject(self._memory, self._bitmap)
            if not previous or previous == ctypes.c_void_p(-1).value:
                raise OSError("SelectObject failed for screen capture")
            self._previous = previous
        except Exception:
            self.close()
            raise
        return self

    def grab(self):
        self.__enter__()
        left, top = self.bbox[:2]
        # Include layered windows so overlays remain visible to recognition.
        if not self._gdi.BitBlt(self._memory, 0, 0, self.width, self.height,
                                self._screen, left, top, 0x40CC0020):
            raise OSError("BitBlt failed for screen capture")
        if not self._gdi.GdiFlush():
            raise OSError("GdiFlush failed for screen capture")
        pixels = ctypes.string_at(self._bits, self.width * self.height * 4)
        return Image.frombytes("RGB", (self.width, self.height), pixels, "raw", "BGRX")

    def close(self):
        self._check_thread()
        if self._previous:
            self._gdi.SelectObject(self._memory, self._previous)
        if self._bitmap:
            self._gdi.DeleteObject(self._bitmap)
        if self._memory:
            self._gdi.DeleteDC(self._memory)
        if self._screen:
            self._user.ReleaseDC(None, self._screen)
        self._screen = self._memory = self._bitmap = self._previous = None
        self._bits = ctypes.c_void_p()
        self._owner = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
