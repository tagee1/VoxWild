"""Shared window placement helpers.

Lives in its own module because app.py, pronunciation.py and settings_window.py
all need it and app.py imports the other two — so they cannot import back from
app.py without a circular import. Three hand-maintained copies had already
drifted apart, and the one in settings_window.py never got a bug fix the other
two did.
"""
import ctypes

import customtkinter as ctk


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", _RECT),
                ("rcWork", _RECT), ("dwFlags", ctypes.c_ulong)]


def work_area(win, cx, cy):
    """Usable desktop — the screen minus the taskbar — as (left, top, right, bottom).

    Reports the monitor under the point (cx, cy) rather than the primary
    display, so a dialog opened while the app sits on a second screen is not
    yanked back to the first. Falls back to the primary work area, then to the
    raw screen size, so this can never be what stops a dialog opening.
    """
    try:
        mon = ctypes.windll.user32.MonitorFromPoint(_POINT(int(cx), int(cy)), 2)
        mi = _MONITORINFO()
        mi.cbSize = ctypes.sizeof(_MONITORINFO)
        if ctypes.windll.user32.GetMonitorInfoW(mon, ctypes.byref(mi)):
            r = mi.rcWork
            if r.right > r.left and r.bottom > r.top:
                return r.left, r.top, r.right, r.bottom
    except Exception:
        pass
    try:
        r = _RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(r), 0):
            if r.right > r.left and r.bottom > r.top:
                return r.left, r.top, r.right, r.bottom
    except Exception:
        pass
    return 0, 0, win.winfo_screenwidth(), win.winfo_screenheight()


def frame_overhead():
    """Pixels the window manager adds around the client area, as (width, height).

    geometry() sizes the CLIENT area only — Windows then draws a title bar and
    a resize border around it, so the real on-screen footprint is bigger. On a
    150% display that is 45px of extra height, which is enough on its own to
    push a tall dialog's footer buttons below the bottom of the screen.
    """
    try:
        u32 = ctypes.windll.user32
        edge = u32.GetSystemMetrics(33) + u32.GetSystemMetrics(92)  # frame + padded border
        return 2 * edge, u32.GetSystemMetrics(4) + edge             # caption + one edge
    except Exception:
        return 16, 45


def scaling(win):
    """The display scaling CustomTkinter will apply to this window's geometry."""
    try:
        return ctk.ScalingTracker.get_window_scaling(win) or 1.0
    except Exception:
        return 1.0


def center_window(win, w, h, parent=None):
    """Size win to w×h and center it over `parent`, kept fully on screen.

    Centers on the parent window rather than the screen so it behaves on
    remote/cloud desktops (Shadow PC) and multi-monitor setups where
    GetSystemMetrics can report the wrong coordinate space.

    w and h are CustomTkinter units, NOT screen pixels. CTk multiplies them by
    the display scaling when it applies the geometry but passes x/y straight
    through, so on a 150% display a 560x580 dialog is really 840x870 on screen.
    Centering on the requested size pushed every dialog down and to the right by
    half the difference — far enough that the taller ones ran off the bottom
    edge and hid their footer buttons.
    """
    win.update_idletasks()
    scale = scaling(win)
    real_w, real_h = round(w * scale), round(h * scale)
    try:
        p = parent if parent is not None else win.master
        px, py = p.winfo_x(), p.winfo_y()
        pw, ph = p.winfo_width(), p.winfo_height()
        x = px + (pw - real_w) // 2
        y = py + (ph - real_h) // 2
        cx, cy = px + pw // 2, py + ph // 2
    except Exception:
        try:
            sw = ctypes.windll.user32.GetSystemMetrics(0)
            sh = ctypes.windll.user32.GetSystemMetrics(1)
        except Exception:
            sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        x, y = (sw - real_w) // 2, (sh - real_h) // 2
        cx, cy = sw // 2, sh // 2
    # Clamp onto the visible desktop. If the parent is hidden or not yet mapped
    # (e.g. still behind the splash) Windows reports its position as a large
    # negative sentinel, which would drop the dialog off-screen — where a modal
    # with grab_set() can neither be reached nor dismissed. Clamp against the
    # work area, not the whole screen, so the taskbar cannot cover a footer, and
    # reserve the title bar's height too or the clamp lets the frame overhang.
    # A dialog too big for the work area lands at the top-left rather than
    # centered, which keeps its title bar reachable so it can still be dragged.
    ow, oh = frame_overhead()
    l, t, r, b = work_area(win, cx, cy)
    x = max(l, min(x, max(l, r - real_w - ow)))
    y = max(t, min(y, max(t, b - real_h - oh)))
    win.geometry(f"{w}x{h}+{x}+{y}")
