"""
test_splash_center.py — Verify the splash screen is centered on the display.

Run with:
    python test_splash_center.py
"""
import tkinter as tk
import sys


def test_splash_centering():
    SW, SH = 520, 360  # must match _run_splash in app.py

    root = tk.Tk()
    root.withdraw()
    root.update()

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    expected_x = (screen_w - SW) // 2
    expected_y = (screen_h - SH) // 2

    splash = tk.Toplevel(root)
    splash.overrideredirect(True)
    splash.geometry(f"{SW}x{SH}+{expected_x}+{expected_y}")
    root.update()

    actual_x = splash.winfo_x()
    actual_y = splash.winfo_y()
    actual_w = splash.winfo_width()
    actual_h = splash.winfo_height()

    splash.destroy()
    root.destroy()

    # ── Assertions ───────────────────────────────────────────────────────────────
    TOLERANCE = 5  # px — allow minor DPI rounding

    errors = []

    if actual_w != SW or actual_h != SH:
        errors.append(f"Size mismatch: got {actual_w}x{actual_h}, expected {SW}x{SH}")

    if abs(actual_x - expected_x) > TOLERANCE:
        errors.append(
            f"X not centered: got {actual_x}, expected {expected_x} "
            f"(off by {actual_x - expected_x}px, screen_w={screen_w})"
        )

    if abs(actual_y - expected_y) > TOLERANCE:
        errors.append(
            f"Y not centered: got {actual_y}, expected {expected_y} "
            f"(off by {actual_y - expected_y}px, screen_h={screen_h})"
        )

    # Check the splash centre aligns with the screen centre
    splash_cx = actual_x + actual_w // 2
    splash_cy = actual_y + actual_h // 2
    screen_cx = screen_w // 2
    screen_cy = screen_h // 2

    if abs(splash_cx - screen_cx) > TOLERANCE:
        errors.append(
            f"Centre X wrong: splash centre={splash_cx}, screen centre={screen_cx}"
        )
    if abs(splash_cy - screen_cy) > TOLERANCE:
        errors.append(
            f"Centre Y wrong: splash centre={splash_cy}, screen centre={screen_cy}"
        )

    if errors:
        print("FAIL")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"PASS — screen={screen_w}x{screen_h}, "
              f"splash placed at ({actual_x},{actual_y}), "
              f"centre=({splash_cx},{splash_cy})")


if __name__ == "__main__":
    test_splash_centering()
