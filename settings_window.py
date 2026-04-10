import ctypes
import customtkinter as ctk
import json
import os
import sys
from tkinter import filedialog


def _center_window(win, w: int, h: int, parent=None) -> None:
    win.update_idletasks()
    try:
        p = parent or win.master
        x = p.winfo_x() + (p.winfo_width()  - w) // 2
        y = p.winfo_y() + (p.winfo_height() - h) // 2
    except Exception:
        try:
            sw = ctypes.windll.user32.GetSystemMetrics(0)
            sh = ctypes.windll.user32.GetSystemMetrics(1)
        except Exception:
            sw = win.winfo_screenwidth()
            sh = win.winfo_screenheight()
        x, y = (sw - w) // 2, (sh - h) // 2
    win.geometry(f"{w}x{h}+{x}+{y}")

SETTINGS_FILE = os.path.join(
    os.environ.get("APPDATA", os.path.expanduser("~")), "TTS Studio", "settings.json"
)

# ── Studio Gold palette (must match app.py) ───────────────────────────────────
C_BG        = "#0d0d0d"
C_SURFACE   = "#171717"
C_CARD      = "#1f1f1f"
C_ELEVATED  = "#2a2a2a"
C_BORDER    = "#383838"
C_ACCENT    = "#e8940a"
C_ACCENT_H  = "#f5aa2a"
C_ACCENT_D  = "#3d2200"
C_TXT       = "#f0ece4"
C_TXT2      = "#9a9290"
C_TXT3      = "#4e4a48"

BTN_GHOST = dict(fg_color="transparent", hover_color=C_ELEVATED,
                 border_width=1, border_color=C_BORDER, text_color=C_TXT2)

DEFAULT_SETTINGS = {
    "default_output_folder": "",
    "default_voice": "🇬🇧 Male - George (Best)",
    "default_speed": 0.85,
    "theme": "dark",
    "notify_on_completion": True,
    "notify_threshold_seconds": 10,
    "auto_clean_text": False,
    "default_profile": "🌙 Calm Narrator",
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in DEFAULT_SETTINGS.items():
                if k not in data:
                    data[k] = v
            return data
        except (OSError, json.JSONDecodeError) as e:
            print(f"[settings] Failed to load settings: {e}", file=sys.stderr)
    return DEFAULT_SETTINGS.copy()

def save_settings(data):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        print(f"[settings] Failed to save settings: {e}", file=sys.stderr)

def open_settings_window(parent, voices: list, profiles: list, on_save_callback=None):
    """Open the settings window."""

    win = ctk.CTkToplevel(parent)
    win.title("Settings")
    _center_window(win, 520, 640)
    win.resizable(False, False)
    win.grab_set()
    win.configure(fg_color=C_BG)

    current = load_settings()

    # ── Header band ───────────────────────────────────────────────────────────
    hdr = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=60)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)

    hdr_inner = ctk.CTkFrame(hdr, fg_color="transparent")
    hdr_inner.pack(side="left", padx=18, pady=10)
    ctk.CTkFrame(hdr_inner, fg_color=C_ACCENT, width=8, height=8,
                 corner_radius=4).pack(side="left", padx=(0, 10))
    ctk.CTkLabel(hdr_inner, text="Settings",
                 font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
                 text_color=C_TXT).pack(side="left")

    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x")

    ctk.CTkLabel(win,
                 text="Changes take effect immediately after saving.",
                 font=ctk.CTkFont(family="Segoe UI", size=11),
                 text_color=C_TXT3).pack(pady=(10, 6))

    # ── Scrollable body ───────────────────────────────────────────────────────
    scroll = ctk.CTkScrollableFrame(win, fg_color=C_BG,
                                    scrollbar_button_color=C_ELEVATED,
                                    scrollbar_button_hover_color=C_ACCENT_D)
    scroll.pack(fill="both", expand=True, padx=16, pady=(0, 8))

    def _section(text):
        row = ctk.CTkFrame(scroll, fg_color="transparent")
        row.pack(anchor="w", padx=4, pady=(16, 4))
        ctk.CTkFrame(row, fg_color=C_ACCENT, width=3, height=10,
                     corner_radius=2).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(row, text=text,
                     font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
                     text_color=C_TXT2, anchor="w").pack(side="left")
        ctk.CTkFrame(scroll, height=1, fg_color=C_BORDER,
                     corner_radius=0).pack(fill="x", pady=(0, 8))

    def _row_label(text, note=""):
        f = ctk.CTkFrame(scroll, fg_color="transparent")
        f.pack(fill="x", padx=4, pady=(0, 2))
        ctk.CTkLabel(f, text=text, anchor="w",
                     font=ctk.CTkFont(family="Segoe UI", size=12),
                     text_color=C_TXT).pack(side="left")
        if note:
            ctk.CTkLabel(f, text=note, anchor="w",
                         font=ctk.CTkFont(family="Segoe UI", size=10),
                         text_color=C_TXT3).pack(side="left", padx=(6, 0))

    # ── General ───────────────────────────────────────────────────────────────
    _section("GENERAL")

    _row_label("Default output folder", "(where files are saved)")
    folder_var = ctk.StringVar(value=current.get("default_output_folder", ""))
    folder_frame = ctk.CTkFrame(scroll, fg_color="transparent")
    folder_frame.pack(fill="x", padx=4, pady=(0, 10))

    folder_entry = ctk.CTkEntry(
        folder_frame, textvariable=folder_var, width=330,
        fg_color=C_CARD, border_color=C_BORDER, text_color=C_TXT,
        placeholder_text="Click Browse to set...",
        placeholder_text_color=C_TXT3)
    folder_entry.pack(side="left", padx=(0, 8))

    def browse_folder():
        path = filedialog.askdirectory()
        if path:
            folder_var.set(path)

    ctk.CTkButton(folder_frame, text="Browse", command=browse_folder,
                  width=88, height=32,
                  font=ctk.CTkFont(family="Segoe UI", size=12),
                  **BTN_GHOST).pack(side="left")

    # ── Voice Defaults ────────────────────────────────────────────────────────
    _section("VOICE DEFAULTS")

    _row_label("Default voice", "(used when app starts)")
    default_voice_var = ctk.StringVar(value=current.get("default_voice", voices[0] if voices else ""))
    ctk.CTkOptionMenu(
        scroll, variable=default_voice_var, values=voices,
        width=340, dynamic_resizing=False,
        fg_color=C_CARD, button_color=C_ACCENT_D, button_hover_color=C_ACCENT,
        text_color=C_TXT, dropdown_fg_color=C_SURFACE,
        dropdown_hover_color=C_ELEVATED, dropdown_text_color=C_TXT
    ).pack(anchor="w", padx=4, pady=(0, 10))

    _row_label("Default speed")
    speed_frame = ctk.CTkFrame(scroll, fg_color="transparent")
    speed_frame.pack(fill="x", padx=4, pady=(0, 10))
    default_speed_var = ctk.DoubleVar(value=current.get("default_speed", 0.85))
    speed_val_lbl = ctk.CTkLabel(
        speed_frame, text=f"{default_speed_var.get():.2f}x", width=54,
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color=C_ACCENT)
    speed_val_lbl.pack(side="right")
    def on_speed(v):
        speed_val_lbl.configure(text=f"{float(v):.2f}x")
    ctk.CTkSlider(speed_frame, from_=0.5, to=2.0, number_of_steps=30,
                  variable=default_speed_var, command=on_speed,
                  width=310).pack(side="left")

    _row_label("Default profile", "(loaded on startup)")
    default_profile_var = ctk.StringVar(
        value=current.get("default_profile", profiles[0] if profiles else ""))
    ctk.CTkOptionMenu(
        scroll, variable=default_profile_var,
        values=profiles if profiles else ["None"],
        width=340, dynamic_resizing=False,
        fg_color=C_CARD, button_color=C_ACCENT_D, button_hover_color=C_ACCENT,
        text_color=C_TXT, dropdown_fg_color=C_SURFACE,
        dropdown_hover_color=C_ELEVATED, dropdown_text_color=C_TXT
    ).pack(anchor="w", padx=4, pady=(0, 10))

    # ── Text Cleaner ──────────────────────────────────────────────────────────
    _section("TEXT CLEANER")

    auto_clean_var = ctk.BooleanVar(value=current.get("auto_clean_text", False))
    ctk.CTkCheckBox(
        scroll,
        text="Auto-clean text when imported or pasted",
        variable=auto_clean_var,
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color=C_TXT
    ).pack(anchor="w", padx=4, pady=(0, 4))
    ctk.CTkLabel(
        scroll,
        text="Removes HTML, fixes spacing, expands abbreviations, and more.",
        font=ctk.CTkFont(family="Segoe UI", size=10),
        text_color=C_TXT3, anchor="w"
    ).pack(fill="x", padx=4, pady=(0, 10))

    # ── Notifications ─────────────────────────────────────────────────────────
    _section("NOTIFICATIONS")

    notify_var = ctk.BooleanVar(value=current.get("notify_on_completion", True))
    ctk.CTkCheckBox(
        scroll,
        text="Show notification when generation finishes",
        variable=notify_var,
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color=C_TXT
    ).pack(anchor="w", padx=4, pady=(0, 8))

    _row_label("Only notify if generation takes longer than:")
    thresh_frame = ctk.CTkFrame(scroll, fg_color="transparent")
    thresh_frame.pack(fill="x", padx=4, pady=(0, 10))
    thresh_var = ctk.IntVar(value=current.get("notify_threshold_seconds", 10))
    thresh_lbl = ctk.CTkLabel(
        thresh_frame, text=f"{thresh_var.get()}s", width=40,
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color=C_ACCENT)
    thresh_lbl.pack(side="right")
    def on_thresh(v):
        thresh_lbl.configure(text=f"{int(float(v))}s")
    ctk.CTkSlider(thresh_frame, from_=5, to=60, number_of_steps=55,
                  variable=thresh_var, command=on_thresh,
                  width=310).pack(side="left")

    # ── Footer buttons ────────────────────────────────────────────────────────
    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x")

    btn_frame = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=60)
    btn_frame.pack(fill="x")
    btn_frame.pack_propagate(False)

    btn_inner = ctk.CTkFrame(btn_frame, fg_color="transparent")
    btn_inner.pack(side="left", padx=16, pady=12)

    def on_save():
        new_settings = {
            "default_output_folder":     folder_var.get(),
            "theme":                     "dark",
            "default_voice":             default_voice_var.get(),
            "default_speed":             round(default_speed_var.get(), 2),
            "default_profile":           default_profile_var.get(),
            "auto_clean_text":           auto_clean_var.get(),
            "notify_on_completion":      notify_var.get(),
            "notify_threshold_seconds":  thresh_var.get(),
        }
        save_settings(new_settings)
        if on_save_callback:
            on_save_callback(new_settings)
        win.destroy()

    ctk.CTkButton(
        btn_inner, text="Save Settings", command=on_save,
        width=148, height=36,
        font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
    ).pack(side="left", padx=(0, 10))

    ctk.CTkButton(
        btn_inner, text="Cancel", command=win.destroy,
        width=96, height=36,
        font=ctk.CTkFont(family="Segoe UI", size=12),
        **BTN_GHOST
    ).pack(side="left")
