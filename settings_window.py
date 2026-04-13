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
    _center_window(win, 520, 680)
    win.resizable(False, False)
    win.grab_set()
    win.configure(fg_color=C_BG)

    current = load_settings()

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=90)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    ctk.CTkLabel(hdr, text="Settings",
                 font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
                 text_color=C_TXT).pack(pady=(18, 0))
    ctk.CTkLabel(hdr, text="Customize your TTS Studio experience",
                 font=ctk.CTkFont(family="Segoe UI", size=11),
                 text_color=C_TXT3).pack(pady=(2, 0))

    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x")

    # ── Scrollable body ───────────────────────────────────────────────────────
    scroll = ctk.CTkScrollableFrame(win, fg_color=C_BG,
                                    scrollbar_button_color=C_ELEVATED,
                                    scrollbar_button_hover_color=C_ACCENT_D)
    scroll.pack(fill="both", expand=True, padx=20, pady=(14, 0))

    def _card():
        """Container for a group of related settings."""
        c = ctk.CTkFrame(scroll, fg_color=C_CARD, corner_radius=10,
                         border_width=1, border_color=C_BORDER)
        c.pack(fill="x", pady=(0, 12))
        return c

    def _section_title(parent, text):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
                     text_color=C_ACCENT, anchor="w").pack(
            anchor="w", padx=16, pady=(12, 8))

    def _field_label(parent, text, hint=""):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=16, pady=(0, 4))
        ctk.CTkLabel(f, text=text, anchor="w",
                     font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                     text_color=C_TXT).pack(side="left")
        if hint:
            ctk.CTkLabel(f, text=hint, anchor="w",
                         font=ctk.CTkFont(family="Segoe UI", size=10),
                         text_color=C_TXT3).pack(side="left", padx=(8, 0))

    def _hint(parent, text):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont(family="Segoe UI", size=10),
                     text_color=C_TXT3, anchor="w", justify="left",
                     wraplength=440).pack(anchor="w", padx=16, pady=(2, 12))

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    output_card = _card()
    _section_title(output_card, "OUTPUT")

    _field_label(output_card, "Default save folder")
    folder_var = ctk.StringVar(value=current.get("default_output_folder", ""))
    folder_frame = ctk.CTkFrame(output_card, fg_color="transparent")
    folder_frame.pack(fill="x", padx=16, pady=(0, 14))

    folder_entry = ctk.CTkEntry(
        folder_frame, textvariable=folder_var,
        fg_color=C_ELEVATED, border_color=C_BORDER, text_color=C_TXT,
        placeholder_text="No folder set — you'll be prompted each time",
        placeholder_text_color=C_TXT3, height=32)
    folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

    def browse_folder():
        path = filedialog.askdirectory(parent=win)
        if path:
            folder_var.set(path)

    ctk.CTkButton(folder_frame, text="Browse", command=browse_folder,
                  width=80, height=32,
                  font=ctk.CTkFont(family="Segoe UI", size=12),
                  **BTN_GHOST).pack(side="left")

    # ── VOICE ─────────────────────────────────────────────────────────────────
    voice_card = _card()
    _section_title(voice_card, "VOICE DEFAULTS")

    _field_label(voice_card, "Voice", "loaded on startup")
    default_voice_var = ctk.StringVar(
        value=current.get("default_voice", voices[0] if voices else ""))
    ctk.CTkOptionMenu(
        voice_card, variable=default_voice_var, values=voices or [""],
        dynamic_resizing=False, height=32,
        fg_color=C_ELEVATED, button_color=C_ACCENT_D, button_hover_color=C_ACCENT,
        text_color=C_TXT, dropdown_fg_color=C_SURFACE,
        dropdown_hover_color=C_ELEVATED, dropdown_text_color=C_TXT
    ).pack(fill="x", padx=16, pady=(0, 12))

    _field_label(voice_card, "Speed")
    speed_row = ctk.CTkFrame(voice_card, fg_color="transparent")
    speed_row.pack(fill="x", padx=16, pady=(0, 12))
    default_speed_var = ctk.DoubleVar(value=current.get("default_speed", 0.85))
    speed_val_lbl = ctk.CTkLabel(
        speed_row, text=f"{default_speed_var.get():.2f}x", width=54,
        font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        text_color=C_ACCENT)
    speed_val_lbl.pack(side="right")
    def on_speed(v):
        speed_val_lbl.configure(text=f"{float(v):.2f}x")
    ctk.CTkSlider(speed_row, from_=0.5, to=2.0, number_of_steps=30,
                  variable=default_speed_var, command=on_speed,
                  progress_color=C_ACCENT, button_color=C_ACCENT,
                  button_hover_color=C_ACCENT_H).pack(
        side="left", fill="x", expand=True, padx=(0, 8))

    _field_label(voice_card, "FX profile", "loaded on startup")
    default_profile_var = ctk.StringVar(
        value=current.get("default_profile", profiles[0] if profiles else ""))
    ctk.CTkOptionMenu(
        voice_card, variable=default_profile_var,
        values=profiles if profiles else ["None"],
        dynamic_resizing=False, height=32,
        fg_color=C_ELEVATED, button_color=C_ACCENT_D, button_hover_color=C_ACCENT,
        text_color=C_TXT, dropdown_fg_color=C_SURFACE,
        dropdown_hover_color=C_ELEVATED, dropdown_text_color=C_TXT
    ).pack(fill="x", padx=16, pady=(0, 14))

    # ── TEXT ──────────────────────────────────────────────────────────────────
    text_card = _card()
    _section_title(text_card, "TEXT HANDLING")

    auto_clean_var = ctk.BooleanVar(value=current.get("auto_clean_text", False))
    ctk.CTkCheckBox(
        text_card,
        text="Auto-clean text on import or paste",
        variable=auto_clean_var,
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color=C_TXT,
        fg_color=C_ACCENT, hover_color=C_ACCENT_H,
        border_color=C_BORDER, checkmark_color=C_BG,
    ).pack(anchor="w", padx=16, pady=(0, 2))
    _hint(text_card,
          "Strips HTML, fixes spacing, expands common abbreviations, "
          "and normalizes quotes.")

    # ── NOTIFICATIONS ─────────────────────────────────────────────────────────
    notif_card = _card()
    _section_title(notif_card, "NOTIFICATIONS")

    notify_var = ctk.BooleanVar(value=current.get("notify_on_completion", True))
    ctk.CTkCheckBox(
        notif_card,
        text="Notify when generation finishes",
        variable=notify_var,
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color=C_TXT,
        fg_color=C_ACCENT, hover_color=C_ACCENT_H,
        border_color=C_BORDER, checkmark_color=C_BG,
    ).pack(anchor="w", padx=16, pady=(0, 10))

    _field_label(notif_card, "Only notify if generation takes longer than")
    thresh_row = ctk.CTkFrame(notif_card, fg_color="transparent")
    thresh_row.pack(fill="x", padx=16, pady=(0, 14))
    thresh_var = ctk.IntVar(value=current.get("notify_threshold_seconds", 10))
    thresh_lbl = ctk.CTkLabel(
        thresh_row, text=f"{thresh_var.get()}s", width=40,
        font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        text_color=C_ACCENT)
    thresh_lbl.pack(side="right")
    def on_thresh(v):
        thresh_lbl.configure(text=f"{int(float(v))}s")
    ctk.CTkSlider(thresh_row, from_=5, to=60, number_of_steps=55,
                  variable=thresh_var, command=on_thresh,
                  progress_color=C_ACCENT, button_color=C_ACCENT,
                  button_hover_color=C_ACCENT_H).pack(
        side="left", fill="x", expand=True, padx=(0, 8))

    # ── Footer ────────────────────────────────────────────────────────────────
    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x")

    foot = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=64)
    foot.pack(fill="x")
    foot.pack_propagate(False)

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

    # Buttons right-aligned, primary Save + ghost Cancel
    ctk.CTkButton(
        foot, text="Save", command=on_save,
        width=110, height=34,
        font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        fg_color=C_ACCENT, hover_color=C_ACCENT_H, text_color=C_BG,
        corner_radius=8,
    ).pack(side="right", padx=(0, 20), pady=15)

    ctk.CTkButton(
        foot, text="Cancel", command=win.destroy,
        width=90, height=34,
        font=ctk.CTkFont(family="Segoe UI", size=12),
        corner_radius=8, **BTN_GHOST
    ).pack(side="right", padx=(0, 8), pady=15)
