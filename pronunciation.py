"""
Pronunciation dictionary — word/phrase substitutions applied before TTS generation.
Entries are stored as a JSON list of {"from": str, "to": str, "case_sensitive": bool}.
"""
import ctypes
import json
import os
import re
import sys


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

PRONUNCIATION_FILE = os.path.join(
    os.environ.get("APPDATA", os.path.expanduser("~")),
    "TTS Studio", "pronunciation.json"
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
C_DANGER    = "#f87171"

BTN_GHOST = dict(fg_color="transparent", hover_color=C_ELEVATED,
                 border_width=1, border_color=C_BORDER, text_color=C_TXT2)

# ── Persistence ───────────────────────────────────────────────────────────────

def load_dictionary():
    """Return list of substitution entries."""
    if os.path.exists(PRONUNCIATION_FILE):
        try:
            with open(PRONUNCIATION_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[pronunciation] Failed to load dictionary: {e}", file=sys.stderr)
    return _default_entries()

def save_dictionary(entries):
    global _dict_cache
    _dict_cache = None  # invalidate cache so next generation re-reads the new entries
    try:
        os.makedirs(os.path.dirname(PRONUNCIATION_FILE), exist_ok=True)
        with open(PRONUNCIATION_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[pronunciation] Failed to save dictionary: {e}", file=sys.stderr)

def _default_entries():
    return [
        {"from": "AWS",   "to": "Amazon Web Services", "case_sensitive": True},
        {"from": "nginx", "to": "engine x",             "case_sensitive": False},
        {"from": "AI",    "to": "A.I.",                 "case_sensitive": True},
        {"from": "API",   "to": "A.P.I.",               "case_sensitive": True},
        {"from": "UI",    "to": "U.I.",                 "case_sensitive": True},
        {"from": "URL",   "to": "U.R.L.",               "case_sensitive": True},
        {"from": "SQL",   "to": "sequel",               "case_sensitive": True},
        {"from": "CLI",   "to": "C.L.I.",               "case_sensitive": True},
        {"from": "i.e.",  "to": "that is,",             "case_sensitive": False},
        {"from": "e.g.",  "to": "for example,",         "case_sensitive": False},
    ]

# ── Text processing ───────────────────────────────────────────────────────────

_dict_cache = None  # module-level sorted-entries cache; invalidated on save

def apply_pronunciation(text):
    """Apply all active substitutions to text, longest 'from' first."""
    global _dict_cache
    if _dict_cache is None:
        entries = load_dictionary()
        _dict_cache = sorted(entries, key=lambda e: len(e["from"]), reverse=True)
    for entry in _dict_cache:
        src = entry["from"]
        dst = entry["to"]
        try:
            pattern = r'\b' + re.escape(src) + r'\b'
            if entry.get("case_sensitive", False):
                text = re.sub(pattern, dst, text)
            else:
                text = re.sub(pattern, dst, text, flags=re.IGNORECASE)
        except re.error:
            pass  # skip malformed entries rather than crashing generation
    return text

# ── UI ────────────────────────────────────────────────────────────────────────

def open_pronunciation_window(parent):
    """Open the pronunciation dictionary editor."""
    import customtkinter as ctk
    from tkinter import messagebox

    win = ctk.CTkToplevel(parent)
    win.title("Pronunciation Dictionary")
    _center_window(win, 560, 580)
    win.resizable(False, False)
    win.grab_set()
    win.configure(fg_color=C_BG)

    entries = load_dictionary()

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=60)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    hdr_inner = ctk.CTkFrame(hdr, fg_color="transparent")
    hdr_inner.pack(side="left", padx=18, pady=10)
    ctk.CTkFrame(hdr_inner, fg_color=C_ACCENT, width=8, height=8,
                 corner_radius=4).pack(side="left", padx=(0, 10))
    ctk.CTkLabel(hdr_inner, text="Pronunciation Dictionary",
                 font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
                 text_color=C_TXT).pack(side="left")
    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x")

    ctk.CTkLabel(win,
                 text="Words and phrases typed here are replaced before speech is generated.",
                 font=ctk.CTkFont(family="Segoe UI", size=11),
                 text_color=C_TXT3).pack(pady=(8, 4))

    # ── Column headers ────────────────────────────────────────────────────────
    col_hdr = ctk.CTkFrame(win, fg_color="transparent")
    col_hdr.pack(fill="x", padx=16)
    ctk.CTkLabel(col_hdr, text="SAY THIS",
                 font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
                 text_color=C_TXT2, width=200, anchor="w").pack(side="left", padx=(0, 8))
    ctk.CTkLabel(col_hdr, text="INSTEAD OF",
                 font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
                 text_color=C_TXT2, width=200, anchor="w").pack(side="left", padx=(0, 8))
    ctk.CTkLabel(col_hdr, text="CASE",
                 font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
                 text_color=C_TXT2, width=50, anchor="w").pack(side="left")

    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x", padx=16, pady=(4, 0))

    # ── Scrollable rows ───────────────────────────────────────────────────────
    scroll = ctk.CTkScrollableFrame(win, fg_color=C_CARD,
                                    scrollbar_button_color=C_BORDER,
                                    scrollbar_button_hover_color=C_ACCENT_D)
    scroll.pack(fill="both", expand=True, padx=16, pady=(0, 6))

    row_widgets = []   # list of (from_var, to_var, cs_var, frame)

    def _add_row(src="", dst="", cs=False):
        row = ctk.CTkFrame(scroll, fg_color="transparent")
        row.pack(fill="x", pady=3)

        from_var = ctk.StringVar(value=src)
        to_var   = ctk.StringVar(value=dst)
        cs_var   = ctk.BooleanVar(value=cs)

        ctk.CTkEntry(row, textvariable=from_var, width=196,
                     fg_color=C_ELEVATED, border_color=C_BORDER,
                     text_color=C_TXT, placeholder_text="word / phrase",
                     placeholder_text_color=C_TXT3).pack(side="left", padx=(0, 8))
        ctk.CTkEntry(row, textvariable=to_var, width=196,
                     fg_color=C_ELEVATED, border_color=C_BORDER,
                     text_color=C_TXT, placeholder_text="replacement",
                     placeholder_text_color=C_TXT3).pack(side="left", padx=(0, 8))
        ctk.CTkCheckBox(row, text="", variable=cs_var, width=24,
                        checkbox_width=16, checkbox_height=16).pack(side="left", padx=(0, 8))

        def _delete(r=row, w=(from_var, to_var, cs_var, row)):
            r.destroy()
            if w in row_widgets:
                row_widgets.remove(w)

        ctk.CTkButton(row, text="✕", width=28, height=28,
                      fg_color="#2a0f0f", hover_color="#3d1515",
                      text_color=C_DANGER, border_width=0,
                      font=ctk.CTkFont(size=11),
                      command=_delete).pack(side="left")

        row_widgets.append((from_var, to_var, cs_var, row))

    # Populate existing entries
    for e in entries:
        _add_row(e["from"], e["to"], e.get("case_sensitive", False))

    # ── Add row / reset ───────────────────────────────────────────────────────
    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x", padx=16)

    mid_row = ctk.CTkFrame(win, fg_color="transparent")
    mid_row.pack(fill="x", padx=16, pady=6)

    ctk.CTkButton(mid_row, text="+ Add Entry", command=lambda: _add_row(),
                  width=110, height=30,
                  font=ctk.CTkFont(family="Segoe UI", size=12),
                  **BTN_GHOST).pack(side="left", padx=(0, 8))

    def _reset_defaults():
        if messagebox.askyesno("Reset", "Replace all entries with defaults?",
                               parent=win):
            for _, _, _, r in row_widgets[:]:
                r.destroy()
            row_widgets.clear()
            for e in _default_entries():
                _add_row(e["from"], e["to"], e.get("case_sensitive", False))

    ctk.CTkButton(mid_row, text="Reset defaults", command=_reset_defaults,
                  width=120, height=30,
                  font=ctk.CTkFont(family="Segoe UI", size=12),
                  **BTN_GHOST).pack(side="left")

    # ── Footer ────────────────────────────────────────────────────────────────
    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x")

    foot = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=56)
    foot.pack(fill="x")
    foot.pack_propagate(False)
    foot_inner = ctk.CTkFrame(foot, fg_color="transparent")
    foot_inner.pack(side="left", padx=16, pady=10)

    def _save():
        new_entries = []
        for from_var, to_var, cs_var, _ in row_widgets:
            src = from_var.get().strip()
            dst = to_var.get().strip()
            if src and dst:
                new_entries.append({
                    "from": src,
                    "to":   dst,
                    "case_sensitive": cs_var.get()
                })
        save_dictionary(new_entries)
        win.destroy()

    ctk.CTkButton(foot_inner, text="Save Dictionary", command=_save,
                  width=148, height=34,
                  font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
                  ).pack(side="left", padx=(0, 10))
    ctk.CTkButton(foot_inner, text="Cancel", command=win.destroy,
                  width=88, height=34,
                  font=ctk.CTkFont(family="Segoe UI", size=12),
                  **BTN_GHOST).pack(side="left")
