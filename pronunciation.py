"""
Pronunciation dictionary — word/phrase substitutions applied before TTS generation.
Entries are stored as a JSON list of {"from": str, "to": str, "case_sensitive": bool}.
"""
import ctypes
import json
import os
import re
import sys

import window_utils


def _center_window(win, w: int, h: int, parent=None) -> None:
    """Center win over its parent at w×h (see window_utils.center_window)."""
    window_utils.center_window(win, w, h, parent=parent)

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
C_ACCENT    = "#00d98b"
C_ACCENT_H  = "#2ee5a0"
C_ACCENT_D  = "#0a3d28"
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
    """Write the dictionary. Returns (ok, error_message).

    It used to swallow the failure into a stderr print — invisible in a windowed
    app — and the editor closed either way, so a save that never happened looked
    exactly like one that did. The caller needs to know.
    """
    global _dict_cache
    _dict_cache = None  # invalidate cache so next generation re-reads the new entries
    try:
        os.makedirs(os.path.dirname(PRONUNCIATION_FILE), exist_ok=True)
        with open(PRONUNCIATION_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        return True, ""
    except OSError as e:
        print(f"[pronunciation] Failed to save dictionary: {e}", file=sys.stderr)
        return False, str(e)

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
        if not src:
            continue   # an empty pattern matches between every character and
                       # would insert the replacement across the whole text. The
                       # editor won't save one, but a hand-edited JSON could.
        try:
            # Only demand a word boundary on a side that actually ENDS in a word
            # character. '\b' after a '.' can never match before a space, so any
            # entry ending in punctuation silently did nothing — including the
            # shipped "i.e." and "e.g." defaults, and anything a user added like
            # "Dr." or "C++".
            left  = r'\b' if (src[:1].isalnum() or src[:1] == '_') else ''
            right = r'\b' if (src[-1:].isalnum() or src[-1:] == '_') else ''
            pattern = left + re.escape(src) + right
            if entry.get("case_sensitive", False):
                text = re.sub(pattern, dst, text)
            else:
                text = re.sub(pattern, dst, text, flags=re.IGNORECASE)
        except re.error:
            pass  # skip malformed entries rather than crashing generation
    return text

# ── UI ────────────────────────────────────────────────────────────────────────

def open_pronunciation_window(parent, info_btn=None):
    """Open the pronunciation dictionary editor.

    info_btn: optional factory from app.py that returns a hoverable "i" icon —
    passed in rather than duplicated here because app.py already owns the tooltip
    class, and importing it back would be circular (app imports this module).
    """
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
                 text_color=C_TXT3).pack(pady=(8, 1))
    # The window was a bare list with no guidance, so the trick that makes it
    # useful — respell phonetically rather than correctly — was never obvious.
    # Keep the example generic; never ship a real user's word here.
    ctk.CTkLabel(win,
                 text="Tip: spell it the way it sounds, not the way it's written — "
                      "type \"GIF\", say \"jif\".",
                 font=ctk.CTkFont(family="Segoe UI", size=11),
                 text_color=C_ACCENT).pack(pady=(0, 5))

    # ── Column headers ────────────────────────────────────────────────────────
    col_hdr = ctk.CTkFrame(win, fg_color="transparent")
    col_hdr.pack(fill="x", padx=16)
    # These read "SAY THIS" / "INSTEAD OF", which described the columns backwards:
    # the first is what you TYPE and the second is what gets SPOKEN, so the built-in
    # AWS -> Amazon Web Services read as "say AWS instead of Amazon Web Services".
    # Anyone filling the window in by reading the headers entered every pair inverted.
    ctk.CTkLabel(col_hdr, text="WHEN YOU TYPE",
                 font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
                 text_color=C_TXT2, width=200, anchor="w").pack(side="left", padx=(0, 8))
    ctk.CTkLabel(col_hdr, text="SAY THIS INSTEAD",
                 font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
                 text_color=C_TXT2, width=200, anchor="w").pack(side="left", padx=(0, 8))
    # "CASE" alone told the user nothing — it was the least obvious control in
    # the window. Spell it out and hang an explanation off it.
    case_hdr = ctk.CTkFrame(col_hdr, fg_color="transparent")
    case_hdr.pack(side="left")
    ctk.CTkLabel(case_hdr, text="EXACT CASE",
                 font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
                 text_color=C_TXT2, anchor="w").pack(side="left")
    if info_btn:
        info_btn(case_hdr,
                 "Tick this to match ONLY the exact capitalisation you typed.\n\n"
                 "Leave it off and “GIF”, “Gif” and “gif” "
                 "all match — usually what you want.\n\n"
                 "Tick it on when your entry is also an everyday word. An entry "
                 "for “US” left off would turn “Give us the figures” "
                 "into “Give U.S. the figures”."
                 ).pack(side="left", padx=(4, 0))

    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x", padx=16, pady=(4, 0))

    # ── Scrollable rows ───────────────────────────────────────────────────────
    scroll = ctk.CTkScrollableFrame(win, fg_color=C_CARD,
                                    scrollbar_button_color=C_BORDER,
                                    scrollbar_button_hover_color=C_ACCENT_D)
    scroll.pack(fill="both", expand=True, padx=16, pady=(0, 6))

    row_widgets = []   # list of (from_var, to_var, cs_var, frame)

    def _add_row(src="", dst="", cs=False, reveal=False):
        row = ctk.CTkFrame(scroll, fg_color="transparent")
        row.pack(fill="x", pady=3)

        from_var = ctk.StringVar(value=src)
        to_var   = ctk.StringVar(value=dst)
        cs_var   = ctk.BooleanVar(value=cs)

        first_entry = ctk.CTkEntry(row, textvariable=from_var, width=196,
                     fg_color=C_ELEVATED, border_color=C_BORDER,
                     text_color=C_TXT, placeholder_text="word you type",
                     placeholder_text_color=C_TXT3)
        first_entry.pack(side="left", padx=(0, 8))
        ctk.CTkEntry(row, textvariable=to_var, width=196,
                     fg_color=C_ELEVATED, border_color=C_BORDER,
                     text_color=C_TXT, placeholder_text="how to say it",
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

        if reveal:
            # A new row lands at the BOTTOM of ten built-in entries, below the
            # fold — clicking "+ Add Entry" looked like it did nothing at all.
            def _show():
                try:
                    scroll.update_idletasks()
                    scroll._parent_canvas.yview_moveto(1.0)
                    first_entry.focus_set()
                except Exception:
                    pass
            win.after(30, _show)

    # Populate existing entries
    for e in entries:
        _add_row(e["from"], e["to"], e.get("case_sensitive", False))

    # ── Add row / reset ───────────────────────────────────────────────────────
    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x", padx=16)

    mid_row = ctk.CTkFrame(win, fg_color="transparent")
    mid_row.pack(fill="x", padx=16, pady=6)

    ctk.CTkButton(mid_row, text="+ Add Entry", command=lambda: _add_row(reveal=True),
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

    def _current_entries():
        """Rows as they stand right now, skipping half-filled ones."""
        out = []
        for from_var, to_var, cs_var, _ in row_widgets:
            src = from_var.get().strip()
            dst = to_var.get().strip()
            if src and dst:
                out.append({"from": src, "to": dst, "case_sensitive": cs_var.get()})
        return out

    _opened_with = _current_entries()   # to detect unsaved edits on close

    def _save():
        ok, err = save_dictionary(_current_entries())
        if not ok:
            # Keep the window open so their typing isn't thrown away.
            messagebox.showerror(
                "Could not save",
                "VoxWild couldn't write the pronunciation file:\n\n"
                f"{PRONUNCIATION_FILE}\n\n{err}\n\n"
                "Your entries are still here — try again.",
                parent=win)
            return
        win.destroy()

    def _close():
        """Nothing here saves as you type. Closing with edits pending used to
        discard them silently, which read as 'my entries disappeared'."""
        if _current_entries() != _opened_with:
            if not messagebox.askyesno(
                    "Discard changes?",
                    "You have unsaved changes to the dictionary.\n\n"
                    "Close without saving them?",
                    parent=win):
                return
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", _close)

    ctk.CTkButton(foot_inner, text="Save Dictionary", command=_save,
                  width=148, height=34,
                  font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
                  ).pack(side="left", padx=(0, 10))
    ctk.CTkButton(foot_inner, text="Cancel", command=_close,
                  width=88, height=34,
                  font=ctk.CTkFont(family="Segoe UI", size=12),
                  **BTN_GHOST).pack(side="left")
