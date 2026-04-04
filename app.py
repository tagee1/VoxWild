import customtkinter as ctk
from kokoro_onnx import Kokoro
import sounddevice as sd
import soundfile as sf
import threading
import numpy as np
import os
import json
import re
import time
import subprocess
import tempfile
from tkinter import filedialog, messagebox
from scipy.signal import butter, sosfilt
import tkinter as tk
from datetime import datetime
from text_cleaner import clean_text, preview_clean
from settings_window import open_settings_window, load_settings, save_settings, DEFAULT_SETTINGS
from pronunciation import open_pronunciation_window, apply_pronunciation
from tts_utils import (
    format_time, chunk_text, parse_dialogue,
    _srt_time, _wrap_for_subtitle, build_srt,
    fmt_err, estimate_audio_duration, GenerationCancelled,
)
from clone_library import (
    load_clone_library  as _lib_load,
    save_clone_library  as _lib_save,
    add_clone_to_library as _lib_add,
)
from audio_utils import trim_silence
import license as _lic

# ── In-memory caches (eliminates repeated disk reads in hot paths) ────────────
_settings_cache     = None
_calibration_cache  = None
_clone_cache        = None

def _get_settings():
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = load_settings()
    return _settings_cache

def _save_settings(s):
    global _settings_cache
    _settings_cache = s
    save_settings(s)

def _get_calibration():
    global _calibration_cache
    if _calibration_cache is None:
        _calibration_cache = load_calibration()
    return _calibration_cache

def _invalidate_calibration():
    global _calibration_cache
    _calibration_cache = None

def _get_clone_library():
    global _clone_cache
    if _clone_cache is None:
        _clone_cache = _lib_load(CLONE_DIR, CLONE_INDEX)
    return _clone_cache

def _invalidate_clone_cache():
    global _clone_cache
    _clone_cache = None

# ── Constants ─────────────────────────────────────────────────────────────────
VERSION          = "1.0.0"
MAX_HISTORY      = 10

# ── User data directory (%APPDATA%\TTS Studio) ────────────────────────────────
_USER_DIR        = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "TTS Studio")
os.makedirs(_USER_DIR, exist_ok=True)

PROFILES_FILE    = os.path.join(_USER_DIR, "tts_profiles.json")
CALIBRATION_FILE = os.path.join(_USER_DIR, "calibration.json")
SETTINGS_FILE    = os.path.join(_USER_DIR, "settings.json")
CLONE_DIR        = os.path.join(_USER_DIR, "voice_clones")
CLONE_INDEX      = os.path.join(CLONE_DIR, "library.json")

# ── One-time migration: copy existing user files from app dir to %APPDATA% ────
def _migrate_user_data():
    """Copy legacy files from the app directory to %APPDATA% on first launch."""
    _app_dir = os.path.dirname(os.path.abspath(__file__))
    _migrations = [
        ("tts_profiles.json",       PROFILES_FILE),
        ("calibration.json",        CALIBRATION_FILE),
        ("settings.json",           SETTINGS_FILE),
    ]
    import shutil as _shutil
    for _src_name, _dst in _migrations:
        _src = os.path.join(_app_dir, _src_name)
        if os.path.exists(_src) and not os.path.exists(_dst):
            try:
                _shutil.copy2(_src, _dst)
            except OSError:
                pass
    # Migrate voice_clones directory
    _src_clones = os.path.join(_app_dir, "voice_clones")
    if os.path.isdir(_src_clones) and not os.path.exists(CLONE_INDEX):
        try:
            _shutil.copytree(_src_clones, CLONE_DIR, dirs_exist_ok=True)
        except OSError:
            pass

_migrate_user_data()

# ── Crash logger ──────────────────────────────────────────────────────────────
import traceback as _traceback
from datetime import datetime as _dt

_CRASH_LOG = os.path.join(_USER_DIR, "crashes.log")
_MAX_CRASH_LOG_BYTES = 512 * 1024  # 512 KB — rotate when exceeded

def _log_crash(e, tb_str=None):
    """Append a crash entry to crashes.log. Never raises."""
    try:
        # Rotate log if it's grown too large
        if os.path.exists(_CRASH_LOG) and os.path.getsize(_CRASH_LOG) > _MAX_CRASH_LOG_BYTES:
            _rotated = _CRASH_LOG + ".old"
            try:
                os.replace(_CRASH_LOG, _rotated)
            except OSError:
                pass

        if tb_str is None:
            tb_str = _traceback.format_exc()

        code = "E099"
        from tts_utils import fmt_err as _fe
        msg = _fe(e)
        import re as _re
        m = _re.search(r'\[(E\d{3})\]', msg)
        if m:
            code = m.group(1)

        timestamp = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"[{timestamp}] {code} — {type(e).__name__}: {str(e)[:200]}\n"
            f"{tb_str.strip()}\n"
            f"{'-' * 60}\n"
        )
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass  # logging must never crash the app

# ── Theme palette (Studio Gold) ───────────────────────────────────────────────
C_BG        = "#0d0d0d"   # carbon black
C_SURFACE   = "#171717"   # header/footer bars
C_CARD      = "#1f1f1f"   # panel / card bg
C_ELEVATED  = "#2a2a2a"   # hover / elevated
C_BORDER    = "#383838"   # subtle neutral border
C_ACCENT    = "#e8940a"   # amber gold
C_ACCENT_H  = "#f5aa2a"   # gold hover
C_ACCENT_D  = "#3d2200"   # dark amber / progress track
C_TXT       = "#f0ece4"   # warm near-white
C_TXT2      = "#9a9290"   # warm medium gray
C_TXT3      = "#4e4a48"   # dark warm gray
C_SUCCESS   = "#22c55e"   # green
C_WARN      = "#f97316"   # orange
C_DANGER    = "#f87171"   # red
C_REC       = "#dc3030"   # record button red

# Button presets
BTN_GHOST   = dict(fg_color="transparent", hover_color=C_ELEVATED,
                   border_width=1, border_color=C_BORDER, text_color=C_TXT2)
BTN_DARK    = dict(fg_color=C_CARD, hover_color=C_ELEVATED, text_color=C_TXT2)
BTN_DANGER  = dict(fg_color="#2a0f0f", hover_color="#3d1515", text_color=C_DANGER,
                   border_width=1, border_color="#3d1515")

# ── App Window ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme(os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme.json"))

app = ctk.CTk()
app.title(f"AI Text to Speech Studio  v{VERSION}")
app.geometry("1380x860")
app.minsize(1100, 720)
app.configure(fg_color=C_BG)
app.withdraw()   # hidden until splash finishes

# Set app icon
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    app.iconbitmap(os.path.join(_APP_DIR, "icon.ico"))
except:
    pass

# ── Logo image (shared across UI) ────────────────────────────────────────────
from PIL import Image as _PILImage
_LOGO_PATH = os.path.join(_APP_DIR, "logo.png")
try:
    _logo_pil = _PILImage.open(_LOGO_PATH).convert("RGBA")
    LOGO_IMG_LG = ctk.CTkImage(_logo_pil, size=(120, 120))  # splash / about
    LOGO_IMG_SM = ctk.CTkImage(_logo_pil, size=(32, 32))    # header
except Exception:
    LOGO_IMG_LG = None
    LOGO_IMG_SM = None

# ── Splash Screen ─────────────────────────────────────────────────────────────
def _run_splash(on_done):
    """Show a splash window, animate a loading bar, then call on_done()."""
    import tkinter as _tk

    SW, SH = 520, 360

    # Use plain tk.Toplevel — ctk.CTkToplevel has deferred-init callbacks that
    # keep resetting geometry, making centering unreliable on Windows.
    splash = _tk.Toplevel(app)
    splash.overrideredirect(True)
    splash.configure(bg=C_BG)
    splash.attributes("-topmost", True)

    # Center on screen using logical-pixel screen dimensions.
    # winfo_screenwidth/height can return physical pixels on HiDPI Windows,
    # but window coordinates are always in logical pixels.  ctypes
    # GetSystemMetrics returns logical-pixel values that match window coords.
    try:
        import ctypes as _ctypes
        _u32 = _ctypes.windll.user32
        sw_screen = _u32.GetSystemMetrics(0)   # SM_CXSCREEN (logical px)
        sh_screen = _u32.GetSystemMetrics(1)   # SM_CYSCREEN (logical px)
    except Exception:
        sw_screen = app.winfo_screenwidth()
        sh_screen = app.winfo_screenheight()
    x = (sw_screen - SW) // 2
    y = (sh_screen - SH) // 2
    splash.geometry(f"{SW}x{SH}+{x}+{y}")
    splash.update_idletasks()

    # Amber border frame
    border = ctk.CTkFrame(splash, fg_color=C_ACCENT, corner_radius=16)
    border.place(relx=0, rely=0, relwidth=1, relheight=1)
    inner = ctk.CTkFrame(border, fg_color=C_BG, corner_radius=14)
    inner.place(relx=0, rely=0, relwidth=1, relheight=1, bordermode="outside")

    # Logo
    if LOGO_IMG_LG:
        ctk.CTkLabel(inner, image=LOGO_IMG_LG, text="").place(relx=0.5, rely=0.28, anchor="center")

    # App name
    ctk.CTkLabel(inner, text="AI Text to Speech Studio",
                 font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
                 text_color=C_TXT).place(relx=0.5, rely=0.60, anchor="center")
    ctk.CTkLabel(inner, text=f"v{VERSION}  ·  Kokoro  ·  Chatterbox",
                 font=ctk.CTkFont(family="Segoe UI", size=12),
                 text_color=C_ACCENT).place(relx=0.5, rely=0.70, anchor="center")

    # Progress bar
    bar_frame = ctk.CTkFrame(inner, fg_color="transparent", width=360)
    bar_frame.place(relx=0.5, rely=0.83, anchor="center")
    splash_bar = ctk.CTkProgressBar(bar_frame, height=6, corner_radius=3,
                                    progress_color=C_ACCENT, fg_color=C_ACCENT_D)
    splash_bar.set(0)
    splash_bar.pack(fill="x")
    splash_status = ctk.CTkLabel(inner, text="Loading...",
                                 font=ctk.CTkFont(family="Segoe UI", size=10),
                                 text_color=C_TXT3)
    splash_status.place(relx=0.5, rely=0.91, anchor="center")

    # Animate bar to ~0.4 quickly, then crawl while Kokoro loads
    _progress = [0.0]

    def _set_bar(val, msg=""):
        _progress[0] = val
        splash_bar.set(val)
        if msg:
            splash_status.configure(text=msg)
        splash.update_idletasks()

    def _animate_to(target, steps=18, delay=30):
        start = _progress[0]
        step_size = (target - start) / steps
        def _step(i=0):
            if i < steps:
                splash_bar.set(start + step_size * (i + 1))
                splash.after(delay, lambda: _step(i + 1))
        _step()

    _animate_to(0.15, steps=12, delay=40)
    splash.after(200, lambda: _set_bar(0.15, "Initializing..."))
    splash.after(400, lambda: _animate_to(0.35, steps=10, delay=50))
    splash.after(800, lambda: _set_bar(0.35, "Loading Kokoro TTS engine..."))

    def _finish_splash():
        _set_bar(1.0, "Ready!")
        splash.after(350, lambda: (_close_splash()))

    def _close_splash():
        splash.destroy()
        on_done()

    # Expose finish hook so main code can call it after Kokoro loads
    splash._finish = _finish_splash
    return splash

# ── Load Kokoro ───────────────────────────────────────────────────────────────
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

_fmt_err = fmt_err  # local alias kept so existing call sites are unchanged

# ── Chatterbox Engine (persistent subprocess) ─────────────────────────────────
class ChatterboxEngine:
    """Manages a persistent chatterbox_worker.py subprocess."""

    WORKER  = os.path.join(os.path.dirname(__file__), "chatterbox_worker.py")
    PYTHON  = os.path.join(os.path.dirname(__file__), "chatterbox_env", "Scripts", "python.exe")

    def __init__(self):
        self._proc   = None
        self._sr     = 24000   # default; updated on ready
        self._lock   = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def start(self, status_cb=None):
        """Start worker and block until model is ready. Raises on failure."""
        with self._lock:
            if self.is_ready:
                return
            if not os.path.exists(self.PYTHON):
                raise FileNotFoundError(
                    "chatterbox_env not found.\n"
                    f"Expected: {self.PYTHON}"
                )
            proc = subprocess.Popen(
                [self.PYTHON, self.WORKER],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for raw in proc.stdout:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue  # skip any non-JSON noise
                if msg["type"] == "status":
                    if status_cb:
                        status_cb(f"🎙️ {msg['msg']}")
                elif msg["type"] == "ready":
                    self._sr   = msg.get("sr", 24000)
                    self._proc = proc
                    return
                elif msg["type"] == "error":
                    proc.kill()
                    raise RuntimeError(msg["msg"])
            proc.kill()
            raise RuntimeError("Chatterbox worker exited during startup.")

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None

    @property
    def is_ready(self):
        return self._proc is not None and self._proc.poll() is None

    @property
    def sr(self):
        return self._sr

    # ── generation ─────────────────────────────────────────────────────────────
    def generate_chunk(self, text, audio_prompt_path=None,
                       exaggeration=0.5, cfg_weight=0.5, temperature=0.8,
                       status_cb=None):
        """Generate one chunk; returns (numpy_samples, sample_rate)."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        req = {
            "cmd": "generate",
            "text": text,
            "output_path": tmp.name,
            "audio_prompt_path": audio_prompt_path,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
        }
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        for raw in self._proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue  # skip non-JSON noise
            if msg["type"] == "status":
                if status_cb:
                    status_cb(f"🎙️ {msg['msg']}")
            elif msg["type"] == "done":
                samples, sr = sf.read(tmp.name)
                os.unlink(tmp.name)
                return samples, sr
            elif msg["type"] == "error":
                os.unlink(tmp.name)
                self.stop()  # mark engine as needing restart so next attempt recovers
                raise RuntimeError(msg["msg"])
        self.stop()
        raise RuntimeError("Chatterbox worker closed unexpectedly.")

chatterbox_engine = ChatterboxEngine()

# ── Voices ────────────────────────────────────────────────────────────────────
VOICES = {
    "🇺🇸 Female - Heart (Best)": "af_heart",
    "🇺🇸 Female - Bella":        "af_bella",
    "🇺🇸 Female - Sarah":        "af_sarah",
    "🇺🇸 Female - Nova":         "af_nova",
    "🇺🇸 Female - Sky":          "af_sky",
    "🇺🇸 Female - Nicole":       "af_nicole",
    "🇺🇸 Female - Jessica":      "af_jessica",
    "🇺🇸 Male - Adam":           "am_adam",
    "🇺🇸 Male - Michael":        "am_michael",
    "🇬🇧 Female - Emma":         "bf_emma",
    "🇬🇧 Female - Isabella":     "bf_isabella",
    "🇬🇧 Male - George (Best)":  "bm_george",
    "🇬🇧 Male - Lewis":          "bm_lewis",
}

# ── Settings helpers (load_settings / save_settings imported from settings_window) ─
def get_default_folder():
    return _get_settings().get("default_output_folder", "")

def set_default_folder(path):
    s = _get_settings()
    s["default_output_folder"] = path
    _save_settings(s)

# ── Calibration ───────────────────────────────────────────────────────────────
def load_calibration():
    data = {"words_per_second": None, "samples": [],
            "cb_words_per_second": None, "cb_samples": []}
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    # One-time migration: remove outlier samples captured during model-load.
    # Values < 0.1 wps are physically impossible during normal generation.
    _changed = False
    for s_key, w_key in [("samples", "words_per_second"),
                          ("cb_samples", "cb_words_per_second")]:
        clean = [s for s in data.get(s_key, []) if s >= 0.1]
        if len(clean) != len(data.get(s_key, [])):
            data[s_key] = clean
            data[w_key] = round(sum(clean) / len(clean), 3) if clean else None
            _changed = True
    if _changed:
        save_calibration(data)

    return data

def save_calibration(data):
    _invalidate_calibration()
    try:
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass  # calibration is non-critical; silently skip if disk write fails

def record_calibration(word_count, elapsed_seconds, use_cb=None):
    data = _get_calibration()
    if use_cb is None:
        use_cb = engine_var.get() == "🎙️ Natural"
    wps = word_count / elapsed_seconds if elapsed_seconds > 0 else (0.5 if use_cb else 55)
    _EMA_ALPHA = 0.4
    if use_cb:
        data.setdefault("cb_samples", [])
        data["cb_samples"].append(round(wps, 3))
        data["cb_samples"] = data["cb_samples"][-5:]
        prior = data.get("cb_words_per_second") or 0.5
        data["cb_words_per_second"] = round(_EMA_ALPHA * wps + (1 - _EMA_ALPHA) * prior, 3)
    else:
        data["samples"].append(round(wps, 3))
        data["samples"] = data["samples"][-5:]
        prior = data.get("words_per_second") or 55
        data["words_per_second"] = round(_EMA_ALPHA * wps + (1 - _EMA_ALPHA) * prior, 3)
    save_calibration(data)

def get_words_per_second():
    data = _get_calibration()
    if engine_var.get() == "🎙️ Natural":
        return data.get("cb_words_per_second") or 0.5
    return data.get("words_per_second") or 55

# ── Audio History ─────────────────────────────────────────────────────────────
audio_history = []   # list of dicts: {samples, sample_rate, text, duration, timestamp, voice}
_active_play_btn = [None]   # currently playing card's Play button, or None

def add_to_history(samples, sample_rate, text, voice_name, segments=None):
    duration = len(samples) / sample_rate
    entry = {
        "samples":     samples,
        "sample_rate": sample_rate,
        "text":        text,
        "duration":    duration,
        "timestamp":   datetime.now().strftime("%H:%M:%S"),
        "voice":       voice_name,
        "segments":    segments,  # list of (start_sec, end_sec, text) for SRT export
    }
    audio_history.insert(0, entry)
    if len(audio_history) > MAX_HISTORY:
        audio_history.pop()
    _prepend_history_card(entry)

def _prepend_history_card(entry):
    """Add one card at the top of the history panel — O(1) widget work."""
    children = history_inner.winfo_children()
    # Remove the "No audio yet." placeholder if present
    if children and isinstance(children[0], ctk.CTkLabel):
        children[0].destroy()
        children = []
    # Drop the oldest card widget if we're at the limit
    if len(children) >= MAX_HISTORY:
        children[-1].destroy()
    # Build the new card and pack it above everything else
    card = _make_history_card(history_inner, 0, entry)
    card.pack(fill="x", padx=6, pady=(0, 3))
    card.lift()  # move to top of stacking/pack order

def refresh_history_panel():
    """Full rebuild — used after deletions."""
    for w in history_inner.winfo_children():
        w.destroy()
    if not audio_history:
        ctk.CTkLabel(history_inner, text="No audio yet.",
                     text_color=C_TXT3, font=ctk.CTkFont(family="Segoe UI", size=12)).pack(pady=30)
        return
    for i, entry in enumerate(audio_history):
        _make_history_card(history_inner, i, entry)

def _make_history_card(parent, idx, entry):
    def _delete(e=entry):
        if e in audio_history:
            audio_history.remove(e)
        refresh_history_panel()

    outer = ctk.CTkFrame(parent, fg_color=C_CARD, corner_radius=6)
    outer.pack(fill="x", padx=6, pady=(0, 3))
    # NOTE: caller (_prepend_history_card) may call outer.lift() after this returns.

    # place() ties relheight to outer's actual rendered size, not the pack allocation
    # (fill="y" inside a CTkScrollableFrame expands to the full viewport height)
    ctk.CTkFrame(outer, fg_color=C_ACCENT, width=3, corner_radius=0).place(
        x=0, y=0, relheight=1)

    content = ctk.CTkFrame(outer, fg_color="transparent")
    content.pack(fill="both", expand=True, padx=(11, 8), pady=6)

    # ── Row 1: timestamp · voice  |  text preview ────────────────────────────
    row1 = ctk.CTkFrame(content, fg_color="transparent")
    row1.pack(fill="x")

    ctk.CTkLabel(row1, text=entry["timestamp"],
                 font=ctk.CTkFont(family="Segoe UI", size=10),
                 text_color=C_TXT3).pack(side="left")

    voice_short = entry["voice"].split(" - ")[-1].replace(" (Best)", "") if " - " in entry["voice"] else entry["voice"]
    ctk.CTkLabel(row1, text=f"  {voice_short}",
                 font=ctk.CTkFont(family="Segoe UI", size=10),
                 text_color=C_ACCENT).pack(side="left")

    preview = entry["text"][:48].replace("\n", " ") + ("…" if len(entry["text"]) > 48 else "")
    ctk.CTkLabel(row1, text=preview,
                 font=ctk.CTkFont(family="Segoe UI", size=11),
                 text_color=C_TXT, anchor="w").pack(side="right", fill="x", expand=True)

    # ── Row 2: Play · Save · SRT ─────────────────────────────────────────────
    row2 = ctk.CTkFrame(content, fg_color="transparent")
    row2.pack(fill="x", pady=(4, 0))

    play_btn = ctk.CTkButton(
        row2, text="▶  Play", width=70, height=24,
        font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
        fg_color=C_ACCENT_D, hover_color=C_ACCENT, text_color=C_TXT, corner_radius=5,
    )
    play_btn.configure(command=lambda e=entry, b=play_btn: _toggle_history_playback(e, b))
    play_btn.pack(side="left", padx=(0, 4))

    ctk.CTkButton(
        row2, text="Save", width=46, height=24,
        font=ctk.CTkFont(family="Segoe UI", size=10),
        **BTN_GHOST, corner_radius=5,
        command=lambda e=entry: download_history_entry(e)
    ).pack(side="left", padx=(0, 4))

    if entry.get("segments"):
        ctk.CTkButton(
            row2, text="SRT", width=38, height=24,
            font=ctk.CTkFont(family="Segoe UI", size=10),
            **BTN_GHOST, corner_radius=5,
            command=lambda e=entry: export_srt_from_entry(e)
        ).pack(side="left", padx=(0, 4))

    # ── Row 3: Delete · duration ──────────────────────────────────────────────
    row3 = ctk.CTkFrame(content, fg_color="transparent")
    row3.pack(fill="x", pady=(2, 0))

    ctk.CTkButton(
        row3, text="Delete", width=52, height=22,
        font=ctk.CTkFont(family="Segoe UI", size=10),
        fg_color="transparent", hover_color="#3d1515",
        text_color=C_DANGER, border_width=1, border_color="#3d1515", corner_radius=5,
        command=_delete
    ).pack(side="left")

    ctk.CTkLabel(row3, text=format_time(int(entry["duration"])),
                 font=ctk.CTkFont(family="Segoe UI", size=10),
                 text_color=C_TXT3).pack(side="right", padx=(0, 2))

    return outer

def _reset_play_btn():
    """Reset the active play button back to ▶ Play (call from any thread)."""
    btn = _active_play_btn[0]
    if btn:
        try:
            btn.configure(text="▶  Play", fg_color=C_ACCENT_D, hover_color=C_ACCENT)
        except Exception:
            pass
        _active_play_btn[0] = None

def _toggle_history_playback(entry, btn):
    # If this button is already the active one, treat as Stop
    if _active_play_btn[0] is btn:
        sd.stop()
        _reset_play_btn()
        status_label.configure(text="⏹ Stopped.")
        return

    # Stop whatever is playing and reset its button
    sd.stop()
    _reset_play_btn()

    # Mark this button as active and switch it to Stop
    _active_play_btn[0] = btn
    btn.configure(text="■  Stop", fg_color="#3d1515", hover_color="#5a1f1f")

    def run():
        try:
            status_label.configure(text=f"🔊 Playing: {entry['text'][:40]}...")
            samples = np.clip(np.nan_to_num(entry["samples"], nan=0.0), -1.0, 1.0)
            sd.play(samples, entry["sample_rate"])
            sd.wait()
            # Only update status if we weren't stopped early
            if _active_play_btn[0] is btn:
                status_label.configure(text="✅ Playback done.")
                _reset_play_btn()
        except Exception as e:
            _log_crash(e)
            status_label.configure(text=f"❌ {_fmt_err(e)}")
            _reset_play_btn()
    threading.Thread(target=run, daemon=True).start()

def play_history_entry(entry):
    """Legacy wrapper — kept for any external callers."""
    _toggle_history_playback(entry, None)

def download_history_entry(entry):
    folder = get_default_folder()
    filepath = filedialog.asksaveasfilename(
        initialdir=folder or None,
        defaultextension=".wav",
        filetypes=[("MP3 files", "*.mp3"), ("WAV files", "*.wav")]
    )
    if not filepath:
        return
    set_default_folder(os.path.dirname(filepath))

    if filepath.lower().endswith(".mp3"):
        _save_as_mp3(entry, filepath)
    else:
        try:
            sf.write(filepath, entry["samples"], entry["sample_rate"])
            status_label.configure(text=f"✅ Saved: {os.path.basename(filepath)}")
        except Exception as e:
            _log_crash(e)
            status_label.configure(text=f"❌ Save failed: {_fmt_err(e)}")

def _save_as_mp3(entry, filepath):
    """Show MP3 metadata dialog then encode and save."""
    import lameenc
    from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, COMM, ID3NoHeaderError
    from mutagen.mp3 import MP3

    win = ctk.CTkToplevel(app)
    win.title("Save as MP3")
    win.geometry("420x340")
    win.resizable(False, False)
    win.grab_set()
    win.configure(fg_color=C_BG)

    # Header
    hdr = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=54)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    hdr_inner = ctk.CTkFrame(hdr, fg_color="transparent")
    hdr_inner.pack(side="left", padx=16, pady=10)
    ctk.CTkFrame(hdr_inner, fg_color=C_ACCENT, width=8, height=8,
                 corner_radius=4).pack(side="left", padx=(0, 10))
    ctk.CTkLabel(hdr_inner, text="MP3 Metadata  (optional)",
                 font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
                 text_color=C_TXT).pack(side="left")
    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x")

    body = ctk.CTkFrame(win, fg_color="transparent")
    body.pack(fill="both", expand=True, padx=20, pady=10)

    def _field(label, placeholder=""):
        ctk.CTkLabel(body, text=label,
                     font=ctk.CTkFont(family="Segoe UI", size=11),
                     text_color=C_TXT2, anchor="w").pack(fill="x", pady=(6, 1))
        var = ctk.StringVar()
        ctk.CTkEntry(body, textvariable=var,
                     fg_color=C_CARD, border_color=C_BORDER, text_color=C_TXT,
                     placeholder_text=placeholder, placeholder_text_color=C_TXT3,
                     height=30).pack(fill="x")
        return var

    title_var  = _field("Title",  entry["text"][:60].replace("\n", " "))
    author_var = _field("Artist / Author", "")
    album_var  = _field("Album / Series", "")
    year_var   = _field("Year", datetime.now().strftime("%Y"))

    # Quality selector
    ctk.CTkLabel(body, text="Quality",
                 font=ctk.CTkFont(family="Segoe UI", size=11),
                 text_color=C_TXT2, anchor="w").pack(fill="x", pady=(8, 1))
    quality_var = ctk.StringVar(value="192 kbps")
    ctk.CTkSegmentedButton(body, values=["128 kbps", "192 kbps", "320 kbps"],
                            variable=quality_var,
                            font=ctk.CTkFont(family="Segoe UI", size=11),
                            width=380).pack(anchor="w")

    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x")

    foot = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=54)
    foot.pack(fill="x")
    foot.pack_propagate(False)
    foot_inner = ctk.CTkFrame(foot, fg_color="transparent")
    foot_inner.pack(side="left", padx=16, pady=10)

    def _do_save():
        win.destroy()
        status_label.configure(text="⏳ Encoding MP3...")
        app.update_idletasks()

        try:
            bitrate_map = {"128 kbps": 128, "192 kbps": 192, "320 kbps": 320}
            bitrate = bitrate_map.get(quality_var.get(), 192)

            samples = entry["samples"]
            sr      = entry["sample_rate"]

            # Convert float32 → int16
            pcm = np.clip(samples, -1.0, 1.0)
            pcm = (pcm * 32767).astype(np.int16)
            # Mono only for now
            if pcm.ndim > 1:
                pcm = pcm.mean(axis=1).astype(np.int16)

            encoder = lameenc.Encoder()
            encoder.set_bit_rate(bitrate)
            encoder.set_in_sample_rate(sr)
            encoder.set_channels(1)
            encoder.set_quality(2)   # 2 = highest
            mp3_data = encoder.encode(pcm.tobytes()) + encoder.flush()

            with open(filepath, "wb") as f:
                f.write(mp3_data)

            # Write ID3 tags
            try:
                tags = ID3(filepath)
            except ID3NoHeaderError:
                tags = ID3()

            title = title_var.get().strip()
            if title:
                tags["TIT2"] = TIT2(encoding=3, text=title)
            author = author_var.get().strip()
            if author:
                tags["TPE1"] = TPE1(encoding=3, text=author)
            album = album_var.get().strip()
            if album:
                tags["TALB"] = TALB(encoding=3, text=album)
            year = year_var.get().strip()
            if year:
                tags["TDRC"] = TDRC(encoding=3, text=year)
            tags["COMM"] = COMM(encoding=3, lang="eng", desc="",
                                text=f"Generated by AI TTS Studio v{VERSION}")
            tags.save(filepath)

            status_label.configure(
                text=f"✅ Saved MP3: {os.path.basename(filepath)}  ({bitrate} kbps)")
        except Exception as e:
            _log_crash(e)
            status_label.configure(text=f"❌ MP3 encode failed: {_fmt_err(e)}")

    ctk.CTkButton(foot_inner, text="Save MP3", command=_do_save,
                  width=120, height=34,
                  font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
                  ).pack(side="left", padx=(0, 10))
    ctk.CTkButton(foot_inner, text="Cancel", command=win.destroy,
                  width=88, height=34,
                  font=ctk.CTkFont(family="Segoe UI", size=12),
                  **BTN_GHOST).pack(side="left")

# ── SRT Export ────────────────────────────────────────────────────────────────
# _srt_time, _wrap_for_subtitle, build_srt imported from tts_utils

def export_srt_from_entry(entry):
    if not entry.get("segments"):
        status_label.configure(text="⚠️ No timing data available for SRT export.")
        return
    folder = get_default_folder()
    filepath = filedialog.asksaveasfilename(
        initialdir=folder or None,
        defaultextension=".srt",
        filetypes=[("SRT subtitles", "*.srt"), ("All files", "*.*")]
    )
    if not filepath:
        return
    try:
        srt_content = build_srt(entry["segments"])
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(srt_content)
        status_label.configure(text=f"✅ SRT saved: {os.path.basename(filepath)}")
    except Exception as e:
        _log_crash(e)
        status_label.configure(text=f"❌ SRT export failed: {_fmt_err(e)}")

# ── Smooth Progress ───────────────────────────────────────────────────────────
class SmoothProgress:
    def __init__(self, bar, time_label):
        self.bar         = bar
        self.time_label  = time_label
        self._current    = 0.0
        self._target     = 0.0
        self._running    = False
        self._start_time = None
        self._est_total  = None

    def start(self, estimated_seconds):
        self._current    = 0.0
        self._target     = 0.0
        self._running    = True
        self._start_time = time.time()
        self._est_total  = max(estimated_seconds, 1)
        self.bar.set(0)
        self.time_label.configure(text=f"⏱ Est. time: ~{format_time(estimated_seconds)}")
        self._tick()

    def set_target(self, value):
        self._target = min(value, 0.99)

    def finish(self):
        # Called from worker threads — schedule all UI work on the main thread
        elapsed = time.time() - self._start_time if self._start_time else 0
        self._running = False
        self.bar.after(0, lambda e=elapsed: self._finish_ui(e))

    def _finish_ui(self, elapsed):
        self._current = 1.0
        self.bar.set(1.0)
        self.time_label.configure(text=f"✅ Done in {format_time(elapsed)}")
        s = _get_settings()
        threshold = s.get("notify_threshold_seconds", 10)
        if s.get("notify_on_completion", True) and elapsed > threshold:
            try:
                from win10toast import ToastNotifier
                ToastNotifier().show_toast(
                    "AI TTS Studio",
                    "Your audio is ready!",
                    duration=4,
                    threaded=True
                )
            except Exception:
                pass  # Notification is optional, never crash for it

    def _tick(self):
        if not self._running:
            return
        elapsed     = time.time() - self._start_time
        remaining   = self._est_total - elapsed
        if remaining > 0:
            time_driven = min(elapsed / self._est_total * 0.90, 0.90)
            ideal       = max(time_driven, self._target)
            self._current += (ideal - self._current) * 0.15
            self._current  = min(self._current, 0.90)
            self.bar.set(self._current)
            self.time_label.configure(text=f"⏱ Est. remaining: ~{format_time(remaining)}")
        else:
            # Estimate exceeded — breathe smoothly between 0.88 and 0.97
            import math
            pulse = 0.925 + 0.045 * math.sin(elapsed * 1.8)
            self.bar.set(pulse)
            self.time_label.configure(text="⏱ Processing…")
        app.after(100, self._tick)

# ── Helpers ───────────────────────────────────────────────────────────────────
# format_time imported from tts_utils

def estimate_processing_time(text):
    return len(text.split()) / get_words_per_second()

# estimate_audio_duration imported from tts_utils
# trim_silence imported from audio_utils
# chunk_text imported from tts_utils

# ── Voice Clone Library ────────────────────────────────────────────────────────
# CLONE_DIR / CLONE_INDEX defined near the top with other user-data paths.

# Thin wrappers bind the module-level paths so all existing call sites are unchanged.
# The in-memory _clone_cache is used for reads; writes invalidate it.
def load_clone_library():
    return _get_clone_library()

def save_clone_library(entries):
    _invalidate_clone_cache()
    _lib_save(entries, CLONE_DIR, CLONE_INDEX)

def add_clone_to_library(name, src_wav_path):
    _invalidate_clone_cache()
    return _lib_add(name, src_wav_path, CLONE_DIR, CLONE_INDEX)

def apply_enhancements(samples, sample_rate):
    out = samples.astype(np.float32).copy()

    # Noise gate
    if noise_gate_var.get():
        threshold = 10 ** (-40 / 20)
        release_samples = int(sample_rate * 0.25)
        gate_open = np.abs(out) > threshold
        # Smooth the gate with a simple release envelope
        envelope = np.zeros(len(out), dtype=np.float32)
        level = 0.0
        for i in range(len(out)):
            if gate_open[i]:
                level = 1.0
            else:
                level = max(0.0, level - 1.0 / release_samples)
            envelope[i] = level
        out *= envelope

    # High-pass filter
    nyq = sample_rate / 2.0
    if highpass_slider.get() > 20:
        hp_freq = min(float(highpass_slider.get()), nyq - 1)
        sos = butter(4, hp_freq, btype="high",
                     fs=sample_rate, output="sos")
        out = sosfilt(sos, out).astype(np.float32)

    # Low-pass filter
    if lowpass_slider.get() < nyq:
        lp_freq = min(float(lowpass_slider.get()), nyq - 1)
        sos = butter(4, lp_freq, btype="low",
                     fs=sample_rate, output="sos")
        out = sosfilt(sos, out).astype(np.float32)

    # Compressor — vectorized via scipy IIR envelope follower (~100x faster than
    # the per-sample Python loop).  Uses the release coefficient for smoothing;
    # attack/release asymmetry is imperceptible on speech-only TTS output.
    if compressor_var.get():
        from scipy.signal import lfilter
        threshold_lin = 10 ** (-20 / 20)
        ratio        = float(compressor_slider.get())
        release_coef = np.exp(-1.0 / (sample_rate * 0.100))   # 100 ms
        # IIR one-pole lowpass: env[n] = r*env[n-1] + (1-r)*|x[n]|
        b = np.array([1.0 - release_coef])
        a = np.array([1.0, -release_coef])
        env = lfilter(b, a, np.abs(out).astype(np.float64)).astype(np.float32)
        env = np.maximum(env, 1e-8)
        gain = np.where(
            env > threshold_lin,
            (threshold_lin + (env - threshold_lin) / ratio) / env,
            1.0,
        ).astype(np.float32)
        out = out * gain

    # Reverb (Freeverb-style Schroeder reverb)
    if reverb_slider.get() > 0:
        rv       = reverb_slider.get()
        wet      = rv * 0.4
        dry      = 1.0 - rv * 0.2
        room     = rv * 0.3
        damping  = 0.7
        # Comb filter delays (in samples) tuned for speech
        comb_delays = [int(sample_rate * d) for d in
                       [0.0297, 0.0371, 0.0411, 0.0437]]
        feedback = 0.5 + room * 0.38
        allpass_delays = [int(sample_rate * d) for d in [0.005, 0.0017]]
        # Run comb filters in parallel
        comb_out = np.zeros_like(out)
        for delay in comb_delays:
            buf   = np.zeros(delay, dtype=np.float32)
            pos   = 0
            filt  = 0.0
            co    = np.empty_like(out)
            for i in range(len(out)):
                buf_out       = buf[pos]
                filt          = buf_out * (1 - damping) + filt * damping
                buf[pos]      = out[i] + filt * feedback
                pos           = (pos + 1) % delay
                co[i]         = buf_out
            comb_out += co
        comb_out /= len(comb_delays)
        # Allpass filters in series
        ap = comb_out.copy()
        for delay in allpass_delays:
            buf = np.zeros(delay, dtype=np.float32)
            pos = 0
            ao  = np.empty_like(ap)
            for i in range(len(ap)):
                buf_out  = buf[pos]
                buf[pos] = ap[i] + buf_out * 0.5
                pos      = (pos + 1) % delay
                ao[i]    = buf_out - ap[i] * 0.5
            ap = ao
        out = dry * out + wet * ap

    # Gain
    gain_db  = float(gain_slider.get())
    out     *= 10 ** (gain_db / 20)

    # Prevent clipping
    max_val = np.max(np.abs(out))
    if max_val > 1.0:
        out /= max_val

    return out

def generate_audio(text, voice, speed, status_cb=None):
    text = apply_pronunciation(text)
    use_chatterbox = engine_var.get() == "🎙️ Natural"

    if use_chatterbox:
        # ── Chatterbox path ────────────────────────────────────────────────────
        if not chatterbox_engine.is_ready:
            if status_cb: status_cb("🎙️ Waiting for Natural mode to finish loading...")
            chatterbox_engine.start(status_cb=status_cb)
        chunks = chunk_text(text)
        all_samples, sample_rate = [], None
        _clone_path = cb_clone_path_var.get()
        if _clone_path and not os.path.exists(_clone_path):
            if status_cb: status_cb("⚠️ Voice clone file not found — using default voice.")
            _clone_path = ""
        prompt = _clone_path or None
        exag   = cb_exag_slider.get()
        cfg    = cb_cfg_slider.get()
        for i, chunk in enumerate(chunks):
            if _cancel_event.is_set():
                raise GenerationCancelled()
            if status_cb: status_cb(f"🎙️ Generating chunk {i+1}/{len(chunks)}...")
            samples, sr = chatterbox_engine.generate_chunk(
                chunk,
                audio_prompt_path=prompt,
                exaggeration=exag,
                cfg_weight=cfg,
                status_cb=status_cb,
            )
            all_samples.append(samples)
            sample_rate = sr
            smooth.set_target((i + 1) / len(chunks) * 0.9)
    else:
        # ── Kokoro path ────────────────────────────────────────────────────────
        chunks = chunk_text(text)
        all_samples, sample_rate = [], None
        for i, chunk in enumerate(chunks):
            if _cancel_event.is_set():
                raise GenerationCancelled()
            if status_cb: status_cb(f"⏳ Generating chunk {i+1}/{len(chunks)}...")
            samples, sr = kokoro.create(chunk, voice=voice, speed=speed)
            all_samples.append(samples)
            sample_rate = sr

    # Build per-chunk timings (before trim/enhance) for SRT
    chunk_timings = []
    offset = 0.0
    for chunk, samp in zip(chunks, all_samples):
        dur = len(samp) / sample_rate
        chunk_timings.append((offset, offset + dur, chunk))
        offset += dur
    pre_total_dur = offset

    combined = np.concatenate(all_samples)

    # Sanitize — NaN/Inf in audio (from a bad voice clone prompt) will crash
    # PortAudio at the C level with no Python exception and no error message.
    if not np.isfinite(combined).all():
        combined = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)
    combined = np.clip(combined, -1.0, 1.0)

    if trim_var.get():
        if status_cb: status_cb("✂️ Trimming silence...")
        combined = trim_silence(combined, sample_rate)
    if status_cb: status_cb("🎛️ Applying enhancements...")
    enhanced = apply_enhancements(combined, sample_rate)

    # Final clip after enhancements (compressor/gain can push above 1.0)
    enhanced = np.clip(enhanced, -1.0, 1.0)

    # Scale timings to match final (post-trim/enhance) audio length
    post_total_dur = len(enhanced) / sample_rate
    scale = post_total_dur / pre_total_dur if pre_total_dur > 0 else 1.0
    segments = [(s * scale, e * scale, t) for s, e, t in chunk_timings]

    return enhanced, sample_rate, segments

# ── Dialogue ──────────────────────────────────────────────────────────────────
# parse_dialogue imported from tts_utils

def generate_dialogue_audio(dialogue_lines, speaker_voices, speed, status_cb=None):
    """
    Generate multi-voice audio from parsed dialogue lines.
    dialogue_lines: list of (speaker, text)
    speaker_voices: dict of speaker_name -> voice display name (key in VOICES)
    Returns: (audio, sample_rate, segments)
    """
    PAUSE_SAME = 0.15   # s between consecutive lines from same speaker
    PAUSE_DIFF = 0.35   # s between different speakers

    parts      = []
    timings    = []
    sample_rate = None
    offset     = 0.0
    voice_keys = list(VOICES.keys())

    for i, (speaker, text) in enumerate(dialogue_lines):
        text = apply_pronunciation(text)
        voice_name = speaker_voices.get(speaker, voice_keys[0])
        voice_id   = VOICES.get(voice_name, list(VOICES.values())[0])
        if status_cb:
            status_cb(f"⏳ Line {i+1}/{len(dialogue_lines)} — {speaker}...")

        line_chunks  = chunk_text(text)
        line_samples = []
        for chunk in line_chunks:
            samp, sr = kokoro.create(chunk, voice=voice_id, speed=speed)
            line_samples.append(samp)
            sample_rate = sr

        seg_audio = np.concatenate(line_samples)
        dur = len(seg_audio) / sample_rate
        timings.append((offset, offset + dur, f"{speaker}: {text}"))
        parts.append(seg_audio)
        offset += dur

        # Pause between lines
        if i < len(dialogue_lines) - 1:
            next_speaker = dialogue_lines[i + 1][0]
            pause = PAUSE_DIFF if next_speaker != speaker else PAUSE_SAME
            parts.append(np.zeros(int(sample_rate * pause), dtype=np.float32))
            offset += pause

    combined = np.concatenate(parts)

    pre_dur = offset
    if trim_var.get():
        if status_cb: status_cb("✂️ Trimming silence...")
        combined = trim_silence(combined, sample_rate)
    if status_cb: status_cb("🎛️ Applying enhancements...")
    enhanced = apply_enhancements(combined, sample_rate)

    post_dur = len(enhanced) / sample_rate
    scale    = post_dur / pre_dur if pre_dur > 0 else 1.0
    segments = [(s * scale, e * scale, t) for s, e, t in timings]

    return enhanced, sample_rate, segments

# ── Profiles ──────────────────────────────────────────────────────────────────
def get_current_settings():
    return {
        "voice": voice_var.get(), "speed": speed_slider.get(),
        "highpass": highpass_slider.get(), "lowpass": lowpass_slider.get(),
        "reverb": reverb_slider.get(), "compressor": compressor_var.get(),
        "compressor_ratio": compressor_slider.get(), "gain": gain_slider.get(),
        "noise_gate": noise_gate_var.get(), "trim": trim_var.get(),
    }

def apply_settings(s):
    voice_var.set(s.get("voice", "🇬🇧 Male - George (Best)"))
    speed_slider.set(s.get("speed", 0.85))
    highpass_slider.set(s.get("highpass", 20))
    lowpass_slider.set(s.get("lowpass", 18000))
    reverb_slider.set(s.get("reverb", 0.0))
    compressor_var.set(s.get("compressor", True))
    compressor_slider.set(s.get("compressor_ratio", 2.0))
    gain_slider.set(s.get("gain", 0))
    noise_gate_var.set(s.get("noise_gate", False))
    trim_var.set(s.get("trim", True))
    update_all_labels()
    update_word_count()
    eq_preset_var.set("Custom")

EQ_PRESETS = {
    "Custom":             None,
    "🎚️ Flat":            {"highpass": 20,  "lowpass": 18000, "compressor": False, "compressor_ratio": 1.0, "reverb": 0.0,  "gain": 0, "noise_gate": False, "trim": False},
    "🎙️ Podcast":         {"highpass": 100, "lowpass": 14000, "compressor": True,  "compressor_ratio": 4.0, "reverb": 0.05, "gain": 3, "noise_gate": True,  "trim": True},
    "📖 Audiobook":       {"highpass": 60,  "lowpass": 16000, "compressor": True,  "compressor_ratio": 2.5, "reverb": 0.08, "gain": 1, "noise_gate": False, "trim": True},
    "📻 Broadcast":       {"highpass": 120, "lowpass": 12000, "compressor": True,  "compressor_ratio": 6.0, "reverb": 0.03, "gain": 4, "noise_gate": True,  "trim": True},
    "🎬 Cinematic":       {"highpass": 40,  "lowpass": 15000, "compressor": True,  "compressor_ratio": 2.0, "reverb": 0.30, "gain": 0, "noise_gate": False, "trim": True},
    "🌙 Warm & Intimate": {"highpass": 80,  "lowpass": 10000, "compressor": True,  "compressor_ratio": 3.0, "reverb": 0.15, "gain": 2, "noise_gate": True,  "trim": True},
}

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "🌙 Calm Narrator": {"voice":"🇬🇧 Male - George (Best)","speed":0.85,"highpass":80,"lowpass":12000,"reverb":0.15,"compressor":True,"compressor_ratio":3.0,"gain":2,"noise_gate":True,"trim":True},
        "🎙️ Podcast":       {"voice":"🇺🇸 Male - Michael","speed":1.0,"highpass":100,"lowpass":14000,"reverb":0.05,"compressor":True,"compressor_ratio":4.0,"gain":3,"noise_gate":True,"trim":True},
        "📖 Audiobook":     {"voice":"🇬🇧 Male - George (Best)","speed":0.9,"highpass":60,"lowpass":16000,"reverb":0.08,"compressor":True,"compressor_ratio":2.5,"gain":1,"noise_gate":False,"trim":True},
    }

def save_profiles(p):
    try:
        with open(PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2)
    except OSError:
        pass

def refresh_profile_menu():
    profile_menu.configure(values=list(load_profiles().keys()))

def save_profile():
    name = profile_name_entry.get().strip()
    if not name:
        status_label.configure(text="⚠️ Enter a profile name first.")
        return
    p = load_profiles()
    p[name] = get_current_settings()
    save_profiles(p)
    refresh_profile_menu()
    profile_var.set(name)
    status_label.configure(text=f"✅ Profile '{name}' saved.")

def load_profile():
    name = profile_var.get()
    p    = load_profiles()
    if name in p:
        apply_settings(p[name])
        status_label.configure(text=f"✅ Profile '{name}' loaded.")

def delete_profile():
    name = profile_var.get()
    p    = load_profiles()
    if name in p:
        if not messagebox.askyesno("Delete Profile", f"Delete '{name}'?"):
            return
        del p[name]
        save_profiles(p)
        refresh_profile_menu()
        remaining = list(load_profiles().keys())
        if remaining: profile_var.set(remaining[0])
        status_label.configure(text=f"🗑 Profile '{name}' deleted.")

_applying_eq_preset = False

def apply_eq_preset(name=None):
    global _applying_eq_preset
    name = name or eq_preset_var.get()
    p = EQ_PRESETS.get(name)
    if not p:
        return
    _applying_eq_preset = True
    highpass_slider.set(p["highpass"])
    lowpass_slider.set(p["lowpass"])
    compressor_var.set(p["compressor"])
    compressor_slider.set(p["compressor_ratio"])
    reverb_slider.set(p["reverb"])
    gain_slider.set(p["gain"])
    noise_gate_var.set(p["noise_gate"])
    trim_var.set(p["trim"])
    _applying_eq_preset = False
    update_all_labels()

# ── Cancellation ─────────────────────────────────────────────────────────────
# GenerationCancelled imported from tts_utils
_cancel_event = threading.Event()

def cancel_generation():
    """Signal the running generation thread to stop after the current chunk."""
    _cancel_event.set()

# ── Queue ─────────────────────────────────────────────────────────────────────
queue_items   = []
is_generating  = False
_queue_counter = 0     # monotonically increasing; not reset on remove/clear

def queue_add():
    global _queue_counter
    text = text_input.get("1.0", "end").strip()
    if not text:
        status_label.configure(text="⚠️ No text to add.")
        return

    _queue_counter += 1
    # Suggest first ~5 words of the text as the default name
    suggested = " ".join(text.split()[:5])
    if len(text.split()) > 5:
        suggested += "…"

    # ── Name popup ────────────────────────────────────────────────────────────
    dlg = ctk.CTkToplevel(app)
    dlg.title("Name this queue item")
    dlg.geometry("360x140")
    dlg.resizable(False, False)
    dlg.configure(fg_color=C_BG)
    dlg.grab_set()
    dlg.transient(app)

    ctk.CTkLabel(dlg, text="Queue item name:",
                 font=ctk.CTkFont(family="Segoe UI", size=12),
                 text_color=C_TXT2).pack(anchor="w", padx=20, pady=(18, 4))

    name_entry = ctk.CTkEntry(dlg, width=320, height=34,
                              font=ctk.CTkFont(family="Segoe UI", size=12))
    name_entry.insert(0, suggested)
    name_entry.select_range(0, "end")
    name_entry.pack(padx=20)
    name_entry.focus_set()

    btn_row = ctk.CTkFrame(dlg, fg_color="transparent")
    btn_row.pack(fill="x", padx=20, pady=(12, 0))

    def _confirm(e=None):
        name = name_entry.get().strip() or suggested
        queue_items.append({"name": name, "text": text})
        refresh_queue_display()
        status_label.configure(text=f"✅ Added '{name}'. {len(queue_items)} item(s) in queue.")
        dlg.destroy()

    ctk.CTkButton(btn_row, text="Add to Queue", width=120, height=32,
                  font=ctk.CTkFont(family="Segoe UI", size=12),
                  fg_color=C_ACCENT, hover_color=C_ACCENT_H, text_color="#000000",
                  command=_confirm).pack(side="left", padx=(0, 8))
    ctk.CTkButton(btn_row, text="Cancel", width=80, height=32,
                  font=ctk.CTkFont(family="Segoe UI", size=12),
                  **BTN_GHOST, command=dlg.destroy).pack(side="left")

    name_entry.bind("<Return>", _confirm)
    dlg.bind("<Escape>", lambda _e: dlg.destroy())

def queue_remove():
    sel = queue_listbox.curselection()
    if not sel:
        status_label.configure(text="⚠️ Click an item first.")
        return
    queue_items.pop(sel[0])
    refresh_queue_display()

def queue_clear():
    queue_items.clear()
    refresh_queue_display()
    status_label.configure(text="🗑 Queue cleared.")

def refresh_queue_display():
    queue_listbox.delete(0, "end")
    for i, item in enumerate(queue_items):
        words = len(item["text"].split())
        proc  = format_time(estimate_processing_time(item["text"]))
        audio = format_time(estimate_audio_duration(item["text"], speed_slider.get()))
        queue_listbox.insert("end",
            f"  {i+1}. {item['name']}  —  {words:,} words  |  Process: ~{proc}  |  Audio: ~{audio}")
    update_queue_estimate()

def update_queue_estimate():
    if not queue_items:
        queue_estimate_label.configure(text="Queue is empty.")
        return
    total_proc  = sum(estimate_processing_time(i["text"]) for i in queue_items)
    total_audio = sum(estimate_audio_duration(i["text"], speed_slider.get()) for i in queue_items)
    total_words = sum(len(i["text"].split()) for i in queue_items)
    queue_estimate_label.configure(
        text=f"📊 {len(queue_items)} items  |  {total_words:,} words  |  "
             f"Processing: ~{format_time(total_proc)}  |  Total audio: ~{format_time(total_audio)}"
    )

def queue_generate_all():
    if not queue_items:
        status_label.configure(text="⚠️ Queue is empty.")
        return
    folder = get_default_folder()
    out_dir = filedialog.askdirectory(title="Choose output folder",
                                      initialdir=folder or None)
    if not out_dir: return
    set_default_folder(out_dir)

    voice       = VOICES[voice_var.get()]
    if engine_var.get() == "🎙️ Natural":
        _sel = cb_clone_var.get()
        voice_name = _sel if _sel != _CLONE_DEFAULT else "Default"
    else:
        voice_name = voice_var.get()
    speed       = round(speed_slider.get(), 2)
    total_items = len(queue_items)
    total_words = sum(len(i["text"].split()) for i in queue_items)
    est_total   = estimate_processing_time(" ".join(i["text"] for i in queue_items))

    queue_gen_btn.configure(state="disabled")
    smooth.start(est_total)

    _cancel_event.clear()

    def run():
        global is_generating
        is_generating = True
        words_done    = 0
        cancelled     = False
        for i, item in enumerate(queue_items):
            if _cancel_event.is_set():
                cancelled = True
                break
            def scb(msg): status_label.configure(text=f"[{i+1}/{total_items}] {msg}")
            scb("Starting...")
            t0 = time.time()
            try:
                samples, sr, segments = generate_audio(item["text"], voice, speed, status_cb=scb)
                words_done += len(item["text"].split())
                smooth.set_target(words_done / total_words)
                out_path = os.path.join(out_dir, f"{i+1:02d}_{item['name'].replace(' ','_')}.wav")
                sf.write(out_path, samples, sr)
                record_calibration(len(item["text"].split()), time.time() - t0)
                add_to_history(samples, sr, item["text"], voice_name, segments=segments)
                scb(f"✅ Saved {item['name']}")
            except GenerationCancelled:
                cancelled = True
                break
            except Exception as e:
                _log_crash(e)
                scb(f"❌ {_fmt_err(e)}")
            time.sleep(0.05)
        smooth.finish()
        is_generating = False
        if cancelled:
            status_label.configure(text=f"⏹ Queue cancelled. {words_done and i or 0} of {total_items} items completed.")
        else:
            status_label.configure(text=f"✅ Queue complete! {total_items} files saved to {out_dir}")
        queue_gen_btn.configure(state="normal")

    threading.Thread(target=run, daemon=True).start()

# ── Word Counter ──────────────────────────────────────────────────────────────
_wc_after_id = None

def update_word_count(*_):
    """Debounced: schedules the actual update 150 ms after the last call."""
    global _wc_after_id
    if _wc_after_id:
        app.after_cancel(_wc_after_id)
    _wc_after_id = app.after(150, _do_word_count)

def _do_word_count():
    global _wc_after_id
    _wc_after_id = None
    text  = text_input.get("1.0", "end").strip()
    words = len(text.split()) if text else 0
    chars = len(text)
    speed = speed_slider.get()
    audio = estimate_audio_duration(text, speed)
    proc  = estimate_processing_time(text)
    cal       = _get_calibration()
    use_cb    = engine_var.get() == "🎙️ Natural"
    n_samples = len(cal.get("cb_samples" if use_cb else "samples") or [])
    if n_samples == 0:
        note = "  (Calibrating — improves after first run)"
    elif n_samples == 1:
        note = "  (Calibrating...)"
    else:
        note = ""
    word_count_label.configure(
        text=f"Words: {words:,}  |  Chars: {chars:,}  |  "
             f"Audio: ~{format_time(audio)}  |  Processing: ~{format_time(proc)}{note}"
    )

# ── Main Actions ──────────────────────────────────────────────────────────────
def generate_and_store():
    """Generate audio, store in history. Does NOT auto-play."""
    global is_generating
    text = text_input.get("1.0", "end").strip()
    if not text:
        status_label.configure(text="⚠️ Please enter some text.")
        return
    voice      = VOICES[voice_var.get()]
    if engine_var.get() == "🎙️ Natural":
        _sel = cb_clone_var.get()
        voice_name = _sel if _sel != _CLONE_DEFAULT else "Default"
    else:
        voice_name = voice_var.get()
    speed      = round(speed_slider.get(), 2)
    words      = len(text.split())
    est        = estimate_processing_time(text)

    _cancel_event.clear()
    play_button.configure(state="disabled", text="⏳ Working...")
    stop_button.configure(state="normal", text="Cancel",
                          fg_color="#2a0f0f", hover_color="#3d1515",
                          text_color=C_DANGER, border_width=1, border_color="#3d1515")
    smooth.start(est)
    is_generating = True

    def run():
        global is_generating
        t0 = time.time()
        try:
            samples, sr, segments = generate_audio(
                text, voice, speed,
                status_cb=lambda m: status_label.configure(text=m)
            )
            elapsed = time.time() - t0
            record_calibration(words, elapsed)
            smooth.finish()
            add_to_history(samples, sr, text, voice_name, segments=segments)
            status_label.configure(text="✅ Audio ready! Click ▶ Play in the history panel.")
        except GenerationCancelled:
            smooth.finish()
            status_label.configure(text="⏹ Generation cancelled.")
        except Exception as e:
            _log_crash(e)
            smooth.finish()
            status_label.configure(text=f"❌ {_fmt_err(e)}")
        finally:
            is_generating = False
            play_button.configure(state="normal", text="▶  Generate")
            stop_button.configure(state="disabled", text="Stop",
                                  fg_color="transparent", hover_color=C_ELEVATED,
                                  text_color=C_TXT2, border_width=1, border_color=C_BORDER)

    threading.Thread(target=run, daemon=True).start()


def stop_audio():
    if is_generating:
        cancel_generation()   # signals the generation thread to stop after current chunk
    else:
        sd.stop()
        status_label.configure(text="⏹ Stopped.")
        play_button.configure(state="normal", text="▶  Generate")
        stop_button.configure(state="disabled", text="Stop",
                              fg_color="transparent", hover_color=C_ELEVATED,
                              text_color=C_TXT2, border_width=1, border_color=C_BORDER)

def preview_voice():
    voice = VOICES[voice_var.get()]
    speed = round(speed_slider.get(), 2)
    preview_button.configure(state="disabled")

    def run():
        try:
            samples, sr = kokoro.create(
                "Hello! This is a preview of the selected voice.", voice=voice, speed=speed)
            enhanced = apply_enhancements(samples, sr)
            sd.play(enhanced, sr)
            sd.wait()
            status_label.configure(text="✅ Preview done!")
        except Exception as e:
            _log_crash(e)
            status_label.configure(text=f"❌ {_fmt_err(e)}")
        finally:
            preview_button.configure(state="normal")

    threading.Thread(target=run, daemon=True).start()

def import_file():
    folder = get_default_folder()
    fp = filedialog.askopenfilename(
        initialdir=folder or None,
        filetypes=[("Text files", "*.txt")]
    )
    if fp:
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                with open(fp, "r", encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            status_label.configure(text="❌ Could not decode file — unknown encoding.")
            return
        if _get_settings().get("auto_clean_text", False):
            content, _ = clean_text(content)
        text_input.delete("1.0", "end")
        text_input.insert("1.0", content)
        update_word_count()
        status_label.configure(text=f"✅ Imported {len(content):,} characters.")

def clear_text():
    text_input.delete("1.0", "end")
    update_word_count()
    status_label.configure(text="Ready.")


def update_all_labels(*_):
    hp_label.configure(text=f"{int(highpass_slider.get())} Hz")
    lp_label.configure(text=f"{int(lowpass_slider.get())} Hz")
    reverb_label.configure(text=f"{reverb_slider.get():.2f}")
    comp_label.configure(text=f"{compressor_slider.get():.1f}x")
    gain_label.configure(text=f"{gain_slider.get():+.0f} dB")
    speed_label.configure(text=f"{speed_slider.get():.2f}x")
    try:
        cb_exag_label.configure(text=f"{cb_exag_slider.get():.2f}")
        cb_cfg_label.configure(text=f"{cb_cfg_slider.get():.2f}")
    except NameError:
        pass  # Chatterbox widgets not yet created
    # NOTE: update_word_count is intentionally NOT called here.
    # Sliders don't change the word count — only text changes do.
    # Calling it here caused a calibration disk read on every slider tick.

def reset_enhancements():
    highpass_slider.set(20);  lowpass_slider.set(18000)
    reverb_slider.set(0.0);   compressor_slider.set(2.0)
    gain_slider.set(0);       compressor_var.set(False)
    noise_gate_var.set(False); trim_var.set(True)
    update_all_labels()
    status_label.configure(text="🔄 Enhancements reset.")

# ── About Window ──────────────────────────────────────────────────────────────

# ── Text Cleaner Window ───────────────────────────────────────────────────────
def show_text_cleaner():
    text = text_input.get("1.0", "end").strip()
    if not text:
        status_label.configure(text="⚠️ No text to clean.")
        return

    win = ctk.CTkToplevel(app)
    win.title("Text Cleaner")
    win.geometry("620x520")
    win.resizable(True, True)
    win.configure(fg_color=C_BG)
    win.grab_set()

    header = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=56)
    header.pack(fill="x")
    header.pack_propagate(False)
    ctk.CTkLabel(header, text="Text Cleaner",
                 font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
                 text_color=C_TXT).pack(side="left", padx=20)

    _, changes = clean_text(text)
    cleaned, _ = clean_text(text)

    if changes:
        summary = "Will fix:  " + "  ·  ".join(changes[:4]) + ("  ·  …" if len(changes) > 4 else "")
        s_color, s_bg = C_WARN, "#2a1f0a"
    else:
        summary = "Text looks clean — no changes needed."
        s_color, s_bg = C_SUCCESS, "#0a2010"

    info = ctk.CTkFrame(win, fg_color=s_bg, corner_radius=0, height=34)
    info.pack(fill="x")
    info.pack_propagate(False)
    ctk.CTkLabel(info, text=summary, text_color=s_color,
                 font=ctk.CTkFont(family="Segoe UI", size=11),
                 anchor="w").pack(side="left", padx=16, pady=8)

    preview_tabs = ctk.CTkTabview(win, fg_color=C_BG)
    preview_tabs.pack(fill="both", expand=True, padx=16, pady=(12, 8))
    preview_tabs.add("Before")
    preview_tabs.add("After")

    font_mono = ctk.CTkFont(family="Consolas", size=12)
    before_box = ctk.CTkTextbox(preview_tabs.tab("Before"), font=font_mono)
    before_box.pack(fill="both", expand=True)
    before_box.insert("1.0", text)
    before_box.configure(state="disabled")

    after_box = ctk.CTkTextbox(preview_tabs.tab("After"), font=font_mono)
    after_box.pack(fill="both", expand=True)
    after_box.insert("1.0", cleaned)
    after_box.configure(state="disabled")

    btn_frame = ctk.CTkFrame(win, fg_color="transparent")
    btn_frame.pack(fill="x", padx=16, pady=(0, 16))

    def apply_clean():
        text_input.delete("1.0", "end")
        text_input.insert("1.0", cleaned)
        update_word_count()
        status_label.configure(text=f"✅ Text cleaned.")
        win.destroy()

    ctk.CTkButton(btn_frame, text="Apply Changes", command=apply_clean,
                  width=150, height=36,
                  font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold")).pack(side="left", padx=(0, 8))
    ctk.CTkButton(btn_frame, text="Cancel", command=win.destroy,
                  width=90, height=36, **BTN_GHOST).pack(side="left")

# ── Settings ──────────────────────────────────────────────────────────────────
def show_settings():
    from settings_window import open_settings_window, load_settings
    profiles = list(load_profiles().keys())

    def on_saved(new_settings):
        global _settings_cache
        _settings_cache = new_settings  # keep cache in sync with what was just saved
        # Apply default voice if changed
        if new_settings.get("default_voice") in VOICES:
            voice_var.set(new_settings["default_voice"])
        # Apply default speed
        speed_slider.set(new_settings.get("default_speed", 0.85))
        update_all_labels()
        status_label.configure(text="✅ Settings saved.")

    open_settings_window(app, list(VOICES.keys()), profiles, on_save_callback=on_saved)


def show_about():
    win = ctk.CTkToplevel(app)
    win.title("About TTS Studio")
    win.geometry("460x620")
    win.resizable(False, False)
    win.configure(fg_color=C_BG)
    win.grab_set()

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=150)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    if LOGO_IMG_LG:
        ctk.CTkLabel(hdr, image=LOGO_IMG_LG, text="").pack(pady=(10, 4))
    ctk.CTkLabel(hdr, text="AI TTS Studio",
                 font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
                 text_color=C_TXT).pack()
    ctk.CTkLabel(hdr, text=f"Version {VERSION}  ·  100% local · no internet required",
                 font=ctk.CTkFont(family="Segoe UI", size=11),
                 text_color=C_ACCENT).pack()

    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x")

    # ── Scrollable body ───────────────────────────────────────────────────────
    scroll = ctk.CTkScrollableFrame(win, fg_color="transparent",
                                    scrollbar_button_color=C_ELEVATED,
                                    scrollbar_button_hover_color=C_ACCENT_D)
    scroll.pack(fill="both", expand=True, padx=20, pady=(12, 0))

    def _section(text):
        ctk.CTkLabel(scroll, text=text,
                     font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
                     text_color=C_TXT3, anchor="w").pack(fill="x", pady=(10, 3))
        ctk.CTkFrame(scroll, fg_color=C_BORDER, height=1,
                     corner_radius=0).pack(fill="x", pady=(0, 6))

    # ── Watermark notice ─────────────────────────────────────────────────────
    notice = ctk.CTkFrame(scroll, fg_color=C_ACCENT_D, corner_radius=8)
    notice.pack(fill="x", pady=(0, 10))
    ctk.CTkLabel(notice,
                 text="Audio Watermarking Notice",
                 font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                 text_color=C_ACCENT, anchor="w").pack(anchor="w", padx=12, pady=(10, 2))
    ctk.CTkLabel(notice,
                 text="Audio generated in Natural mode contains an imperceptible\n"
                      "neural watermark applied by Resemble AI's Perth library.\n"
                      "This watermark is inaudible and does not affect quality.",
                 font=ctk.CTkFont(family="Segoe UI", size=11),
                 text_color=C_TXT2, anchor="w", justify="left").pack(anchor="w", padx=12, pady=(0, 10))

    # ── Keyboard shortcuts ────────────────────────────────────────────────────
    _section("KEYBOARD SHORTCUTS")
    shortcuts = [
        ("Ctrl + Enter", "Generate audio"),
        ("Ctrl + P",     "Play latest audio"),
        ("Ctrl + S",     "Save latest audio"),
        ("Ctrl + I",     "Import text file"),
        ("Ctrl + Q",     "Add to queue"),
        ("Escape",       "Stop playback"),
        ("Ctrl + /",     "Open About"),
    ]
    for key, desc in shortcuts:
        row = ctk.CTkFrame(scroll, fg_color=C_CARD, corner_radius=6)
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=key, width=120, anchor="w",
                     font=ctk.CTkFont(family="Consolas", size=11),
                     text_color=C_ACCENT).pack(side="left", padx=10, pady=5)
        ctk.CTkLabel(row, text=desc, anchor="w",
                     font=ctk.CTkFont(family="Segoe UI", size=11),
                     text_color=C_TXT2).pack(side="left")

    # ── Open source credits ───────────────────────────────────────────────────
    _section("OPEN SOURCE CREDITS")
    credits = [
        ("Kokoro TTS v1.0",    "Apache 2.0",  "hexgrad/kokoro"),
        ("Chatterbox TTS",     "MIT",          "ResembleAI/chatterbox"),
        ("Perth Watermarker",  "MIT",          "resemble-ai/perth"),
        ("CustomTkinter",      "MIT",          "TomSchimansky/CustomTkinter"),
        ("ONNX Runtime",       "MIT",          "microsoft/onnxruntime"),
        ("SciPy",              "BSD-3-Clause", "scipy/scipy"),
        ("sounddevice",        "MIT",          "spatialaudio/python-sounddevice"),
        ("soundfile",          "BSD-3-Clause", "bastibe/python-soundfile"),
        ("NumPy",              "BSD-3-Clause", "numpy/numpy"),
        ("PyTorch",            "BSD-3-Clause", "pytorch/pytorch"),
        ("torchaudio",         "BSD-2-Clause", "pytorch/audio"),
    ]
    for lib, lic, repo in credits:
        row = ctk.CTkFrame(scroll, fg_color="transparent")
        row.pack(fill="x", pady=1)
        ctk.CTkLabel(row, text=lib, width=190, anchor="w",
                     font=ctk.CTkFont(family="Segoe UI", size=11),
                     text_color=C_TXT).pack(side="left")
        ctk.CTkLabel(row, text=lic, width=120, anchor="w",
                     font=ctk.CTkFont(family="Segoe UI", size=10),
                     text_color=C_TXT2).pack(side="left")
        ctk.CTkLabel(row, text=repo, anchor="w",
                     font=ctk.CTkFont(family="Consolas", size=10),
                     text_color=C_TXT3).pack(side="left")

    ctk.CTkLabel(scroll, text=" ", text_color=C_BG).pack()  # bottom padding

    # ── Footer ────────────────────────────────────────────────────────────────
    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x")
    foot = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=52)
    foot.pack(fill="x")
    foot.pack_propagate(False)
    ctk.CTkButton(foot, text="Close", command=win.destroy,
                  width=100, height=32, **BTN_GHOST).pack(side="right", padx=16, pady=10)

# ── Close Handler ─────────────────────────────────────────────────────────────
def on_close():
    if is_generating:
        if not messagebox.askyesno(
            "Generation in progress",
            "Audio is still being generated.\nAre you sure you want to quit? It will be cancelled."
        ):
            return
    sd.stop()
    chatterbox_engine.stop()
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_close)

# ── Keyboard Shortcuts ────────────────────────────────────────────────────────
def _shortcut_generate(e=None): generate_and_store()
def _shortcut_save(e=None):
    if audio_history: download_history_entry(audio_history[0])
def _shortcut_play_latest(e=None):
    if audio_history: play_history_entry(audio_history[0])
def _shortcut_import(e=None): import_file()
def _shortcut_queue(e=None):  queue_add()
def _shortcut_stop(e=None):   stop_audio()
def _shortcut_about(e=None):  show_about()

app.bind("<Control-Return>",    _shortcut_generate)
app.bind("<Control-s>",         _shortcut_save)
app.bind("<Control-S>",         _shortcut_save)
app.bind("<Control-p>",         _shortcut_play_latest)
app.bind("<Control-P>",         _shortcut_play_latest)
app.bind("<Control-i>",         _shortcut_import)
app.bind("<Control-I>",         _shortcut_import)
app.bind("<Control-q>",         _shortcut_queue)
app.bind("<Control-Q>",         _shortcut_queue)
app.bind("<Escape>",            _shortcut_stop)
app.bind("<Control-slash>",     _shortcut_about)

# ══════════════════════════════════════════════════════════════════════════════
# UI — Studio Gold Theme
# ══════════════════════════════════════════════════════════════════════════════

def _sep(parent, pady=0):
    """Thin horizontal divider line."""
    ctk.CTkFrame(parent, fg_color=C_BORDER, height=1,
                 corner_radius=0).pack(fill="x", pady=pady)

def _section_label(parent, text, padx=14, pady=(12, 6)):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(anchor="w", padx=padx, pady=pady)
    ctk.CTkFrame(row, fg_color=C_ACCENT, width=3, height=10,
                 corner_radius=2).pack(side="left", padx=(0, 6))
    ctk.CTkLabel(row, text=text,
                 font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                 text_color=C_TXT2, anchor="w").pack(side="left")

# ── Status Bar (packed first so it stays at bottom) ───────────────────────────
status_bar = ctk.CTkFrame(app, fg_color=C_SURFACE, corner_radius=0, height=32)
status_bar.pack(fill="x", side="bottom")
status_bar.pack_propagate(False)
_sep(status_bar)
status_label = ctk.CTkLabel(
    status_bar,
    text="Ready  ·  Ctrl+Enter to generate  ·  Ctrl+P to play  ·  Esc to stop",
    anchor="w", text_color=C_TXT3,
    font=ctk.CTkFont(family="Segoe UI", size=12))
status_label.pack(side="left", padx=16, pady=6)

def _copy_status():
    txt = status_label.cget("text")
    app.clipboard_clear()
    app.clipboard_append(txt)

_copy_btn = ctk.CTkButton(
    status_bar, text="⎘", width=24, height=20,
    font=ctk.CTkFont(family="Segoe UI", size=11),
    fg_color="transparent", hover_color=C_BORDER,
    text_color=C_TXT3, corner_radius=4,
    command=_copy_status)
_copy_btn.pack(side="right", padx=(0, 8), pady=6)

# ── Header ────────────────────────────────────────────────────────────────────
header = ctk.CTkFrame(app, fg_color=C_SURFACE, corner_radius=0, height=58)
header.pack(fill="x")
header.pack_propagate(False)

# Left: logo + title + version
title_left = ctk.CTkFrame(header, fg_color="transparent")
title_left.pack(side="left", padx=18, pady=10)
if LOGO_IMG_SM:
    ctk.CTkLabel(title_left, image=LOGO_IMG_SM, text="").pack(side="left", padx=(0, 10))
else:
    ctk.CTkFrame(title_left, fg_color=C_ACCENT, width=10, height=10,
                 corner_radius=5).pack(side="left", padx=(0, 10))
ctk.CTkLabel(title_left, text="AI Text to Speech Studio",
             font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"),
             text_color=C_TXT).pack(side="left")
ctk.CTkLabel(title_left, text=f" v{VERSION}",
             font=ctk.CTkFont(family="Segoe UI", size=11),
             text_color=C_TXT3).pack(side="left")

# Right: action buttons
ctk.CTkButton(header, text="About", command=show_about,
              width=72, height=30,
              font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_GHOST).pack(side="right", padx=(0, 14), pady=14)
ctk.CTkButton(header, text="Settings", command=show_settings,
              width=84, height=30,
              font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_GHOST).pack(side="right", padx=(0, 6), pady=14)
_activate_btn = ctk.CTkButton(
    header, text="🔑 Activate", command=lambda: _show_activation_modal(can_skip=True),
    width=98, height=30,
    font=ctk.CTkFont(family="Segoe UI", size=12),
    **BTN_GHOST)
# Only show when not yet activated
if not _lic.load_license().get("activated"):
    _activate_btn.pack(side="right", padx=(0, 6), pady=14)

_sep(app)

# ── Progress row ──────────────────────────────────────────────────────────────
prog_row = ctk.CTkFrame(app, fg_color=C_SURFACE, corner_radius=0, height=34)
prog_row.pack(fill="x")
prog_row.pack_propagate(False)
progress_bar = ctk.CTkProgressBar(prog_row, height=6, corner_radius=3)
progress_bar.set(0)
progress_bar.pack(side="left", fill="x", expand=True, padx=(16, 10), pady=14)
progress_time_label = ctk.CTkLabel(
    prog_row, text="", width=260, anchor="w",
    font=ctk.CTkFont(family="Segoe UI", size=12), text_color=C_TXT3)
progress_time_label.pack(side="left", padx=(0, 16))

smooth = SmoothProgress(progress_bar, progress_time_label)

_sep(app)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = ctk.CTkTabview(app, fg_color=C_BG)
tabs.pack(fill="both", expand=True, padx=0, pady=0)
tabs.add("  Studio  ")
tabs.add("  Queue  ")
tabs.add("  Dialogue  ")
tabs.add("  Profiles  ")

# ══════════════════════════════════════════════════════════════════════════════
# STUDIO TAB
# ══════════════════════════════════════════════════════════════════════════════
studio = tabs.tab("  Studio  ")
studio.configure(fg_color=C_BG)
studio.grid_columnconfigure(0, weight=3)
studio.grid_columnconfigure(1, minsize=242)
studio.grid_columnconfigure(2, minsize=248)
studio.grid_columnconfigure(3, minsize=224)
studio.grid_rowconfigure(0, weight=1)

# helper: make a panel card
def _panel(parent, col, padright=8):
    f = ctk.CTkFrame(parent, fg_color=C_CARD, corner_radius=12)
    f.grid(row=0, column=col, sticky="nsew", padx=(0, padright), pady=6)
    return f

# ── Text panel ────────────────────────────────────────────────────────────────
text_panel = _panel(studio, 0)

_section_label(text_panel, "TEXT INPUT", padx=14, pady=(12, 4))

text_input = ctk.CTkTextbox(
    text_panel,
    font=ctk.CTkFont(family="Segoe UI", size=13),
    wrap="word",
    fg_color=C_ELEVATED, border_width=0, corner_radius=8,
    text_color=C_TXT,
    scrollbar_button_color=C_BORDER,
    scrollbar_button_hover_color=C_ACCENT_D)
text_input.pack(fill="both", expand=True, padx=12, pady=(0, 6))
text_input.bind("<KeyRelease>", update_word_count)

def _on_paste(e=None):
    app.after(10, update_word_count)  # after paste content lands
    if _get_settings().get("auto_clean_text", False):
        def _clean():
            raw = text_input.get("1.0", "end").strip()
            cleaned, changes = clean_text(raw)
            if changes:
                text_input.delete("1.0", "end")
                text_input.insert("1.0", cleaned)
                update_word_count()
                status_label.configure(text=f"✅ Auto-cleaned: {', '.join(changes[:3])}")
        app.after(20, _clean)

text_input.bind("<<Paste>>", _on_paste)

word_count_label = ctk.CTkLabel(
    text_panel,
    text="Words: 0  ·  Chars: 0  ·  Audio: ~0s  ·  Processing: ~0s",
    font=ctk.CTkFont(family="Segoe UI", size=11), text_color=C_TXT3)
word_count_label.pack(anchor="w", padx=14, pady=(2, 8))

_sep(text_panel, pady=0)

txt_btns = ctk.CTkFrame(text_panel, fg_color="transparent")
txt_btns.pack(fill="x", padx=10, pady=8)
ctk.CTkButton(txt_btns, text="Import", command=import_file,
              width=76, height=30, font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_GHOST).pack(side="left", padx=(0, 5))
ctk.CTkButton(txt_btns, text="+ Queue", command=queue_add,
              width=76, height=30, font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_GHOST).pack(side="left", padx=(0, 5))
ctk.CTkButton(txt_btns, text="Clean", command=show_text_cleaner,
              width=66, height=30, font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_GHOST).pack(side="left", padx=(0, 5))
ctk.CTkButton(txt_btns, text="Dict",
              command=lambda: open_pronunciation_window(app),
              width=54, height=30, font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_GHOST).pack(side="left", padx=(0, 5))
ctk.CTkButton(txt_btns, text="Clear", command=clear_text,
              width=54, height=30, font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_DARK).pack(side="left")

# ── Voice + Engine panel ──────────────────────────────────────────────────────
mid_panel = _panel(studio, 1)

_section_label(mid_panel, "ENGINE")
engine_var = ctk.StringVar(value="⚡ Fast")
engine_toggle = ctk.CTkSegmentedButton(
    mid_panel, values=["⚡ Fast", "🎙️ Natural"],
    variable=engine_var,
    font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
    width=214)
engine_toggle.pack(padx=14, pady=(0, 12))

_sep(mid_panel)

# Container that swaps between Kokoro and Chatterbox settings
voice_container = ctk.CTkFrame(mid_panel, fg_color="transparent")
voice_container.pack(fill="x")

# ── Kokoro section ──
kokoro_frame = ctk.CTkFrame(voice_container, fg_color="transparent")
kokoro_frame.pack(fill="x")

_section_label(kokoro_frame, "VOICE", pady=(10, 4))
voice_var = ctk.StringVar(value="🇬🇧 Male - George (Best)")
ctk.CTkOptionMenu(kokoro_frame, variable=voice_var, values=list(VOICES.keys()),
                  width=214, dynamic_resizing=False,
                  font=ctk.CTkFont(family="Segoe UI", size=12)).pack(padx=14, pady=(0, 6))
preview_button = ctk.CTkButton(
    kokoro_frame, text="Preview Voice", command=preview_voice,
    width=214, height=30,
    font=ctk.CTkFont(family="Segoe UI", size=12), **BTN_GHOST)
preview_button.pack(padx=14, pady=(0, 8))

# ── Chatterbox section (hidden initially) ──
cb_frame = ctk.CTkFrame(voice_container, fg_color="transparent")

_section_label(cb_frame, "VOICE CLONE", pady=(10, 4))
cb_clone_path_var = ctk.StringVar(value="")

# ── Clone library dropdown ────────────────────────────────────────────────────
_CLONE_DEFAULT = "Default voice"

def _clone_display_names():
    return [_CLONE_DEFAULT] + [e["name"] for e in load_clone_library()]

cb_clone_var = ctk.StringVar(value=_CLONE_DEFAULT)
cb_clone_menu = ctk.CTkOptionMenu(
    cb_frame, variable=cb_clone_var,
    values=_clone_display_names(),
    font=ctk.CTkFont(family="Segoe UI", size=11),
    width=214, dynamic_resizing=False)
cb_clone_menu.pack(padx=14, pady=(0, 6))

def _apply_clone_selection(name=None):
    name = name or cb_clone_var.get()
    if name == _CLONE_DEFAULT:
        cb_clone_path_var.set("")
        status_label.configure(text="Voice clone: Default voice")
    else:
        lib = load_clone_library()
        entry = next((e for e in lib if e["name"] == name), None)
        if entry and os.path.exists(entry["file"]):
            cb_clone_path_var.set(entry["file"])
            status_label.configure(text=f"Voice clone: {name}")
        else:
            cb_clone_path_var.set("")
            status_label.configure(text="Clone file missing — using default voice.")
    s = _get_settings(); s["selected_clone"] = name; _save_settings(s)

cb_clone_var.trace_add("write", lambda *_: _apply_clone_selection())

def _refresh_clone_menu():
    names = _clone_display_names()
    cb_clone_menu.configure(values=names)
    if cb_clone_var.get() not in names:
        cb_clone_var.set(_CLONE_DEFAULT)

# ── Clone action buttons ──────────────────────────────────────────────────────
cb_clone_btns = ctk.CTkFrame(cb_frame, fg_color="transparent")
cb_clone_btns.pack(fill="x", padx=14, pady=(0, 6))

def _delete_clone():
    name = cb_clone_var.get()
    if name == _CLONE_DEFAULT:
        status_label.configure(text="⚠️ Default voice cannot be deleted.")
        return
    if not messagebox.askyesno("Delete Voice Clone", f"Delete '{name}'? This cannot be undone."):
        return
    lib = load_clone_library()
    entry = next((e for e in lib if e["name"] == name), None)
    if entry:
        try:
            os.remove(entry["file"])
        except Exception:
            pass
        lib = [e for e in lib if e["name"] != name]
        save_clone_library(lib)
    cb_clone_var.set(_CLONE_DEFAULT)
    _refresh_clone_menu()
    status_label.configure(text=f"Deleted clone: {name}")

def show_voice_recorder():
    SAMPLE_RATE = 44100
    CLONE_SCRIPT = (
        "The warm light of the setting sun spilled across the old stone bridge, "
        "painting the river gold. A gentle breeze carried the scent of pine through "
        "the quiet valley below."
    )

    win = ctk.CTkToplevel(app)
    win.title("Record Voice Clone")
    win.geometry("560x520")
    win.resizable(False, False)
    win.configure(fg_color=C_BG)
    win.grab_set()

    hdr = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=56)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    ctk.CTkLabel(hdr, text="Record Voice Clone",
                 font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
                 text_color=C_TXT).pack(side="left", padx=20)
    ctk.CTkLabel(hdr, text="Aim for 8–12 seconds",
                 font=ctk.CTkFont(family="Segoe UI", size=11),
                 text_color=C_TXT3).pack(side="right", padx=20)

    script_card = ctk.CTkFrame(win, fg_color=C_CARD, corner_radius=10)
    script_card.pack(fill="x", padx=20, pady=(16, 8))
    ctk.CTkLabel(script_card, text="Read this aloud",
                 font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
                 text_color=C_ACCENT, anchor="w").pack(anchor="w", padx=14, pady=(12, 4))
    ctk.CTkLabel(script_card, text=CLONE_SCRIPT,
                 font=ctk.CTkFont(family="Segoe UI", size=13), wraplength=488,
                 text_color=C_TXT, justify="left", anchor="w").pack(fill="x", padx=14, pady=(0, 14))

    tips_frame = ctk.CTkFrame(win, fg_color="transparent")
    tips_frame.pack(fill="x", padx=20, pady=(0, 6))
    for tip in ("Speak naturally — same tone you want cloned",
                "Quiet room, no background noise",
                "15–30 cm from the microphone"):
        ctk.CTkLabel(tips_frame, text=f"  {tip}",
                     font=ctk.CTkFont(family="Segoe UI", size=10),
                     text_color=C_TXT3, anchor="w").pack(anchor="w", pady=1)

    # Name field
    name_row = ctk.CTkFrame(win, fg_color="transparent")
    name_row.pack(fill="x", padx=20, pady=(4, 6))
    ctk.CTkLabel(name_row, text="Name:",
                 font=ctk.CTkFont(family="Segoe UI", size=11),
                 text_color=C_TXT2, width=46, anchor="w").pack(side="left")
    name_entry = ctk.CTkEntry(name_row, placeholder_text="e.g. My Voice",
                               font=ctk.CTkFont(family="Segoe UI", size=11),
                               fg_color=C_ELEVATED, border_color=C_BORDER,
                               text_color=C_TXT, height=28)
    name_entry.pack(side="left", fill="x", expand=True)

    ctrl = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=10)
    ctrl.pack(fill="x", padx=20, pady=(0, 8))

    timer_lbl = ctk.CTkLabel(ctrl, text="0.0s", width=52,
                              font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
                              text_color=C_TXT3)
    timer_lbl.pack(side="left", padx=(14, 8), pady=12)

    record_btn = ctk.CTkButton(ctrl, text="⏺  Record", width=120, height=36,
                                font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                                fg_color=C_REC, hover_color="#a52020", corner_radius=8)
    record_btn.pack(side="left", padx=(0, 6), pady=12)

    play_btn = ctk.CTkButton(ctrl, text="▶  Preview", width=100, height=36,
                              font=ctk.CTkFont(family="Segoe UI", size=12),
                              **BTN_GHOST, corner_radius=8, state="disabled")
    play_btn.pack(side="left", padx=(0, 6), pady=12)

    save_btn = ctk.CTkButton(ctrl, text="Save to Library", width=120, height=36,
                              font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                              corner_radius=8, state="disabled")
    save_btn.pack(side="left", pady=12)

    rec_status = ctk.CTkLabel(win, text="Press Record when you're ready.",
                               font=ctk.CTkFont(family="Segoe UI", size=11),
                               text_color=C_TXT3)
    rec_status.pack(pady=(4, 0))

    _recording = threading.Event()
    _chunks    = []
    _samples   = [None]
    _stream    = [None]
    _start     = [0.0]
    _tmp_path  = [None]

    def _tick():
        if _recording.is_set():
            t = time.time() - _start[0]
            timer_lbl.configure(text=f"{t:.1f}s",
                                 text_color=C_REC if t <= 15 else C_WARN)
            win.after(100, _tick)

    def _start_rec():
        _chunks.clear()
        _recording.set()
        _start[0] = time.time()
        record_btn.configure(text="⏹  Stop")
        play_btn.configure(state="disabled")
        save_btn.configure(state="disabled")
        rec_status.configure(text="Recording...", text_color=C_REC)
        _tick()
        def _cb(indata, frames, time_info, st):
            if _recording.is_set():
                _chunks.append(indata.copy())
        s = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                           dtype="float32", callback=_cb)
        s.start()
        _stream[0] = s

    def _stop_rec():
        _recording.clear()
        if _stream[0]:
            _stream[0].stop(); _stream[0].close(); _stream[0] = None
        timer_lbl.configure(text_color=C_TXT3)
        record_btn.configure(text="⏺  Record Again")
        if not _chunks:
            rec_status.configure(text="Nothing recorded.", text_color=C_WARN)
            return
        _samples[0] = np.concatenate(_chunks).flatten()
        dur = len(_samples[0]) / SAMPLE_RATE
        peak = float(np.abs(_samples[0]).max()) if len(_samples[0]) else 0.0
        if dur < 3:
            rec_status.configure(text=f"Only {dur:.1f}s — try for at least 5s.", text_color=C_WARN)
            play_btn.configure(state="normal")
        elif peak < 0.01:
            rec_status.configure(
                text=f"Recording is too quiet (level: {peak:.4f}) — check your mic gain in Windows Sound settings and re-record.",
                text_color=C_WARN)
            play_btn.configure(state="normal")
        else:
            # Write to a temp file so preview works before library save
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            sf.write(tmp.name, _samples[0], SAMPLE_RATE)
            _tmp_path[0] = tmp.name
            rec_status.configure(text=f"{dur:.1f}s recorded — name it and save to library.",
                                  text_color=C_SUCCESS)
            play_btn.configure(state="normal")
            save_btn.configure(state="normal")

    def _toggle():
        if _recording.is_set(): _stop_rec()
        else: _start_rec()

    def _preview():
        if _samples[0] is not None:
            threading.Thread(
                target=lambda: (sd.stop(), sd.play(_samples[0], SAMPLE_RATE), sd.wait()),
                daemon=True).start()

    def _save_to_library():
        if _tmp_path[0] is None or not os.path.exists(_tmp_path[0]):
            return
        name = name_entry.get().strip() or "Recorded Voice"
        # Ensure unique name
        lib = load_clone_library()
        existing = {e["name"] for e in lib}
        base, n = name, 2
        while name in existing:
            name = f"{base} ({n})"; n += 1
        add_clone_to_library(name, _tmp_path[0])
        try:
            os.remove(_tmp_path[0])
        except Exception:
            pass
        _tmp_path[0] = None
        _refresh_clone_menu()
        cb_clone_var.set(name)
        status_label.configure(text=f"Saved voice clone: {name}")
        win.destroy()

    def _on_close():
        _recording.clear()
        if _stream[0]:
            _stream[0].stop(); _stream[0].close()
        if _tmp_path[0]:
            try: os.remove(_tmp_path[0])
            except Exception: pass
        sd.stop()
        win.destroy()

    record_btn.configure(command=_toggle)
    play_btn.configure(command=_preview)
    save_btn.configure(command=_save_to_library)
    win.protocol("WM_DELETE_WINDOW", _on_close)

def _browse_and_add_clone():
    fp = filedialog.askopenfilename(
        title="Select voice sample (5–30s WAV/MP3)",
        filetypes=[("Audio files", "*.wav *.mp3")])
    if not fp:
        return
    # Ask for a name via a simple dialog
    name_win = ctk.CTkToplevel(app)
    name_win.title("Name this voice clone")
    name_win.geometry("340x130")
    name_win.resizable(False, False)
    name_win.configure(fg_color=C_BG)
    name_win.grab_set()
    ctk.CTkLabel(name_win, text="Name for this voice clone:",
                 font=ctk.CTkFont(family="Segoe UI", size=12),
                 text_color=C_TXT).pack(padx=20, pady=(18, 6), anchor="w")
    entry = ctk.CTkEntry(name_win, font=ctk.CTkFont(family="Segoe UI", size=12),
                         fg_color=C_ELEVATED, border_color=C_BORDER,
                         text_color=C_TXT, height=32, width=300)
    entry.insert(0, os.path.splitext(os.path.basename(fp))[0])
    entry.pack(padx=20)
    def _confirm():
        name = entry.get().strip() or os.path.splitext(os.path.basename(fp))[0]
        lib = load_clone_library()
        existing = {e["name"] for e in lib}
        base, n = name, 2
        while name in existing:
            name = f"{base} ({n})"; n += 1
        add_clone_to_library(name, fp)
        _refresh_clone_menu()
        cb_clone_var.set(name)
        status_label.configure(text=f"Added voice clone: {name}")
        name_win.destroy()
    ctk.CTkButton(name_win, text="Add to Library", command=_confirm,
                  height=30, font=ctk.CTkFont(family="Segoe UI", size=12)).pack(pady=10)
    name_win.bind("<Return>", lambda _: _confirm())

ctk.CTkButton(cb_clone_btns, text="Record New", command=show_voice_recorder,
              width=96, height=28, font=ctk.CTkFont(family="Segoe UI", size=11),
              fg_color=C_ACCENT_D, hover_color=C_ACCENT, text_color=C_TXT).pack(side="left", padx=(0, 5))
ctk.CTkButton(cb_clone_btns, text="Browse", command=_browse_and_add_clone,
              width=76, height=28, font=ctk.CTkFont(family="Segoe UI", size=11),
              **BTN_GHOST).pack(side="left", padx=(0, 5))
ctk.CTkButton(cb_clone_btns, text="Delete", command=_delete_clone,
              width=62, height=28, font=ctk.CTkFont(family="Segoe UI", size=11),
              fg_color="transparent", hover_color="#3d1515",
              text_color=C_DANGER, border_width=1, border_color="#3d1515").pack(side="left")

def make_slider(parent, label, from_, to, steps, default, width=214):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=14, pady=(4, 2))
    ctk.CTkLabel(row, text=label,
                 font=ctk.CTkFont(family="Segoe UI", size=11),
                 text_color=C_TXT2, anchor="w").pack(side="left")
    val = ctk.CTkLabel(row, text="",
                       font=ctk.CTkFont(family="Segoe UI", size=11),
                       text_color=C_ACCENT, width=58, anchor="e")
    val.pack(side="right")
    s = ctk.CTkSlider(parent, from_=from_, to=to, number_of_steps=steps,
                      command=lambda _: update_all_labels(),
                      width=width, height=14)
    s.set(default)
    s.pack(padx=14, pady=(0, 8))
    return s, val

_section_label(cb_frame, "PARAMETERS", pady=(10, 4))
cb_exag_slider, cb_exag_label = make_slider(cb_frame, "Exaggeration", 0.0, 1.0, 20, 0.5)
cb_cfg_slider,  cb_cfg_label  = make_slider(cb_frame, "CFG Weight",   0.0, 1.0, 20, 0.0)

# Speed (shared)
_sep(mid_panel)
_section_label(mid_panel, "SPEED", pady=(8, 4))
speed_slider, speed_label = make_slider(mid_panel, "Playback Speed", 0.5, 2.0, 30, 0.85)

# Engine switch logic
def _on_engine_change(*_):
    if engine_var.get() == "🎙️ Natural":
        kokoro_frame.pack_forget()
        cb_frame.pack(fill="x")
        if not chatterbox_engine.is_ready:
            threading.Thread(target=_load_chatterbox_bg, daemon=True).start()
    else:
        cb_frame.pack_forget()
        kokoro_frame.pack(fill="x")
        if chatterbox_engine.is_ready:
            status_label.configure(text="⚡ Fast mode active — Natural mode unloaded.")
            threading.Thread(target=chatterbox_engine.stop, daemon=True).start()

engine_var.trace_add("write", _on_engine_change)

# Generate / Stop
_sep(mid_panel)
gen_btns = ctk.CTkFrame(mid_panel, fg_color="transparent")
gen_btns.pack(fill="x", padx=14, pady=10)

play_button = ctk.CTkButton(
    gen_btns, text="▶  Generate", command=generate_and_store,
    width=214, height=46,
    font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
    corner_radius=10)
play_button.pack(pady=(0, 6))

stop_button = ctk.CTkButton(
    gen_btns, text="Stop", command=stop_audio,
    width=214, height=30,
    font=ctk.CTkFont(family="Segoe UI", size=12),
    state="disabled", **BTN_GHOST, corner_radius=8)
stop_button.pack()

ctk.CTkLabel(mid_panel, text="Ctrl+Enter  ·  Esc to stop",
             font=ctk.CTkFont(family="Segoe UI", size=10),
             text_color=C_TXT3).pack(pady=(4, 8))

# ── Enhancement panel ─────────────────────────────────────────────────────────
enh_panel = _panel(studio, 2)

_section_label(enh_panel, "AUDIO FX")

eq_preset_var = ctk.StringVar(value="Custom")
eq_preset_menu = ctk.CTkOptionMenu(
    enh_panel, variable=eq_preset_var,
    values=list(EQ_PRESETS.keys()),
    font=ctk.CTkFont(family="Segoe UI", size=11),
    width=214, dynamic_resizing=False)
eq_preset_menu.pack(padx=14, pady=(0, 8))
eq_preset_var.trace_add("write", lambda *_: apply_eq_preset())

checks = ctk.CTkFrame(enh_panel, fg_color="transparent")
checks.pack(fill="x", padx=14, pady=(0, 8))
noise_gate_var = ctk.BooleanVar(value=False)
ctk.CTkCheckBox(checks, text="Noise Gate", variable=noise_gate_var,
                font=ctk.CTkFont(family="Segoe UI", size=12)).pack(anchor="w", pady=5)
trim_var = ctk.BooleanVar(value=True)
ctk.CTkCheckBox(checks, text="Trim Silence", variable=trim_var,
                font=ctk.CTkFont(family="Segoe UI", size=12)).pack(anchor="w", pady=5)
compressor_var = ctk.BooleanVar(value=True)
ctk.CTkCheckBox(checks, text="Compressor", variable=compressor_var,
                font=ctk.CTkFont(family="Segoe UI", size=12)).pack(anchor="w", pady=5)

compressor_slider, comp_label   = make_slider(enh_panel, "Comp Ratio", 1.0,   8.0,  14, 2.0)
highpass_slider,   hp_label     = make_slider(enh_panel, "High Pass",   20,   500,  48,  20)
lowpass_slider,    lp_label     = make_slider(enh_panel, "Low Pass",  4000, 18000,  56, 18000)
reverb_slider,     reverb_label = make_slider(enh_panel, "Reverb",    0.0,   1.0,  20, 0.0)
gain_slider,       gain_label   = make_slider(enh_panel, "Gain",       -12,    12,  24,   0)

def _on_eq_manual_change(_=None):
    if not _applying_eq_preset:
        eq_preset_var.set("Custom")
    update_all_labels()

for _eq_s in (compressor_slider, highpass_slider, lowpass_slider, reverb_slider, gain_slider):
    _eq_s.configure(command=_on_eq_manual_change)

noise_gate_var.trace_add("write",  lambda *_: eq_preset_var.set("Custom") if not _applying_eq_preset else None)
trim_var.trace_add("write",        lambda *_: eq_preset_var.set("Custom") if not _applying_eq_preset else None)
compressor_var.trace_add("write",  lambda *_: eq_preset_var.set("Custom") if not _applying_eq_preset else None)

ctk.CTkButton(enh_panel, text="Reset to defaults", command=reset_enhancements,
              width=214, height=28,
              font=ctk.CTkFont(family="Segoe UI", size=11),
              **BTN_GHOST).pack(padx=14, pady=(0, 6))


# ── Audio History panel ───────────────────────────────────────────────────────
hist_panel = _panel(studio, 3, padright=0)

hist_header = ctk.CTkFrame(hist_panel, fg_color="transparent")
hist_header.pack(fill="x", padx=14, pady=(10, 4))
_hist_left = ctk.CTkFrame(hist_header, fg_color="transparent")
_hist_left.pack(side="left")
ctk.CTkFrame(_hist_left, fg_color=C_ACCENT, width=3, height=10,
             corner_radius=2).pack(side="left", padx=(0, 6))
ctk.CTkLabel(_hist_left, text="HISTORY",
             font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
             text_color=C_TXT2).pack(side="left")
ctk.CTkLabel(hist_header, text=f"last {MAX_HISTORY}",
             font=ctk.CTkFont(family="Segoe UI", size=11),
             text_color=C_TXT3).pack(side="right")

_sep(hist_panel)

history_scroll = ctk.CTkScrollableFrame(
    hist_panel, fg_color=C_CARD,
    scrollbar_button_color=C_BORDER,
    scrollbar_button_hover_color=C_ACCENT_D)
history_scroll.pack(fill="both", expand=True, padx=4, pady=(4, 6))
history_inner = history_scroll

ctk.CTkLabel(history_inner, text="No audio yet.",
             text_color=C_TXT3, font=ctk.CTkFont(family="Segoe UI", size=12)).pack(pady=30)

# ══════════════════════════════════════════════════════════════════════════════
# QUEUE TAB
# ══════════════════════════════════════════════════════════════════════════════
queue_tab = tabs.tab("  Queue  ")
queue_tab.configure(fg_color=C_BG)

q_card = ctk.CTkFrame(queue_tab, fg_color=C_CARD, corner_radius=12)
q_card.pack(fill="both", expand=True, padx=6, pady=6)

q_top = ctk.CTkFrame(q_card, fg_color="transparent")
q_top.pack(fill="x", padx=14, pady=(12, 6))
_q_left = ctk.CTkFrame(q_top, fg_color="transparent")
_q_left.pack(side="left")
ctk.CTkFrame(_q_left, fg_color=C_ACCENT, width=3, height=10,
             corner_radius=2).pack(side="left", padx=(0, 6))
ctk.CTkLabel(_q_left, text="BATCH QUEUE",
             font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
             text_color=C_TXT2).pack(side="left")
queue_estimate_label = ctk.CTkLabel(q_top, text="Queue is empty",
                                     font=ctk.CTkFont(family="Segoe UI", size=11),
                                     text_color=C_TXT2)
queue_estimate_label.pack(side="right")

_sep(q_card)

queue_listbox = tk.Listbox(
    q_card,
    font=("Consolas", 11),
    bg=C_ELEVATED, fg=C_TXT,
    selectbackground=C_ACCENT_D, selectforeground=C_TXT,
    borderwidth=0, highlightthickness=0,
    activestyle="none",
    relief="flat")
queue_listbox.pack(fill="both", expand=True, padx=12, pady=(8, 4))

_sep(q_card)

q_btns = ctk.CTkFrame(q_card, fg_color="transparent")
q_btns.pack(fill="x", padx=12, pady=10)
queue_gen_btn = ctk.CTkButton(
    q_btns, text="Generate All & Save",
    command=queue_generate_all, width=200, height=36,
    font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"))
queue_gen_btn.pack(side="left", padx=(0, 8))
ctk.CTkButton(q_btns, text="Remove Selected", command=queue_remove,
              width=140, height=36,
              font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_GHOST).pack(side="left", padx=(0, 6))
ctk.CTkButton(q_btns, text="Clear All", command=queue_clear,
              width=90, height=36,
              font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_DARK).pack(side="left")

# ══════════════════════════════════════════════════════════════════════════════
# PROFILES TAB
# ══════════════════════════════════════════════════════════════════════════════
prof_tab = tabs.tab("  Profiles  ")
prof_tab.configure(fg_color=C_BG)

p_outer = ctk.CTkFrame(prof_tab, fg_color="transparent")
p_outer.pack(fill="both", expand=True, padx=6, pady=6)

# Load card
load_card = ctk.CTkFrame(p_outer, fg_color=C_CARD, corner_radius=12)
load_card.pack(fill="x", pady=(0, 10))

_section_label(load_card, "LOAD PROFILE")
_sep(load_card)

load_row = ctk.CTkFrame(load_card, fg_color="transparent")
load_row.pack(fill="x", padx=14, pady=14)
profile_var = ctk.StringVar(value="🌙 Calm Narrator")
profile_menu = ctk.CTkOptionMenu(
    load_row, variable=profile_var,
    values=list(load_profiles().keys()), width=260,
    font=ctk.CTkFont(family="Segoe UI", size=12),
    dynamic_resizing=False)
profile_menu.pack(side="left", padx=(0, 8))
ctk.CTkButton(load_row, text="Load", command=load_profile,
              width=80, height=34,
              font=ctk.CTkFont(family="Segoe UI", size=12)).pack(side="left", padx=(0, 6))
ctk.CTkButton(load_row, text="Delete", command=delete_profile,
              width=80, height=34,
              font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_DANGER).pack(side="left")

# Save card
save_card = ctk.CTkFrame(p_outer, fg_color=C_CARD, corner_radius=12)
save_card.pack(fill="x", pady=(0, 10))

_section_label(save_card, "SAVE CURRENT SETTINGS")
_sep(save_card)

save_row = ctk.CTkFrame(save_card, fg_color="transparent")
save_row.pack(fill="x", padx=14, pady=14)
profile_name_entry = ctk.CTkEntry(
    save_row, width=260,
    placeholder_text="Profile name...",
    font=ctk.CTkFont(family="Segoe UI", size=12))
profile_name_entry.pack(side="left", padx=(0, 8))
profile_name_entry.bind("<Return>", lambda e: save_profile())
ctk.CTkButton(save_row, text="Save", command=save_profile,
              width=80, height=34,
              font=ctk.CTkFont(family="Segoe UI", size=12)).pack(side="left")

# Tips card
tip_card = ctk.CTkFrame(p_outer, fg_color=C_CARD, corner_radius=12)
tip_card.pack(fill="x")
ctk.CTkLabel(tip_card,
             text="The Calm Narrator preset is optimized for science/documentary narration.\n"
                  "George (Best) at 0.85× speed with subtle compression produces the cleanest results.",
             font=ctk.CTkFont(family="Segoe UI", size=11), text_color=C_TXT2,
             justify="left", anchor="w").pack(anchor="w", padx=14, pady=14)

# ══════════════════════════════════════════════════════════════════════════════
# DIALOGUE TAB
# ══════════════════════════════════════════════════════════════════════════════
dlg_tab = tabs.tab("  Dialogue  ")
dlg_tab.configure(fg_color=C_BG)
dlg_tab.grid_columnconfigure(0, weight=3)
dlg_tab.grid_columnconfigure(1, minsize=290)
dlg_tab.grid_rowconfigure(0, weight=1)

# ── Script panel ──────────────────────────────────────────────────────────────
dlg_script_panel = ctk.CTkFrame(dlg_tab, fg_color=C_CARD, corner_radius=12)
dlg_script_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=6)

_section_label(dlg_script_panel, "DIALOGUE SCRIPT", padx=14, pady=(12, 2))
ctk.CTkLabel(dlg_script_panel,
             text="One cue per line  ·  SPEAKER (all caps) followed by a colon, then the spoken text",
             font=ctk.CTkFont(family="Segoe UI", size=10),
             text_color=C_TXT3, anchor="w").pack(anchor="w", padx=14, pady=(0, 8))

dlg_text = ctk.CTkTextbox(
    dlg_script_panel,
    font=ctk.CTkFont(family="Consolas", size=12),
    wrap="word",
    fg_color=C_ELEVATED, border_width=0, corner_radius=8,
    text_color=C_TXT)
dlg_text.pack(fill="both", expand=True, padx=12, pady=(0, 6))
dlg_text.insert("1.0",
    "NARRATOR: In the beginning, there was silence.\n"
    "ALICE: But silence never lasts forever.\n"
    "NARRATOR: And so the story began.")

_sep(dlg_script_panel, pady=0)
dlg_txt_btns = ctk.CTkFrame(dlg_script_panel, fg_color="transparent")
dlg_txt_btns.pack(fill="x", padx=10, pady=8)

def dlg_import_file():
    folder = get_default_folder()
    fp = filedialog.askopenfilename(
        initialdir=folder or None,
        filetypes=[("Text files", "*.txt")]
    )
    if fp:
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                with open(fp, "r", encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            status_label.configure(text="❌ Could not decode file — unknown encoding.")
            return
        dlg_text.delete("1.0", "end")
        dlg_text.insert("1.0", content)
        status_label.configure(text=f"✅ Imported {len(content):,} characters.")

ctk.CTkButton(dlg_txt_btns, text="Import", command=dlg_import_file,
              width=76, height=30, font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_GHOST).pack(side="left", padx=(0, 5))
ctk.CTkButton(dlg_txt_btns, text="Clear",
              command=lambda: (dlg_text.delete("1.0", "end")),
              width=60, height=30, font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_DARK).pack(side="left")

# ── Speakers panel ────────────────────────────────────────────────────────────
dlg_right = ctk.CTkFrame(dlg_tab, fg_color=C_CARD, corner_radius=12)
dlg_right.grid(row=0, column=1, sticky="nsew", padx=0, pady=6)

_section_label(dlg_right, "SPEAKERS")

dlg_detect_btn = ctk.CTkButton(dlg_right, text="Detect Speakers",
              width=256, height=30,
              font=ctk.CTkFont(family="Segoe UI", size=12),
              **BTN_GHOST)
dlg_detect_btn.pack(padx=16, pady=(0, 8))

_sep(dlg_right)

dlg_speakers_scroll = ctk.CTkScrollableFrame(
    dlg_right, fg_color="transparent",
    scrollbar_button_color=C_BORDER,
    scrollbar_button_hover_color=C_ACCENT_D)
dlg_speakers_scroll.pack(fill="both", expand=True, padx=6, pady=6)

_sep(dlg_right)

dlg_gen_frame = ctk.CTkFrame(dlg_right, fg_color="transparent")
dlg_gen_frame.pack(fill="x", padx=16, pady=10)

dlg_gen_btn = ctk.CTkButton(
    dlg_gen_frame, text="Generate Dialogue",
    width=256, height=46,
    font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
    corner_radius=10)
dlg_gen_btn.pack(pady=(0, 6))

ctk.CTkLabel(dlg_right,
             text="Fast mode (Kokoro) · output appears in Studio history",
             font=ctk.CTkFont(family="Segoe UI", size=10),
             text_color=C_TXT3).pack(pady=(0, 10))

# ── Dialogue logic ────────────────────────────────────────────────────────────
dlg_speaker_vars = {}   # speaker_name → StringVar (voice display name)

def dlg_detect_speakers():
    text    = dlg_text.get("1.0", "end").strip()
    d_lines = parse_dialogue(text)
    speakers = []
    for sp, _ in d_lines:
        if sp not in speakers:
            speakers.append(sp)

    for w in dlg_speakers_scroll.winfo_children():
        w.destroy()
    dlg_speaker_vars.clear()

    if not speakers:
        ctk.CTkLabel(dlg_speakers_scroll,
                     text="No speakers found.\nFormat: SPEAKER: text",
                     text_color=C_TXT3,
                     font=ctk.CTkFont(family="Segoe UI", size=11),
                     justify="center").pack(pady=24)
        return

    voice_list = list(VOICES.keys())
    for i, speaker in enumerate(speakers):
        row = ctk.CTkFrame(dlg_speakers_scroll, fg_color=C_ELEVATED, corner_radius=8)
        row.pack(fill="x", padx=2, pady=(0, 6))
        ctk.CTkLabel(row, text=speaker, width=88, anchor="w",
                     font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                     text_color=C_TXT).pack(side="left", padx=(10, 4), pady=8)
        var = ctk.StringVar(value=voice_list[i % len(voice_list)])
        ctk.CTkOptionMenu(row, variable=var, values=voice_list,
                          width=150, dynamic_resizing=False,
                          font=ctk.CTkFont(family="Segoe UI", size=11)
                          ).pack(side="left", padx=(0, 8), pady=8)
        dlg_speaker_vars[speaker] = var

    status_label.configure(text=f"✅ {len(speakers)} speaker(s) detected. Assign voices, then Generate.")

def dlg_generate():
    text    = dlg_text.get("1.0", "end").strip()
    d_lines = parse_dialogue(text)
    if not d_lines:
        status_label.configure(text="⚠️ No dialogue lines detected. Format: SPEAKER: text")
        return

    # Auto-detect if speaker vars not populated
    if not dlg_speaker_vars:
        dlg_detect_speakers()

    speaker_voices = {sp: var.get() for sp, var in dlg_speaker_vars.items()}
    speed          = round(speed_slider.get(), 2)
    est            = estimate_processing_time(" ".join(t for _, t in d_lines))

    dlg_gen_btn.configure(state="disabled", text="⏳ Generating...")
    dlg_detect_btn.configure(state="disabled")
    smooth.start(est)

    def run():
        global is_generating
        is_generating = True
        try:
            samples, sr, segments = generate_dialogue_audio(
                d_lines, speaker_voices, speed,
                status_cb=lambda m: status_label.configure(text=m)
            )
            smooth.finish()
            label = "Dialogue: " + " · ".join(dlg_speaker_vars.keys())
            add_to_history(samples, sr, text, label, segments=segments)
            status_label.configure(text="✅ Dialogue ready! View in Studio → History panel.")
        except Exception as e:
            _log_crash(e)
            status_label.configure(text=f"❌ {_fmt_err(e)}")
            smooth.finish()
        finally:
            is_generating = False
            dlg_gen_btn.configure(state="normal", text="Generate Dialogue")
            dlg_detect_btn.configure(state="normal")

    threading.Thread(target=run, daemon=True).start()

# Wire up buttons now that the functions exist
dlg_detect_btn.configure(command=dlg_detect_speakers)
dlg_gen_btn.configure(command=dlg_generate)

# ── Init ──────────────────────────────────────────────────────────────────────
_startup_settings = _get_settings()
ctk.set_appearance_mode(_startup_settings.get("theme", "dark"))

# Apply saved default profile (voice, speed, enhancements)
_default_profile = _startup_settings.get("default_profile", "")
if _default_profile and _default_profile in load_profiles():
    apply_settings(load_profiles()[_default_profile])

update_all_labels()

# Preload last-used voice clone
_saved_clone = _startup_settings.get("selected_clone", _CLONE_DEFAULT)
if _saved_clone and _saved_clone != _CLONE_DEFAULT:
    _lib = load_clone_library()
    _match = next((e for e in _lib if e["name"] == _saved_clone), None)
    if _match and os.path.exists(_match["file"]):
        cb_clone_var.set(_saved_clone)
        status_label.configure(text="Ready  ·  Ctrl+Enter to generate  ·  Ctrl+P to play  ·  Esc to stop")
    # if not found (deleted), dropdown stays at Default

def _load_chatterbox_bg():
    """Load Chatterbox in a background thread; called when user switches to Natural mode."""
    def _st(msg):
        app.after(0, lambda m=msg: status_label.configure(text=m))
    app.after(0, lambda: play_button.configure(state="disabled"))
    try:
        _st("🎙️ Loading Natural mode — this may take a few minutes...")
        app.after(0, lambda: engine_toggle.configure(state="disabled"))
        chatterbox_engine.start(status_cb=_st)
        app.after(0, lambda: engine_toggle.configure(state="normal"))
        app.after(0, lambda: play_button.configure(state="normal"))
        _st("✅ Natural mode ready  ·  Ctrl+Enter to generate")
    except Exception as e:
        _log_crash(e)
        err = _fmt_err(e)
        app.after(0, lambda: engine_var.set("⚡ Fast"))   # revert the toggle
        app.after(0, lambda: engine_toggle.configure(state="normal"))
        app.after(0, lambda: play_button.configure(state="normal"))
        _st(f"❌ Natural mode failed: {err}")

# ── License activation modal ──────────────────────────────────────────────────
def _show_activation_modal(can_skip=True, remaining=0):
    """
    Show the license key activation dialog.
    can_skip  → whether the user can dismiss without activating.
    remaining → launches left in grace period (for the countdown message).
    """

    win = ctk.CTkToplevel(app)
    win.title("Activate TTS Studio")
    win.geometry("460x310")
    win.resizable(False, False)
    win.configure(fg_color=C_BG)
    win.grab_set()
    win.transient(app)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=54)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    hdr_inner = ctk.CTkFrame(hdr, fg_color="transparent")
    hdr_inner.pack(side="left", padx=16, pady=10)
    ctk.CTkFrame(hdr_inner, fg_color=C_ACCENT, width=4, height=20,
                 corner_radius=2).pack(side="left", padx=(0, 10))
    ctk.CTkLabel(hdr_inner, text="Activate TTS Studio",
                 font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
                 text_color=C_TXT).pack(side="left")
    ctk.CTkFrame(win, fg_color=C_BORDER, height=1, corner_radius=0).pack(fill="x")

    # ── Body ──────────────────────────────────────────────────────────────────
    body = ctk.CTkFrame(win, fg_color="transparent")
    body.pack(fill="both", expand=True, padx=24, pady=(16, 8))

    if can_skip and remaining > 0:
        note = f"{remaining} free launch{'es' if remaining != 1 else ''} left after this one."
        ctk.CTkLabel(body, text=note,
                     font=ctk.CTkFont(family="Segoe UI", size=11),
                     text_color=C_WARN).pack(anchor="w", pady=(0, 10))
    elif not can_skip:
        ctk.CTkLabel(body, text="Your free trial has ended. Enter your license key to continue.",
                     font=ctk.CTkFont(family="Segoe UI", size=11),
                     text_color=C_WARN).pack(anchor="w", pady=(0, 10))

    ctk.CTkLabel(body, text="License Key",
                 font=ctk.CTkFont(family="Segoe UI", size=12),
                 text_color=C_TXT2).pack(anchor="w")
    key_entry = ctk.CTkEntry(
        body, width=410, height=36,
        placeholder_text="XXXX-XXXX-XXXX-XXXX",
        font=ctk.CTkFont(family="Segoe UI", size=13))
    key_entry.pack(pady=(4, 8))

    # Pre-fill if there's a saved (unactivated) key
    saved_key = _lic.load_license().get("key") or ""
    if saved_key:
        key_entry.insert(0, saved_key)

    msg_label = ctk.CTkLabel(body, text="",
                             font=ctk.CTkFont(family="Segoe UI", size=11),
                             text_color=C_DANGER)
    msg_label.pack(anchor="w")

    # ── Buttons ───────────────────────────────────────────────────────────────
    btn_row = ctk.CTkFrame(win, fg_color=C_SURFACE, corner_radius=0, height=52)
    btn_row.pack(fill="x", side="bottom")
    btn_row.pack_propagate(False)
    ctk.CTkFrame(btn_row, fg_color=C_BORDER, height=1,
                 corner_radius=0).pack(fill="x", side="top")

    def _do_activate():
        activate_btn.configure(state="disabled", text="Activating…")
        msg_label.configure(text="", text_color=C_DANGER)
        key = key_entry.get().strip()

        def _run():
            ok, msg = _lic.activate_license(key)
            def _done():
                if ok:
                    msg_label.configure(text=f"✅ {msg}", text_color=C_SUCCESS)
                    activate_btn.configure(text="Activate")
                    _activate_btn.pack_forget()   # hide header button
                    app.after(900, win.destroy)
                else:
                    msg_label.configure(text=f"❌ {msg}", text_color=C_DANGER)
                    activate_btn.configure(state="normal", text="Activate")
            app.after(0, _done)
        threading.Thread(target=_run, daemon=True).start()

    activate_btn = ctk.CTkButton(
        btn_row, text="Activate", width=110, height=34,
        font=ctk.CTkFont(family="Segoe UI", size=13),
        fg_color=C_ACCENT, hover_color=C_ACCENT_H, text_color="#000000",
        command=_do_activate)
    activate_btn.pack(side="left", padx=14, pady=9)
    key_entry.bind("<Return>", lambda _e: _do_activate())

    def _open_store():
        import webbrowser
        webbrowser.open(_lic.STORE_URL)

    ctk.CTkButton(
        btn_row, text="Buy a License", width=110, height=34,
        font=ctk.CTkFont(family="Segoe UI", size=12),
        **BTN_GHOST, command=_open_store
    ).pack(side="left", padx=(0, 6), pady=9)

    if can_skip:
        ctk.CTkButton(
            btn_row, text="Later", width=72, height=34,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            **BTN_GHOST, command=win.destroy
        ).pack(side="right", padx=14, pady=9)
    else:
        ctk.CTkButton(
            btn_row, text="Exit App", width=80, height=34,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="transparent", hover_color="#3d1515",
            text_color=C_DANGER, border_width=1, border_color="#3d1515",
            command=app.destroy
        ).pack(side="right", padx=14, pady=9)


def _on_splash_done():
    app.deiconify()
    app.state("zoomed")   # maximize (fullscreen with taskbar visible)
    _handle_license_on_startup()


def _handle_license_on_startup():
    """Check license state and show banner or blocking modal as appropriate."""
    lic = _lic.load_license()

    # Already activated — silently re-validate in background once per session
    if lic.get("activated"):
        def _revalidate():
            valid = _lic.validate_license_silent(lic.get("key"), lic.get("instance_id"))
            if not valid:
                app.after(0, lambda: status_label.configure(
                    text="⚠️ License validation failed. Please re-activate via the 🔑 Activate button."))
                app.after(0, lambda: _activate_btn.pack(side="right", padx=(0, 6), pady=14))
        threading.Thread(target=_revalidate, daemon=True).start()
        return

    state, remaining = _lic.check_startup()

    if state == "grace":
        pl = "launch" if remaining == 1 else "launches"
        status_label.configure(
            text=f"🔑 {remaining} free {pl} remaining — click 🔑 Activate in the toolbar to unlock.")
        # Also allow opening the modal from the status bar message click (optional)
        # The header button already handles this.
    elif state == "required":
        # Slight delay so the window is fully rendered before the modal appears
        app.after(300, lambda: _show_activation_modal(can_skip=False, remaining=0))

# Seed Kokoro calibration on first-ever launch using a quick inference.
# Skipped if calibration data already exists. Takes ~150-250ms — imperceptible.
def _run_kokoro_benchmark():
    _BENCH_TEXT = "The quick brown fox jumps over the lazy dog today."
    try:
        if load_calibration().get("words_per_second"):
            return  # already calibrated
        _t0 = time.time()
        kokoro.create(_BENCH_TEXT, voice="af_heart", speed=1.0)
        record_calibration(len(_BENCH_TEXT.split()), time.time() - _t0, use_cb=False)
    except Exception:
        pass  # non-critical; never block startup

_run_kokoro_benchmark()

_splash = _run_splash(_on_splash_done)
# Kokoro is already loaded at this point — complete the splash bar
app.after(100, _splash._finish)

app.mainloop()