# app.spec — PyInstaller build spec for VoxWild (Fast mode only)
#
# Build command (from C:\tts-app):
#   pyinstaller app.spec
#
# Output: dist\VoxWild\VoxWild.exe  (one-dir build)
#
# Chatterbox (Natural mode) is NOT bundled here. The chatterbox_env folder
# and chatterbox_worker.py are deployed by the Inno Setup installer alongside
# the PyInstaller output.

import os
import sys
import site
import glob as _glob

# Site-packages directory for the current Python interpreter
# getsitepackages() returns [python_root, Lib\site-packages] on Windows — pick the one ending in site-packages
SP = next(p for p in site.getsitepackages() if p.endswith("site-packages"))

# ── Helper: expand glob patterns in datas entries ─────────────────────────────
def _glob_datas(pattern, dest):
    return [(f, dest) for f in _glob.glob(pattern)]


# ── Data files to bundle ──────────────────────────────────────────────────────
_datas = [
    # ── App data files (placed at root of the bundle) ──────────────────────
    ("kokoro-v1.0.onnx",      "."),
    ("voices-v1.0.bin",       "."),
    ("icon.ico",              "."),
    ("logo.png",              "."),
    ("theme.json",            "."),
    ("chatterbox_worker.py",  "."),  # run by chatterbox_env python, not imported
    ("enhance_worker.py",     "."),  # run by python_embed python, not imported
    ("vcomp140.dll",          "."),  # OpenMP runtime — copied to torchaudio\lib during Natural mode setup

    # ── certifi CA bundle — HTTPS verification in frozen app ────────────────
    (os.path.join(SP, "certifi", "cacert.pem"), "certifi"),

    # ── customtkinter (themes, images, assets) ─────────────────────────────
    (os.path.join(SP, "customtkinter"), "customtkinter"),

    # ── kokoro_onnx config ─────────────────────────────────────────────────
    (os.path.join(SP, "kokoro_onnx", "config.json"), "kokoro_onnx"),

    # ── espeak-ng (phonemizer backend) ─────────────────────────────────────
    (os.path.join(SP, "espeakng_loader", "espeak-ng-data"),
     os.path.join("espeakng_loader", "espeak-ng-data")),
    (os.path.join(SP, "espeakng_loader", "espeak-ng.dll"),
     "espeakng_loader"),

    # ── sounddevice — PortAudio binaries ───────────────────────────────────
    (os.path.join(SP, "_sounddevice_data"), "_sounddevice_data"),

    # ── soundfile — libsndfile ─────────────────────────────────────────────
    (os.path.join(SP, "_soundfile_data"), "_soundfile_data"),

    # ── onnxruntime DLLs ───────────────────────────────────────────────────
    *_glob_datas(os.path.join(SP, "onnxruntime", "capi", "*.dll"),
                 os.path.join("onnxruntime", "capi")),

    # ── language_tags JSON data (required by csvw → segments → phonemizer → kokoro) ─
    (os.path.join(SP, "language_tags", "data"), os.path.join("language_tags", "data")),

    # ── scipy OpenBLAS ─────────────────────────────────────────────────────
    (os.path.join(SP, "scipy.libs"), "scipy.libs"),

    # ── numpy OpenBLAS ─────────────────────────────────────────────────────
    (os.path.join(SP, "numpy.libs"), "numpy.libs"),
]

# ── Hidden imports PyInstaller misses by static analysis ─────────────────────
_hidden = [
    # Local modules
    "text_cleaner",
    "settings_window",
    "pronunciation",
    "tts_utils",
    "clone_library",
    "audio_utils",
    "license",

    # scipy — heavily uses ctypes/C extensions that confuse the analyser
    "scipy.signal",
    "scipy.signal._sosfilt",
    "scipy.signal.bsplines",
    "scipy._lib.messagestream",
    "scipy._lib._ccallback_c",
    "scipy.special._ufuncs",
    "scipy.special._ufuncs_cxx",
    "scipy.special.cython_special",

    # Audio I/O
    "sounddevice",
    "soundfile",

    # Pedalboard (audio effects)
    "pedalboard",
    "pedalboard.io",

    # kokoro / onnx
    "kokoro_onnx",
    "onnxruntime",
    "onnxruntime.capi",

    # Tkinter dialogs used at runtime
    "tkinter.filedialog",
    "tkinter.messagebox",

    # SSL CA bundle — needed for HTTPS in frozen app (GitHub API, Gumroad)
    "certifi",

    # Standard library bits the analyser sometimes misses
    "ctypes.wintypes",
    "ctypes.windll",
    "threading",
    "ssl",
    "socket",
    "queue",
]

# ── Packages to exclude (Chatterbox / torch — not in fast-mode bundle) ────────
_excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "chatterbox",
    "gradio",
    "fastapi",
    "uvicorn",
    "huggingface_hub",
    "diffusers",
    "librosa",
    "numba",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "setuptools",
    "pip",
]

a = Analysis(
    ["app.py"],
    pathex=[os.path.dirname(os.path.abspath(SPEC))],
    binaries=[],
    datas=_datas,
    hiddenimports=_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoxWild",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        # Don't UPX-compress DLLs that self-verify their layout at runtime
        "onnxruntime.dll",
        "libportaudio64bit.dll",
        "libsndfile_x64.dll",
    ],
    console=False,            # no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        "onnxruntime.dll",
        "libportaudio64bit.dll",
        "libsndfile_x64.dll",
    ],
    name="VoxWild",
)
