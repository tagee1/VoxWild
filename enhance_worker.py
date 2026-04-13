"""
Resemble Enhance subprocess worker.
Runs under python_embed Python (3.11).

Protocol (line-delimited JSON over stdin/stdout — same as chatterbox_worker):
  Startup: emits {"type": "status", "msg": "..."} then {"type": "ready"}
  Request: {"cmd": "enhance", "input_path": "...", "output_path": "...",
            "device": "cpu"}
  Response: {"type": "status", "msg": "..."} ...
            {"type": "done", "sr": <int>, "rms_delta_db": <float>}
            or {"type": "error", "msg": "..."}
  Quit:    {"cmd": "quit"}
"""
import sys
import os
import json
import logging

# ── Binary-mode stdio for IPC ────────────────────────────────────────────────
# Grab raw binary buffers BEFORE any redirects. Same pattern as chatterbox_worker.
if hasattr(sys.stdout, "buffer"):
    _proto_bin = sys.stdout.buffer
else:
    _proto_bin = sys.stdout

if hasattr(sys.stdin, "buffer"):
    _stdin_bin = sys.stdin.buffer
else:
    _stdin_bin = sys.stdin

# Redirect stdout during imports so print() spam doesn't corrupt protocol.
sys.stdout = open(os.devnull, "w")

# Suppress noisy loggers
logging.disable(logging.WARNING)
os.environ["TQDM_DISABLE"] = "1"


def emit(obj):
    """Write a JSON line to the parent over the binary pipe."""
    line = json.dumps(obj, ensure_ascii=True) + "\n"
    _proto_bin.write(line.encode("utf-8"))
    _proto_bin.flush()


def _stub_deepspeed():
    """Intercept deepspeed imports with no-op stubs (Windows can't build it)."""
    try:
        import deepspeed  # noqa
        return
    except ImportError:
        pass

    if any(getattr(f, "_is_deepspeed_stub", False) for f in sys.meta_path):
        return

    import types

    class _Anything:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return _Anything()
        def __getattr__(self, n): return _Anything()
        def __iter__(self): return iter([])
        def __len__(self): return 0
        def __bool__(self): return True
        def __int__(self): return 0
        def __float__(self): return 0.0
        def __str__(self): return ""
        def __repr__(self): return "<deepspeed-stub>"
        def __contains__(self, item): return False
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __mro_entries__(self, bases): return ()
        def __class_getitem__(cls, item): return cls

    class _DeepSpeedStubFinder:
        _is_deepspeed_stub = True

        def find_module(self, fullname, path=None):
            if fullname == "deepspeed" or fullname.startswith("deepspeed."):
                return self
            return None

        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            mod = types.ModuleType(fullname)
            mod.__path__ = []
            mod.__package__ = fullname
            mod.__file__ = None
            mod.__getattr__ = lambda name: _Anything()
            if fullname == "deepspeed":
                def _initialize(*args, **kwargs):
                    model = args[0] if args else _Anything()
                    return model, _Anything(), _Anything(), _Anything()
                mod.initialize = _initialize
                mod.DeepSpeedEngine = _Anything
            sys.modules[fullname] = mod
            return mod

    sys.meta_path.insert(0, _DeepSpeedStubFinder())


def _log_path():
    return os.path.join(
        os.environ.get("APPDATA", ""), "TTS Studio", "enhance_error.log"
    )


def _write_log(text):
    try:
        with open(_log_path(), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def _ensure_tabulate():
    """Install tabulate if missing — needed by Enhancer.__init__ → summarize → to_markdown."""
    try:
        import tabulate  # noqa: F401
    except ImportError:
        import subprocess as _sp
        _sp.run(
            [sys.executable, "-m", "pip", "install", "tabulate"],
            capture_output=True,
        )


def main():
    emit({"type": "status", "msg": "Starting AI Enhancement engine..."})

    # Stub deepspeed before importing resemble_enhance
    _stub_deepspeed()
    _ensure_tabulate()

    # Windows: patch PosixPath so Linux-saved checkpoints load
    import platform
    if platform.system() == "Windows":
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath

    try:
        import torch
        import numpy as np
        from resemble_enhance.enhancer.inference import enhance as _re_enhance

        # ── Patch model download to use huggingface_hub instead of git ────────
        # resemble-enhance's download.py shells out to `git clone` + `git lfs`,
        # which fails on machines without git installed. huggingface_hub is
        # already available (installed for chatterbox/Natural mode) and handles
        # downloads natively via HTTP — no git needed.
        import resemble_enhance.enhancer.download as _dl_mod
        import resemble_enhance.enhancer.inference as _inf_mod
        from pathlib import Path

        def _hf_download():
            """Download model weights via huggingface_hub (no git required)."""
            emit({"type": "status", "msg": "Downloading enhancement model (~450 MB)..."})
            from huggingface_hub import snapshot_download
            local = snapshot_download("ResembleAI/resemble-enhance")
            return Path(local) / "enhancer_stage2"

        _dl_mod.download = _hf_download
        _inf_mod.download = _hf_download
        # ─────────────────────────────────────────────────────────────────────

    except Exception as e:
        import traceback
        _write_log(traceback.format_exc())
        emit({"type": "error", "msg": f"Failed to load enhancement engine: {e}"})
        return

    emit({"type": "ready"})

    # ── Request loop ──────────────────────────────────────────────────────────
    for raw_bytes in _stdin_bin:
        raw = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception:
            continue

        cmd = req.get("cmd")
        if cmd == "quit":
            break

        if cmd == "enhance":
            try:
                import traceback as _tb
                import soundfile as sf

                input_path  = req.get("input_path", "")
                output_path = req.get("output_path", "")
                device      = req.get("device", "cpu")

                if not input_path or not output_path:
                    emit({"type": "error", "msg": "Malformed enhance request (missing paths)."})
                    continue

                # Device check
                if device == "cuda" and not torch.cuda.is_available():
                    device = "cpu"

                # Read input audio
                emit({"type": "status", "msg": "Reading audio..."})
                samples, sample_rate = sf.read(input_path, dtype="float32")

                # Run enhancement
                emit({"type": "status", "msg": "Enhancing audio (first run downloads ~450 MB)..."})

                # Handle None stdout/stderr (tqdm inside resemble-enhance)
                _stdout_saved, _stderr_saved = sys.stdout, sys.stderr
                if sys.stdout is None:
                    sys.stdout = open(os.devnull, "w")
                if sys.stderr is None:
                    sys.stderr = open(os.devnull, "w")

                try:
                    wav = torch.from_numpy(samples.astype(np.float32))
                    enhanced_wav, out_sr = _re_enhance(
                        wav, sample_rate,
                        run_dir=None, nfe=64, solver="midpoint",
                        lambd=0.5, tau=0.5, device=device,
                    )
                    enhanced = enhanced_wav.cpu().numpy().astype(np.float32)
                finally:
                    sys.stdout, sys.stderr = _stdout_saved, _stderr_saved

                enhanced_clipped = np.clip(enhanced, -1.0, 1.0)
                out_sr = int(out_sr)

                # Verify enhancement actually changed the audio
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(sample_rate, out_sr)
                orig_resampled = resample_poly(
                    samples, out_sr // g, sample_rate // g
                ).astype(np.float32)
                min_len = min(len(orig_resampled), len(enhanced_clipped))
                mad = float(np.mean(np.abs(
                    orig_resampled[:min_len] - enhanced_clipped[:min_len]
                )))

                if mad < 1e-6:
                    emit({"type": "error",
                          "msg": "Enhancement produced no change — model weights may have "
                                 "failed to load. Try reinstalling resemble-enhance.  [E013]"})
                    continue

                # Calculate RMS delta for UI
                rms_orig = float(np.sqrt(np.mean(orig_resampled[:min_len] ** 2)))
                rms_enh  = float(np.sqrt(np.mean(enhanced_clipped[:min_len] ** 2)))
                rms_delta_db = 20 * np.log10(rms_enh / rms_orig) if rms_orig > 1e-9 else 0.0

                # Save output
                emit({"type": "status", "msg": "Saving enhanced audio..."})
                sf.write(output_path, enhanced_clipped, out_sr)

                emit({"type": "done", "sr": out_sr, "rms_delta_db": round(rms_delta_db, 1)})

            except Exception as e:
                import traceback as _tb
                _full_tb = _tb.format_exc()
                print(_full_tb, file=sys.stderr, flush=True)
                _write_log(f"Enhancement error: {type(e).__name__}: {e}\n\n{_full_tb}")

                # Categorize errors for user-friendly messages
                msg = str(e)
                msg_lower = msg.lower()
                if "out of memory" in msg_lower:
                    emit({"type": "error", "msg": "GPU ran out of memory — try CPU mode.  [E013]"})
                elif "cuda" in msg_lower or "cudnn" in msg_lower:
                    emit({"type": "error", "msg": f"CUDA error — try CPU mode. Detail: {msg}  [E013]"})
                else:
                    emit({"type": "error", "msg": f"Enhancement failed: {msg}  [E013]"})


if len(sys.argv) > 1 and sys.argv[1] == "--check":
    # Lightweight import check — used by the installer to verify deps.
    # Reuses the same deepspeed stub + PosixPath fix as the real worker,
    # so it tests the exact code path that will run at enhancement time.
    _stub_deepspeed()
    import platform
    if platform.system() == "Windows":
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        from resemble_enhance.enhancer.inference import enhance  # noqa: F401
        _proto_bin.write(b"ok\n")
        _proto_bin.flush()
    except Exception as e:
        _proto_bin.write(f"fail: {e}\n".encode("utf-8"))
        _proto_bin.flush()
        sys.exit(1)
else:
    main()
