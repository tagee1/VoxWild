"""
audio_utils.py — Pure audio-processing functions (no UI dependencies).
Requires numpy.
"""
import sys
import types
import numpy as np


def _stub_deepspeed():
    """Intercept all deepspeed imports with a no-op stub if deepspeed isn't installed.

    resemble-enhance imports deepspeed and its submodules (deepspeed.accelerator,
    deepspeed.comm, etc.) at module level for training code.  Inference never
    calls any of those paths at runtime, but the imports still fail on Windows
    where deepspeed can't be built from source.

    Rather than listing every submodule individually, we install a meta path
    finder that auto-stubs any 'deepspeed' or 'deepspeed.*' import on demand.
    The finder is a no-op if deepspeed is genuinely installed.
    """
    try:
        import deepspeed  # noqa — real deepspeed is installed, nothing to do
        return
    except ImportError:
        pass

    # Already patched by a previous call
    if any(getattr(f, "_is_deepspeed_stub", False) for f in sys.meta_path):
        return

    class _Anything:
        """Permissive stub — any attribute access or call returns another stub.

        Special dunder methods are defined explicitly because Python looks them
        up on the type, not the instance, so __getattr__ alone isn't enough.
        """
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
        # Used when _Anything() appears as a base class in a class definition
        def __mro_entries__(self, bases): return ()
        # Used for generic subscript: deepspeed.SomeType[int]
        def __class_getitem__(cls, item): return cls

    class _DeepSpeedStubFinder:
        """sys.meta_path finder — intercepts every deepspeed.* import."""
        _is_deepspeed_stub = True

        def find_module(self, fullname, path=None):  # Python 3 legacy API, still works
            if fullname == "deepspeed" or fullname.startswith("deepspeed."):
                return self
            return None

        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            mod = types.ModuleType(fullname)
            mod.__path__    = []          # marks it as a package so sub-imports work
            mod.__package__ = fullname
            mod.__file__    = None        # prevents inspect.getfile() from getting _Anything
            # Do NOT set __loader__ or __spec__ — setting __loader__ = self causes
            # Python to treat the module as built-in, which breaks reimports.
            mod.__getattr__ = lambda name: _Anything()
            if fullname == "deepspeed":
                def _initialize(*args, **kwargs):
                    model = args[0] if args else _Anything()
                    return model, _Anything(), _Anything(), _Anything()
                mod.initialize      = _initialize
                mod.DeepSpeedEngine = _Anything
            sys.modules[fullname] = mod
            return mod

    # Prepend so we intercept before Python's normal import machinery
    sys.meta_path.insert(0, _DeepSpeedStubFinder())


def enhance_audio(samples, sample_rate, device="cpu"):
    """Run Resemble Enhance on a float32 numpy array.

    Uses the resemble-enhance library (pip install resemble-enhance --no-deps)
    to apply AI-based speech enhancement (denoising + diffusion-based
    enhancement).  Noticeably improves naturalness and presence of
    TTS-generated audio.

    device — "cpu" or "cuda" (GPU).

    Returns (enhanced_samples: float32 ndarray, new_sample_rate: int).
    The output sample rate may differ from the input (resemble-enhance
    typically upsamples to 44100 Hz).

    Raises descriptive RuntimeError (tagged E013) on any failure so the
    caller can surface a useful message directly to the user.
    """
    # ── Dependency checks ────────────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "torch is not installed — enhancement requires PyTorch. "
            "Run: pip install torch  [E013]"
        )

    _stub_deepspeed()   # must come before any resemble_enhance import

    try:
        from resemble_enhance.enhancer.inference import enhance as _re_enhance
    except ImportError as e:
        raise RuntimeError(
            f"resemble-enhance is not installed or failed to import ({e}). "
            "Re-enable the checkbox to trigger auto-install.  [E013]"
        ) from e

    # ── Device checks ────────────────────────────────────────────────────────
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU mode selected but CUDA is not available on this machine. "
                "Switch to CPU or Async mode in the AI Enhancement panel.  [E013]"
            )

    # ── Windows: patch PosixPath so Linux-saved checkpoints load correctly ──────
    import platform
    if platform.system() == "Windows":
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath

    # ── Run enhancement ──────────────────────────────────────────────────────
    # tqdm (used inside resemble-enhance) calls sys.stderr.write().
    # When launched without a console sys.stdout/stderr are None → AttributeError.
    # Redirect to devnull for the duration of the call then restore.
    import os as _os
    _stdout_saved, _stderr_saved = sys.stdout, sys.stderr
    if sys.stdout is None:
        sys.stdout = open(_os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(_os.devnull, "w")

    try:
        wav = torch.from_numpy(samples.astype(np.float32))
        enhanced_wav, out_sr = _re_enhance(
            wav, sample_rate,
            run_dir=None, nfe=64, solver="midpoint",
            lambd=0.5, tau=0.5, device=device,
        )
        return enhanced_wav.cpu().numpy().astype(np.float32), int(out_sr)

    except RuntimeError as e:
        msg = str(e)
        msg_lower = msg.lower()

        if "out of memory" in msg_lower or "cuda out of memory" in msg_lower:
            raise RuntimeError(
                "GPU ran out of memory during enhancement. "
                "Switch to CPU mode and try again.  [E013]"
            ) from e
        if "cuda" in msg_lower or "cudnn" in msg_lower or "cublas" in msg_lower:
            raise RuntimeError(
                f"CUDA error during enhancement — try switching to CPU mode. "
                f"Detail: {msg}  [E013]"
            ) from e
        if "posixpath" in msg_lower or "cannot instantiate" in msg_lower:
            raise RuntimeError(
                "Model checkpoint contains Linux paths (PosixPath) which can't load on Windows. "
                "This is a known issue — please report it.  [E013]"
            ) from e
        if any(k in msg_lower for k in ("connection", "network", "timeout",
                                         "download", "http", "url", "certificate")):
            raise RuntimeError(
                f"Network error while downloading model weights — check your "
                f"internet connection and try again. Detail: {msg}  [E013]"
            ) from e
        if "no such file" in msg_lower or "filenotfounderror" in msg_lower:
            raise RuntimeError(
                f"Model file not found — the model weights may not have downloaded "
                f"correctly. Delete the cache folder and try again. "
                f"Detail: {msg}  [E013]"
            ) from e
        raise RuntimeError(f"Enhancement model error: {msg}  [E013]") from e

    except Exception as e:
        raise RuntimeError(
            f"Unexpected enhancement error ({type(e).__name__}): {e}  [E013]"
        ) from e

    finally:
        sys.stdout, sys.stderr = _stdout_saved, _stderr_saved


def trim_silence(samples, sample_rate, threshold_db=-50):
    """Trim leading/trailing silence from a numpy sample array.

    Keeps a 50 ms pre-roll before the first non-silent sample and a
    300 ms tail after the last non-silent sample so soft fade-outs
    and trailing consonants are not clipped.
    """
    threshold  = 10 ** (threshold_db / 20)
    non_silent = np.where(np.abs(samples) > threshold)[0]
    if len(non_silent) == 0:
        return samples
    start = max(0, non_silent[0]  - int(sample_rate * 0.05))
    end   = min(len(samples), non_silent[-1] + int(sample_rate * 0.30))
    return samples[start:end]
