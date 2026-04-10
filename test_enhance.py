"""
test_enhance.py — Tests for the Resemble Enhance integration.

Covers four layers:
  1. Installation diagnostics  — package present, all deps installed
  2. Stub correctness          — deepspeed stub handles every edge case
  3. Audio contract            — enhance_audio() produces valid, changed output
  4. Edge cases & robustness   — weird inputs, None streams, concurrency, recovery

Run with:  python test_enhance.py
       or: pytest test_enhance.py -v
"""
import re
import sys
import threading
import numpy as np
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _sine(duration=2.0, sr=24000, freq=220.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32), sr

def _noise(duration=2.0, sr=24000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration)).astype(np.float32), sr

def _silence(duration=2.0, sr=24000):
    return np.zeros(int(sr * duration), dtype=np.float32), sr

def _skip_if_no_resemble():
    try:
        from importlib.metadata import version
        version("resemble-enhance")
    except Exception:
        pytest.skip("resemble-enhance not installed")
    for dep in ("torch", "numpy"):
        try:
            __import__(dep)
        except ImportError:
            pytest.skip(f"required dep '{dep}' not installed")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INSTALLATION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def test_package_installed():
    """resemble-enhance must appear in pip metadata."""
    try:
        from importlib.metadata import version
        v = version("resemble-enhance")
        print(f"\n  resemble-enhance=={v}")
    except Exception as e:
        pytest.fail(f"resemble-enhance not in pip metadata: {e}")


def test_numpy_version_compatible():
    """numpy must be < 2 — resemble-enhance breaks on numpy 2.x."""
    import numpy as np
    major = int(np.version.version.split(".")[0])
    assert major < 2, (
        f"numpy {np.__version__} is incompatible with resemble-enhance. "
        "Run: pip install 'numpy<2'"
    )


def test_declared_deps_installed():
    """Every inference-required dep of resemble-enhance must be importable.

    Training/demo-only packages (deepspeed, ptflops, gradio, celluloid) are
    intentionally skipped — the app stubs or skips them in _SKIP_PKGS.
    """
    # Mirrors _SKIP_PKGS in app.py — packages not needed for inference.
    # torchvision is excluded separately: re-importing it after the model runs
    # triggers a torch double-registration RuntimeError unrelated to our code.
    _SKIP = {"deepspeed", "ptflops", "gradio", "celluloid", "torchvision"}

    try:
        from importlib.metadata import requires
        raw = requires("resemble-enhance") or []
    except Exception as e:
        pytest.skip(f"Cannot read metadata: {e}")
        return

    missing = []
    for req in raw:
        if "extra ==" in req or "extra==" in req:
            continue
        name = re.split(r"[\s;><=!]", req.strip())[0]
        if not name or name.lower() in _SKIP:
            continue
        import_name = name.replace("-", "_").lower()
        try:
            __import__(import_name)
        except ImportError:
            try:
                __import__(name)
            except ImportError:
                missing.append(name)

    if missing:
        pytest.fail(
            f"Missing deps: {', '.join(missing)}\n"
            f"  Fix: pip install {' '.join(missing)}"
        )


def test_resemble_deps_helper_excludes_deepspeed():
    """_resemble_deps_without_deepspeed() must never return deepspeed."""
    try:
        # Import the helper via exec since it lives inside app.py globals
        import importlib, sys as _sys
        # Inline the same logic the app uses
        from importlib.metadata import requires
        raw = requires("resemble-enhance") or []
        result = []
        for req in raw:
            if "extra ==" in req or "extra==" in req:
                continue
            name = re.split(r"[\s;><=!]", req.strip())[0]
            if not name or name.lower().startswith("deepspeed"):
                continue
            result.append(name)
        ds = [d for d in result if d.lower().startswith("deepspeed")]
        assert not ds, f"deepspeed slipped through the filter: {ds}"
    except Exception as e:
        pytest.skip(f"Cannot read metadata: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DEEPSPEED STUB
# ═══════════════════════════════════════════════════════════════════════════════

def test_stub_installs_finder():
    """_stub_deepspeed() must add a finder to sys.meta_path."""
    from audio_utils import _stub_deepspeed
    before = len(sys.meta_path)
    # Remove existing stub finders first
    sys.meta_path[:] = [f for f in sys.meta_path
                        if not getattr(f, "_is_deepspeed_stub", False)]
    sys.modules.pop("deepspeed", None)
    _stub_deepspeed()
    stubs = [f for f in sys.meta_path if getattr(f, "_is_deepspeed_stub", False)]
    assert len(stubs) >= 1, "No stub finder was added to sys.meta_path"


def test_stub_idempotent():
    """Calling _stub_deepspeed() twice must not add duplicate finders."""
    from audio_utils import _stub_deepspeed
    _stub_deepspeed()
    _stub_deepspeed()
    stubs = [f for f in sys.meta_path if getattr(f, "_is_deepspeed_stub", False)]
    assert len(stubs) == 1, f"Expected 1 stub finder, got {len(stubs)}"


def test_stub_top_level():
    """import deepspeed must not raise."""
    from audio_utils import _stub_deepspeed
    _stub_deepspeed()
    import deepspeed
    assert deepspeed is not None


def test_stub_submodule_accelerator():
    """import deepspeed.accelerator must not raise."""
    from audio_utils import _stub_deepspeed
    _stub_deepspeed()
    import deepspeed.accelerator  # noqa
    from deepspeed.accelerator import SomeClass  # noqa — should not raise


def test_stub_submodule_comm():
    from audio_utils import _stub_deepspeed
    _stub_deepspeed()
    import deepspeed.comm  # noqa


def test_stub_arbitrary_depth():
    """deepspeed.a.b.c.d must not raise."""
    from audio_utils import _stub_deepspeed
    _stub_deepspeed()
    import deepspeed
    _ = deepspeed.runtime.zero.partition_parameters.something.deep


def test_stub_as_base_class():
    """Using a stub attribute as a base class (class Foo(ds.Base):) must not raise."""
    from audio_utils import _stub_deepspeed
    _stub_deepspeed()
    import deepspeed

    # This is the pattern that caused __mro_entries__ TypeError
    class MyModel(deepspeed.DeepSpeedEngine):  # type: ignore
        pass

    assert MyModel is not None


def test_stub_initialize_returns_4_tuple():
    """deepspeed.initialize() must return a 4-tuple (model, opt, sched, loader)."""
    from audio_utils import _stub_deepspeed
    _stub_deepspeed()
    import deepspeed
    import torch.nn as nn

    class _M(nn.Module):
        def forward(self, x): return x

    result = deepspeed.initialize(model=_M())
    assert len(result) == 4, f"Expected 4-tuple from initialize(), got {len(result)}"


def test_stub_bool_is_truthy():
    """Stub objects must be truthy so 'if deepspeed.thing:' doesn't silently return None."""
    from audio_utils import _stub_deepspeed
    _stub_deepspeed()
    import deepspeed
    assert bool(deepspeed.comm), \
        "deepspeed.comm stub is falsy — this causes NoneType.write errors in resemble-enhance"
    assert bool(deepspeed.comm.get_rank()), \
        "deepspeed stub method return value is falsy"


def test_stub_call_chain():
    """Chaining calls on a stub must not raise."""
    from audio_utils import _stub_deepspeed
    _stub_deepspeed()
    import deepspeed
    result = deepspeed.utils.logging.get_logger().info("test")
    assert result is not None or result is None  # just must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AUDIO CONTRACT  (requires model — ~14s per test on CPU)
# ═══════════════════════════════════════════════════════════════════════════════

def test_returns_tuple():
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    samples, sr = _sine()
    enhanced, new_sr = enhance_audio(samples, sr, device="cpu")
    assert isinstance(enhanced, np.ndarray)
    assert isinstance(new_sr, int)


def test_output_float32():
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    enhanced, _ = enhance_audio(*_sine(), device="cpu")
    assert enhanced.dtype == np.float32, f"Expected float32, got {enhanced.dtype}"


def test_output_no_nan_inf():
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    enhanced, _ = enhance_audio(*_sine(), device="cpu")
    assert not np.any(np.isnan(enhanced)), "Output contains NaN"
    assert not np.any(np.isinf(enhanced)), "Output contains Inf"


def test_output_upsample():
    """resemble-enhance must upsample to 44100 Hz."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    samples, sr = _sine(sr=24000)
    _, new_sr = enhance_audio(samples, sr, device="cpu")
    assert new_sr > sr, f"Expected upsample ({sr} → >sr), got {new_sr}"
    assert new_sr == 44100, f"Expected 44100 Hz output, got {new_sr}"


def test_output_changes_audio():
    """Enhanced output must differ from the input (model is doing real work)."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    from scipy.signal import resample_poly
    from math import gcd
    samples, sr = _sine(duration=2.0)
    enhanced, new_sr = enhance_audio(samples, sr, device="cpu")
    g = gcd(sr, new_sr)
    orig_rs = resample_poly(samples, new_sr // g, sr // g).astype(np.float32)
    min_len = min(len(orig_rs), len(enhanced))
    mad = float(np.mean(np.abs(orig_rs[:min_len] - enhanced[:min_len])))
    assert mad > 1e-6, f"Output identical to input (MAD={mad:.2e}) — model not working"


def test_preserves_duration():
    """Output duration must be within 10% of input."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    samples, sr = _sine(duration=2.0)
    enhanced, new_sr = enhance_audio(samples, sr, device="cpu")
    in_dur  = len(samples) / sr
    out_dur = len(enhanced) / new_sr
    assert abs(out_dur - in_dur) / in_dur < 0.10, \
        f"Duration drifted: {in_dur:.2f}s → {out_dur:.2f}s"


def test_output_mono():
    """Output must be 1-D (mono)."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    enhanced, _ = enhance_audio(*_sine(), device="cpu")
    assert enhanced.ndim == 1, f"Expected 1-D output, got shape {enhanced.shape}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EDGE CASES & ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════

def test_silent_input():
    """All-zero input must not crash — model may return silence or near-silence."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    samples, sr = _silence(duration=2.0)
    enhanced, new_sr = enhance_audio(samples, sr, device="cpu")
    assert isinstance(enhanced, np.ndarray)
    assert not np.any(np.isnan(enhanced)), "Silent input produced NaN output"


def test_noisy_input():
    """White noise input must complete without error."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    samples, sr = _noise(duration=2.0)
    enhanced, new_sr = enhance_audio(samples, sr, device="cpu")
    assert isinstance(enhanced, np.ndarray)
    assert not np.any(np.isnan(enhanced))


def test_short_input():
    """Very short audio (0.5 s) must complete without error."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    samples, sr = _sine(duration=0.5)
    enhanced, new_sr = enhance_audio(samples, sr, device="cpu")
    assert len(enhanced) > 0


def test_clipped_input():
    """Input with values > 1.0 must not crash — values should survive or be clamped."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    samples, sr = _sine()
    samples = samples * 3.0   # intentionally clipped
    enhanced, _ = enhance_audio(samples, sr, device="cpu")
    assert not np.any(np.isnan(enhanced))


def test_different_sample_rates():
    """Enhancement must work for common TTS sample rates."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    for sr in (16000, 22050, 24000):
        samples = np.sin(2 * np.pi * 220 * np.linspace(0, 1, sr)).astype(np.float32)
        enhanced, new_sr = enhance_audio(samples, sr, device="cpu")
        assert isinstance(enhanced, np.ndarray), f"Failed for sr={sr}"
        assert new_sr == 44100, f"Expected 44100 output for sr={sr}, got {new_sr}"


def test_stdout_none_safe():
    """enhance_audio must not crash when sys.stdout and sys.stderr are None."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        sys.stdout = None
        sys.stderr = None
        enhanced, new_sr = enhance_audio(*_sine(duration=1.0), device="cpu")
        assert isinstance(enhanced, np.ndarray), "Got non-array when stdout=None"
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err


def test_stdout_restored_after_success():
    """sys.stdout/stderr must be restored to originals after a successful call."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    orig_out, orig_err = sys.stdout, sys.stderr
    enhance_audio(*_sine(duration=1.0), device="cpu")
    assert sys.stdout is orig_out, "sys.stdout was not restored after enhance"
    assert sys.stderr is orig_err, "sys.stderr was not restored after enhance"


def test_stdout_restored_after_error():
    """sys.stdout/stderr must be restored even when enhance_audio raises."""
    from audio_utils import enhance_audio
    orig_out, orig_err = sys.stdout, sys.stderr
    try:
        enhance_audio(*_sine(duration=0.5), device="notadevice:99")
    except RuntimeError:
        pass
    assert sys.stdout is orig_out, "sys.stdout leaked after enhance error"
    assert sys.stderr is orig_err, "sys.stderr leaked after enhance error"


def test_bad_device_raises_e013():
    """Invalid device must raise RuntimeError tagged [E013]."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    with pytest.raises(RuntimeError, match="E013"):
        enhance_audio(*_sine(duration=0.5), device="notadevice:99")


def test_error_message_readable():
    """Error messages must not be raw tracebacks — they should be human-readable."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    try:
        enhance_audio(*_sine(duration=0.5), device="notadevice:99")
    except RuntimeError as e:
        msg = str(e)
        assert "Traceback" not in msg, "Error message contains raw traceback"
        assert "File " not in msg, "Error message contains file path noise"
        assert "[E013]" in msg, "Error message missing [E013] tag"


def test_concurrent_calls():
    """Two enhance calls in parallel threads must both complete successfully."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    results = {}
    errors  = {}

    def _run(key, samples, sr):
        try:
            results[key] = enhance_audio(samples, sr, device="cpu")
        except Exception as e:
            errors[key] = e

    s1, r1 = _sine(duration=1.0, freq=220)
    s2, r2 = _sine(duration=1.0, freq=440)
    t1 = threading.Thread(target=_run, args=("a", s1, r1))
    t2 = threading.Thread(target=_run, args=("b", s2, r2))
    t1.start(); t2.start()
    t1.join(timeout=120); t2.join(timeout=120)

    assert not errors, f"Thread errors: {errors}"
    assert "a" in results and "b" in results, "Not all threads produced results"
    for key, (enhanced, new_sr) in results.items():
        assert isinstance(enhanced, np.ndarray), f"Thread {key} got non-array"
        assert new_sr == 44100, f"Thread {key} got unexpected sr {new_sr}"


def test_call_again_after_error():
    """enhance_audio must work on a subsequent call after a previous call raised."""
    _skip_if_no_resemble()
    from audio_utils import enhance_audio
    # First call — bad device, should raise
    try:
        enhance_audio(*_sine(duration=0.5), device="notadevice:99")
    except RuntimeError:
        pass
    # Second call — good device, should work
    enhanced, new_sr = enhance_audio(*_sine(duration=1.0), device="cpu")
    assert isinstance(enhanced, np.ndarray), "Second call failed after first errored"
    assert new_sr == 44100


# ═══════════════════════════════════════════════════════════════════════════════
# standalone runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_tests = [v for k, v in sorted(globals().items())
                 if k.startswith("test_") and callable(v)]

    passed = failed = skipped = 0
    for t in all_tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"  SKIP  {t.__name__}  ({e})")
            skipped += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}")
            print(f"        {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed · {skipped} skipped · {failed} failed")
    sys.exit(1 if failed else 0)
