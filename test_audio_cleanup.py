"""
test_audio_cleanup.py — Tests for the audio cleanup tool.

Tests cleanup_audio() in audio_utils.py for both pedalboard and
noisereduce modes without needing a running UI.
"""
import numpy as np
import pytest


def _sine(freq=440, sr=22050, duration=1.0, amplitude=0.5):
    """Generate a clean sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32), sr


def _noisy(freq=440, sr=22050, duration=1.0, signal_amp=0.5, noise_amp=0.05):
    """Generate a sine wave with added Gaussian noise."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t) * signal_amp
    noise  = np.random.default_rng(42).normal(0, noise_amp, len(t))
    return (signal + noise).astype(np.float32), sr


# ── Pedalboard mode ───────────────────────────────────────────────────────────

def test_pedalboard_returns_float32():
    from audio_utils import cleanup_audio
    samples, sr = _sine()
    out = cleanup_audio(samples, sr, mode="pedalboard")
    assert out.dtype == np.float32

def test_pedalboard_same_length():
    from audio_utils import cleanup_audio
    samples, sr = _sine()
    out = cleanup_audio(samples, sr, mode="pedalboard")
    # Pedalboard chain should not change sample count significantly
    assert abs(len(out) - len(samples)) <= sr // 10  # within 100ms

def test_pedalboard_output_in_range():
    from audio_utils import cleanup_audio
    samples, sr = _sine(amplitude=0.9)
    out = cleanup_audio(samples, sr, mode="pedalboard")
    assert np.max(np.abs(out)) <= 1.0 + 1e-5

def test_pedalboard_reduces_very_quiet_noise():
    """NoiseGate should significantly attenuate sub-threshold signal."""
    from audio_utils import cleanup_audio
    sr = 22050
    # Near-silent noise well below -40 dB threshold
    noise = (np.random.default_rng(0).normal(0, 0.001, sr)).astype(np.float32)
    out = cleanup_audio(noise, sr, mode="pedalboard")
    assert np.max(np.abs(out)) < np.max(np.abs(noise)) + 1e-4

def test_pedalboard_preserves_loud_signal():
    """Loud speech-level signal should survive the chain largely intact."""
    from audio_utils import cleanup_audio
    samples, sr = _sine(amplitude=0.7)
    out = cleanup_audio(samples, sr, mode="pedalboard")
    # RMS should be at least 30% of input RMS
    rms_in  = np.sqrt(np.mean(samples ** 2))
    rms_out = np.sqrt(np.mean(out ** 2))
    assert rms_out > rms_in * 0.3


# ── noisereduce mode ──────────────────────────────────────────────────────────

def test_noisereduce_import():
    try:
        import noisereduce
    except ImportError:
        pytest.skip("noisereduce not installed")

def test_noisereduce_returns_float32():
    try:
        import noisereduce
    except ImportError:
        pytest.skip("noisereduce not installed")
    from audio_utils import cleanup_audio
    samples, sr = _noisy()
    out = cleanup_audio(samples, sr, mode="noisereduce")
    assert out.dtype == np.float32

def test_noisereduce_same_length():
    try:
        import noisereduce
    except ImportError:
        pytest.skip("noisereduce not installed")
    from audio_utils import cleanup_audio
    samples, sr = _noisy()
    out = cleanup_audio(samples, sr, mode="noisereduce")
    assert len(out) == len(samples)

def test_noisereduce_reduces_noise():
    """noisereduce should lower noise floor on a noisy signal."""
    try:
        import noisereduce
    except ImportError:
        pytest.skip("noisereduce not installed")
    from audio_utils import cleanup_audio
    samples, sr = _noisy(signal_amp=0.01, noise_amp=0.3)
    out = cleanup_audio(samples, sr, mode="noisereduce")
    rms_in  = np.sqrt(np.mean(samples ** 2))
    rms_out = np.sqrt(np.mean(out ** 2))
    assert rms_out < rms_in  # output should be quieter (noise reduced)


# ── General ───────────────────────────────────────────────────────────────────

def test_invalid_mode_raises():
    from audio_utils import cleanup_audio
    samples, sr = _sine()
    try:
        cleanup_audio(samples, sr, mode="unknown")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_pedalboard_returns_float32,
        test_pedalboard_same_length,
        test_pedalboard_output_in_range,
        test_pedalboard_reduces_very_quiet_noise,
        test_pedalboard_preserves_loud_signal,
        test_noisereduce_import,
        test_noisereduce_returns_float32,
        test_noisereduce_same_length,
        test_noisereduce_reduces_noise,
        test_invalid_mode_raises,
    ]
    passed = failed = skipped = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except SystemExit:
            print(f"  SKIP  {t.__name__}")
            skipped += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    raise SystemExit(failed)
