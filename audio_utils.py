"""
audio_utils.py — Pure audio-processing functions (no UI dependencies).
Requires numpy.
"""
import numpy as np


def cleanup_audio(samples, sample_rate, mode="pedalboard"):
    """Apply audio cleanup to a numpy float32 sample array.

    mode="pedalboard" — HPF 80 Hz → NoiseGate → Compressor chain using
        the already-installed pedalboard library. Zero extra dependencies.
        Good for taming breath noise and evening out loudness.

    mode="noisereduce" — Spectral subtraction via the noisereduce library
        (pip install noisereduce). Better for steady background hiss/hum.
        Requires noisereduce to be installed; raises ImportError if missing.

    Returns a float32 numpy array at the same sample rate.
    """
    if mode == "pedalboard":
        return _cleanup_pedalboard(samples, sample_rate)
    elif mode == "noisereduce":
        return _cleanup_noisereduce(samples, sample_rate)
    else:
        raise ValueError(f"Unknown cleanup mode: {mode!r}")


def _cleanup_pedalboard(samples, sample_rate):
    try:
        from pedalboard import Pedalboard, HighpassFilter, NoiseGate, Compressor
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80.0),
            NoiseGate(threshold_db=-40.0, ratio=4.0, attack_ms=5.0, release_ms=100.0),
            Compressor(threshold_db=-18.0, ratio=3.0, attack_ms=5.0, release_ms=100.0),
        ])
        # pedalboard expects shape (channels, samples); our arrays are 1-D (mono)
        out = board(samples.reshape(1, -1).astype(np.float32), sample_rate)
        return out.squeeze().astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Audio cleanup (pedalboard) failed: {e}  [E013]") from e


def _cleanup_noisereduce(samples, sample_rate):
    try:
        import noisereduce as nr
        reduced = nr.reduce_noise(y=samples.astype(np.float32), sr=sample_rate,
                                  stationary=True)
        return reduced.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Audio cleanup (noisereduce) failed: {e}  [E013]") from e


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
