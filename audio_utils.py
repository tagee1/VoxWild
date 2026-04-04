"""
audio_utils.py — Pure audio-processing functions (no UI dependencies).
Requires numpy.
"""
import numpy as np


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
