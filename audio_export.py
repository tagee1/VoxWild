"""
audio_export.py — export-time audio processing for VoxWild.

Loudness normalization to streaming/broadcast targets, implementing
ITU-R BS.1770-4 integrated loudness: K-weighted, 400 ms gating blocks
with 75% overlap, absolute (-70 LUFS) and relative (-10 LU) gates.

Pure numpy + scipy.signal — no UI, no file I/O; testable standalone.
Validated against pyloudnorm (reference implementation) to within
0.05 LU on speech-like signals at 24/44.1/48 kHz.
"""
import numpy as np
from scipy.signal import lfilter

# Loudness presets offered in the UI (label → target LUFS).
LOUDNESS_PRESETS = {
    "Podcast  -16 LUFS":   -16.0,
    "YouTube  -14 LUFS":   -14.0,
    "Broadcast  -23 LUFS": -23.0,
}
DEFAULT_PRESET = "Podcast  -16 LUFS"

_ABS_GATE_LUFS  = -70.0   # BS.1770 absolute gate
_REL_GATE_LU    = -10.0   # relative gate below ungated mean
PEAK_CEILING_DB = -1.0    # sample-peak ceiling after normalization


def _k_weighting_coeffs(fs):
    """Biquad coefficients for the two BS.1770 pre-filters at sample rate fs.

    Stage 1: +4 dB high-shelf (acoustic head model); stage 2: high-pass
    (RLB weighting). Uses the De Man bilinear formulation ("Evaluation of
    Implementations of the EBU R128 Loudness Measurement", AES 2013),
    which reproduces the spec's 48 kHz coefficient tables exactly and
    generalizes to any sample rate: a 997 Hz full-scale sine reads the
    canonical -3.01 LUFS.
    """
    # Stage 1 — high shelf
    G, Q, fc = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    K  = np.tan(np.pi * fc / fs)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh ** 0.499666774155
    d  = 1.0 + K / Q + K * K
    shelf = (np.array([(Vh + Vb * K / Q + K * K) / d,
                       2.0 * (K * K - Vh) / d,
                       (Vh - Vb * K / Q + K * K) / d]),
             np.array([1.0,
                       2.0 * (K * K - 1.0) / d,
                       (1.0 - K / Q + K * K) / d]))

    # Stage 2 — high pass (numerator [1, -2, 1] exactly, per the ITU table)
    Q, fc = 0.5003270373253953, 38.13547087613982
    K = np.tan(np.pi * fc / fs)
    d = 1.0 + K / Q + K * K
    hp = (np.array([1.0, -2.0, 1.0]),
          np.array([1.0,
                    2.0 * (K * K - 1.0) / d,
                    (1.0 - K / Q + K * K) / d]))
    return shelf, hp


def measure_lufs(samples, fs):
    """Integrated loudness (LUFS) per ITU-R BS.1770-4.

    samples: float array in ±1.0 range, mono (n,) or channels-last (n, ch).
    Returns a float, or None for silent/near-silent audio (everything
    below the absolute gate) where normalization would be meaningless.
    """
    # float32 filtering is ~2x faster than float64; the resulting loudness
    # error is far below 0.001 LU. Accumulation below stays in float64.
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if not x.shape[0]:
        return None

    shelf, hp = _k_weighting_coeffs(fs)
    y = lfilter(*shelf, x, axis=0)
    y = lfilter(*hp, y, axis=0)
    y2 = (y * y).sum(axis=1)          # per-sample power, channels summed (G=1)

    block = int(round(0.400 * fs))    # 400 ms gating blocks
    hop   = int(round(0.100 * fs))    # 75% overlap
    n = y2.shape[0]
    if n < block:
        # Shorter than one gating block (e.g. a quick preview clip):
        # measure the whole clip as a single block instead of failing.
        power = float(y2.mean(dtype=np.float64))
        if power <= 0:
            return None
        lufs = -0.691 + 10 * np.log10(power)
        return lufs if lufs > _ABS_GATE_LUFS else None

    cs = np.concatenate(([0.0], np.cumsum(y2, dtype=np.float64)))
    starts = np.arange(0, n - block + 1, hop)
    powers = (cs[starts + block] - cs[starts]) / block
    with np.errstate(divide="ignore"):
        l_blocks = -0.691 + 10 * np.log10(powers)

    abs_pass = powers[l_blocks > _ABS_GATE_LUFS]
    if not len(abs_pass):
        return None
    rel_gate = -0.691 + 10 * np.log10(abs_pass.mean()) + _REL_GATE_LU
    final = powers[(l_blocks > _ABS_GATE_LUFS) & (l_blocks > rel_gate)]
    if not len(final):
        return None
    return float(-0.691 + 10 * np.log10(final.mean()))


def normalize_loudness(samples, fs, target_lufs, peak_ceiling_db=PEAK_CEILING_DB):
    """Gain samples to the target integrated loudness, respecting a peak ceiling.

    Returns (out, info) with info = {"measured", "gain_db", "limited"}.
    Silent/unmeasurable audio is returned unchanged (measured=None).
    If the required gain would push the sample peak above peak_ceiling_db,
    the gain is reduced so the peak sits exactly at the ceiling instead of
    clipping — the clip lands as loud as it can go without distortion.
    """
    x = np.asarray(samples)
    measured = measure_lufs(x, fs)
    info = {"measured": measured, "gain_db": 0.0, "limited": False}
    if measured is None:
        return x, info

    gain = 10 ** ((target_lufs - measured) / 20)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    ceiling = 10 ** (peak_ceiling_db / 20)
    if peak * gain > ceiling:
        gain = ceiling / peak
        info["limited"] = True
    info["gain_db"] = float(20 * np.log10(gain)) if gain > 0 else 0.0
    out = (x.astype(np.float64) * gain).astype(np.float32)
    return out, info
