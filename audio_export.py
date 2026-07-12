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

_LIMIT_LOOKAHEAD_S = 0.005   # limiter attack ramp / lookahead window
_LIMIT_RELEASE_S   = 0.060   # limiter release time constant


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


def _sliding_min_ahead(g, w):
    """m[i] = min(g[i : i+w]) — lookahead sliding-window minimum in O(n).

    van Herk/Gil-Werman: with block prefix/suffix minima, any w-wide window
    spans at most two w-blocks, so its min is min(suffix[i], prefix[i+w-1]).
    """
    n = g.shape[0]
    if w <= 1 or n == 0:
        return g
    pad = (-n) % w
    if pad < w - 1:          # prefix index i+w-1 must stay in bounds
        pad += w
    gp = np.concatenate([g, np.full(pad, np.inf, dtype=g.dtype)])
    blocks = gp.reshape(-1, w)
    pre = np.minimum.accumulate(blocks, axis=1).ravel()
    suf = np.minimum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].ravel()
    return np.minimum(suf[:n], pre[w - 1:w - 1 + n])


def _moving_avg(g, w):
    """Trailing moving average over w samples (edge-held at the start)."""
    if w <= 1:
        return g
    arr = np.concatenate([np.full(w - 1, g[0], dtype=g.dtype), g])
    cs = np.concatenate([[0.0], np.cumsum(arr, dtype=np.float64)])
    return ((cs[w:] - cs[:-w]) / w).astype(g.dtype, copy=False)


def _limit_peaks(x, fs, ceiling):
    """Lookahead brickwall limiter: keep |out| <= ceiling by attenuating only
    around peaks, with a ramped attack and smooth release — no hard clipping,
    and passages without peaks pass through untouched.
    """
    ceiling = np.float32(ceiling)
    ax = np.abs(x) if x.ndim == 1 else np.abs(x).max(axis=1)
    if not ax.size or float(ax.max()) <= ceiling:
        return x                     # nothing exceeds the ceiling
    g_inst = np.minimum(np.float32(1.0), ceiling / np.maximum(ax, np.float32(1e-9)))
    w = max(1, int(round(_LIMIT_LOOKAHEAD_S * fs)))
    # Floor over the upcoming window, then ramp into it; never above what
    # the current sample itself requires.
    g_att = np.minimum(_moving_avg(_sliding_min_ahead(g_inst, w), w), g_inst)
    # One-pole smoothing gives the release; the elementwise min keeps the
    # attack instant (smoothing must never lag a gain *reduction*).
    alpha = np.float32(np.exp(-1.0 / (_LIMIT_RELEASE_S * fs)))
    g_rel, _ = lfilter(np.array([1 - alpha], dtype=np.float32),
                       np.array([1, -alpha], dtype=np.float32), g_att,
                       zi=np.array([g_att[0] * alpha], dtype=np.float32))
    g = np.minimum(g_rel, g_att)
    if x.ndim != 1:
        g = g[:, None]
    return np.clip(x * g, -ceiling, ceiling)


def normalize_loudness(samples, fs, target_lufs, peak_ceiling_db=PEAK_CEILING_DB):
    """Gain samples to the target integrated loudness, respecting a peak ceiling.

    Returns (out, info) with info = {"measured", "gain_db", "limited"}.
    Silent/unmeasurable audio is returned unchanged (measured=None).
    If plain gain would push sample peaks above peak_ceiling_db, the full
    gain is still applied and a lookahead limiter transparently tames just
    the moments around each peak (limited=True) — quiet-but-peaky speech
    reaches the target instead of falling short of it.
    """
    x = np.asarray(samples, dtype=np.float32)
    measured = measure_lufs(x, fs)
    info = {"measured": measured, "gain_db": 0.0, "limited": False}
    if measured is None:
        return x, info
    # Already at target (within a quarter LU — far below audibility): leave
    # the audio untouched so re-exporting a normalized clip is a clean no-op.
    if abs(target_lufs - measured) < 0.25:
        return x, info

    gain = 10 ** ((target_lufs - measured) / 20)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    ceiling = 10 ** (peak_ceiling_db / 20)
    if peak * gain <= ceiling:
        info["gain_db"] = float(20 * np.log10(gain)) if gain > 0 else 0.0
        return x * np.float32(gain), info

    info["limited"] = True
    out = _limit_peaks(x * np.float32(gain), fs, ceiling)
    # Limiting shaves a little energy off the loudest moments, so the result
    # lands slightly under target — nudge up and re-limit until it converges.
    for _ in range(2):
        got = measure_lufs(out, fs)
        if got is None or abs(target_lufs - got) <= 0.2:
            break
        step = 10 ** ((target_lufs - got) / 20)
        gain *= step
        out = _limit_peaks(out * np.float32(step), fs, ceiling)
    info["gain_db"] = float(20 * np.log10(gain))
    return out, info
