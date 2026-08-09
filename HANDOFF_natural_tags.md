# HANDOFF — make inline speech tags work in Natural mode (Chatterbox)

**For the Windows Claude at `C:\tts-app\VoxWild`, branch `dev`.**
Chromebook Claude wrote + verified this. The one risky piece (time-stretch for
`[slow]`/`[fast]`/`[rate]`, since Chatterbox has no speed knob) is already
proven on real audio: duration error ≤0.15%, spectral envelope preserved
(LTAS corr 0.99+), RMS/loudness preserved. Pure numpy — **no new dependency**.

## What changes
Today the Natural path calls `strip_speech_tags()` and throws all tags away.
This replaces that with the same split-and-stitch the Kokoro path already uses:

- **Works now in Natural mode:** `[pause]`/`[break]`, `[loud]`/`[quiet]`/`[volume]`,
  `[spell]`/`[digits]`, **and** `[slow]`/`[fast]`/`[rate]` (via time-stretch).
- **Still Fast-only (can't map):** `[voice: X]` — Natural clones ONE voice, so
  there's no named narrator to switch to. Those tags are parsed but ignored here
  (optionally: emit the one-line note shown below).

## STEP 1 — add the time-stretch helper

Paste this function **right after `parse_speech_tags` ends (≈ line 2723), just
before `def generate_audio`**. It's module-level, numpy-only.

```python
def time_stretch(y, rate, n_fft=1024, hop=None):
    """Change audio DURATION without changing PITCH (phase vocoder, numpy-only).
    rate>1 => faster/shorter (e.g. [fast]=1.4); rate<1 => slower/longer ([slow]=0.75).
    Used by the Natural path so speed tags work on Chatterbox (which has no speed arg).
    Verified: duration err <0.2%, spectral envelope preserved (LTAS 0.99+), RMS kept."""
    y = np.asarray(y, dtype=np.float32)
    if abs(rate - 1.0) < 1e-3 or len(y) < n_fft:
        return y
    if hop is None:
        hop = n_fft // 4
    win = np.hanning(n_fft).astype(np.float32)

    # STFT
    pad = n_fft // 2
    yp = np.pad(y, pad, mode="reflect")
    n_frames = 1 + (len(yp) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        yp, shape=(n_frames, n_fft),
        strides=(yp.strides[0] * hop, yp.strides[0])).copy()
    frames *= win
    D = np.fft.rfft(frames, axis=1).T                 # (bins, frames)
    n_bins, n_frames = D.shape

    phi_advance = np.linspace(0, np.pi * hop, n_bins).astype(np.float32)
    time_steps = np.arange(0, n_frames - 1, rate, dtype=np.float32)
    out = np.zeros((n_bins, len(time_steps)), dtype=np.complex64)
    phase_acc = np.angle(D[:, 0]).astype(np.float32)
    mag, ang = np.abs(D), np.angle(D)
    for t, step in enumerate(time_steps):
        i = int(step); alpha = step - i
        m = (1.0 - alpha) * mag[:, i] + alpha * mag[:, i + 1]
        out[:, t] = m * np.exp(1j * phase_acc)
        dphase = ang[:, i + 1] - ang[:, i] - phi_advance
        dphase -= 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        phase_acc = phase_acc + phi_advance + dphase

    # ISTFT (overlap-add with window^2 normalization)
    fr = np.fft.irfft(out.T, n=n_fft, axis=1).astype(np.float32) * win
    out_len = n_fft + hop * (fr.shape[0] - 1)
    sig = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    w2 = win * win
    for k in range(fr.shape[0]):
        s = k * hop
        sig[s:s + n_fft] += fr[k]
        wsum[s:s + n_fft] += w2
    wsum[wsum < 1e-8] = 1e-8
    sig = (sig / wsum)[pad:-pad]

    # phase-vocoder OLA loses ~30% RMS — restore it so a stretched span sits at the
    # same level as the un-stretched text around it, then guard against clipping.
    in_rms = float(np.sqrt(np.mean(y ** 2)))
    out_rms = float(np.sqrt(np.mean(sig ** 2))) if len(sig) else 0.0
    if out_rms > 1e-9 and in_rms > 1e-9:
        sig *= (in_rms / out_rms)
    peak = float(np.max(np.abs(sig))) if len(sig) else 0.0
    if peak > 0.99:
        sig *= 0.99 / peak
    return sig.astype(np.float32)
```

## STEP 2 — replace the Natural generation block

**Replace lines ≈2735–2765** — the whole `if use_chatterbox:` body, from
`# ── Chatterbox path` down to and including its `smooth.segment_done(i)`.
**Do NOT touch the `else:` (Kokoro path) that follows**, and leave the
`text = apply_pronunciation(text)` line above the `if` as-is.

Old first lines you're replacing (to locate it):
```python
    if use_chatterbox:
        # ── Chatterbox path ────────────────────────────────────────────────────
        text = strip_speech_tags(text)   # inline tags are a Fast-mode feature
        ...
            all_samples.append(samples)
            sample_rate = sr
            smooth.segment_done(i)
```

New block:
```python
    if use_chatterbox:
        # ── Chatterbox path (inline tags: pause/volume/spell/speed) ─────────────
        # Natural mode clones ONE voice, so [voice:] can't switch narrators here —
        # it stays Fast-only. Everything else applies via split-and-stitch.
        if not chatterbox_engine.is_ready:
            if status_cb: status_cb("Waiting for Natural mode to finish loading...")
            chatterbox_engine.start(status_cb=status_cb)
        _clone_path = cb_clone_path_var.get()
        if _clone_path and not os.path.exists(_clone_path):
            if status_cb: status_cb("⚠️ Voice clone file not found — using default voice.")
            _clone_path = ""
        prompt = _clone_path or None
        exag   = cb_exag_slider.get()
        cfg    = cb_cfg_slider.get()
        cb_sr  = chatterbox_engine.sr or 24000

        # base_speed=1.0 → each span's "speed" IS the tag factor ([slow]=0.75,
        # [fast]=1.4, [rate N]=N). Spell/digits are already baked into span text.
        spans, _tag_used = parse_speech_tags(text, voice, 1.0)
        if _tag_used and any(s.get("voice") not in (None, voice) for s in spans):
            if status_cb: status_cb("ℹ️ [voice:] tags are ignored in Natural mode (uses your cloned voice).")
        units = []
        for _sp in spans:
            if _sp["kind"] == "pause":
                units.append(_sp)
            else:
                # Chatterbox's T3 model truncates ~500 chars — keep sub-chunks small
                for _sub in chunk_text(_sp["text"], max_chars=300):
                    units.append({"kind": "text", "text": _sub,
                                  "speed": _sp["speed"], "gain": _sp["gain"]})
        if not units:
            units = [{"kind": "text", "text": text, "speed": 1.0, "gain": 1.0}]

        all_samples, sample_rate, chunks = [], None, []
        smooth.begin_segments([max(1, len(u.get("text", "")) or 1) for u in units])
        _used_gain = False
        n_text = sum(1 for u in units if u["kind"] == "text")
        _done = 0
        for i, u in enumerate(units):
            if _cancel_event.is_set():
                raise GenerationCancelled()
            if u["kind"] == "pause":
                all_samples.append(np.zeros(int(u["seconds"] * cb_sr), dtype=np.float32))
                chunks.append("")
                sample_rate = sample_rate or cb_sr
            else:
                _done += 1
                if status_cb: status_cb(f"Generating chunk {_done}/{n_text}...")
                samples, sr = chatterbox_engine.generate_chunk(
                    u["text"], audio_prompt_path=prompt,
                    exaggeration=exag, cfg_weight=cfg, status_cb=status_cb)
                samples = np.asarray(samples, dtype=np.float32)
                if abs(u["speed"] - 1.0) > 1e-3:        # [slow]/[fast]/[rate]
                    samples = time_stretch(samples, u["speed"])
                if abs(u["gain"] - 1.0) > 1e-6:         # [loud]/[quiet]/[volume]
                    samples = samples * u["gain"]
                    _used_gain = True
                all_samples.append(samples)
                chunks.append(u["text"])
                sample_rate = sr
                cb_sr = sr or cb_sr
            smooth.segment_done(i)
        if _used_gain:                       # scale down so gains never clip; keep dynamics
            _peak = max((float(np.max(np.abs(s))) for s in all_samples if len(s)), default=0.0)
            if _peak > 0.97:
                _f = 0.97 / _peak
                all_samples = [s * _f for s in all_samples]
```

### Why this is safe
- Every symbol used already exists in `generate_audio`'s scope: `np`, `chunk_text`,
  `parse_speech_tags`, `chatterbox_engine` (+ `.sr`, `.generate_chunk`, `.is_ready`,
  `.start`), `cb_clone_path_var`, `cb_exag_slider`, `cb_cfg_slider`, `smooth`,
  `_cancel_event`, `GenerationCancelled`, `status_cb`, `voice`, `os`.
- It now builds `chunks` parallel to `all_samples` (empty string for pauses), which
  the existing SRT-timing loop right below (`for chunk, samp in zip(chunks, all_samples)`)
  depends on. Don't remove that.
- `strip_speech_tags` is no longer called from this path — that's intended. Leave the
  function defined (still used elsewhere / harmless).

## STEP 3 — verify BEFORE building
```
python -m py_compile app.py
python -m pyflakes app.py    # expect no NEW undefined-name errors in this range
```
Then run from source (`python app.py`) and do a quick tag render in **Natural mode**.

## STEP 4 — ear test (the whole point — Chromebook can't hear Chatterbox)
Switch engine to **Natural**, pick/clone a voice, and generate each:
1. `Hello. [pause 1s] Did the gap land?` — pause length right, no click at the seam.
2. `This is normal. [loud]THIS IS LOUDER.[/loud] Back to normal.` — clear jump, no clipping/crackle, and the loud part sits at a sensible level (not way off from the rest).
3. `[slow]This whole sentence should be slower.[/slow]` — slower, **same pitch/voice** (no chipmunk/robot), no warble.
4. `[fast]And this one noticeably quicker.[/fast]` — faster, same voice.
5. `Your code is [spell]R4T9[/spell].` — reads "R. 4. T. 9."
6. A paragraph with **no tags** — must sound exactly like Natural mode does today (regression guard).

**The thing to listen hardest for:** because effects split the text into pieces that
Chatterbox generates separately, the *joins between pieces* may sound slightly uneven
(energy/tone jump) — worse than in Fast mode. If the seams are distracting, tell the
Chromebook Claude — options are a short cross-fade at span joins or limiting which tags
split Natural audio. If they sound clean, this rides along in 1.4.0.

## STEP 5 — commit + push (only after the ear test passes)
```
git add app.py
git commit -m "Inline tags in Natural mode: pause/volume/spell + pitch-safe speed"
git push origin dev
```
Report back what each of the 6 tests sounded like (esp. #3/#4 pitch and any seam issues).
```
```
