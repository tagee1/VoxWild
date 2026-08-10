# HANDOFF — 4 Natural-mode fixes from user testing (2026-08-09)

**For the Windows Claude at `C:\tts-app\VoxWild`, branch `dev`.** First: `git pull` (you should
land on commit `1424357` or later — the 1.4.0 batch + enhance/progress fixes). Then work through
the four items below. **You can run and hear the app; the Chromebook Claude can't** — so YOU are the
judge of how things sound and look. Chromebook diagnosed each issue against the code and (for the
hard one) hands you a numerically-verified drop-in. When done: `python -m py_compile app.py`, run
from source, user re-tests, then commit + push.

Context — the user's exact words, so nothing is lost:
> Natural mode bottom bar (Import / +Queue / etc.): the two buttons next to Clear are small and
> don't tell me what they do; there's no toggle like Fast mode's "sentence/word". The progress bar's
> "still working" section is good — keep it for long jobs — but the bar should mostly load accurately
> to real time; it goes the same speed then sits at 98% with just "still working on section" for 2+
> min. A 3-sentence gen loaded in 3 sections, slower/more accurate, but still "still working" 4+ min.
> Slow mode on Natural is robotic/scary; fast mode also robotic, not as bad. The "R4T9" spell test
> sounded good but had weird spacing between "4" and "T".

The user's PC: **8 GB RAM, Natural runs Enhance/Chatterbox on CPU** — slow. Keep that in mind.

---

## ISSUE 1 (biggest) — Natural [slow]/[fast] sound robotic

**Cause:** Natural mode has no speed knob, so we time-stretch the generated audio. The current
stretch is a **phase vocoder** (`def time_stretch` in app.py) — mathematically pitch-correct but it
adds that metallic/robotic "phasiness", worst at bigger stretches (so [slow] is worse than [fast]).

**Fix:** replace it with **WSOLA** (waveform-similarity overlap-add — time-domain, no phasiness,
the standard choice for speech). Below is a drop-in the Chromebook already verified for correctness
(duration err 0.00%, energy 1.000, spectral envelope LTAS 0.999 — cleaner than the phase vocoder).
**Only the SOUND is unverified — that's your job to judge by ear.**

### Step 1a — add this function next to `time_stretch` (paste right after `time_stretch` ends)
```python
def wsola(x, rate, frame=1024, search=360):
    """Time-stretch WITHOUT changing pitch, time-domain (speech-friendly, no robotic phasiness).
    rate>1 => faster/shorter ([fast]=1.4); rate<1 => slower/longer ([slow]=0.75)."""
    x = np.asarray(x, dtype=np.float32)
    if abs(rate - 1.0) < 1e-3 or len(x) < frame * 2:
        return x
    Hs = frame // 2
    Ha = rate * Hs
    win = np.hanning(frame).astype(np.float32)
    out_len = int(np.ceil(len(x) / rate)) + frame
    out = np.zeros(out_len, dtype=np.float32)
    ow = np.zeros(out_len, dtype=np.float32)
    out[:frame] += x[:frame] * win
    ow[:frame] += win
    prev_read = 0
    read = float(Ha)
    syn = Hs
    while syn + frame < out_len:
        tgt = prev_read + Hs
        if tgt + frame > len(x):
            break
        target = x[tgt:tgt + frame]
        center = int(round(read))
        lo = max(0, center - search)
        hi = min(len(x) - frame, center + search)
        if hi <= lo:
            break
        best, best_corr = lo, -1e30
        for s in range(lo, hi + 1, 2):
            c = float(np.dot(x[s:s + frame], target))
            if c > best_corr:
                best_corr, best = c, s
        out[syn:syn + frame] += x[best:best + frame] * win
        ow[syn:syn + frame] += win
        prev_read = best
        read += Ha
        syn += Hs
    ow[ow < 1e-6] = 1e-6
    out = (out / ow)[:int(round(len(x) / rate))]
    ir = float(np.sqrt(np.mean(x ** 2)))
    orr = float(np.sqrt(np.mean(out ** 2))) if len(out) else 0.0
    if orr > 1e-9 and ir > 1e-9:
        out *= ir / orr
    p = float(np.max(np.abs(out))) if len(out) else 0.0
    if p > 0.99:
        out *= 0.99 / p
    return out.astype(np.float32)
```

### Step 1b — use it in the Natural path
Find this line (in `generate_audio`, the Chatterbox branch):
```python
                if abs(u["speed"] - 1.0) > 1e-3:        # [slow]/[fast]/[rate]
                    samples = time_stretch(samples, u["speed"])
```
Change the call to `wsola`:
```python
                if abs(u["speed"] - 1.0) > 1e-3:        # [slow]/[fast]/[rate]
                    samples = wsola(samples, u["speed"])
```
(Leave `time_stretch` defined — harmless. Fast mode never uses either; it uses Kokoro's native speed.)

### Step 1c — LISTEN, then decide
Generate in Natural mode: `[slow]This should be slower.[/slow]` and `[fast]And this quicker.[/fast]`
- **If WSOLA sounds natural** → keep it. Done.
- **If it STILL sounds robotic/bad** → Chatterbox just doesn't take stretching well. Don't ship it —
  make speed tags a **no-op in Natural** (like `[voice:]` already is): replace the two lines from 1b with
```python
                # speed tags stay Fast-only — stretching Chatterbox sounds unnatural
```
  and tell the user speed is Fast-mode-only. Pause/loud/quiet/spell still work in Natural regardless.

---

## ISSUE 2 — progress bar races to 98% then sits for minutes (Natural)

**Cause:** the estimate IS engine-aware — for Natural it uses `cb_words_per_second`, but its
**default is 0.5 wps, way too fast for CPU Chatterbox on this 8 GB PC.** The user saw ~1 sentence
take 2+ min (~0.08 wps). So the bar fills to 98% in its estimated ~15 s, then parks in "still
working" for the real remaining time. It gets better over runs because the calibration EMA learns —
but slowly, and from too-fast a starting point.

**Fix (tune on THIS machine — you can measure real timing, Chromebook can't):**
1. Generate one ~10-word sentence in Natural mode, time it. Compute wps = words / seconds
   (e.g. 10 words / 130 s ≈ 0.077).
2. In `app.py`, lower the Natural default to match — both places that read it:
   - `get_words_per_second()`: `return data.get("cb_words_per_second") or 0.5` → change `0.5` to your measured value (≈ `0.08`).
   - `record_calibration()`: `prior = data.get("cb_words_per_second") or 0.5` → same value.
   - Also the fallback in that same function: `(0.5 if use_cb else 8)` → `(0.08 if use_cb else 8)`.
3. Make Natural converge faster: in `record_calibration`, the `_EMA_ALPHA = 0.4` is shared; for the
   `use_cb` branch bump responsiveness, e.g. use `0.6` just for cb (a longer-term calibration weight
   so it trusts the machine's real speed sooner). Optional but helps the first few runs.
4. **Keep the "still working" timer** — the user explicitly wants it for genuinely long jobs. This
   fix just makes the bar *pace correctly* so "still working" is the exception, not every run.

(Deeper truth, FYI: Chatterbox time is dominated by fixed per-chunk model overhead, not word count,
so words/sec is a rough model. A conservative default + the "still working" backstop is the pragmatic
fix; a per-chunk time model is a bigger future change.)

---

## ISSUE 3 — two cryptic buttons on the bottom bar; no self-explaining labels

**What they are** (shared bar `txt_btns`, so they show in BOTH modes — the user just noticed them in
Natural): the buttons between `Dict`/`Clear` region are **`Clean`** (`command=show_text_cleaner` — the
text cleanup tool) and **`Dict`** (`command=... open_pronunciation_window` — the pronunciation
dictionary). Neither has a tooltip, unlike the Read-along switch.

**Fix:** give them tooltips (and optionally clearer labels). Assign each to a variable and add
`_Tooltip` (the class already used elsewhere). Find the `text="Clean"` and `text="Dict"` buttons and
change to:
```python
_clean_btn = ctk.CTkButton(txt_btns, text="Clean", command=show_text_cleaner,
              width=66, height=30, font=ctk.CTkFont(family="Segoe UI", size=12), **BTN_GHOST)
_clean_btn.pack(side="left", padx=(0, 5))
_Tooltip(_clean_btn, "Clean up pasted text — fix smart quotes, odd spacing, stray characters")
_dict_btn = ctk.CTkButton(txt_btns, text="Dict",
              command=lambda: open_pronunciation_window(app),
              width=54, height=30, font=ctk.CTkFont(family="Segoe UI", size=12), **BTN_GHOST)
_dict_btn.pack(side="left", padx=(0, 5))
_Tooltip(_dict_btn, "Pronunciation dictionary — teach VoxWild how to say specific words")
```
(Confirm those tooltip descriptions match what the tools actually do, then keep or reword.)

**Also verify visually:** the Read-along **Sentence / Word** toggle lives in the SAME shared bar
(`ra_ctrl` → `ra_mode_seg`), so per the code it should appear in Natural mode too. Confirm it's
actually visible in Natural and not being pushed off-screen / clipped when the Chatterbox controls
are shown. If it's clipped, that's a layout fix in the bottom row.

---

## ISSUE 4 (minor) — `[spell]` uneven spacing (the "4"→"T" gap)

`_tag_spell` turns `R4T9` into `R. 4. T. 9.` by appending `'. '` after each character. In Natural
mode, Chatterbox's prosody on those periods is uneven — the digit→letter boundary ("4. T.") gets an
odd gap. Low priority; **tune by ear.** Try, in `_tag_spell`, a gentler separator than `'. '` — e.g.
a comma-space `', '`, or a single space, or a short SSML-ish pause — and pick what reads evenest on
Chatterbox. (Fast mode already sounds fine, so whatever you change, re-check Fast too.)

---

## Wrap up
```
python -m py_compile app.py
python -m pyflakes app.py     # no new undefined names
python app.py                 # run from source, user re-tests all four
```
Then commit + push:
```
git add app.py
git commit -m "Natural-mode fixes: WSOLA speed (or Fast-only), pace progress bar, tooltip Clean/Dict, [spell] spacing"
git push origin dev
```
Report back what slow/fast sound like now (WSOLA vs disabled), whether the bar paces right, and the
[spell] result. Delete this file when the fixes are in.
