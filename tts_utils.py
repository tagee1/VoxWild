"""
tts_utils.py — pure logic functions with no UI or audio dependencies.
Extracted from app.py so they can be imported independently for testing.
"""
import re


def format_time(seconds):
    seconds = int(seconds)
    if seconds <= 0: return "0s"
    if seconds < 60: return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"


def chunk_text(text, max_chars=800, min_chars=80):
    sentences = []
    # Protect ellipses ("..." and "..") before splitting on ". " so that
    # "Just listen... and rest" is not torn into "Just listen.." + "and rest".
    _normalized = (text
        .replace("...", "\x00ELLIPSIS3\x00")
        .replace("..", "\x00ELLIPSIS2\x00")
        .replace("!\n", "! ")
        .replace("?\n", "? ")
        .replace(".\n", ". "))
    for s in _normalized.split(". "):
        s = (s.strip()
               .replace("\x00ELLIPSIS3\x00", "...")
               .replace("\x00ELLIPSIS2\x00", ".."))
        if s: sentences.append(s)
    chunks, current = [], ""
    for s in sentences:
        # If the sentence already ends with punctuation (e.g. "Let's begin.")
        # don't append another period — that produces "Let's begin.." which
        # confuses neural TTS models and causes them to drop the last line.
        suffix = " " if s and s[-1] in ".!?" else ". "
        if len(current) + len(s) < max_chars:
            current += s + suffix
        else:
            if current: chunks.append(current.strip())
            current = s + suffix
    if current: chunks.append(current.strip())
    # Merge a short final chunk into the previous one to avoid Chatterbox producing
    # distorted audio on tiny inputs (e.g. "Let's begin." as an orphan chunk).
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()
    return chunks or [text]


def parse_dialogue(text):
    """Parse SPEAKER: text lines. Returns list of (speaker, text) tuples."""
    result = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^([A-Z][A-Z0-9 _\-]*)\s*:\s*(.+)$', line)
        if m:
            result.append((m.group(1).strip(), m.group(2).strip()))
        elif result:
            result[-1] = (result[-1][0], result[-1][1] + ' ' + line)
    return result


def _srt_time(seconds):
    # round() avoids float-precision truncation errors (e.g. 0.9999... → 1 not 0)
    ms = min(round((seconds % 1) * 1000), 999)
    s  = int(seconds) % 60
    m  = (int(seconds) // 60) % 60
    h  = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wrap_for_subtitle(text, max_line=42):
    """Word-wrap text into 2-line subtitle blocks."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if len(test) <= max_line:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    # Pair into 2-line blocks
    blocks = []
    for i in range(0, len(lines), 2):
        blocks.append("\n".join(lines[i:i+2]))
    return blocks or [text]


class GenerationCancelled(Exception):
    """Raised inside generate_audio() when the user cancels mid-generation."""


def fmt_err(e):
    """Return a plain-English error string with a crash code for the status label.

    Format: "Human readable message  [E###]"
    Codes:
      E001 — Out of memory / RAM
      E002 — Audio device not found
      E003 — File permission denied
      E004 — Disk full
      E005 — File not found
      E006 — Network / connection error
      E007 — Model tensor mismatch (bad audio prompt)
      E008 — Model device mismatch
      E009 — Unsupported audio file format
      E010 — Voice clone recording too quiet
      E011 — Voice clone recording too short
      E012 — Failed to reset voice conditioning (default voice switch)
      E013 — Audio enhancement failed (resemble-enhance error)
      E099 — Unknown / unrecognised error
    """
    raw  = str(e).lower()
    full = str(e)

    def _r(msg, code):
        return f"{msg}  [{code}]"

    # ── Memory / RAM ──────────────────────────────────────────────────────────
    if any(k in raw for k in ("out of memory", "not enough memory", "not enough ram",
                               "memoryerror", "paging file", "winerror 1455", "cannot allocate")):
        return _r("Not enough RAM — close other apps and try again", "E001")

    # ── Audio device ──────────────────────────────────────────────────────────
    if any(k in raw for k in ("no default output", "invalid device", "portaudio",
                               "device unavailable", "hostapierror")):
        return _r("Audio device not found — check your speakers/headphones in Windows Sound settings", "E002")

    # ── File permission / disk ────────────────────────────────────────────────
    if "permissionerror" in raw or "access is denied" in raw:
        return _r("Permission denied — the file may be open in another app, or the folder is read-only", "E003")
    if any(k in raw for k in ("disk full", "no space left", "there is not enough space")):
        return _r("Disk full — free up space and try again", "E004")

    # ── File not found ────────────────────────────────────────────────────────
    if "filenotfounderror" in raw or "no such file" in raw:
        return _r("File not found — it may have been moved or deleted", "E005")

    # ── Network / download ────────────────────────────────────────────────────
    if any(k in raw for k in ("connectionerror", "connection refused", "timed out",
                               "network", "ssl", "certificate")):
        return _r("Network error — check your internet connection and try again", "E006")

    # ── Model / tensor errors (Chatterbox) ───────────────────────────────────
    if "sizes of tensors must match" in raw or "size does not match" in raw:
        return _r("Model error — audio prompt may be incompatible. Try a different voice sample", "E007")
    if "expected all tensors to be on the same device" in raw:
        return _r("Model error — device mismatch. Restart the app and try again", "E008")

    # ── Voice clone quality / state ───────────────────────────────────────────
    if "too quiet" in raw or "e010" in raw:
        return _r("Voice clone recording is too quiet — re-record in a louder environment", "E010")
    if "too short" in raw or "voice clone" in raw and "short" in raw:
        return _r("Voice clone recording is too short — use at least 6 seconds of clear speech", "E011")
    if "failed to reset voice conditioning" in raw or "e012" in raw:
        return _r("Failed to reset to default voice — restart Natural mode to recover", "E012")
    if "e013" in raw or "enhancement" in raw or "resemble" in raw:
        # Re-surface the descriptive message from enhance_audio rather than
        # collapsing it — strip the tag and pass through the detail.
        detail = full.replace("  [E013]", "").replace(" [E013]", "")
        first = next((l.strip() for l in detail.splitlines() if l.strip()), detail)
        return _r(first[:180], "E013")

    # ── Audio file format ─────────────────────────────────────────────────────
    if any(k in raw for k in ("sndfile", "unknown format", "could not read",
                               "unrecognized audio", "unsupported format")):
        return _r("Unsupported audio format — use a WAV or MP3 file", "E009")

    # ── Fallback: trim to first meaningful line, strip stack trace noise ──────
    first_line = next((l.strip() for l in full.splitlines() if l.strip()), full)
    clean = "".join(ch if ch.isprintable() or ch == " " else "?" for ch in first_line)
    if clean.startswith(("RuntimeError:", "ValueError:", "OSError:", "TypeError:")):
        clean = clean.split(":", 1)[-1].strip()
    return _r(clean[:100], "E099")


def estimate_audio_duration(text, speed):
    """Estimate playback duration in seconds for TTS output at the given speed multiplier."""
    words = len(text.split())
    wpm   = 150 * speed
    return (words / wpm) * 60 if wpm > 0 else 0


def build_srt(segments):
    """Build SRT string from list of (start_sec, end_sec, text)."""
    entries = []
    idx = 1
    for start, end, text in segments:
        blocks = _wrap_for_subtitle(text)
        dur = max(end - start, 0.5)
        total_words = len(text.split()) or 1
        t = start
        for block in blocks:
            block_words = len(block.split()) or 1
            block_dur   = max(dur * block_words / total_words, 0.5)
            entries.append((idx, t, t + block_dur, block))
            t += block_dur
            idx += 1
    lines = []
    for idx, start, end, text in entries:
        lines.append(str(idx))
        lines.append(f"{_srt_time(start)} --> {_srt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


# ── History card display helpers ──────────────────────────────────────────────

def history_card_preview(text: str, max_chars: int = 100) -> str:
    """Return a single-line, length-capped preview string for history cards."""
    preview = text.replace("\n", " ").strip()
    if len(preview) > max_chars:
        preview = preview[:max_chars] + "…"
    return preview


def history_card_voice_label(voice: str, max_len: int = 18) -> str:
    """Shorten a voice profile name for display in a narrow history card.

    Rules (applied in order):
    1. If the name contains ' - ', take only the part after the last ' - '.
    2. Strip ' (Best)' suffix.
    3. If still longer than max_len, truncate with '…'.
    """
    short = voice.split(" - ")[-1].replace(" (Best)", "") if " - " in voice else voice
    short = short.strip()
    if len(short) > max_len:
        short = short[: max_len - 1] + "…"
    return short
