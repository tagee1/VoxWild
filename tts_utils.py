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


def chunk_text(text, max_chars=800):
    sentences = []
    for s in text.replace("!\n", "! ").replace("?\n", "? ").replace(".\n", ". ").split(". "):
        s = s.strip()
        if s: sentences.append(s)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < max_chars:
            current += s + ". "
        else:
            if current: chunks.append(current.strip())
            current = s + ". "
    if current: chunks.append(current.strip())
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
    """Return a single-line, printable error string suitable for a status label."""
    msg = str(e)
    first_line = next((l.strip() for l in msg.splitlines() if l.strip()), msg)
    return "".join(ch if ch.isprintable() or ch == " " else "?" for ch in first_line)


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
