import re

def clean_text(text: str) -> tuple[str, list[str]]:
    """
    Clean and normalize text for TTS.
    Returns (cleaned_text, list_of_changes_made)
    """
    changes = []
    original = text

    # ── Fix encoding artifacts ────────────────────────────────────────────────
    replacements = {
        "\u2018": "'", "\u2019": "'",   # curly single quotes
        "\u201c": '"', "\u201d": '"',   # curly double quotes
        "\u2013": "-", "\u2014": "-",   # en dash, em dash
        "\u2026": "...",                # ellipsis character
        "\u00a0": " ",                  # non-breaking space
        "\u200b": "",                   # zero-width space
        "\ufffd": "",                   # replacement character
        "\u2022": "",                   # bullet point
        "\u25cf": "",                   # filled circle bullet
        "\u2023": "",                   # triangle bullet
        "\u00e2\u0080\u0099": "'",      # mangled apostrophe from UTF-8
    }
    fixed_chars = False
    for bad, good in replacements.items():
        if bad in text:
            text = text.replace(bad, good)
            fixed_chars = True
    if fixed_chars:
        changes.append("Fixed special characters")

    # ── Remove HTML tags ──────────────────────────────────────────────────────
    html_cleaned = re.sub(r"<[^>]+>", " ", text)
    if html_cleaned != text:
        text = html_cleaned
        changes.append("Removed HTML tags")

    # ── Remove URLs ───────────────────────────────────────────────────────────
    url_cleaned = re.sub(r"http[s]?://\S+", "", text)
    if url_cleaned != text:
        text = url_cleaned
        changes.append("Removed URLs")

    # ── Remove markdown formatting ────────────────────────────────────────────
    md_cleaned = text
    md_cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", md_cleaned)   # bold
    md_cleaned = re.sub(r"\*(.+?)\*",     r"\1", md_cleaned)   # italic
    md_cleaned = re.sub(r"__(.+?)__",     r"\1", md_cleaned)   # bold
    md_cleaned = re.sub(r"_(.+?)_",       r"\1", md_cleaned)   # italic
    md_cleaned = re.sub(r"#{1,6}\s*",     "",    md_cleaned)   # headers
    md_cleaned = re.sub(r"`(.+?)`",       r"\1", md_cleaned)   # inline code
    md_cleaned = re.sub(r"^\s*[-*+]\s+",  "",    md_cleaned, flags=re.MULTILINE)  # bullets
    md_cleaned = re.sub(r"^\s*\d+\.\s+",  "",    md_cleaned, flags=re.MULTILINE)  # numbered lists
    if md_cleaned != text:
        text = md_cleaned
        changes.append("Removed markdown formatting")

    # ── Remove excessive punctuation ──────────────────────────────────────────
    punct_cleaned = re.sub(r"\.{4,}", "...", text)         # 4+ dots -> ellipsis
    punct_cleaned = re.sub(r"!{2,}", "!", punct_cleaned)   # multiple !
    punct_cleaned = re.sub(r"\?{2,}", "?", punct_cleaned)  # multiple ?
    punct_cleaned = re.sub(r"-{3,}", "-", punct_cleaned)   # long dashes
    if punct_cleaned != text:
        text = punct_cleaned
        changes.append("Fixed excessive punctuation")

    # ── Fix spacing ───────────────────────────────────────────────────────────
    space_cleaned = re.sub(r" {2,}", " ", text)            # double spaces
    space_cleaned = re.sub(r"\t+", " ", space_cleaned)     # tabs to spaces
    space_cleaned = re.sub(r"\n{3,}", "\n\n", space_cleaned)  # 3+ newlines -> 2
    space_cleaned = re.sub(r" +\n", "\n", space_cleaned)   # trailing spaces
    space_cleaned = re.sub(r"\n +", "\n", space_cleaned)   # leading spaces on lines
    if space_cleaned != text:
        text = space_cleaned
        changes.append("Fixed spacing issues")

    # ── Fix missing space after punctuation ───────────────────────────────────
    punct_space = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
    if punct_space != text:
        text = punct_space
        changes.append("Fixed missing spaces after punctuation")

    # ── Remove bracketed content like [1] [citation] [image] ─────────────────
    bracket_cleaned = re.sub(r"\[\d+\]", "", text)         # footnote numbers [1]
    bracket_cleaned = re.sub(r"\[image\]", "", bracket_cleaned, flags=re.IGNORECASE)
    bracket_cleaned = re.sub(r"\[photo\]", "", bracket_cleaned, flags=re.IGNORECASE)
    bracket_cleaned = re.sub(r"\[video\]", "", bracket_cleaned, flags=re.IGNORECASE)
    bracket_cleaned = re.sub(r"\[edit\]",  "", bracket_cleaned, flags=re.IGNORECASE)
    if bracket_cleaned != text:
        text = bracket_cleaned
        changes.append("Removed bracketed references")

    # ── Expand common abbreviations for better TTS ────────────────────────────
    abbrevs = {
        r"\bDr\.":   "Doctor",
        r"\bMr\.":   "Mister",
        r"\bMrs\.":  "Missus",
        r"\bMs\.":   "Miss",
        r"\bProf\.": "Professor",
        r"\bSt\.":   "Saint",
        r"\betc\.":  "etcetera",
        r"\bvs\.":   "versus",
        r"\be\.g\.": "for example",
        r"\bi\.e\.": "that is",
    }
    expanded_any = False
    for pattern, replacement in abbrevs.items():
        expanded = re.sub(pattern, replacement, text)
        if expanded != text:
            text = expanded
            expanded_any = True
    if expanded_any:
        changes.append("Expanded abbreviations")

    # ── Strip leading/trailing whitespace ─────────────────────────────────────
    text = text.strip()

    # ── Deduplicate changes list ───────────────────────────────────────────────
    changes = list(dict.fromkeys(changes))

    return text, changes


def preview_clean(text: str) -> str:
    """Return a summary of what would be changed without applying."""
    _, changes = clean_text(text)
    if not changes:
        return "✅ Text looks clean — no changes needed."
    return "Would fix:\n• " + "\n• ".join(changes)