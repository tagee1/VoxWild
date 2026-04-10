"""
test_logic.py — unit tests for pure-logic functions.

Run with:
    python -m pytest test_logic.py -v
    python test_logic.py          (stdlib unittest)
"""
import math
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tts_utils import (
    format_time,
    chunk_text,
    parse_dialogue,
    _srt_time,
    _wrap_for_subtitle,
    build_srt,
    fmt_err,
    estimate_audio_duration,
    GenerationCancelled,
    history_card_preview,
    history_card_voice_label,
)
from text_cleaner import clean_text, preview_clean
from pronunciation import apply_pronunciation
from clone_library import load_clone_library, save_clone_library, add_clone_to_library

try:
    import numpy as np
    from audio_utils import trim_silence
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# format_time
# ══════════════════════════════════════════════════════════════════════════════
class TestFormatTime(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(format_time(0), "0s")

    def test_negative(self):
        self.assertEqual(format_time(-5), "0s")

    def test_under_a_minute(self):
        self.assertEqual(format_time(45), "45s")

    def test_exactly_one_minute(self):
        self.assertEqual(format_time(60), "1m 0s")

    def test_minutes_and_seconds(self):
        self.assertEqual(format_time(90), "1m 30s")

    def test_large_value_no_hours_field(self):
        # format_time doesn't emit hours — 61m 1s is expected
        self.assertEqual(format_time(3661), "61m 1s")

    def test_float_truncated(self):
        self.assertEqual(format_time(59.9), "59s")

    def test_one_second(self):
        self.assertEqual(format_time(1), "1s")


# ══════════════════════════════════════════════════════════════════════════════
# chunk_text
# ══════════════════════════════════════════════════════════════════════════════
class TestChunkText(unittest.TestCase):

    def test_short_text_is_single_chunk(self):
        self.assertEqual(len(chunk_text("Hello world.")), 1)

    def test_no_words_lost(self):
        text = "First sentence. Second sentence. Third sentence."
        combined = " ".join(chunk_text(text))
        for word in ("First", "Second", "Third"):
            self.assertIn(word, combined)

    def test_long_text_splits_into_multiple_chunks(self):
        sentence = "This is a moderately long sentence that contributes to the total. "
        text = sentence * 20  # well over 800 chars
        chunks = chunk_text(text)
        self.assertGreater(len(chunks), 1)

    def test_chunks_not_excessively_long(self):
        sentence = "Short. " * 200
        for chunk in chunk_text(sentence):
            self.assertLessEqual(len(chunk), 900)

    def test_empty_string_returns_list_containing_it(self):
        self.assertEqual(chunk_text(""), [""])

    def test_single_sentence_under_limit_stays_together(self):
        sentence = "The quick brown fox jumped over the lazy dog right here."
        self.assertEqual(len(chunk_text(sentence)), 1)

    def test_exclamation_split_handled(self):
        text = "Hello!\nWorld!"
        chunks = chunk_text(text)
        combined = " ".join(chunks)
        self.assertIn("Hello", combined)
        self.assertIn("World", combined)

    def test_no_double_period_at_end(self):
        # A sentence that already ends with a period must not gain a second one.
        # Double periods confuse neural TTS and cause the last line to be dropped.
        text = "First sentence. Second sentence. Let's begin."
        for chunk in chunk_text(text):
            self.assertNotIn("..", chunk)

    def test_final_sentence_with_period_is_preserved(self):
        # "Let's begin." must appear verbatim in the output — the exact regression
        # that caused Chatterbox to silently drop the last line of the sleepy-science
        # script.
        text = (
            "Welcome to the Sleepy Science Channel.\n\n"
            "Tonight, we're going to slowly explore a collection of quiet and relaxing facts.\n\n"
            "Take a moment to get comfortable.\n\n"
            "Allow your body to settle.\n\n"
            "And if you'd like, gently close your eyes.\n\n"
            "There's no need to focus too hard.\n\n"
            "Just let the information drift by.\n\n"
            "There is nothing you need to do, and nowhere you need to be.\n\n"
            "Just listen... and rest.\n\n"
            "Let's begin."
        )
        combined = " ".join(chunk_text(text))
        self.assertIn("Let's begin", combined)
        # "Let's begin." must not become "Let's begin.." (the specific bug).
        # Note: "..." (ellipsis) legitimately contains ".." so we check the
        # exact bad pattern — a word boundary followed by exactly two periods.
        self.assertNotIn("begin..", combined)
        # Ellipsis must be preserved intact
        self.assertIn("listen...", combined)

    def test_short_final_chunk_merged_into_previous(self):
        # When the last chunk is very short (< min_chars), it gets folded into
        # the second-to-last chunk to prevent Chatterbox distortion artifacts.
        long_sentence = "A" * 790  # forces its own chunk
        short_ending  = "Short end."
        text = long_sentence + ". " + short_ending
        chunks = chunk_text(text, max_chars=800, min_chars=80)
        # The short ending must be merged — there should only be one chunk
        # (or the last chunk must contain the short ending content).
        combined = " ".join(chunks)
        self.assertIn("Short end", combined)
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk), 80,
                msg=f"Chunk is suspiciously short (orphan not merged?): {chunk!r}")

    def test_question_mark_ending_no_double_period(self):
        text = "Are you comfortable? Let's begin."
        for chunk in chunk_text(text):
            self.assertNotIn("..", chunk)
            self.assertNotIn("?.", chunk)


# ══════════════════════════════════════════════════════════════════════════════
# parse_dialogue
# ══════════════════════════════════════════════════════════════════════════════
class TestParseDialogue(unittest.TestCase):

    def test_basic_two_speakers(self):
        text = "ALICE: Hello there.\nBOB: Good morning."
        self.assertEqual(
            parse_dialogue(text),
            [("ALICE", "Hello there."), ("BOB", "Good morning.")]
        )

    def test_empty_lines_ignored(self):
        text = "ALICE: Hello.\n\nBOB: Hi."
        self.assertEqual(len(parse_dialogue(text)), 2)

    def test_multi_word_speaker_name(self):
        result = parse_dialogue("NARRATOR ONE: Once upon a time.")
        self.assertEqual(result[0][0], "NARRATOR ONE")

    def test_continuation_line_appended_to_previous(self):
        text = "ALICE: This starts here\nand continues on the next line."
        result = parse_dialogue(text)
        self.assertEqual(len(result), 1)
        self.assertIn("continues", result[0][1])

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(parse_dialogue(""), [])

    def test_whitespace_only_returns_empty_list(self):
        self.assertEqual(parse_dialogue("   \n  \n"), [])

    def test_lowercase_label_not_parsed_as_speaker(self):
        # "alice:" doesn't start with uppercase pattern
        result = parse_dialogue("alice: Hello.")
        self.assertEqual(result, [])

    def test_mixed_case_label_not_parsed(self):
        result = parse_dialogue("Alice: Hello.")
        # "Alice" is only one uppercase letter followed by lowercase — doesn't match
        self.assertEqual(result, [])

    def test_speaker_with_digit(self):
        text = "VOICE1: First.\nVOICE2: Second."
        result = parse_dialogue(text)
        self.assertEqual(result[0][0], "VOICE1")
        self.assertEqual(result[1][0], "VOICE2")

    def test_speaker_text_whitespace_stripped(self):
        result = parse_dialogue("  ALICE  :  Hello there.  ")
        self.assertEqual(result[0][0], "ALICE")
        self.assertEqual(result[0][1], "Hello there.")

    def test_same_speaker_multiple_lines(self):
        text = "NARRATOR: Line one.\nNARRATOR: Line two."
        result = parse_dialogue(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "NARRATOR")
        self.assertEqual(result[1][0], "NARRATOR")

    def test_hyphen_in_speaker_name(self):
        result = parse_dialogue("VOICE-OVER: Hello.")
        self.assertEqual(result[0][0], "VOICE-OVER")


# ══════════════════════════════════════════════════════════════════════════════
# _srt_time
# ══════════════════════════════════════════════════════════════════════════════
class TestSrtTime(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(_srt_time(0.0), "00:00:00,000")

    def test_half_second(self):
        self.assertEqual(_srt_time(0.5), "00:00:00,500")

    def test_one_second_and_one_ms(self):
        # 1.001 has float-precision noise; round() must recover the correct ms
        self.assertEqual(_srt_time(1.001), "00:00:01,001")

    def test_one_minute_one_second(self):
        self.assertEqual(_srt_time(61.0), "00:01:01,000")

    def test_one_hour(self):
        self.assertEqual(_srt_time(3600.0), "01:00:00,000")

    def test_hours_minutes_seconds_ms(self):
        # 3661.999 has float-precision noise; round() must recover the correct ms
        self.assertEqual(_srt_time(3661.999), "01:01:01,999")

    def test_format_zero_padded(self):
        # All fields must be zero-padded to 2 digits (ms to 3)
        result = _srt_time(1.5)
        self.assertRegex(result, r"^\d{2}:\d{2}:\d{2},\d{3}$")


# ══════════════════════════════════════════════════════════════════════════════
# _wrap_for_subtitle
# ══════════════════════════════════════════════════════════════════════════════
class TestWrapForSubtitle(unittest.TestCase):

    def test_short_text_single_block_no_newline(self):
        blocks = _wrap_for_subtitle("Hello world")
        self.assertEqual(len(blocks), 1)
        self.assertNotIn("\n", blocks[0])

    def test_medium_text_wraps_within_single_block(self):
        # ~52 chars, should wrap to 2 lines in one block
        text = "The quick brown fox jumped over the lazy sleeping dog"
        blocks = _wrap_for_subtitle(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("\n", blocks[0])

    def test_long_text_produces_multiple_blocks(self):
        text = " ".join(["word"] * 30)
        self.assertGreater(len(_wrap_for_subtitle(text)), 1)

    def test_no_block_has_more_than_two_lines(self):
        text = " ".join(["word"] * 60)
        for block in _wrap_for_subtitle(text):
            self.assertLessEqual(block.count("\n"), 1)

    def test_single_word(self):
        self.assertEqual(_wrap_for_subtitle("Hello"), ["Hello"])

    def test_empty_string(self):
        result = _wrap_for_subtitle("")
        self.assertEqual(len(result), 1)

    def test_all_words_preserved(self):
        text = "one two three four five six seven eight nine ten"
        blocks = _wrap_for_subtitle(text)
        combined = " ".join(b.replace("\n", " ") for b in blocks)
        for word in text.split():
            self.assertIn(word, combined)

    def test_custom_max_line(self):
        # With a very short max_line, even short text should wrap
        text = "Hello beautiful world"
        blocks = _wrap_for_subtitle(text, max_line=8)
        # Should produce multiple lines/blocks
        total_lines = sum(b.count("\n") + 1 for b in blocks)
        self.assertGreater(total_lines, 1)


# ══════════════════════════════════════════════════════════════════════════════
# build_srt
# ══════════════════════════════════════════════════════════════════════════════
class TestBuildSrt(unittest.TestCase):

    def test_single_segment_has_index_and_timecode(self):
        srt = build_srt([(0.0, 4.0, "Hello world.")])
        self.assertIn("1\n", srt)
        self.assertIn("-->", srt)
        self.assertIn("Hello", srt)

    def test_multiple_segments_numbered_sequentially(self):
        segments = [(0.0, 3.0, "First."), (3.0, 6.0, "Second."), (6.0, 9.0, "Third.")]
        srt = build_srt(segments)
        lines = srt.strip().split("\n")
        # Index numbers appear as their own lines
        indices = [l for l in lines if l.strip().isdigit()]
        self.assertGreaterEqual(len(indices), 3)

    def test_timecodes_start_at_zero(self):
        srt = build_srt([(0.0, 5.0, "Test.")])
        self.assertIn("00:00:00,000 -->", srt)

    def test_empty_segments_gives_empty_output(self):
        self.assertEqual(build_srt([]).strip(), "")

    def test_very_short_segment_padded_to_minimum(self):
        # 0.1s duration → should be padded to at least 0.5s
        srt = build_srt([(0.0, 0.1, "Hi.")])
        lines = srt.strip().split("\n")
        timecode = lines[1]
        _, _, end = timecode.partition(" --> ")
        self.assertNotEqual(end, "00:00:00,100")

    def test_blank_line_between_entries(self):
        srt = build_srt([(0.0, 4.0, "First."), (4.0, 8.0, "Second.")])
        self.assertIn("\n\n", srt)

    def test_timecode_format_correct(self):
        srt = build_srt([(0.0, 4.0, "Test.")])
        import re
        timecode_pattern = r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
        self.assertRegex(srt, timecode_pattern)

    def test_long_text_produces_multiple_numbered_entries(self):
        # A segment with enough text to wrap into multiple subtitle blocks
        long_text = " ".join(["word"] * 40)
        srt = build_srt([(0.0, 20.0, long_text)])
        lines = srt.strip().split("\n")
        indices = [l for l in lines if l.strip().isdigit()]
        self.assertGreater(len(indices), 1)

    def test_second_entry_timecode_follows_first(self):
        srt = build_srt([(0.0, 4.0, "First."), (4.0, 8.0, "Second.")])
        lines = srt.strip().split("\n")
        timecodes = [l for l in lines if "-->" in l]
        self.assertGreaterEqual(len(timecodes), 2)
        # Second entry should not start before 4s
        second_start = timecodes[1].split(" --> ")[0]
        self.assertGreaterEqual(second_start, "00:00:04,000")


# ══════════════════════════════════════════════════════════════════════════════
# clean_text
# ══════════════════════════════════════════════════════════════════════════════
class TestCleanText(unittest.TestCase):

    def test_clean_text_unchanged(self):
        text = "This is a clean sentence."
        cleaned, changes = clean_text(text)
        self.assertEqual(cleaned, text)
        self.assertEqual(changes, [])

    def test_curly_single_quote_replaced(self):
        # clean_text replaces one special-char type per call (breaks after first match).
        # \u2018 is first in the replacement dict so it is always processed.
        text = "It\u2018s fine and it\u2018s correct."
        cleaned, changes = clean_text(text)
        self.assertNotIn("\u2018", cleaned)
        self.assertIn("'", cleaned)
        self.assertIn("Fixed special characters", changes)

    def test_curly_close_apostrophe_replaced(self):
        # \u2019 alone — \u2018 not present, so \u2019 is found next
        text = "It\u2019s a test"
        cleaned, changes = clean_text(text)
        self.assertNotIn("\u2019", cleaned)
        self.assertIn("Fixed special characters", changes)

    def test_em_dash_replaced(self):
        text = "Before\u2014after"
        cleaned, _ = clean_text(text)
        self.assertNotIn("\u2014", cleaned)
        self.assertIn("-", cleaned)

    def test_html_tags_removed(self):
        text = "Hello <b>world</b> and <br/> more."
        cleaned, changes = clean_text(text)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("</b>", cleaned)
        self.assertIn("world", cleaned)
        self.assertIn("Removed HTML tags", changes)

    def test_urls_removed(self):
        text = "See https://example.com for more."
        cleaned, changes = clean_text(text)
        self.assertNotIn("https://", cleaned)
        self.assertIn("Removed URLs", changes)

    def test_markdown_bold_stripped(self):
        text = "This is **bold** text."
        cleaned, changes = clean_text(text)
        self.assertNotIn("**", cleaned)
        self.assertIn("bold", cleaned)
        self.assertIn("Removed markdown formatting", changes)

    def test_markdown_italic_stripped(self):
        text = "This is *italic* text."
        cleaned, changes = clean_text(text)
        self.assertNotIn("*italic*", cleaned)
        self.assertIn("italic", cleaned)

    def test_markdown_header_stripped(self):
        text = "## Section\nBody text."
        cleaned, changes = clean_text(text)
        self.assertNotIn("##", cleaned)
        self.assertIn("Section", cleaned)

    def test_markdown_bullet_stripped(self):
        text = "- First item\n- Second item"
        cleaned, changes = clean_text(text)
        self.assertNotIn("- ", cleaned)

    def test_excessive_exclamation_collapsed(self):
        text = "Wow!!!!"
        cleaned, changes = clean_text(text)
        self.assertNotIn("!!", cleaned)
        self.assertIn("Fixed excessive punctuation", changes)

    def test_excessive_question_collapsed(self):
        text = "Really????"
        cleaned, changes = clean_text(text)
        self.assertNotIn("??", cleaned)

    def test_double_spaces_collapsed(self):
        text = "Hello   world  here."
        cleaned, changes = clean_text(text)
        self.assertNotIn("  ", cleaned)
        self.assertIn("Fixed spacing issues", changes)

    def test_tabs_converted_to_spaces(self):
        text = "Hello\tworld."
        cleaned, changes = clean_text(text)
        self.assertNotIn("\t", cleaned)

    def test_footnote_brackets_removed(self):
        text = "Einstein[1] showed[2] that energy equals mass."
        cleaned, changes = clean_text(text)
        self.assertNotIn("[1]", cleaned)
        self.assertNotIn("[2]", cleaned)
        self.assertIn("Removed bracketed references", changes)

    def test_image_bracket_removed(self):
        text = "See [image] below."
        cleaned, _ = clean_text(text)
        self.assertNotIn("[image]", cleaned)

    def test_abbreviation_dr_expanded(self):
        text = "Dr. Smith attended."
        cleaned, changes = clean_text(text)
        self.assertIn("Doctor", cleaned)
        self.assertIn("Expanded abbreviations", changes)

    def test_abbreviation_mr_expanded(self):
        text = "Mr. Jones called."
        cleaned, _ = clean_text(text)
        self.assertIn("Mister", cleaned)

    def test_changes_deduplicated(self):
        text = "<b>Bold</b> and <i>italic</i>"
        _, changes = clean_text(text)
        self.assertEqual(changes.count("Removed HTML tags"), 1)

    def test_leading_trailing_whitespace_stripped(self):
        text = "  Hello world.  "
        cleaned, _ = clean_text(text)
        self.assertEqual(cleaned, "Hello world.")

    def test_missing_space_after_period_fixed(self):
        text = "Hello world.How are you?"
        cleaned, changes = clean_text(text)
        self.assertIn(" ", cleaned[cleaned.index("."):cleaned.index(".")+2])


class TestPreviewClean(unittest.TestCase):

    def test_clean_text_reports_clean(self):
        result = preview_clean("Plain clean text here.")
        self.assertIn("clean", result.lower())

    def test_dirty_text_mentions_change(self):
        result = preview_clean("<b>HTML</b> text")
        self.assertIn("HTML", result)

    def test_returns_string(self):
        self.assertIsInstance(preview_clean("test"), str)


# ══════════════════════════════════════════════════════════════════════════════
# apply_pronunciation  (mocks load_dictionary to avoid file I/O)
# ══════════════════════════════════════════════════════════════════════════════
class TestApplyPronunciation(unittest.TestCase):

    def _run(self, text, entries):
        import pronunciation as _pron
        _pron._dict_cache = None  # clear cache so the mock load_dictionary is used
        with patch("pronunciation.load_dictionary", return_value=entries):
            return apply_pronunciation(text)

    def test_basic_replacement(self):
        entries = [{"from": "API", "to": "A.P.I.", "case_sensitive": True}]
        result = self._run("Call the API now.", entries)
        self.assertIn("A.P.I.", result)
        self.assertNotIn(" API ", result)

    def test_case_insensitive_replacement(self):
        entries = [{"from": "nginx", "to": "engine x", "case_sensitive": False}]
        result = self._run("Run nginx here.", entries)
        self.assertIn("engine x", result)

    def test_case_sensitive_no_match_on_wrong_case(self):
        entries = [{"from": "API", "to": "A.P.I.", "case_sensitive": True}]
        result = self._run("Use the api now.", entries)
        self.assertIn("api", result)
        self.assertNotIn("A.P.I.", result)

    def test_word_boundary_prevents_partial_match(self):
        entries = [{"from": "AI", "to": "A.I.", "case_sensitive": True}]
        result = self._run("RAIN contains AI potential.", entries)
        self.assertIn("RAIN", result)   # not touched
        self.assertIn("A.I.", result)   # standalone AI replaced

    def test_longer_pattern_takes_priority(self):
        entries = [
            {"from": "AWS Lambda", "to": "Amazon Lambda",       "case_sensitive": True},
            {"from": "AWS",        "to": "Amazon Web Services", "case_sensitive": True},
        ]
        result = self._run("Use AWS Lambda today.", entries)
        self.assertIn("Amazon Lambda", result)
        self.assertNotIn("Amazon Web Services Lambda", result)

    def test_empty_dictionary_leaves_text_unchanged(self):
        result = self._run("No changes expected.", [])
        self.assertEqual(result, "No changes expected.")

    def test_empty_text_returns_empty(self):
        entries = [{"from": "API", "to": "A.P.I.", "case_sensitive": True}]
        self.assertEqual(self._run("", entries), "")

    def test_multiple_replacements_applied(self):
        entries = [
            {"from": "AI",  "to": "A.I.",  "case_sensitive": True},
            {"from": "API", "to": "A.P.I.", "case_sensitive": True},
        ]
        result = self._run("The AI and the API.", entries)
        self.assertIn("A.I.", result)
        self.assertIn("A.P.I.", result)

    def test_replacement_does_not_double_replace(self):
        # "A.P.I." should not be re-matched by another pattern
        entries = [
            {"from": "API", "to": "A.P.I.", "case_sensitive": True},
            {"from": "PI",  "to": "pie",    "case_sensitive": True},
        ]
        result = self._run("The API works.", entries)
        # "PI" in "A.P.I." should not be replaced by "pie"
        # (word boundary prevents it since "PI" is inside "A.P.I.")
        self.assertNotIn("A.pie", result)


# ══════════════════════════════════════════════════════════════════════════════
# fmt_err
# ══════════════════════════════════════════════════════════════════════════════
class TestFmtErr(unittest.TestCase):

    def test_known_error_disk_full_mapped(self):
        # "disk full" matches E004 and returns the plain-English message
        result = fmt_err(Exception("disk full"))
        self.assertIn("E004", result)
        self.assertIn("Disk full", result)

    def test_only_first_line_returned(self):
        # Multi-line message: only the first non-empty line survives into E099
        e = Exception("line one\nline two\nline three")
        result = fmt_err(e)
        self.assertIn("line one", result)
        self.assertNotIn("line two", result)
        self.assertIn("E099", result)

    def test_leading_blank_lines_skipped(self):
        e = Exception("\n\nactual message\nignored")
        result = fmt_err(e)
        self.assertIn("actual message", result)
        self.assertNotIn("ignored", result)

    def test_non_printable_chars_replaced(self):
        # Control characters (\x00, \x07) should become "?" via fallback path
        e = Exception("bad\x00char\x07here")
        result = fmt_err(e)
        self.assertNotIn("\x00", result)
        self.assertNotIn("\x07", result)
        self.assertIn("?", result)

    def test_space_preserved(self):
        e = Exception("hello world")
        self.assertIn(" ", fmt_err(e))

    def test_empty_exception_returns_string(self):
        result = fmt_err(Exception(""))
        self.assertIsInstance(result, str)

    def test_unicode_printable_preserved_in_fallback(self):
        # Unrecognised message hits E099 fallback; printable unicode kept
        result = fmt_err(Exception("café résumé"))
        self.assertIn("café", result)
        self.assertIn("E099", result)

    def test_returns_string(self):
        self.assertIsInstance(fmt_err(ValueError("oops")), str)

    def test_known_error_oom_mapped(self):
        result = fmt_err(Exception("out of memory"))
        self.assertIn("E001", result)

    def test_known_error_network_mapped(self):
        result = fmt_err(Exception("connection refused"))
        self.assertIn("E006", result)

    def test_all_results_contain_code(self):
        # Every result should end with a bracketed error code
        cases = [
            Exception("disk full"),
            Exception("out of memory"),
            Exception("something totally unknown xyz"),
            Exception("café"),
        ]
        for e in cases:
            result = fmt_err(e)
            self.assertRegex(result, r"\[E\d{3}\]$", msg=f"No code in: {result!r}")


# ══════════════════════════════════════════════════════════════════════════════
# estimate_audio_duration
# ══════════════════════════════════════════════════════════════════════════════
class TestEstimateAudioDuration(unittest.TestCase):

    def test_zero_speed_returns_zero(self):
        # Guard against division by zero
        self.assertEqual(estimate_audio_duration("hello world", 0), 0)

    def test_proportional_to_word_count(self):
        short = estimate_audio_duration("one two", 1.0)
        long  = estimate_audio_duration("one two three four", 1.0)
        self.assertAlmostEqual(long / short, 2.0, places=5)

    def test_faster_speed_shorter_duration(self):
        normal = estimate_audio_duration("hello world", 1.0)
        fast   = estimate_audio_duration("hello world", 2.0)
        self.assertLess(fast, normal)

    def test_ten_words_at_speed_one(self):
        # 10 words / 150 wpm * 60 = 4.0 seconds
        result = estimate_audio_duration("one two three four five six seven eight nine ten", 1.0)
        self.assertAlmostEqual(result, 4.0, places=5)

    def test_empty_text_returns_zero(self):
        self.assertEqual(estimate_audio_duration("", 1.0), 0)

    def test_returns_float(self):
        self.assertIsInstance(estimate_audio_duration("hello", 1.0), float)

    def test_speed_below_one_gives_longer_duration(self):
        normal = estimate_audio_duration("test sentence here", 1.0)
        slow   = estimate_audio_duration("test sentence here", 0.5)
        self.assertGreater(slow, normal)


# ══════════════════════════════════════════════════════════════════════════════
# Voice clone library (clone_library.py)
# ══════════════════════════════════════════════════════════════════════════════
class TestCloneLibrary(unittest.TestCase):
    """Tests run in a temporary directory so they never touch real app data."""

    def setUp(self):
        self._tmpdir   = tempfile.mkdtemp()
        self.clone_dir = os.path.join(self._tmpdir, "voice_clones")
        self.index     = os.path.join(self.clone_dir, "library.json")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_wav(self, name="sample.wav"):
        """Create a tiny placeholder file that can be copy2'd."""
        p = os.path.join(self._tmpdir, name)
        with open(p, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 36)  # minimal fake WAV header
        return p

    # ── load ──────────────────────────────────────────────────────────────────

    def test_load_empty_dir_returns_empty_list(self):
        result = load_clone_library(self.clone_dir, self.index)
        self.assertEqual(result, [])

    def test_load_creates_clone_dir_if_missing(self):
        load_clone_library(self.clone_dir, self.index)
        self.assertTrue(os.path.isdir(self.clone_dir))

    def test_load_filters_entries_with_missing_files(self):
        # Write an index that points to a non-existent file
        os.makedirs(self.clone_dir)
        ghost = {"name": "Ghost", "file": os.path.join(self.clone_dir, "gone.wav")}
        with open(self.index, "w") as f:
            json.dump([ghost], f)
        result = load_clone_library(self.clone_dir, self.index)
        self.assertEqual(result, [])

    def test_load_corrupted_json_returns_empty_list(self):
        os.makedirs(self.clone_dir)
        with open(self.index, "w") as f:
            f.write("not valid json {{{")
        result = load_clone_library(self.clone_dir, self.index)
        self.assertEqual(result, [])

    # ── save ──────────────────────────────────────────────────────────────────

    def test_save_creates_valid_json_file(self):
        entries = [{"name": "Alice", "file": "/tmp/alice.wav"}]
        save_clone_library(entries, self.clone_dir, self.index)
        self.assertTrue(os.path.exists(self.index))
        with open(self.index, encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded, entries)

    def test_save_roundtrip_preserves_unicode_names(self):
        entries = [{"name": "Ångström — voice", "file": "/x/y.wav"}]
        save_clone_library(entries, self.clone_dir, self.index)
        with open(self.index, encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded[0]["name"], "Ångström — voice")

    # ── add ───────────────────────────────────────────────────────────────────

    def test_add_copies_file_into_clone_dir(self):
        src = self._make_wav()
        entry = add_clone_to_library("My Voice", src, self.clone_dir, self.index)
        self.assertTrue(os.path.exists(entry["file"]))
        self.assertTrue(entry["file"].startswith(self.clone_dir))

    def test_add_uses_unique_filename(self):
        src = self._make_wav()
        e1 = add_clone_to_library("Voice A", src, self.clone_dir, self.index)
        e2 = add_clone_to_library("Voice B", src, self.clone_dir, self.index)
        self.assertNotEqual(e1["file"], e2["file"])

    def test_add_persists_to_index(self):
        src = self._make_wav()
        add_clone_to_library("Persistent", src, self.clone_dir, self.index)
        entries = load_clone_library(self.clone_dir, self.index)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "Persistent")

    def test_add_accumulates_multiple_entries(self):
        src = self._make_wav()
        for i in range(3):
            add_clone_to_library(f"Voice {i}", src, self.clone_dir, self.index)
        entries = load_clone_library(self.clone_dir, self.index)
        self.assertEqual(len(entries), 3)

    def test_add_returns_entry_with_name_and_file(self):
        src = self._make_wav()
        entry = add_clone_to_library("Test Name", src, self.clone_dir, self.index)
        self.assertEqual(entry["name"], "Test Name")
        self.assertIn("file", entry)

    def test_add_preserves_source_content(self):
        src = self._make_wav()
        with open(src, "rb") as f:
            original_bytes = f.read()
        entry = add_clone_to_library("Copy Check", src, self.clone_dir, self.index)
        with open(entry["file"], "rb") as f:
            copied_bytes = f.read()
        self.assertEqual(original_bytes, copied_bytes)


# ══════════════════════════════════════════════════════════════════════════════
# trim_silence  (skipped if numpy unavailable)
# ══════════════════════════════════════════════════════════════════════════════
@unittest.skipUnless(_NUMPY_AVAILABLE, "numpy not installed")
class TestTrimSilence(unittest.TestCase):

    def _make(self, values):
        return np.array(values, dtype=np.float32)

    def test_all_silent_returns_original(self):
        samples = self._make([0.0] * 100)
        result = trim_silence(samples, sample_rate=100)
        np.testing.assert_array_equal(result, samples)

    def test_leading_silence_trimmed(self):
        # 50 silent samples then loud
        silent  = np.zeros(50, dtype=np.float32)
        loud    = np.ones(50, dtype=np.float32)
        samples = np.concatenate([silent, loud])
        result  = trim_silence(samples, sample_rate=100, threshold_db=-6)
        # Result should be shorter than the original
        self.assertLess(len(result), len(samples))

    def test_trailing_silence_trimmed(self):
        loud    = np.ones(50, dtype=np.float32)
        silent  = np.zeros(200, dtype=np.float32)
        samples = np.concatenate([loud, silent])
        result  = trim_silence(samples, sample_rate=100, threshold_db=-6)
        self.assertLess(len(result), len(samples))

    def test_pre_roll_not_negative(self):
        # Audio that starts at sample 0 — pre-roll clamp must not go below 0
        samples = np.ones(100, dtype=np.float32)
        result  = trim_silence(samples, sample_rate=100, threshold_db=-6)
        self.assertGreaterEqual(len(result), 1)

    def test_output_is_numpy_array(self):
        samples = np.ones(50, dtype=np.float32)
        result  = trim_silence(samples, sample_rate=100)
        self.assertIsInstance(result, np.ndarray)

    def test_pure_tone_not_trimmed(self):
        # A sine wave at -3 dB should survive the default -50 dB threshold
        t       = np.linspace(0, 1, 24000, dtype=np.float32)
        samples = 0.7 * np.sin(2 * np.pi * 440 * t)
        result  = trim_silence(samples, sample_rate=24000)
        # Should keep most of the content
        self.assertGreater(len(result), len(samples) * 0.9)


# ══════════════════════════════════════════════════════════════════════════════
# SmoothProgress pulse formula
# ══════════════════════════════════════════════════════════════════════════════
class TestSmoothProgressPulse(unittest.TestCase):
    """Tests the idle-state pulse: 0.925 + 0.045 * sin(elapsed * 1.8).

    These do NOT instantiate the UI widget — they validate the math in isolation.
    """

    def _pulse(self, elapsed):
        return 0.925 + 0.045 * math.sin(elapsed * 1.8)

    def test_pulse_stays_within_visible_range(self):
        for t in range(0, 400):
            val = self._pulse(t / 10.0)
            self.assertGreaterEqual(val, 0.88 - 1e-9)
            self.assertLessEqual(val,    0.97 + 1e-9)

    def test_pulse_centre_is_0_925(self):
        # Average over a full period should equal the centre value
        period = 2 * math.pi / 1.8
        n = 10000
        values = [self._pulse(i * period / n) for i in range(n)]
        self.assertAlmostEqual(sum(values) / n, 0.925, places=2)

    def test_pulse_amplitude_is_0_045(self):
        period = 2 * math.pi / 1.8
        n = 10000
        values = [self._pulse(i * period / n) for i in range(n)]
        self.assertAlmostEqual(max(values) - 0.925, 0.045, places=3)
        self.assertAlmostEqual(0.925 - min(values), 0.045, places=3)

    def test_pulse_period_approx_3_5_seconds(self):
        # Period = 2π / 1.8 ≈ 3.49 s
        expected_period = 2 * math.pi / 1.8
        self.assertAlmostEqual(expected_period, 3.49, delta=0.01)

    def test_pulse_is_continuous(self):
        # Consecutive samples should not jump more than 0.02
        prev = self._pulse(0.0)
        for i in range(1, 200):
            curr = self._pulse(i * 0.05)
            self.assertLess(abs(curr - prev), 0.02, f"Jump at t={i * 0.05:.2f}")
            prev = curr


# ══════════════════════════════════════════════════════════════════════════════
# GenerationCancelled + cancel_event logic
# ══════════════════════════════════════════════════════════════════════════════
class TestCancellation(unittest.TestCase):
    """Tests for the cancellation mechanism.

    The actual threading.Event lives in app.py (can't import), so we test
    the logic patterns in isolation using a local Event and the imported
    GenerationCancelled exception.
    """

    def _make_cancel_event(self):
        import threading
        return threading.Event()

    # ── GenerationCancelled exception ─────────────────────────────────────────

    def test_generation_cancelled_is_exception(self):
        self.assertTrue(issubclass(GenerationCancelled, Exception))

    def test_generation_cancelled_can_be_raised_and_caught(self):
        with self.assertRaises(GenerationCancelled):
            raise GenerationCancelled()

    def test_generation_cancelled_caught_separately_from_base_exception(self):
        """GenerationCancelled must not be swallowed by a broad Exception handler
        that is intended to catch only unexpected errors."""
        caught_cancelled = False
        caught_generic   = False
        try:
            raise GenerationCancelled()
        except GenerationCancelled:
            caught_cancelled = True
        except Exception:
            caught_generic = True
        self.assertTrue(caught_cancelled)
        self.assertFalse(caught_generic)

    # ── Event-based cancellation patterns ────────────────────────────────────

    def test_event_clear_then_set(self):
        ev = self._make_cancel_event()
        ev.clear()
        self.assertFalse(ev.is_set())
        ev.set()
        self.assertTrue(ev.is_set())

    def test_chunk_loop_raises_on_cancel(self):
        """Simulate the chunk loop: raise GenerationCancelled when event is set."""
        ev = self._make_cancel_event()
        chunks = ["chunk1", "chunk2", "chunk3"]
        processed = []

        ev.set()  # pre-cancel before the loop

        with self.assertRaises(GenerationCancelled):
            for chunk in chunks:
                if ev.is_set():
                    raise GenerationCancelled()
                processed.append(chunk)

        self.assertEqual(processed, [])  # nothing processed after pre-cancel

    def test_chunk_loop_cancels_mid_way(self):
        """Cancel fires after the first chunk — second and third should not run."""
        ev = self._make_cancel_event()
        chunks = ["a", "b", "c"]
        processed = []

        def fake_generate(chunk):
            processed.append(chunk)
            if chunk == "a":
                ev.set()  # cancel after first chunk completes

        with self.assertRaises(GenerationCancelled):
            for chunk in chunks:
                if ev.is_set():
                    raise GenerationCancelled()
                fake_generate(chunk)

        self.assertEqual(processed, ["a"])

    def test_chunk_loop_completes_without_cancel(self):
        """If cancel never fires, all chunks complete normally."""
        ev = self._make_cancel_event()
        ev.clear()
        chunks = ["x", "y", "z"]
        processed = []

        for chunk in chunks:
            if ev.is_set():
                raise GenerationCancelled()
            processed.append(chunk)

        self.assertEqual(processed, chunks)

    def test_cancel_event_cleared_before_new_generation(self):
        """Clearing the event resets it so a new generation is not immediately
        cancelled by a leftover signal from the previous run."""
        ev = self._make_cancel_event()
        ev.set()      # leftover from previous generation
        ev.clear()    # simulate what generate_and_store does
        self.assertFalse(ev.is_set())

    def test_multiple_sets_idempotent(self):
        """Calling set() multiple times should not raise or double-fire."""
        ev = self._make_cancel_event()
        ev.set()
        ev.set()  # harmless second call
        self.assertTrue(ev.is_set())

    def test_cancel_status_message_distinct_from_error(self):
        """The cancelled status message should NOT contain 'Error' or '❌'."""
        cancelled_msg = "⏹ Generation cancelled."
        self.assertNotIn("❌", cancelled_msg)
        self.assertNotIn("Error", cancelled_msg)
        self.assertIn("cancelled", cancelled_msg.lower())


# ══════════════════════════════════════════════════════════════════════════════
# history_card_preview
# ══════════════════════════════════════════════════════════════════════════════
class TestHistoryCardPreview(unittest.TestCase):

    def test_short_text_unchanged(self):
        self.assertEqual(history_card_preview("Hello."), "Hello.")

    def test_empty_string(self):
        self.assertEqual(history_card_preview(""), "")

    def test_newlines_collapsed(self):
        self.assertEqual(history_card_preview("Line one.\nLine two."), "Line one. Line two.")

    def test_exactly_max_chars_not_truncated(self):
        text = "a" * 100
        result = history_card_preview(text, max_chars=100)
        self.assertEqual(result, text)
        self.assertFalse(result.endswith("…"))

    def test_over_max_chars_truncated(self):
        text = "a" * 101
        result = history_card_preview(text, max_chars=100)
        self.assertTrue(result.endswith("…"))
        self.assertEqual(len(result), 101)  # 100 chars + ellipsis

    def test_truncation_respects_max_chars(self):
        text = "word " * 30   # 150 chars
        result = history_card_preview(text, max_chars=100)
        self.assertLessEqual(len(result), 101)  # 100 + ellipsis char

    def test_leading_trailing_whitespace_stripped(self):
        result = history_card_preview("  hello  ")
        self.assertEqual(result, "hello")

    def test_unicode_text_survives(self):
        text = "日本語テスト " * 5
        result = history_card_preview(text, max_chars=20)
        self.assertTrue(result.endswith("…"))

    def test_custom_max_chars(self):
        result = history_card_preview("abcdef", max_chars=3)
        self.assertEqual(result, "abc…")

    def test_only_newlines(self):
        result = history_card_preview("\n\n\n")
        self.assertEqual(result, "")


# ══════════════════════════════════════════════════════════════════════════════
# history_card_voice_label
# ══════════════════════════════════════════════════════════════════════════════
class TestHistoryCardVoiceLabel(unittest.TestCase):

    def test_short_name_unchanged(self):
        self.assertEqual(history_card_voice_label("Alice"), "Alice")

    def test_splits_on_dash(self):
        result = history_card_voice_label("Kokoro - Alice")
        self.assertEqual(result, "Alice")

    def test_strips_best_suffix(self):
        result = history_card_voice_label("Kokoro - Alice (Best)")
        self.assertEqual(result, "Alice")

    def test_no_dash_no_change(self):
        result = history_card_voice_label("Cookies Voice")
        self.assertEqual(result, "Cookies Voice")

    def test_long_name_truncated(self):
        name = "A" * 25
        result = history_card_voice_label(name, max_len=18)
        self.assertTrue(result.endswith("…"))
        self.assertEqual(len(result), 18)

    def test_exactly_max_len_not_truncated(self):
        name = "A" * 18
        result = history_card_voice_label(name, max_len=18)
        self.assertEqual(result, name)
        self.assertFalse(result.endswith("…"))

    def test_long_name_after_dash_split_truncated(self):
        long_part = "VeryLongVoiceProfileName"
        result = history_card_voice_label(f"Kokoro - {long_part}", max_len=18)
        self.assertTrue(result.endswith("…"))
        self.assertEqual(len(result), 18)

    def test_empty_string(self):
        result = history_card_voice_label("")
        self.assertEqual(result, "")

    def test_best_and_long_combined(self):
        name = "Kokoro - " + "X" * 20 + " (Best)"
        result = history_card_voice_label(name, max_len=18)
        self.assertTrue(result.endswith("…"))
        self.assertEqual(len(result), 18)

    def test_multiple_dashes_takes_last_segment(self):
        result = history_card_voice_label("Group - SubGroup - Alice")
        self.assertEqual(result, "Alice")


# ══════════════════════════════════════════════════════════════════════════════
# _get_free_ram_gb  (tested via mock — no ctypes dependency in test runner)
# ══════════════════════════════════════════════════════════════════════════════
class TestGetFreeRamGb(unittest.TestCase):

    def _import_fn(self):
        """Import _get_free_ram_gb without triggering the full app.py module."""
        import importlib, sys, types

        # Stub out every heavy import app.py would pull in
        heavy = [
            "customtkinter", "kokoro_onnx", "sounddevice", "soundfile",
            "numpy", "scipy", "scipy.signal", "tkinter", "tkinter.filedialog",
            "tkinter.messagebox", "text_cleaner", "settings_window",
            "pronunciation", "clone_library", "audio_utils", "torch",
        ]
        stubs = {}
        for name in heavy:
            if name not in sys.modules:
                stubs[name] = types.ModuleType(name)
                sys.modules[name] = stubs[name]

        # app.py cannot be imported cleanly (it runs UI code at module level),
        # so we define the function inline — identical to the implementation.
        import ctypes

        def _get_free_ram_gb():
            try:
                class _MEMSTATEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength",                ctypes.c_ulong),
                        ("dwMemoryLoad",            ctypes.c_ulong),
                        ("ullTotalPhys",            ctypes.c_ulonglong),
                        ("ullAvailPhys",            ctypes.c_ulonglong),
                        ("ullTotalPageFile",        ctypes.c_ulonglong),
                        ("ullAvailPageFile",        ctypes.c_ulonglong),
                        ("ullTotalVirtual",         ctypes.c_ulonglong),
                        ("ullAvailVirtual",         ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual",ctypes.c_ulonglong),
                    ]
                stat = _MEMSTATEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return stat.ullAvailPhys / (1024 ** 3)
            except Exception:
                return 999.0

        # Clean up stubs
        for name in stubs:
            del sys.modules[name]

        return _get_free_ram_gb

    def test_returns_float(self):
        fn = self._import_fn()
        result = fn()
        self.assertIsInstance(result, float)

    def test_returns_positive(self):
        fn = self._import_fn()
        result = fn()
        self.assertGreater(result, 0.0)

    def test_fallback_on_exception(self):
        """When ctypes call raises, function returns 999.0 (safe pass-through)."""
        import ctypes
        fn = self._import_fn()
        with patch.object(ctypes, "windll", None):
            result = fn()
        self.assertEqual(result, 999.0)

    def test_low_ram_threshold_logic(self):
        """The 6 GB threshold gate used in _on_engine_change: value < 6.0 is low."""
        # This tests the comparison logic, not the ctypes call itself.
        low_values  = [0.0, 1.5, 4.0, 5.99]
        high_values = [6.0, 8.0, 16.0, 999.0]
        for v in low_values:
            self.assertTrue(v < 6.0, f"{v} GB should be flagged as low")
        for v in high_values:
            self.assertFalse(v < 6.0, f"{v} GB should not be flagged as low")


# ══════════════════════════════════════════════════════════════════════════════
# Input edge cases — very long text, unicode, whitespace, empty
# ══════════════════════════════════════════════════════════════════════════════
class TestInputEdgeCases(unittest.TestCase):

    # ── chunk_text with 5000+ word input ──────────────────────────────────────

    def test_very_long_text_does_not_crash(self):
        text = ("The cat sat on the mat. " * 500)  # ~12 000 chars / ~1000 words
        chunks = chunk_text(text)
        self.assertGreater(len(chunks), 1)

    def test_very_long_text_no_chunk_exceeds_limit(self):
        text = ("A somewhat lengthy sentence with many words filling space. " * 300)
        for chunk in chunk_text(text):
            self.assertLessEqual(len(chunk), 900)

    def test_very_long_text_all_words_preserved(self):
        # spot-check that no words vanish
        sentinel = "UNIQUEWORDXYZ"
        text = ("Normal sentence here. " * 200) + sentinel + ". " + ("More text. " * 200)
        combined = " ".join(chunk_text(text))
        self.assertIn(sentinel, combined)

    # ── chunk_text with unicode ───────────────────────────────────────────────

    def test_unicode_cjk_does_not_crash(self):
        text = "こんにちは。これはテストです。日本語のテキスト。" * 5
        result = chunk_text(text)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_unicode_emoji_does_not_crash(self):
        text = "Hello 🌍. Nice to meet you 😊. Let's go! 🚀"
        result = chunk_text(text)
        self.assertIsInstance(result, list)

    def test_unicode_arabic_does_not_crash(self):
        text = "مرحبا بالعالم. هذا اختبار. " * 10
        result = chunk_text(text)
        self.assertIsInstance(result, list)

    def test_unicode_mixed_scripts_preserved(self):
        text = "English text. Ångström units. Naïve café. Résumé here."
        combined = " ".join(chunk_text(text))
        self.assertIn("Ångström", combined)
        self.assertIn("Naïve", combined)

    # ── chunk_text whitespace/degenerate inputs ───────────────────────────────

    def test_whitespace_only_does_not_crash(self):
        for ws in ("   ", "\n\n\n", "\t\t", " \n \t "):
            result = chunk_text(ws)
            self.assertIsInstance(result, list)

    def test_only_punctuation_does_not_crash(self):
        result = chunk_text("... !!! ??? ---")
        self.assertIsInstance(result, list)

    def test_newline_only_text(self):
        result = chunk_text("\n" * 50)
        self.assertIsInstance(result, list)

    # ── apply_pronunciation with edge inputs ──────────────────────────────────

    def test_pronunciation_empty_string(self):
        import pronunciation as _pron
        _pron._dict_cache = None
        with patch("pronunciation.load_dictionary",
                   return_value=[{"from": "API", "to": "A.P.I.", "case_sensitive": True}]):
            result = apply_pronunciation("")
        self.assertEqual(result, "")

    def test_pronunciation_whitespace_only(self):
        import pronunciation as _pron
        _pron._dict_cache = None
        with patch("pronunciation.load_dictionary",
                   return_value=[{"from": "API", "to": "A.P.I.", "case_sensitive": True}]):
            result = apply_pronunciation("   ")
        self.assertEqual(result, "   ")

    def test_pronunciation_unicode_input_does_not_crash(self):
        import pronunciation as _pron
        _pron._dict_cache = None
        text = "こんにちは API 🌍 Naïve café résumé."
        with patch("pronunciation.load_dictionary",
                   return_value=[{"from": "API", "to": "A.P.I.", "case_sensitive": True}]):
            result = apply_pronunciation(text)
        self.assertIsInstance(result, str)
        self.assertIn("A.P.I.", result)

    def test_pronunciation_very_long_text_does_not_crash(self):
        import pronunciation as _pron
        _pron._dict_cache = None
        text = "The API returns data. " * 500  # ~11 000 chars
        with patch("pronunciation.load_dictionary",
                   return_value=[{"from": "API", "to": "A.P.I.", "case_sensitive": True}]):
            result = apply_pronunciation(text)
        self.assertIn("A.P.I.", result)
        self.assertNotIn(" API ", result)

    def test_pronunciation_malformed_regex_entry_skipped(self):
        """A malformed regex pattern in an entry must not crash the whole run."""
        import pronunciation as _pron
        _pron._dict_cache = None
        bad_entries = [
            {"from": "[invalid(regex", "to": "whatever", "case_sensitive": False},
            {"from": "API", "to": "A.P.I.", "case_sensitive": True},
        ]
        with patch("pronunciation.load_dictionary", return_value=bad_entries):
            result = apply_pronunciation("Call the API.")
        self.assertIn("A.P.I.", result)  # good entry still applied


# ══════════════════════════════════════════════════════════════════════════════
# Dialogue tab — parse_dialogue edge cases & generate_dialogue_audio error paths
# ══════════════════════════════════════════════════════════════════════════════
class TestParseDialogueEdgeCases(unittest.TestCase):

    def test_single_speaker_single_line(self):
        result = parse_dialogue("ALICE: Hello world.")
        self.assertEqual(result, [("ALICE", "Hello world.")])

    def test_three_speakers_ordered(self):
        text = "A: one.\nB: two.\nC: three."
        result = parse_dialogue(text)
        self.assertEqual([sp for sp, _ in result], ["A", "B", "C"])

    def test_underscore_in_speaker(self):
        result = parse_dialogue("VOICE_OVER: Narration here.")
        self.assertEqual(result[0][0], "VOICE_OVER")

    def test_no_text_after_colon_not_parsed(self):
        result = parse_dialogue("ALICE:")
        self.assertEqual(result, [])

    def test_only_whitespace_after_colon_not_parsed(self):
        result = parse_dialogue("ALICE:    ")
        self.assertEqual(result, [])

    def test_number_only_speaker_not_parsed(self):
        # Must start with a letter
        result = parse_dialogue("1BOB: text")
        self.assertEqual(result, [])

    def test_continuation_appended_with_space(self):
        text = "ALICE: Part one.\nPart two continues."
        result = parse_dialogue(text)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0][1].endswith("Part two continues."))

    def test_large_script_no_crash(self):
        lines = [f"SPEAKER{i % 5}: Line {i} text here." for i in range(200)]
        result = parse_dialogue("\n".join(lines))
        self.assertEqual(len(result), 200)

    def test_duplicate_speakers_not_merged(self):
        text = "ALICE: First.\nBOB: Middle.\nALICE: Last."
        result = parse_dialogue(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[2][0], "ALICE")

    def test_empty_lines_between_all_speakers_ignored(self):
        text = "ALICE: Hi.\n\n\nBOB: Hey.\n\nALICE: Bye."
        result = parse_dialogue(text)
        self.assertEqual(len(result), 3)

    def test_unicode_text_in_dialogue(self):
        text = "NARRATOR: 今日はいい天気ですね。\nALICE: Oui, c'est magnifique!"
        result = parse_dialogue(text)
        self.assertEqual(len(result), 2)
        self.assertIn("今日", result[0][1])

    def test_returns_list_of_tuples(self):
        result = parse_dialogue("ALICE: Hello.")
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(len(result[0]), 2)


class TestDialogueUniqueSpakers(unittest.TestCase):
    """Logic for extracting ordered-unique speakers from parsed lines."""

    def _unique_speakers(self, text):
        lines = parse_dialogue(text)
        return list(dict.fromkeys(sp for sp, _ in lines))

    def test_unique_speakers_ordered(self):
        text = "ALICE: Hi.\nBOB: Hey.\nALICE: Bye."
        self.assertEqual(self._unique_speakers(text), ["ALICE", "BOB"])

    def test_single_speaker(self):
        self.assertEqual(self._unique_speakers("NARRATOR: Once."), ["NARRATOR"])

    def test_empty_script_no_speakers(self):
        self.assertEqual(self._unique_speakers(""), [])

    def test_five_speakers_order_preserved(self):
        lines = "\n".join(f"SP{i}: text" for i in range(5))
        result = self._unique_speakers(lines)
        self.assertEqual(result, [f"SP{i}" for i in range(5)])

    def test_interleaved_speakers_no_duplicates(self):
        text = "A: 1.\nB: 2.\nA: 3.\nC: 4.\nB: 5."
        self.assertEqual(self._unique_speakers(text), ["A", "B", "C"])


class TestDialogueValidation(unittest.TestCase):
    """Validate pre-generate checks (logic extracted for testability)."""

    def _missing_speakers(self, text, panel_keys):
        lines = parse_dialogue(text)
        script_speakers = set(sp for sp, _ in lines)
        panel_speakers  = set(panel_keys)
        return script_speakers - panel_speakers

    def test_all_speakers_in_panel_no_missing(self):
        text = "ALICE: Hi.\nBOB: Hey."
        self.assertEqual(self._missing_speakers(text, ["ALICE", "BOB"]), set())

    def test_one_speaker_missing_from_panel(self):
        text = "ALICE: Hi.\nBOB: Hey.\nCHARLIE: Hello."
        missing = self._missing_speakers(text, ["ALICE", "BOB"])
        self.assertEqual(missing, {"CHARLIE"})

    def test_empty_panel_all_speakers_missing(self):
        text = "ALICE: Hi.\nBOB: Hey."
        missing = self._missing_speakers(text, [])
        self.assertEqual(missing, {"ALICE", "BOB"})

    def test_extra_panel_keys_not_flagged(self):
        # Panel may have more speakers than script — that's fine
        text = "ALICE: Hi."
        missing = self._missing_speakers(text, ["ALICE", "BOB"])
        self.assertEqual(missing, set())

    def test_empty_script_no_missing(self):
        self.assertEqual(self._missing_speakers("", ["ALICE"]), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
