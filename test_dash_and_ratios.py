"""
test_dash_and_ratios.py — a dash becomes real silence; "2:1" is read as a ratio.

Two defects from full-pass testing, section E. Both are cases where an engine
rewrote our text after we handed it over, so the repair never arrived.

  1. "2:1" was spoken as "two thousand one" in Natural mode. Chatterbox
     normalizes punctuation before tokenizing (chatterbox/tts.py, punc_norm)
     and rewrites ":" as ",", so the model received "2,1" and read it as a
     thousands-grouped number. Fast was unaffected — espeak keeps the colon.

  2. "wait--stop" added no pause. We replaced a dash with "... " and trusted
     the engine to render it. punc_norm ALSO rewrites "..." as ", ", so in
     Natural the beat was gone before synthesis. Measured on one sentence:

                           Fast     Natural
         dash removed      0.359s   0.225s
         "... " (old fix)  0.468s   0.220s   <- under the do-nothing baseline
         ". "              0.434s   0.169s
         0.30s silence     0.607s   0.525s

     Only real silence registers, so a dash is now a [pause] span.

  3. Found while fixing 2: auto-clean flattened en/em dashes to a single
     hyphen, which is deliberately NOT a pause ("well-known"). Pasting from
     Word silently removed every dash pause before we ever saw the text.

The helpers live in app.py, which builds a window on import, so the text-repair
block is extracted and exec'd — the same approach test_dll_fix.py uses.

Run with:
    python -m pytest test_dash_and_ratios.py -v
    python test_dash_and_ratios.py          (stdlib unittest)
"""
import io
import os
import re
import unittest

from text_cleaner import clean_text

HERE   = os.path.dirname(os.path.abspath(__file__))
APP_PY = os.path.join(HERE, "app.py")

with io.open(APP_PY, encoding="utf-8") as _f:
    SRC = _f.read()


def _load_text_repairs():
    """Exec the inline-tag/text-repair block of app.py in a bare namespace.

    Runs from "_TAG_RE = re.compile" to just before _tag_year, which is
    self-contained apart from VOICES (used only to build a name lookup).
    """
    start = SRC.index("_TAG_RE = re.compile")
    end   = SRC.index("\ndef _tag_year(", start)
    ns = {"re": re, "VOICES": {}}
    exec(SRC[start:end], ns)
    return ns


R = _load_text_repairs()
speak_times  = R["_speak_times"]
speak_ratios = R["_speak_ratios"]
to_pause_tag = R["_dash_to_pause_tag"]
DASH_RE      = R["_DASH_RE"]
PAUSE_SECS   = R["_DASH_PAUSE_SECONDS"]


def repaired(text):
    """What Fast and Natural both send to the model, in order."""
    return speak_ratios(speak_times(text))


class TestRatios(unittest.TestCase):

    def test_the_reported_case(self):
        self.assertEqual(repaired("The ratio is 2:1."), "The ratio is two to one.")

    def test_aspect_ratio(self):
        self.assertEqual(repaired("It is 16:9 widescreen."),
                         "It is sixteen to nine widescreen.")

    def test_score(self):
        self.assertEqual(repaired("We won 3:2."), "We won three to two.")

    def test_two_digit_names_are_hyphenated_words(self):
        self.assertEqual(repaired("A 21:9 monitor."), "A twenty-one to nine monitor.")

    def test_no_colon_survives_a_ratio(self):
        """The whole point: a colon reaching Chatterbox becomes a comma, and
        "2,1" is read as a thousands-grouped number."""
        self.assertNotIn(":", repaired("Shot at 2:1 and 16:9 and 4:3."))


class TestTimesStillWinFirst(unittest.TestCase):
    """_speak_times runs first and has first claim on clock-shaped input."""

    def test_a_clock_time_is_not_a_ratio(self):
        out = repaired("Meet me at 3:30.")
        self.assertIn("three thirty", out)
        self.assertNotIn(" to ", out)

    def test_a_verse_reference_keeps_the_time_reading(self):
        """"John 3:16" is HH:MM-shaped, so the time rule claims it and says
        "three sixteen" — which is also how the verse is read aloud."""
        out = repaired("John 3:16 says so.")
        self.assertIn("three sixteen", out)
        self.assertNotIn("to sixteen", out)

    def test_pm_still_survives(self):
        self.assertIn("pm", repaired("At 3:30 pm."))

    def test_oclock_is_untouched_by_the_ratio_rule(self):
        self.assertIn("o'clock", repaired("It is 12:00."))


class TestRatiosStayNarrow(unittest.TestCase):

    def test_prose_colons_are_left_alone(self):
        for t in ("Note: this is fine.", "Chapter 4: The Return", "a ratio of x:y"):
            self.assertEqual(repaired(t), t)

    def test_three_digit_numbers_are_not_ratios(self):
        """A resolution or a long reference is more likely than a spoken "to"."""
        self.assertEqual(repaired("1920:1080"), "1920:1080")

    def test_a_timestamp_is_not_split(self):
        """HH:MM:SS is guarded by the time rule's lookarounds; the ratio rule
        must not pick the leftovers apart either."""
        self.assertEqual(repaired("30:00:00"), "30:00:00")

    def test_decimals_are_untouched(self):
        self.assertEqual(repaired("version 2.5:1.5"), "version 2.5:1.5")


class TestDashBecomesAPause(unittest.TestCase):

    def test_the_reported_case(self):
        self.assertIn(f"[pause {PAUSE_SECS}]", to_pause_tag("wait--stop"))

    def test_every_dash_style(self):
        for t in ("wait--stop", "wait—stop", "wait–stop", "wait -- stop"):
            self.assertIn("[pause", to_pause_tag(t), f"{t!r} produced no pause")

    def test_a_single_hyphen_is_not_a_pause(self):
        """"well-known" is one word, not a beat."""
        self.assertEqual(to_pause_tag("a well-known fact"), "a well-known fact")

    def test_the_words_either_side_survive(self):
        out = to_pause_tag("I told her the truth--but then I left.")
        self.assertIn("the truth", out)
        self.assertIn("but then I left.", out)

    def test_several_dashes_all_become_pauses(self):
        self.assertEqual(to_pause_tag("a--b--c").count("[pause"), 2)

    def test_the_pause_is_a_tag_the_parser_recognizes(self):
        """It has to match parse_speech_tags' own [pause N] grammar, or it
        would be left in the text and read out loud."""
        tag = to_pause_tag("a--b").split("[", 1)[1].split("]", 1)[0]
        self.assertRegex(tag.lower(), r'^(pause|break)\b\s*\d+(\.\d+)?\s*(ms|s)?$')

    def test_a_dash_inside_a_tag_does_not_split_the_tag(self):
        """Substituting blindly would turn [voice: a--b] into two broken tags."""
        self.assertEqual(to_pause_tag("[voice: a--b] hi"), "[voice: a--b] hi")

    def test_a_typed_pause_tag_is_left_alone(self):
        self.assertEqual(to_pause_tag("a [pause 1.5] b"), "a [pause 1.5] b")


class TestAutoCleanKeepsTheBeat(unittest.TestCase):
    """Auto-clean must never change how the text sounds."""

    def test_an_em_dash_survives_cleaning_as_a_pause(self):
        cleaned, _ = clean_text("Wait—stop, hold on.")
        self.assertTrue(DASH_RE.search(cleaned),
                        "cleaning flattened the dash, so the pause is gone")

    def test_every_dash_style_survives_cleaning(self):
        for ch in ("–", "—", "--"):
            cleaned, _ = clean_text(f"before{ch}after")
            self.assertTrue(DASH_RE.search(cleaned), f"{ch!r} did not survive")

    def test_cleaning_still_removes_the_unicode_character(self):
        cleaned, _ = clean_text("Before—after")
        self.assertNotIn("—", cleaned)

    def test_a_hyphenated_word_is_not_turned_into_a_pause(self):
        cleaned, _ = clean_text("a well-known fact")
        self.assertIsNone(DASH_RE.search(cleaned))


class TestWiring(unittest.TestCase):

    def fn(self, header):
        self.assertIn(header, SRC, f"{header} not found")
        return SRC.split(header, 1)[1].split("\ndef ", 1)[0]

    def test_the_tag_parser_converts_dashes(self):
        """One funnel: Studio in both engines and Dialogue all parse tags."""
        self.assertIn("_dash_to_pause_tag(text)", self.fn("def parse_speech_tags("))

    def test_the_audiobook_path_is_covered_too(self):
        """It synthesizes raw chunks and never reaches parse_speech_tags, so
        kokoro_create splits on the dash itself and splices in silence."""
        fn = self.fn("def kokoro_create(")
        self.assertIn("_DASH_RE.split(text)", fn)
        self.assertIn("np.zeros(int(_DASH_PAUSE_SECONDS", fn)

    def test_a_line_with_no_dash_takes_the_unchanged_path(self):
        """The overwhelmingly common case must not gain an extra code path."""
        fn = self.fn("def kokoro_create(")
        self.assertIn("if len(pieces) < 2:", fn)

    def test_both_engines_speak_ratios(self):
        self.assertIn("_speak_ratios(", self.fn("def _kokoro_one("))      # Fast
        self.assertIn("_speak_ratios(_speak_times(", SRC)                 # Natural

    def test_ratios_are_applied_after_times_everywhere(self):
        """Order is the whole guard against "3:30" becoming "three to thirty"."""
        fast = self.fn("def _kokoro_one(")
        self.assertLess(fast.find("_speak_times("), fast.find("_speak_ratios("))
        cb = SRC.split("_cb_text = ", 1)[1].split("\n", 1)[0]
        self.assertEqual(cb, "_speak_ratios(_speak_times(u[\"text\"]))")

    def test_the_old_punctuation_fix_is_gone(self):
        """Leaving it in place would add "... " on top of the real silence."""
        self.assertNotIn("_dash_pause", SRC)
        self.assertNotIn("_DASH_PAUSE = ", SRC)

    def test_the_cleaner_maps_a_dash_to_the_pause_form(self):
        """Asserted on behaviour, not source text — the file writes these as
        \\uXXXX escapes, so grepping for the literal character is fragile."""
        self.assertEqual(clean_text("—")[0], "--")
        self.assertEqual(clean_text("–")[0], "--")


if __name__ == "__main__":
    unittest.main(verbosity=2)
