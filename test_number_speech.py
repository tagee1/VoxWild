"""
test_number_speech.py — fractions, ranges and degrees reach the engines as words.

Section D of the full test pass reported three separate faults that turned out
to be one: numbers and symbols were handed to both engines raw, and each engine
mangled them a different way.

                    Fast (espeak)                  Natural (Chatterbox)
    "3/4"           "three SLASH four"             "three four"
    "22-25"         "twenty two DASH twenty five"  "twenty two twenty five"
    "195°F"         correct                        "nineteen five degrees F"

espeak reads the punctuation out as a word; Chatterbox has no number
normalization at all — every one of these symbols is in its vocabulary, so the
model simply improvises. Same family as "2:1" being read as "two thousand one".

The degree sign is worse than it looks: only U+00B0 works. The look-alikes
people actually paste are indistinguishable on screen and fail silently —
U+00BA says "oh", U+02DA is dropped, U+2103/U+2109 vanish with the number.

app.py builds a window on import, so the repair block is extracted and exec'd.

Run with:
    python -m pytest test_number_speech.py -v
    python test_number_speech.py          (stdlib unittest)
"""
import io
import os
import re
import unittest

HERE   = os.path.dirname(os.path.abspath(__file__))
APP_PY = os.path.join(HERE, "app.py")

with io.open(APP_PY, encoding="utf-8") as _f:
    SRC = _f.read()

_ns = {"re": re, "VOICES": {}}
exec(SRC[SRC.index("_TAG_RE = re.compile"):SRC.index("\ndef _tag_year(")], _ns)

speak      = _ns["_speak_numbers"]
num_words  = _ns["_num_to_words"]
fractions  = _ns["_speak_fractions"]
ranges     = _ns["_speak_ranges"]
degrees    = _ns["_speak_degrees"]
phones     = _ns["_speak_phones"]
dates      = _ns["_speak_dates"]
ordinal    = _ns["_ordinal_words"]


class TestNumToWords(unittest.TestCase):

    def test_units_and_teens(self):
        self.assertEqual(num_words(0), "zero")
        self.assertEqual(num_words(7), "seven")
        self.assertEqual(num_words(19), "nineteen")

    def test_tens(self):
        self.assertEqual(num_words(20), "twenty")
        self.assertEqual(num_words(42), "forty-two")

    def test_hundreds(self):
        self.assertEqual(num_words(100), "one hundred")
        self.assertEqual(num_words(195), "one hundred ninety-five")
        self.assertEqual(num_words(350), "three hundred fifty")

    def test_thousands(self):
        self.assertEqual(num_words(1000), "one thousand")
        self.assertEqual(num_words(2500), "two thousand five hundred")

    def test_beyond_range_is_left_as_digits(self):
        """Past six figures a voice reads the digits better than a word salad."""
        self.assertEqual(num_words(1234567), "1234567")


class TestFractions(unittest.TestCase):

    def test_the_reported_cases(self):
        self.assertEqual(speak("2 1/2 cups of flour"), "two and a half cups of flour")
        self.assertEqual(speak("3/4 of the way"), "three quarters of the way")

    def test_common_denominators(self):
        self.assertIn("half",     fractions("1/2"))
        self.assertIn("third",    fractions("1/3"))
        self.assertIn("quarter",  fractions("1/4"))
        self.assertIn("eighth",   fractions("1/8"))

    def test_plurals(self):
        self.assertEqual(fractions("2/3"), "two thirds")
        self.assertEqual(fractions("3/8"), "three eighths")

    def test_a_mixed_number_reads_as_a_person_would_say_it(self):
        self.assertEqual(fractions("2 1/2"), "two and a half")
        self.assertEqual(fractions("1 3/4"), "one and three quarters")

    def test_a_date_is_not_read_as_a_fraction(self):
        """"3/4/2026" contains "3/4"; the date rule runs first and claims it."""
        self.assertNotIn("quarters", speak("On 3/4/2026 we met."))

    def test_an_improper_fraction_is_read_arithmetically(self):
        """Covered in full by TestImproperFractions; pinned here because this
        used to be deliberately skipped and was changed on request."""
        self.assertEqual(fractions("5/4 time"), "five fourths time")

    def test_an_unusual_denominator_is_left_alone(self):
        self.assertEqual(fractions("3/13 of it"), "3/13 of it")

    def test_no_slash_survives_a_converted_fraction(self):
        """A slash reaching espeak is spoken as the word "slash"."""
        self.assertNotIn("/", speak("Add 1/2 cup and 3/4 tsp."))


class TestPhoneNumbers(unittest.TestCase):
    """espeak read "555-1234" as "five hundred fifty five DASH one thousand two
    hundred thirty four". Digits in groups is how a person reads one aloud."""

    def test_seven_digit(self):
        self.assertEqual(speak("Call 555-1234 now."),
                         "Call five five five, one two three four now.")

    def test_ten_digit(self):
        self.assertEqual(phones("555-123-4567"),
                         "five five five, one two three, four five six seven")

    def test_bracketed_area_code(self):
        self.assertEqual(phones("(555) 123-4567"),
                         "five five five, one two three, four five six seven")

    def test_no_digits_are_lost(self):
        for n in ("555-1234", "555-123-4567", "(555) 123-4567"):
            words = phones(n).replace(",", "").split()
            self.assertEqual(len(words), sum(c.isdigit() for c in n))

    def test_a_phone_is_never_read_as_a_range(self):
        """Phones run before ranges precisely so this cannot happen."""
        self.assertNotIn(" to ", speak("Call 555-1234 now."))

    def test_a_range_is_still_a_range(self):
        self.assertIn(" to ", speak("22-25 minutes"))


class TestDates(unittest.TestCase):

    def test_the_reported_case(self):
        self.assertEqual(speak("On 3/4/2026 we met."),
                         "On March fourth twenty twenty-six we met.")

    def test_a_two_digit_day_gets_a_compound_ordinal(self):
        self.assertEqual(dates("12/25/1999"),
                         "December twenty-fifth nineteen ninety-nine")

    def test_an_impossible_month_is_not_a_date(self):
        self.assertEqual(speak("13/45/2026"), "13/45/2026")

    def test_a_date_is_not_eaten_by_the_fraction_rule(self):
        """Dates run first for exactly this reason — "3/4" is inside "3/4/2026"."""
        out = speak("On 3/4/2026 we met.")
        self.assertNotIn("quarters", out)

    def test_a_bare_fraction_is_still_a_fraction(self):
        self.assertIn("three quarters", speak("3/4 of the way"))

    def test_no_slash_survives_a_date(self):
        self.assertNotIn("/", speak("On 3/4/2026 we met."))


class TestImproperFractions(unittest.TestCase):
    """Reported: "5/4" should read "five fourths"."""

    def test_the_reported_case(self):
        self.assertEqual(fractions("5/4"), "five fourths")

    def test_halves_are_irregular_in_both_directions(self):
        """The ordinal for 2 is "second" and the plural of "half" is not
        "halfs" — "7/2" is "seven halves", never "seven seconds"."""
        self.assertEqual(fractions("7/2"), "seven halves")
        self.assertEqual(fractions("3/2"), "three halves")

    def test_thirds_and_eighths(self):
        self.assertEqual(fractions("5/3"), "five thirds")
        self.assertEqual(fractions("9/8"), "nine eighths")

    def test_proper_fractions_keep_their_idiomatic_names(self):
        """Below the whole, "3/4" is "three quarters", not "three fourths"."""
        self.assertEqual(fractions("3/4"), "three quarters")
        self.assertEqual(fractions("1/2"), "one half")


class TestRanges(unittest.TestCase):

    def test_the_reported_case(self):
        self.assertEqual(speak("22-25 minutes"), "twenty-two to twenty-five minutes")

    def test_spaces_around_the_hyphen_are_allowed(self):
        self.assertEqual(ranges("10 - 20"), "ten to twenty")

    def test_a_phone_number_is_not_a_range(self):
        """It is now read as a phone number instead — but never as a span."""
        self.assertNotIn(" to ", speak("Call 555-1234 now."))

    def test_a_descending_pair_is_not_a_range(self):
        self.assertEqual(ranges("pages 100-99"), "pages 100-99")

    def test_a_hyphenated_word_is_untouched(self):
        self.assertEqual(speak("a well-known fact"), "a well-known fact")

    def test_no_hyphen_is_left_between_the_numbers(self):
        """espeak reads a bare hyphen between digits as the word "dash"."""
        self.assertNotIn("-25", speak("22-25 minutes"))


class TestDegrees(unittest.TestCase):

    def test_the_reported_case(self):
        self.assertEqual(speak("Heat to 195°F."),
                         "Heat to one hundred ninety-five degrees Fahrenheit.")

    def test_celsius(self):
        self.assertIn("degrees Celsius", speak("Set it to 200°C now."))

    def test_a_bare_degree_keeps_the_following_word(self):
        """The space before C/F sits inside the optional group; consuming it
        unconditionally produced "three hundred fifty degreesfor 20 minutes"."""
        self.assertEqual(speak("Bake at 350° for 20 minutes."),
                         "Bake at three hundred fifty degrees for 20 minutes.")

    def test_the_f_of_a_following_word_is_not_read_as_fahrenheit(self):
        self.assertNotIn("Fahrenheit", speak("Bake at 350° for 20 minutes."))

    def test_lowercase_scale_letters_work(self):
        self.assertIn("Fahrenheit", degrees("350°f is hot"))

    def test_lookalike_symbols_are_normalized(self):
        """All of these render identically on screen but behave differently:
        º says "oh", ˚ is dropped, ℃ and ℉ vanish along with nothing at all."""
        for bad in ("195ºF", "195˚F", "195ᵒF"):
            self.assertIn("degrees", degrees(bad), f"{bad!r} lost its degrees")
        self.assertIn("degrees Celsius", degrees("20℃"))
        self.assertIn("degrees Fahrenheit", degrees("195℉"))

    def test_the_number_is_spelled_out(self):
        """Chatterbox guessed at the digits — it read 195 as "nineteen five"."""
        self.assertNotIn("195", speak("It hit 195°F."))


class TestTheRulesDoNotCollide(unittest.TestCase):
    """Order is the whole design: each rule runs on what the previous declined."""

    def test_a_time_still_wins_over_everything(self):
        out = speak("Meet me at 3:30 pm.")
        self.assertIn("three thirty", out)
        self.assertNotIn(" to ", out)

    def test_a_ratio_still_works(self):
        self.assertEqual(speak("The ratio is 2:1."), "The ratio is two to one.")

    def test_a_verse_reference_is_untouched_by_the_new_rules(self):
        self.assertIn("three sixteen", speak("John 3:16 says so."))

    def test_all_the_repairs_can_share_one_sentence(self):
        out = speak("At 3:30 pm, bake 2 1/2 cups at 350°F for 22-25 minutes.")
        for expect in ("three thirty", "two and a half",
                       "three hundred fifty degrees Fahrenheit",
                       "twenty-two to twenty-five"):
            self.assertIn(expect, out, f"missing {expect!r} in {out!r}")

    def test_plain_prose_is_never_touched(self):
        for t in ("Chapter 7", "Note: this is fine.", "a well-known fact",
                  "Send it to room 214.", "Model X15 arrived."):
            self.assertEqual(speak(t), t)

    def test_non_english_is_skipped_entirely(self):
        """"quarters" and "degrees Fahrenheit" have no place in Spanish."""
        t = "Son 2 1/2 tazas a 195°F."
        self.assertEqual(speak(t, "es"), t)


class TestWiring(unittest.TestCase):

    def fn(self, header):
        self.assertIn(header, SRC, f"{header} not found")
        return SRC.split(header, 1)[1].split("\ndef ", 1)[0]

    def test_both_engines_use_the_single_funnel(self):
        """Adding a rule at one call site and forgetting the other is exactly
        how "2:1" stayed broken in Natural while Fast was fine."""
        self.assertIn("_speak_numbers(text, lang)", self.fn("def _kokoro_one("))
        self.assertIn("_cb_text = _speak_numbers(", SRC)

    def test_the_funnel_applies_every_rule(self):
        body = self.fn("def _speak_numbers(")
        for rule in ("_speak_phones", "_speak_dates", "_speak_times",
                     "_speak_ratios", "_speak_fractions", "_speak_ranges",
                     "_speak_degrees"):
            self.assertIn(rule + "(", body, f"{rule} is not in the funnel")

    def test_phones_run_before_ranges(self):
        """Otherwise "123-4567" is a numeric span before it is ever a phone."""
        body = self.fn("def _speak_numbers(")
        self.assertLess(body.find("_speak_phones("), body.find("_speak_ranges("))

    def test_dates_run_before_fractions(self):
        """"3/4/2026" contains "3/4"; whichever runs first wins it."""
        body = self.fn("def _speak_numbers(")
        self.assertLess(body.find("_speak_dates("), body.find("_speak_fractions("))

    def test_times_run_before_ratios(self):
        body = self.fn("def _speak_numbers(")
        self.assertLess(body.find("_speak_times("), body.find("_speak_ratios("),
                        '"3:30" would become "three to thirty"')

    def test_the_old_per_site_calls_are_gone(self):
        """Two sites each assembling their own chain is what drifted before."""
        self.assertNotIn("_speak_ratios(_speak_times(", SRC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
