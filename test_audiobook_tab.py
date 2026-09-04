"""
test_audiobook_tab.py — Clear resets the whole tab; book details fill themselves in.

Two defects found in full-pass testing, both on the Audiobook tab:

  1. Clear emptied the text box only. The detected chapters, the summary line,
     the title/author/year and the cover art all stayed behind, so a second book
     inherited the first one's chapters and cover.

  2. Title/Author/Year never filled themselves in, even when the text carried
     labels. Nothing read them — the feature did not exist.

TestParseBookMetadata covers the parser (pure logic, no UI). The wiring classes
read app.py as text, because importing it would build the whole window.

Run with:
    python -m pytest test_audiobook_tab.py -v
    python test_audiobook_tab.py          (stdlib unittest)
"""
import io
import os
import unittest

from tts_utils import parse_book_metadata

HERE   = os.path.dirname(os.path.abspath(__file__))
APP_PY = os.path.join(HERE, "app.py")


class TestParseBookMetadata(unittest.TestCase):

    def test_full_header_is_read_and_removed(self):
        fields, rest = parse_book_metadata(
            "Title: The Long Road\nAuthor: Jane Doe\nYear: 1998\n\nIt began.\n")
        self.assertEqual(fields, {"title": "The Long Road",
                                  "author": "Jane Doe", "year": "1998"})
        self.assertEqual(rest.strip(), "It began.")

    def test_by_is_accepted_as_author(self):
        fields, _ = parse_book_metadata("By: Jane Doe\n\nIt began.")
        self.assertEqual(fields.get("author"), "Jane Doe")

    def test_author_wins_over_a_later_by(self):
        fields, _ = parse_book_metadata("Author: Jane Doe\nBy: Someone Else\n\nx")
        self.assertEqual(fields["author"], "Jane Doe")

    def test_labels_are_case_insensitive(self):
        fields, _ = parse_book_metadata("TITLE: X\nauthor: Y\n\nz")
        self.assertEqual(fields, {"title": "X", "author": "Y"})

    def test_blank_lines_inside_the_header_are_allowed(self):
        fields, rest = parse_book_metadata("Title: X\n\nAuthor: Y\n\nBody.")
        self.assertEqual(fields, {"title": "X", "author": "Y"})
        self.assertEqual(rest.strip(), "Body.")

    def test_prose_starting_with_by_is_not_a_header(self):
        """'By the time I got there...' must not become an author."""
        text = "By the time I got there, it was dark.\n"
        fields, rest = parse_book_metadata(text)
        self.assertEqual(fields, {})
        self.assertEqual(rest, text)

    def test_a_label_below_real_content_is_ignored(self):
        """The scan stops at the first line of prose, so mid-book text is safe."""
        text = "# Chapter One\n\nTitle: a line of dialogue\n"
        fields, rest = parse_book_metadata(text)
        self.assertEqual(fields, {})
        self.assertEqual(rest, text)

    def test_text_with_no_header_is_returned_untouched(self):
        text = "Just a book.\n\nWith two paragraphs.\n"
        fields, rest = parse_book_metadata(text)
        self.assertEqual(fields, {})
        self.assertEqual(rest, text)

    def test_year_is_extracted_from_a_longer_value(self):
        fields, _ = parse_book_metadata("Year: first published 1998\n\nx")
        self.assertEqual(fields["year"], "1998")

    def test_unparseable_year_does_not_lose_the_title_under_it(self):
        fields, rest = parse_book_metadata("Year: next spring\nTitle: X\n\nBody.")
        self.assertNotIn("year", fields)
        self.assertEqual(fields["title"], "X")
        self.assertEqual(rest.strip(), "Body.")

    def test_label_with_no_value_sets_nothing(self):
        fields, _ = parse_book_metadata("Title:   \nAuthor: Jane\n\nx")
        self.assertNotIn("title", fields)
        self.assertEqual(fields["author"], "Jane")

    def test_header_only_input_leaves_no_body(self):
        fields, rest = parse_book_metadata("Title: X\nAuthor: Y\n")
        self.assertEqual(fields, {"title": "X", "author": "Y"})
        self.assertEqual(rest.strip(), "")

    def test_empty_input(self):
        self.assertEqual(parse_book_metadata(""), ({}, ""))
        self.assertEqual(parse_book_metadata("   "), ({}, "   "))

    def test_chapter_headings_after_the_header_survive(self):
        """Chapter detection runs on the remaining text, so '#' must stay."""
        _, rest = parse_book_metadata("Title: X\n\n# Chapter One\n\nBody.\n")
        self.assertTrue(rest.lstrip().startswith("# Chapter One"))

    def test_only_the_four_agreed_labels_are_recognised(self):
        """Publisher:, ISBN: and friends are prose as far as this is concerned."""
        text = "Publisher: Penguin\nTitle: X\n"
        fields, rest = parse_book_metadata(text)
        self.assertEqual(fields, {})
        self.assertEqual(rest, text)


class TestClearWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with io.open(APP_PY, encoding="utf-8") as f:
            cls.src = f.read()

    def test_clear_is_a_named_function_not_an_inline_lambda(self):
        self.assertIn("def ab_clear_all(", self.src)
        self.assertIn("ab_clear_btn.configure(command=ab_clear_all)", self.src)

    def test_clear_asks_first(self):
        fn = self.src.split("def ab_clear_all(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("askyesno", fn,
                      "Clear wipes typed-in details now, so it must confirm")

    def test_clear_skips_the_prompt_when_there_is_nothing_to_clear(self):
        fn = self.src.split("def ab_clear_all(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_ab_is_empty()", fn)
        self.assertLess(fn.find("_ab_is_empty()"), fn.find("askyesno"),
                        "the empty check must come before the prompt")

    def test_clear_resets_every_column(self):
        """The whole point: nothing may survive into the next book."""
        fn = self.src.split("def ab_clear_all(", 1)[1].split("\ndef ", 1)[0]
        for what, needle in (
                ("book text",       'ab_text.delete("1.0", "end")'),
                ("chapter rows",    "_ab_clear_rows()"),
                ("summary line",    "ab_chap_summary.configure"),
                ("title",           "ab_title_var.set("),
                ("author",          "ab_author_var.set("),
                ("year",            "ab_year_var.set("),
                ("cover bytes",     'ab_cover["bytes"]'),
                ("cover label",     "ab_cover_label.configure")):
            self.assertIn(needle, fn, f"Clear leaves the {what} behind")

    def test_the_old_text_only_lambda_is_gone(self):
        self.assertNotIn('command=lambda: ab_text.delete("1.0", "end")', self.src)


class TestDetailsAutofillWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with io.open(APP_PY, encoding="utf-8") as f:
            cls.src = f.read()

    def test_detect_chapters_fills_the_details(self):
        fn = self.src.split("def ab_detect_chapters(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_ab_fill_details(", fn)

    def test_autofill_never_overwrites_what_the_user_typed(self):
        fn = self.src.split("def _ab_fill_details(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("not ab_title_var.get().strip()", fn)
        self.assertIn("not ab_author_var.get().strip()", fn)

    def test_year_treats_the_current_year_as_unset(self):
        """It is pre-filled with this year, which nobody chose."""
        fn = self.src.split("def _ab_fill_details(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('datetime.now().strftime("%Y")', fn)

    def test_the_header_is_not_read_aloud(self):
        """_ab_fill_details returns the text WITHOUT the header block, and
        detect uses that — otherwise chapter one opens with 'Title colon...'."""
        fill = self.src.split("def _ab_fill_details(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("return remaining", fill)
        det = self.src.split("def ab_detect_chapters(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("text = _ab_fill_details(text)", det,
                      "the stripped text must replace the original")
        self.assertLess(det.find("_ab_fill_details(text)"),
                        det.find("split_into_chapters("),
                        "details must be stripped before chapters are split")

    def test_header_only_input_does_not_build_a_chapter(self):
        det = self.src.split("def ab_detect_chapters(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("if not text.strip():", det)


if __name__ == "__main__":
    unittest.main(verbosity=2)
