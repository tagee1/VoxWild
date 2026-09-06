"""
test_pack_order.py — an expand=True widget must be packed LAST in its row.

Tk's pack hands out space in the order widgets are added, and a widget with
expand=True claims everything still unclaimed. Anything packed after it gets
whatever is left, which is often nothing. The widget is created correctly, gets
laid out at a width of 1px, and simply never appears.

This has now been the cause of five separate "it's not there" reports in this
app, each found only by measuring the rendered widget:

  * Settings window footer — Save/Cancel pushed off the bottom
  * Studio "Tag guide" button
  * Dialogue speaker "Rename" button — measured at x=0, width=1
  * History card "Orig" button — measured at x=0, width=1
  * Progress row time label — squeezed 285px -> 249px, clipping the text

The last one is the subtlest: it only misbehaves once SegmentBar holds more than
one cell, because each cell carries minsize=8, so the bar's MINIMUM width grows
with the chunk count. A one-chunk generation looked fine; a real one did not.
The label is on screen precisely while that is true, so "⏱ Assembling audio…"
showed up as a fragment mid-generation.

Run with:
    python -m pytest test_pack_order.py -v
    python test_pack_order.py          (stdlib unittest)
"""
import io
import os
import re
import unittest

HERE   = os.path.dirname(os.path.abspath(__file__))
APP_PY = os.path.join(HERE, "app.py")

with io.open(APP_PY, encoding="utf-8") as _f:
    SRC = _f.read()


class TestProgressRow(unittest.TestCase):
    """The row is: percent · segment bar (expands) · time remaining."""

    def setUp(self):
        block = SRC.split("# ── Progress row ", 1)[1]
        self.block = block.split("smooth = SmoothProgress", 1)[0]

    def _pack_line(self, widget):
        m = re.search(rf"^{re.escape(widget)}\.pack\((.*)$", self.block, re.MULTILINE)
        self.assertIsNotNone(m, f"{widget} is never packed")
        return m

    def test_the_expanding_bar_is_packed_after_the_time_label(self):
        """The whole fix. Reversed, the bar takes the row and the label starves."""
        self.assertLess(self.block.index("progress_time_label.pack("),
                        self.block.index("progress_bar.pack("),
                        "the expanding bar must be packed last in this row")

    def test_the_bar_is_still_the_one_that_expands(self):
        self.assertIn("expand=True", self._pack_line("progress_bar").group(1))

    def test_the_time_label_does_not_expand(self):
        """A fixed width is what keeps the text from being clipped."""
        line = self._pack_line("progress_time_label").group(1)
        self.assertNotIn("expand=True", line)

    def test_the_time_label_is_pinned_to_the_right(self):
        """Packed before the bar but still drawn at the far right, so the
        reading order on screen is unchanged."""
        self.assertIn('side="right"', self._pack_line("progress_time_label").group(1))

    def test_the_label_is_wide_enough_for_its_longest_string(self):
        """Measured at 150% scaling: the box is 210*1.5 = 315px and the longest
        string the label ever holds is "🎧 Assembling audiobook…" at 302px."""
        m = re.search(r"progress_time_label = ctk\.CTkLabel\(\s*prog_row,[^)]*?width=(\d+)",
                      self.block, re.DOTALL)
        self.assertIsNotNone(m, "could not read the time label width")
        self.assertGreaterEqual(int(m.group(1)), 210,
                                "too narrow for '🎧 Assembling audiobook…'")


class TestTheOtherRowsStayFixed(unittest.TestCase):
    """Regression guards for the four earlier occurrences."""

    def fn(self, header):
        self.assertIn(header, SRC, f"{header} not found")
        return SRC.split(header, 1)[1].split("\ndef ", 1)[0]

    def test_speaker_row_packs_rename_before_the_expanding_name_box(self):
        panel = self.fn("def dlg_detect_speakers(")
        self.assertLess(panel.find('text="Rename"'),
                        panel.find("name_entry = ctk.CTkEntry("))

    def test_history_action_row_reserves_delete_first(self):
        """Delete is packed to the right before anything claims the row."""
        block = SRC.split("row_act = ctk.CTkFrame(", 1)[1].split("return outer", 1)[0]
        self.assertLess(block.find('text="Delete"'), block.find('text="Save"'))

    def test_orig_has_its_own_row(self):
        block = SRC.rsplit('if entry.get("original_samples") is not None:', 1)[1]
        self.assertIn("row_orig = ctk.CTkFrame(", block)


class TestSegmentBarIsWhyThisRowIsDifferent(unittest.TestCase):
    """The bar's minimum width is not fixed — it grows with the chunk count."""

    def test_each_cell_carries_a_minimum_width(self):
        build = SRC.split("def _build(self, weights):", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("minsize=8", build,
                      "if this ever goes, re-check what starves the time label")

    def test_one_cell_per_chunk(self):
        build = SRC.split("def _build(self, weights):", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("for i, w in enumerate(weights):", build)


if __name__ == "__main__":
    unittest.main(verbosity=2)
