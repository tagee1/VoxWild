"""
test_queue_select.py — the queue list allows several items, and lets go of them.

Reported: once a queued item was selected the green bar could not be cleared,
and only one item could ever be selected.

Both came from one omission — the Listbox was created without a selectmode, so
Tk applied its default of "browse": exactly one row selected, always. There is
no way to reach zero rows and no way to reach two. Remove then compounded it by
only ever deleting sel[0].

Run with:
    python -m pytest test_queue_select.py -v
    python test_queue_select.py          (stdlib unittest)
"""
import io
import os
import unittest

HERE   = os.path.dirname(os.path.abspath(__file__))
APP_PY = os.path.join(HERE, "app.py")


def _remove_selected(items, sel):
    """The deletion rule from queue_remove, on plain data."""
    for idx in sorted(sel, reverse=True):
        if 0 <= idx < len(items):
            items.pop(idx)
    return items


class TestRemovalOrder(unittest.TestCase):
    """Popping several indices only works from the back."""

    def setUp(self):
        self.items = [f"item {i + 1}" for i in range(5)]

    def test_removing_one(self):
        self.assertEqual(_remove_selected(list(self.items), (2,)),
                         ["item 1", "item 2", "item 4", "item 5"])

    def test_removing_several_scattered(self):
        self.assertEqual(_remove_selected(list(self.items), (0, 2, 4)),
                         ["item 2", "item 4"])

    def test_removing_a_contiguous_range(self):
        self.assertEqual(_remove_selected(list(self.items), (1, 2, 3)),
                         ["item 1", "item 5"])

    def test_removing_everything(self):
        self.assertEqual(_remove_selected(list(self.items), (0, 1, 2, 3, 4)), [])

    def test_front_first_would_delete_the_wrong_rows(self):
        """Why sorted(..., reverse=True) is load-bearing, not tidiness."""
        items = list(self.items)
        for idx in sorted((0, 2, 4)):          # deliberately the wrong order
            if 0 <= idx < len(items):
                items.pop(idx)
        self.assertEqual(items, ["item 2", "item 3", "item 5"])
        self.assertNotEqual(items, _remove_selected(list(self.items), (0, 2, 4)))

    def test_selection_order_from_the_widget_does_not_matter(self):
        """curselection() is ascending, but Ctrl+click order is the user's."""
        self.assertEqual(_remove_selected(list(self.items), (4, 0, 2)),
                         _remove_selected(list(self.items), (0, 2, 4)))

    def test_a_stale_index_cannot_raise(self):
        self.assertEqual(_remove_selected(["a", "b"], (0, 7)), ["b"])


class TestQueueWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with io.open(APP_PY, encoding="utf-8") as f:
            cls.src = f.read()

    def fn(self, header):
        self.assertIn(header, self.src, f"{header} not found")
        return self.src.split(header, 1)[1].split("\ndef ", 1)[0]

    def test_the_listbox_allows_more_than_one_row(self):
        block = self.src.split("queue_listbox = tk.Listbox(", 1)[1].split("relief=", 1)[0]
        self.assertIn('selectmode="extended"', block,
                      "without this Tk defaults to browse: one row, always")

    def test_remove_deletes_every_selected_row(self):
        body = self.fn("def queue_remove(")
        self.assertNotIn("queue_items.pop(sel[0])", body,
                         "only the first selected item would be removed")
        self.assertIn("sorted(sel, reverse=True)", body,
                      "front-first popping shifts the indices and deletes wrong rows")

    def test_remove_guards_against_a_stale_index(self):
        body = self.fn("def queue_remove(")
        self.assertIn("0 <= idx < len(queue_items)", body)

    def test_remove_says_how_many_went(self):
        body = self.fn("def queue_remove(")
        self.assertIn("status_label.configure", body)

    def test_the_selection_can_be_cleared(self):
        """'extended' still offers no way to deselect the last row by clicking
        it, and a green bar that will not clear reads as stuck."""
        self.assertIn("def _queue_click_empty(", self.src)
        self.assertIn('queue_listbox.bind("<Button-1>", _queue_click_empty', self.src)
        self.assertIn('queue_listbox.bind("<Escape>"', self.src)

    def test_multi_select_is_signposted(self):
        """Nothing on screen says a list takes Ctrl+click until you know it does."""
        self.assertIn("Ctrl+click", self.src)

    def test_both_buttons_are_still_there(self):
        self.assertIn('text="Remove Selected"', self.src)
        self.assertIn('text="Clear All"', self.src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
