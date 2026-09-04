"""
test_autoclean_notify.py — auto-clean on every tab, and the completion chime.

Two features from the same round of testing:

  TestKeepHeadings     the audiobook tab must clean text WITHOUT eating the '#'
                       marks that chapter detection depends on. Real functions,
                       no UI.

  TestChapterSplitting the point of the above: a cleaned book must still split
                       into the same chapters it would have before.

  TestAutoCleanWiring  every text box has to be wired up. Reads app.py as text,
  TestNotifyWiring     because the alternative is launching the whole UI. Cheap,
                       and it catches the failure that actually happened here:
                       a feature wired into one tab and silently absent from the
                       others, and a Settings control wired to nothing at all.

Run with:
    python -m pytest test_autoclean_notify.py -v
    python test_autoclean_notify.py          (stdlib unittest)
"""
import io
import os
import unittest

from text_cleaner import clean_text

HERE   = os.path.dirname(os.path.abspath(__file__))
APP_PY = os.path.join(HERE, "app.py")

BOOK = (
    "# Chapter One\n"
    "\n"
    "Dr. Smith said  “hello”!!!! Visit https://example.com now.\n"
    "\n"
    "## Chapter Two\n"
    "\n"
    "The **next** day — e.g. Tuesday.\n"
)


class TestKeepHeadings(unittest.TestCase):

    def test_headings_are_stripped_by_default(self):
        out, _ = clean_text(BOOK)
        self.assertNotIn("#", out)
        self.assertIn("Chapter One", out)

    def test_headings_survive_when_asked(self):
        out, _ = clean_text(BOOK, keep_headings=True)
        self.assertIn("# Chapter One", out)
        self.assertIn("## Chapter Two", out)

    def test_everything_else_still_cleans_with_headings_kept(self):
        out, changes = clean_text(BOOK, keep_headings=True)
        self.assertNotIn("https://", out)      # URL
        self.assertNotIn("**", out)            # bold
        self.assertNotIn("!!!!", out)          # excess punctuation
        self.assertNotIn("“", out)        # curly quote
        self.assertIn("Doctor Smith", out)     # abbreviation expanded
        self.assertIn("for example", out)      # e.g. expanded
        self.assertGreater(len(changes), 3)

    def test_both_modes_agree_once_the_hashes_are_removed(self):
        """keep_headings must change ONLY the heading marks, nothing else."""
        plain, _ = clean_text(BOOK)
        kept, _  = clean_text(BOOK, keep_headings=True)
        stripped = "\n".join(ln.lstrip("# ").rstrip() for ln in kept.splitlines())
        self.assertEqual(
            [ln for ln in stripped.splitlines() if ln.strip()],
            [ln for ln in plain.splitlines() if ln.strip()])

    def test_plain_text_is_unaffected_by_the_flag(self):
        sample = "Just a  sentence with e.g. an abbreviation!!!!"
        self.assertEqual(clean_text(sample)[0],
                         clean_text(sample, keep_headings=True)[0])


class TestChapterSplitting(unittest.TestCase):
    """The reason keep_headings exists at all."""

    def _split(self, text):
        # split_into_chapters lives in app.py, which imports the whole UI stack.
        # Reimplementing its heading rule here would test the copy, not the app,
        # so assert on the input it needs instead: the marks have to survive.
        return [ln for ln in text.splitlines() if ln.lstrip().startswith("#")]

    def test_cleaning_normally_would_destroy_the_chapter_marks(self):
        cleaned, _ = clean_text(BOOK)
        self.assertEqual(self._split(cleaned), [],
                         "sanity check: plain cleaning removes every heading")

    def test_cleaning_for_the_audiobook_tab_keeps_them(self):
        cleaned, _ = clean_text(BOOK, keep_headings=True)
        self.assertEqual(len(self._split(cleaned)), 2,
                         "a two-chapter book must still have two headings")


class TestAutoCleanWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with io.open(APP_PY, encoding="utf-8") as f:
            cls.src = f.read()

    def test_shared_helpers_exist(self):
        for name in ("_auto_clean_enabled", "_auto_clean_imported",
                     "_auto_clean_after_paste"):
            self.assertIn(f"def {name}(", self.src, f"{name} is missing")

    def test_every_text_box_cleans_on_paste(self):
        for box in ("text_input", "dlg_text", "ab_text"):
            self.assertIn(f"_auto_clean_after_paste({box}", self.src,
                          f"{box} does not auto-clean on paste")

    def test_every_import_path_cleans(self):
        # Three importers: Studio, Dialogue, Audiobook.
        self.assertEqual(self.src.count("_auto_clean_imported("), 4,
                         "expected 3 import call sites plus the definition")

    def test_audiobook_keeps_headings_on_both_paths(self):
        """Paste and import both have to preserve chapter marks."""
        paste = self.src.split("ab_text.bind(", 1)[1][:300]
        self.assertIn("keep_headings=True", paste,
                      "audiobook paste would strip chapter headings")
        # Scope to the import handler itself. Searching for a bare ab_text.delete
        # finds the Clear button's lambda first, which proves nothing.
        imp = self.src.split("def ab_import_file(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_auto_clean_imported(", imp,
                      "audiobook import does not auto-clean at all")
        self.assertIn("keep_headings=True", imp,
                      "audiobook import would strip chapter headings")

    def test_studio_and_dialogue_imports_strip_headings(self):
        for fn_name in ("def import_file(", "def dlg_import_file("):
            fn = self.src.split(fn_name, 1)[1].split("\ndef ", 1)[0]
            self.assertIn("_auto_clean_imported(", fn,
                          f"{fn_name} does not auto-clean")
            self.assertNotIn("keep_headings=True", fn,
                             f"{fn_name} should strip headings like before")

    def test_studio_and_dialogue_do_not_keep_headings(self):
        """Only the audiobook needs the marks; elsewhere they are noise."""
        for box in ("text_input", "dlg_text"):
            i = self.src.find(f"_auto_clean_after_paste({box}")
            self.assertGreater(i, -1)
            call = self.src[i:i + 160]
            self.assertNotIn("keep_headings=True", call,
                             f"{box} should strip headings like before")

    def test_the_old_inline_clean_is_gone(self):
        """One implementation only, or the copies drift apart again."""
        self.assertEqual(self.src.count("clean_text(raw"), 1)


class TestNotifyWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with io.open(APP_PY, encoding="utf-8") as f:
            cls.src = f.read()

    def test_win10toast_is_gone(self):
        """It was never installed, never in requirements, never in the spec.

        The import sat inside a bare except, so the Settings checkbox and the
        threshold slider controlled nothing on any machine.

        Matches the import and the call, not the bare word — the comment that
        records why it was removed is meant to stay.
        """
        self.assertNotIn("import win10toast", self.src)
        self.assertNotIn("from win10toast", self.src)
        self.assertNotIn("ToastNotifier(", self.src)

    def test_chime_uses_the_apps_own_audio_output(self):
        """winsound alone was inaudible: PlaySound goes to Windows' separate
        "System Sounds" mixer channel, not the one VoxWild's speech plays on."""
        fn = self.src.split("def _play_chime(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("sd.play(", fn, "the chime must use the app's audio path")
        self.assertIn("winsound.PlaySound(", fn,
                      "keep a fallback for a busy/missing device")
        # Compare the CALLS, not the words — the docstring names winsound first.
        self.assertLess(fn.find("sd.play("), fn.find("winsound.PlaySound("),
                        "sounddevice must be tried first, winsound only as fallback")

    def test_chime_is_synthesised_not_a_bundled_file(self):
        """No asset to ship, nothing to go missing from an install."""
        fn = self.src.split("def _build_chime(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("np.sin", fn)
        self.assertNotIn(".wav", fn)

    def test_chime_never_blocks_the_ui(self):
        """sd.play returns immediately; sd.wait would freeze the whole window."""
        fn = self.src.split("def _play_chime(", 1)[1].split("\ndef ", 1)[0]
        self.assertNotIn("sd.wait(", fn)

    def test_chime_failure_can_never_break_a_generation(self):
        fn = self.src.split("def _play_chime(", 1)[1].split("\ndef ", 1)[0]
        self.assertEqual(fn.count("except Exception:"), 2,
                         "both the primary and the fallback must be guarded")

    def test_notify_honours_both_settings(self):
        fn = self.src.split("def notify_done(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("notify_on_completion", fn)
        self.assertIn("notify_threshold_seconds", fn)

    def test_taskbar_flash_exists_and_skips_the_foreground_case(self):
        fn = self.src.split("def _flash_taskbar(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("FlashWindowEx", fn)
        self.assertIn("GetForegroundWindow", fn,
                      "flashing while the window is already focused is pointless")

    def test_finish_takes_an_outcome(self):
        self.assertIn("def finish(self, ok=True):", self.src)
        self.assertIn("def _finish_ui(self, elapsed, ok=True):", self.src)

    def test_cancel_and_error_paths_do_not_chime(self):
        """finish() runs on every exit path, so each must say which it was."""
        self.assertGreaterEqual(self.src.count("smooth.finish(ok=False)"), 5,
                                "a cancel or error path still reports success")
        # The queue and audiobook decide at runtime rather than statically.
        self.assertIn("smooth.finish(ok=not cancelled)", self.src)
        self.assertIn('smooth.finish(ok=not res.get("cancelled"))', self.src)

    def test_only_the_success_path_notifies(self):
        fn = self.src.split("def _finish_ui(", 1)[1].split("\n    def ", 1)[0]
        before, after = fn.split("notify_done(", 1)
        self.assertIn("if not ok:", before)
        self.assertIn("return", before,
                      "the not-ok branch must return before reaching notify_done")


if __name__ == "__main__":
    unittest.main(verbosity=2)
