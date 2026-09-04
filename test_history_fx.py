"""
test_history_fx.py — the Orig button is reachable; FX settings actually persist.

Two defects from full-pass testing, section H.

  1. Enhanced clips showed the ✨ badge but no Orig button. It was being
     created — the badge and the button are gated on the same flag — but pack
     starves whoever is added last, and Orig was last. Measured at 150% display
     scaling, the history card's action row has 249px:

         Delete  x=171 w=78    Save x=0 w=64    📁 x=70 w=47
         SRT     x=123 w=42 (wants 56)
         Orig    x=  0 w= 1 (wants 69)   <- one pixel wide, invisible

     SRT is only built when the entry has segments, which every Studio
     generation has, so real clips always hit the crowded case. Orig now has a
     row of its own: 218px of the 249 available, still fitting once Pause and
     Stop appear during playback.

  2. FX settings never saved. _save_fx_settings ran from exactly one place —
     the window-close handler — and no fx_* key had ever reached settings.json.
     Any failure was swallowed by a bare `except Exception: pass`, so there was
     nothing to find. It now saves whenever a control changes, and logs.

Run with:
    python -m pytest test_history_fx.py -v
    python test_history_fx.py          (stdlib unittest)
"""
import io
import os
import re
import unittest

HERE   = os.path.dirname(os.path.abspath(__file__))
APP_PY = os.path.join(HERE, "app.py")

with io.open(APP_PY, encoding="utf-8") as _f:
    SRC = _f.read()


class _Stub:
    """Stands in for a slider / BooleanVar / StringVar.

    on_set mimics a Tk write trace, so a test can prove that restoring the
    panel does not fire the save the way a user edit does.
    """
    on_set = None

    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value
    def set(self, v):
        self.value = v
        if self.on_set:
            self.on_set()


def _load_fx():
    """Exec the FX save/restore block with the UI replaced by stubs."""
    start = SRC.index("_FX_DEFAULTS = {")
    end   = SRC.index("# ── Calibration", start)
    saved = {}
    ns = {
        "DEFAULT_LOUDNESS_PRESET": "Podcast",
        "LOUDNESS_PRESETS": {"Podcast": -16.0, "Audiobook": -18.0},
        "_get_settings": lambda: saved,
        "_save_settings": lambda s: saved.update(s),
        "_log_crash": lambda e: ns.setdefault("_errors", []).append(e),
        "update_all_labels": lambda: None,
        "app": None,                       # no event loop in a test
        "highpass_slider":      _Stub(120),
        "lowpass_slider":       _Stub(14000),
        "reverb_slider":        _Stub(0.15),
        "compressor_slider":    _Stub(3.5),
        "gain_slider":          _Stub(6),
        "compressor_var":       _Stub(False),
        "noise_gate_var":       _Stub(True),
        "trim_var":             _Stub(False),
        "enhance_mode":         _Stub("Sync"),
        "normalize_var":        _Stub(True),
        "normalize_preset_var": _Stub("Audiobook"),
    }
    exec(SRC[start:end], ns)
    return ns, saved


class TestFxRoundTrip(unittest.TestCase):
    """The bug was never reproduced by reading the code — it saved nothing at
    all and said nothing about why. These run the real functions."""

    def test_saving_writes_every_key(self):
        ns, saved = _load_fx()
        ns["_save_fx_settings"]()
        for key in ns["_FX_DEFAULTS"]:
            self.assertIn(key, saved, f"{key} never reaches settings.json")

    def test_saving_writes_the_actual_values(self):
        ns, saved = _load_fx()
        ns["_save_fx_settings"]()
        self.assertEqual(saved["fx_highpass"], 120)
        self.assertEqual(saved["fx_lowpass"], 14000)
        self.assertEqual(saved["fx_gain"], 6)
        self.assertEqual(saved["fx_noise_gate"], True)
        self.assertEqual(saved["fx_trim"], False)
        self.assertEqual(saved["fx_normalize_preset"], "Audiobook")

    def test_it_reports_a_failure_instead_of_hiding_it(self):
        """A silent `pass` is why this went unnoticed for months."""
        ns, _ = _load_fx()
        ns["gain_slider"] = None                    # force an AttributeError
        ns["_save_fx_settings"]()
        self.assertTrue(ns.get("_errors"), "the failure was swallowed again")

    def test_a_failure_still_cannot_stop_the_app_closing(self):
        ns, _ = _load_fx()
        ns["normalize_var"] = None
        ns["_save_fx_settings"]()                   # must not raise

    def test_restore_puts_the_saved_values_back(self):
        ns, saved = _load_fx()
        ns["_save_fx_settings"]()
        for name in ("highpass_slider", "gain_slider", "trim_var", "normalize_var"):
            ns[name].set(None)                      # wipe the panel
        ns["_restore_fx_settings"]()
        self.assertEqual(ns["highpass_slider"].get(), 120)
        self.assertEqual(ns["gain_slider"].get(), 6)
        self.assertEqual(ns["trim_var"].get(), False)
        self.assertEqual(ns["normalize_var"].get(), True)

    def test_restore_falls_back_to_defaults_on_a_fresh_install(self):
        ns, _ = _load_fx()
        ns["_restore_fx_settings"]()
        self.assertEqual(ns["highpass_slider"].get(), ns["_FX_DEFAULTS"]["fx_highpass"])
        self.assertEqual(ns["trim_var"].get(), ns["_FX_DEFAULTS"]["fx_trim"])

    def test_restore_does_not_trigger_a_save(self):
        """Every .set() during restore fires the same trace a user edit does.
        Without the guard, start-up writes the file back over itself."""
        ns, _ = _load_fx()
        seen = []
        ns["_save_fx_settings"] = lambda: seen.append(1)
        # Make each stub behave like a traced variable.
        for name, w in list(ns.items()):
            if isinstance(w, _Stub):
                w.on_set = ns["_schedule_fx_save"]
        ns["_restore_fx_settings"]()
        self.assertEqual(seen, [], "start-up scheduled a redundant write")

    def test_a_real_edit_after_start_up_does_save(self):
        """The guard must not leave saving switched off."""
        ns, _ = _load_fx()
        seen = []
        ns["_save_fx_settings"] = lambda: seen.append(1)
        ns["_restore_fx_settings"]()
        ns["_schedule_fx_save"]()                   # as a user edit would
        self.assertEqual(seen, [1], "no save happens after start-up")

    def test_the_guard_is_released_even_if_restore_fails(self):
        ns, _ = _load_fx()
        ns["highpass_slider"] = None                # blow up mid-restore
        ns["_restore_fx_settings"]()
        self.assertFalse(ns["_fx_restoring"][0],
                         "saving would stay disabled for the whole session")

    def test_the_enhance_checkbox_is_never_persisted(self):
        """Restoring it re-runs pip against python_embed at start-up."""
        ns, saved = _load_fx()
        ns["_save_fx_settings"]()
        self.assertFalse(saved["fx_enhance"])
        restore = SRC.split("def _restore_fx_settings(", 1)[1].split("\ndef ", 1)[0]
        self.assertNotIn("\n        enhance_var.set(", restore)


class TestFxWiring(unittest.TestCase):

    def fn(self, header):
        self.assertIn(header, SRC, f"{header} not found")
        return SRC.split(header, 1)[1].split("\ndef ", 1)[0]

    def test_changing_a_control_saves(self):
        """Close-only saving lost the panel to any close the handler missed."""
        self.assertIn("_schedule_fx_save()", self.fn("def _on_eq_manual_change("))

    def test_every_checkbox_and_dropdown_is_wired(self):
        traced = set()
        for block in re.findall(r"for _fx_v in \(([^)]*)\)", SRC):
            traced |= {n.strip() for n in block.split(",") if n.strip()}
        for name in ("noise_gate_var", "trim_var", "compressor_var",
                     "enhance_mode", "normalize_var", "normalize_preset_var"):
            self.assertIn(name, traced, f"{name} changes are still not saved")

    def test_each_variable_is_defined_before_its_trace(self):
        """A trace can't be hung on a name that doesn't exist yet — getting this
        wrong is an immediate crash at start-up, not a subtle bug."""
        for block in re.finditer(r"for _fx_v in \(([^)]*)\):", SRC):
            for name in (n.strip() for n in block.group(1).split(",") if n.strip()):
                defined = re.search(rf"^{re.escape(name)} = ", SRC, re.MULTILINE)
                self.assertIsNotNone(defined, f"{name} is never defined")
                self.assertLess(defined.start(), block.start(),
                                f"{name} is traced before it exists")

    def test_programmatic_slider_changes_also_save(self):
        """A CTkSlider fires its command on a drag, not on .set(), so every
        place that moves a slider in code has to ask for the save itself."""
        for header in ("def apply_eq_preset(",      # EQ preset dropdown
                       "def reset_enhancements(",   # Reset to defaults
                       "def apply_settings("):      # loading a profile
            self.assertIn("_schedule_fx_save()", self.fn(header),
                          f"{header} changes the panel without persisting it")

    def test_the_write_is_debounced(self):
        """Sliders fire continuously while dragging."""
        body = self.fn("def _schedule_fx_save(")
        self.assertIn("after_cancel", body)
        self.assertIn("app.after(", body)

    def test_closing_still_saves_as_a_backstop(self):
        self.assertIn("_save_fx_settings()", self.fn("def on_close("))

    def test_no_silent_pass_is_left_in_either_function(self):
        for header in ("def _save_fx_settings(", "def _restore_fx_settings("):
            body = self.fn(header)
            self.assertNotIn("except Exception:\n        pass", body,
                             f"{header} still hides its failures")
            self.assertIn("_log_crash(", body)


class TestOrigButtonHasItsOwnRow(unittest.TestCase):

    COND = 'entry.get("original_samples") is not None'

    def setUp(self):
        # The condition appears three times: saving the original to disk, the
        # ✨ badge, and the button block. Anchor on the last one — splitting on
        # the first lands in the persistence code and picks up the main play
        # row's Pause button instead.
        self.blk = SRC.rsplit(f"if {self.COND}:", 1)[1].split("return outer", 1)[0]

    def test_a_separate_row_is_created(self):
        self.assertIn("row_orig = ctk.CTkFrame(", self.blk)
        self.assertIn('row_orig.pack(fill="x"', self.blk)

    def test_all_three_buttons_moved_off_the_crowded_row(self):
        for btn in ("Orig", "Pause", "Stop"):
            m = re.search(rf'ctk\.CTkButton\(\s*(\w+), text="{btn}"', self.blk)
            self.assertIsNotNone(m, f"the {btn} button is gone")
            self.assertEqual(m.group(1), "row_orig",
                             f"{btn} is still packed into the full action row")

    def test_the_badge_and_the_button_still_share_one_condition(self):
        """If they ever diverge, a clip shows ✨ with no way to hear the original
        — which is exactly what this looked like from the outside."""
        badge = SRC.split('text="✨ enhanced"', 1)[0][-200:]
        self.assertIn(f"elif {self.COND}:", badge,
                      "the ✨ badge no longer keys off the original audio")
        self.assertIn(f"if {self.COND}:", SRC.rsplit("row_orig = ", 1)[0],
                      "the Orig row is no longer gated on the original audio")

    def test_pause_and_stop_are_still_hidden_until_playback(self):
        """They read as empty boxes when disabled."""
        self.assertNotIn('orig_pause_btn.pack(', self.blk)
        self.assertNotIn('orig_stop_btn.pack(', self.blk)
        self.assertIn('orig_play_btn.pack(', self.blk)

    def test_the_button_says_what_it_does(self):
        self.assertIn("_Tooltip(orig_play_btn", self.blk)


if __name__ == "__main__":
    unittest.main(verbosity=2)
