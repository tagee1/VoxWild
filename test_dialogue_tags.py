"""
test_dialogue_tags.py — inline tags work in the Dialogue script, and look like it.

Reported: typing [voice: Heart] hello [/voice] in a dialogue line did nothing,
and the tag never coloured green or orange the way it does in Studio.

Those were two separate faults that looked like one:

  * the tag UI (colouring, '[' autocomplete) was hard-wired to the Studio box,
    so the Dialogue box could never light up;
  * generate_dialogue_audio never parsed tags at all, so the line went to the
    engine verbatim and "[pause 500ms]" was read out loud.

Fixing either alone still looks broken, so both are pinned here. Also covers
three faults found in the Speakers panel while investigating.

Run with:
    python -m pytest test_dialogue_tags.py -v
    python test_dialogue_tags.py          (stdlib unittest)
"""
import io
import os
import re
import unittest

HERE   = os.path.dirname(os.path.abspath(__file__))
APP_PY = os.path.join(HERE, "app.py")


class _Src(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with io.open(APP_PY, encoding="utf-8") as f:
            cls.src = f.read()

    def fn(self, header, end="\ndef "):
        """Body of a top-level function, by its 'def name(' header."""
        self.assertIn(header, self.src, f"{header} not found")
        return self.src.split(header, 1)[1].split(end, 1)[0]


class TestTagsAreHonouredInDialogue(_Src):

    def test_dialogue_parses_speech_tags(self):
        body = self.fn("def generate_dialogue_audio(")
        self.assertIn("parse_speech_tags(", body,
                      "dialogue still sends raw text to the engine, so a tag is spoken")

    def test_the_speakers_voice_is_the_base_for_the_line(self):
        """[voice:] overrides for its span; the Speakers panel still rules the rest."""
        body = self.fn("def generate_dialogue_audio(")
        self.assertIn("parse_speech_tags(proc_text, voice_id, speed)", body)

    def test_pause_tags_become_real_silence(self):
        body = self.fn("def generate_dialogue_audio(")
        self.assertIn('u["kind"] == "pause"', body)
        self.assertIn("np.zeros(", body)

    def test_each_span_uses_its_own_voice_and_speed(self):
        body = self.fn("def generate_dialogue_audio(")
        self.assertIn('voice=u["voice"]', body)
        self.assertIn('speed=u["speed"]', body)

    def test_volume_tags_are_applied_and_normalised(self):
        """[loud] can clip; the whole script is scaled by one factor, not per line."""
        body = self.fn("def generate_dialogue_audio(")
        self.assertIn("apply_tag_gain(", body)
        self.assertIn("used_gain", body)
        self.assertIn("0.97", body)

    def test_pronunciation_still_runs_before_tags(self):
        """Dictionary substitutions must not be able to eat a tag's brackets."""
        body = self.fn("def generate_dialogue_audio(")
        self.assertLess(body.find("apply_pronunciation("),
                        body.find("parse_speech_tags("))

    def test_a_line_of_nothing_but_tags_is_skipped_not_crashed(self):
        body = self.fn("def generate_dialogue_audio(")
        self.assertIn("if not line_samples:", body)

    def test_cancel_still_lands_between_spans(self):
        body = self.fn("def generate_dialogue_audio(")
        self.assertIn("cancel_event.is_set()", body)


class TestTagStrengths(_Src):
    """The four strength tags, tuned by ear. These assert direction and the
    relative-vs-absolute design, not the exact numbers — those get retuned."""

    def parse(self):
        import numpy as np
        ns = {"re": re, "np": np, "VOICES": {}}
        exec(self.src[self.src.index("_TAG_RE = re.compile"):
                      self.src.index("\ndef time_stretch(")], ns)
        return ns

    def span(self, ns, tag, base):
        spans, _ = ns["parse_speech_tags"](f"a [{tag}]b[/{tag}] c", "af_heart", base)
        return [s for s in spans if s["text"].strip() == "b"][0]

    def test_each_tag_pushes_the_right_way(self):
        ns = self.parse()
        self.assertGreater(self.span(ns, "fast",  1.0)["speed"], 1.0)
        self.assertLess(   self.span(ns, "slow",  1.0)["speed"], 1.0)
        self.assertGreater(self.span(ns, "loud",  1.0)["gain"],  1.0)
        self.assertLess(   self.span(ns, "quiet", 1.0)["gain"],  1.0)

    def test_speed_tags_scale_with_the_slider_rather_than_replacing_it(self):
        """A fixed speed would do nothing at a matching slider setting, and
        would SLOW text down above it — the opposite of the tag's name."""
        ns = self.parse()
        slow_slider = self.span(ns, "fast", 0.85)["speed"]
        fast_slider = self.span(ns, "fast", 1.00)["speed"]
        self.assertLess(slow_slider, fast_slider)
        ratio_a = slow_slider / 0.85
        ratio_b = fast_slider / 1.00
        self.assertAlmostEqual(ratio_a, ratio_b, places=6,
                               msg="the contrast must not depend on the slider")

    def test_speed_stays_inside_a_sane_range(self):
        """Past 2.0 speech stops being intelligible; below 0.5 it drawls.

        Checked across every span, not just the tagged one: at a slider already
        at the ceiling the tag clamps to its neighbours' speed, the spans become
        identical and the parser merges them — so there is no "b" span left to
        look up. That merge is correct, and it is why this asserts on all spans.
        """
        ns = self.parse()
        for base, tag in ((2.0, "fast"), (0.5, "slow")):
            spans, _ = ns["parse_speech_tags"](f"a [{tag}]b[/{tag}] c",
                                               "af_heart", base)
            for s in spans:
                if s["kind"] == "text":
                    self.assertGreaterEqual(s["speed"], 0.5)
                    self.assertLessEqual(s["speed"], 2.0)

    def test_a_tag_at_the_ceiling_collapses_instead_of_overshooting(self):
        """At a maxed slider [fast] has nowhere to go, so it must read as plain
        text rather than being clamped into a separate, identical-sounding span."""
        ns = self.parse()
        spans, _ = ns["parse_speech_tags"]("a [fast]b[/fast] c", "af_heart", 2.0)
        texts = [s["text"] for s in spans if s["kind"] == "text"]
        self.assertEqual(len(texts), 1, f"expected one merged span, got {texts}")

    def test_quiet_is_well_below_half_volume(self):
        """-6 dB was not quiet enough against Natural's output; now -12 dB."""
        ns = self.parse()
        self.assertLessEqual(ns["_TAG_QUIET"], 0.3)


class TestRateTagWithdrawn(_Src):
    """[rate N] was withdrawn on request — it multiplied the speed slider
    instead of setting a speed, so "[rate 1.5]" at a 0.85 slider only reached
    1.28 and read as doing nothing. [slow]/[fast] cover the same ground.

    It is still CONSUMED rather than deleted from the parser: an unrecognized
    tag is left in the text and read out loud, so simply removing it would make
    every saved script start announcing "rate one point five".
    """

    def parse(self):
        import numpy as np
        ns = {"re": re, "np": np, "VOICES": {}}
        exec(self.src[self.src.index("_TAG_RE = re.compile"):
                      self.src.index("\ndef time_stretch(")], ns)
        return ns

    def test_it_is_gone_from_the_tag_guide(self):
        self.assertNotIn("[rate 0.8] … [/rate]", self.src)

    def test_it_is_gone_from_the_insert_menu(self):
        self.assertNotIn("Rate (exact)", self.src)
        self.assertNotIn('"[rate 0.8]‸[/rate]"', self.src)

    def test_it_is_never_read_out_loud(self):
        """The reason it is consumed rather than dropped."""
        ns = self.parse()
        spans, _ = ns["parse_speech_tags"]("a [rate 1.5]b[/rate] c", "af_heart", 1.0)
        spoken = "".join(s["text"] for s in spans if s["kind"] == "text")
        self.assertNotIn("rate", spoken.lower())
        self.assertNotIn("[", spoken)

    def test_it_changes_nothing(self):
        ns = self.parse()
        spans, used = ns["parse_speech_tags"]("a [rate 1.5]b[/rate] c",
                                              "af_heart", 0.85)
        for s in spans:
            if s["kind"] == "text":
                self.assertAlmostEqual(s["speed"], 0.85, places=6)
        self.assertFalse(used, "a withdrawn tag must not count as an effect")

    def test_the_text_is_not_split_into_extra_synthesis_calls(self):
        """A no-op that still split spans would cost an extra engine call per
        tag — minutes, in Natural."""
        ns = self.parse()
        spans, _ = ns["parse_speech_tags"]("a [rate 1.5]b[/rate] c", "af_heart", 1.0)
        self.assertEqual(len([s for s in spans if s["kind"] == "text"]), 1)

    def test_it_is_transparent_inside_another_speed_tag(self):
        """Pushing a flat 1.0 would cancel an enclosing [fast]; it pushes a copy
        of the current factor instead."""
        ns = self.parse()
        spans, _ = ns["parse_speech_tags"]("[fast]a [rate 2]b[/rate] c[/fast]",
                                           "af_heart", 1.0)
        for s in spans:
            if s["kind"] == "text":
                self.assertAlmostEqual(s["speed"], ns["_TAG_FAST"], places=6)

    def test_volume_was_not_removed_with_it(self):
        """[volume N] shared the same parser branch and had to survive it."""
        ns = self.parse()
        spans, _ = ns["parse_speech_tags"]("a [volume 2]b[/volume] c", "af_heart", 1.0)
        gains = [s["gain"] for s in spans if s["kind"] == "text"]
        self.assertIn(2.0, gains)

    def test_the_other_speed_tags_still_work(self):
        ns = self.parse()
        for tag, expect in (("fast", ns["_TAG_FAST"]), ("slow", ns["_TAG_SLOW"])):
            spans, _ = ns["parse_speech_tags"](f"a [{tag}]b[/{tag}] c", "af_heart", 1.0)
            mid = [s for s in spans if s["text"].strip() == "b"][0]
            self.assertAlmostEqual(mid["speed"], expect, places=6)


class TestTagUiIsShared(_Src):

    def test_the_tag_ui_is_no_longer_bound_to_one_box(self):
        self.assertNotIn("_tag_inner", self.src,
                         "the hard-wired Studio-only box is back")
        self.assertIn("_tag_active = [None]", self.src)

    def test_attach_helper_exists(self):
        self.assertIn("def _tag_attach(", self.src)

    def test_both_boxes_are_attached(self):
        self.assertIn("_tag_attach(text_input)", self.src)
        self.assertIn("_tag_attach(dlg_text)", self.src)

    def test_attach_sets_up_colouring_and_the_menu(self):
        body = self.fn("def _tag_attach(")
        self.assertIn("tag_ok", body)
        self.assertIn("tag_bad", body)
        self.assertIn("_tag_on_keyrelease", body)
        self.assertIn("_tag_key_nav", body)

    def test_focus_switches_the_active_box(self):
        body = self.fn("def _tag_attach(")
        self.assertIn("<FocusIn>", body,
                      "without this the menu would edit whichever box was last attached")

    def test_helpers_cope_with_no_active_box(self):
        for header in ("def _tag_recolor(", "def _tag_ctx(",
                       "def _tag_menu_show(", "def _tag_insert_snippet("):
            body = self.fn(header)
            self.assertIn("if inner is None:", body,
                          f"{header} would raise before any box has focus")

    def test_dialogue_has_the_tag_guide(self):
        """Tags do something there now, so the help belongs there too."""
        self.assertEqual(self.src.count('text="?  Tag guide"'), 2)

    def test_dialogue_hint_no_longer_demands_all_caps(self):
        """The rule was relaxed; the hint kept sending people off to shout.

        Checks the label the user actually reads, not the surrounding source —
        the comment recording why it changed is meant to say "all caps".
        """
        panel = self.src.split("_section_label(dlg_script_panel", 1)[1][:900]
        label = panel.split("text=", 1)[1].split("font=", 1)[0].lower()
        self.assertNotIn("all caps", label)
        self.assertIn("capital", label)
        self.assertIn("[tags]", label, "the hint should say tags work here")


class TestDialogueCanBeStopped(_Src):
    """Reported as "I can't stop a dialogue generation".

    The underlying mechanism was fine — measured at 0.16s from press to stop.
    Both ways of reaching it were broken instead: the tab's own Cancel button was
    invisible, and the header Stop was never enabled.
    """

    def test_the_cancel_button_is_not_the_same_colour_as_its_panel(self):
        """BTN_DARK fills with C_CARD, which IS the panel colour, with dim text
        and no border. The button rendered correctly and could not be seen."""
        # Up to the next statement, not the first ")" — that lands inside
        # CTkFont(...) and cuts the style argument off.
        block = self.src.split("dlg_cancel_btn = ctk.CTkButton(", 1)[1].split("\n#", 1)[0]
        self.assertNotIn("BTN_DARK", block,
                         "a C_CARD button on a C_CARD panel is invisible")
        self.assertIn("BTN_DANGER", block)

    def test_the_header_stop_is_armed_during_a_dialogue_run(self):
        """It sits in the tab row, so it is on screen from the Dialogue tab —
        but dialogue never enabled it, so it stayed greyed out all run."""
        fn = self.fn("def dlg_generate(")
        self.assertIn("_set_stop_live()", fn)

    def test_the_header_stop_is_released_afterwards(self):
        fn = self.fn("def dlg_generate(")
        self.assertIn("_reset_stop_button", fn)

    def test_header_stop_sets_the_dialogue_flag_too(self):
        """Dialogue checks its own event between lines; aborting the inference
        alone would stop one line and then carry on to the next speaker."""
        fn = self.fn("def cancel_generation(")
        self.assertIn("_dlg_cancel_event.set()", fn)
        self.assertIn("_abort_kokoro()", fn)

    def test_that_flag_set_cannot_break_start_up(self):
        """cancel_generation is defined long before the Dialogue tab is built."""
        fn = self.fn("def cancel_generation(")
        self.assertIn("except NameError:", fn)

    def test_the_dialogue_cancel_button_aborts_the_running_line(self):
        fn = self.fn("def dlg_generate(")
        self.assertIn("_dlg_cancel_event.set()", fn)
        self.assertIn("_abort_kokoro()", fn)


class TestSpeakerPanelFixes(_Src):

    def test_the_rename_button_says_rename(self):
        """It was labelled 'Reset' while renaming the speaker through the script."""
        self.assertIn('text="Rename"', self.src)
        panel = self.fn("def dlg_detect_speakers(")
        self.assertNotIn('text="Reset"', panel)

    def test_the_speaker_row_is_split_over_two_lines(self):
        """One line wanted 471px inside a 294px panel, so Rename never rendered.

        Measured at 150% scaling: the button came back with x=0 and width=1 —
        Tk had no room left to lay it out at all.
        """
        panel = self.fn("def dlg_detect_speakers(")
        self.assertIn("line1", panel)
        self.assertIn("line2", panel)
        self.assertIn('menu.pack(fill="x")', panel,
                      "the voice dropdown should own the second line")

    def test_rename_is_packed_before_the_name_box(self):
        """Pack order decides who gets starved; the entry expands, so it goes last.

        Packed after, the button was squeezed to 45px and its label clipped —
        the same failure the Studio 'Tag guide' button had.
        """
        panel = self.fn("def dlg_detect_speakers(")
        self.assertLess(panel.find('text="Rename"'),
                        panel.find("name_entry = ctk.CTkEntry("),
                        "the expand=True name box must be packed after the button")

    def test_the_row_widgets_do_not_carry_fixed_widths_that_overflow(self):
        """The name box and dropdown stretch now instead of asking for a fixed size."""
        panel = self.fn("def dlg_detect_speakers(")
        entry = panel.split("name_entry = ctk.CTkEntry(", 1)[1].split(")", 1)[0]
        self.assertNotIn("width=", entry)
        menu = panel.split("menu = ctk.CTkOptionMenu(", 1)[1].split(")", 1)[0]
        self.assertNotIn("width=", menu)

    def test_rename_does_not_force_all_caps(self):
        panel = self.fn("def dlg_detect_speakers(")
        self.assertNotIn("nv.get().strip().upper()", panel,
                         'renaming to "Dr Smith" would produce "DR SMITH"')

    def test_redetecting_keeps_the_voices_already_chosen(self):
        panel = self.fn("def dlg_detect_speakers(")
        self.assertIn("keep_voices", panel)
        self.assertIn("keep_voices.get(speaker)", panel)

    def test_detect_still_works_as_a_plain_button_command(self):
        """Tk calls it with no arguments, so the new parameter must be optional."""
        self.assertIn("def dlg_detect_speakers(keep_voices=None):", self.src)
        self.assertIn("dlg_detect_btn.configure(command=dlg_detect_speakers)", self.src)

    def test_rename_carries_the_voice_onto_the_new_name(self):
        panel = self.fn("def dlg_detect_speakers(")
        self.assertIn("dlg_detect_speakers(keep_voices=_keep)", panel)

    def test_rename_always_says_something(self):
        """Every early exit used to be a silent return, so pressing Rename
        without editing the name first looked like a dead button."""
        panel = self.fn("def dlg_detect_speakers(")
        do = panel.split("def _do(", 1)[1].split("return _do", 1)[0]
        for i, ln in enumerate(do.splitlines()):
            if ln.strip() == "return":
                prior = "\n".join(do.splitlines()[max(0, i - 4):i])
                self.assertIn("status_label.configure", prior,
                              "a Rename exit path gives the user nothing")

    def test_rename_reports_how_many_lines_changed(self):
        panel = self.fn("def dlg_detect_speakers(")
        self.assertIn("re.subn(", panel,
                      "use subn so a rename that matched nothing can be reported")

    def test_enter_commits_a_rename(self):
        panel = self.fn("def dlg_detect_speakers(")
        self.assertIn('name_entry.bind("<Return>"', panel)

    def test_merging_into_an_existing_speaker_keeps_their_voice(self):
        panel = self.fn("def dlg_detect_speakers(")
        self.assertIn("_merged", panel,
                      "renaming onto an existing speaker must not clobber their voice")

    def test_an_unknown_kept_voice_falls_back_to_a_default(self):
        """A voice that no longer exists must not end up in the dropdown."""
        panel = self.fn("def dlg_detect_speakers(")
        self.assertIn("_kept in VOICES", panel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
