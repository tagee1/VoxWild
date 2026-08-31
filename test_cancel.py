"""
test_cancel.py — Cancel must stop work immediately AND leave the app usable.

Two halves:

  TestKokoroAbort   exercises the real ONNX abort against the real Kokoro model.
                    Slow (a generation on a 2-core box is tens of seconds) and
                    skipped automatically if the model files aren't present.

  TestCancelWiring  reads app.py as text. Cheap, and it catches the regression
                    that actually matters: the ONNX terminate flag is sticky, so
                    a new generation path that forgets to arm it would not fail
                    at cancel time — it would break generating entirely, on
                    every run, with a confusing ONNX error.

Run with:
    python -m pytest test_cancel.py -v
    python test_cancel.py          (stdlib unittest)
"""
import io
import os
import re
import threading
import time
import unittest

HERE   = os.path.dirname(os.path.abspath(__file__))
MODEL  = os.path.join(HERE, "kokoro-v1.0.onnx")
VOICES = os.path.join(HERE, "voices-v1.0.bin")
APP_PY = os.path.join(HERE, "app.py")

_HAVE_MODEL = os.path.exists(MODEL) and os.path.exists(VOICES)


# ══════════════════════════════════════════════════════════════════════════════
# The real thing: abort a running inference, then prove the engine still works
# ══════════════════════════════════════════════════════════════════════════════
@unittest.skipUnless(_HAVE_MODEL, "Kokoro model files not present")
class TestKokoroAbort(unittest.TestCase):
    """Mirrors exactly what app.py does — see _abort_kokoro / _arm_kokoro."""

    @classmethod
    def setUpClass(cls):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        cls.kokoro   = Kokoro(MODEL, VOICES)
        cls.run_opts = ort.RunOptions()

        # kokoro_onnx calls sess.run(None, inputs) with no run options of its
        # own, so wrap run() to slip ours in. Same wrapper app.py installs.
        _orig = cls.kokoro.sess.run

        def _run(output_names, input_feed, run_options=None):
            return _orig(output_names, input_feed, run_options or cls.run_opts)

        cls.kokoro.sess.run = _run

    def _arm(self):
        self.run_opts.terminate = False

    def _abort(self):
        self.run_opts.terminate = True

    def _generate(self, text):
        return self.kokoro.create(text, voice="am_michael", speed=1.0, lang="en-us")

    def test_abort_stops_a_running_generation_quickly(self):
        self._arm()
        long_text = "The quick brown fox jumped over the lazy dog. " * 18
        outcome = {}

        def worker():
            try:
                self._generate(long_text)
                outcome["raised"] = None
            except Exception as e:                     # noqa: BLE001 - recording it
                outcome["raised"] = e

        th = threading.Thread(target=worker)
        th.start()
        time.sleep(3.0)          # let it get properly under way
        t0 = time.time()
        self._abort()
        th.join(timeout=60)
        stopped_after = time.time() - t0

        self.assertFalse(th.is_alive(), "generation did not stop after abort")
        self.assertIsNotNone(outcome["raised"],
                             "generation ran to completion — abort did nothing")
        self.assertIn("terminate flag", str(outcome["raised"]).lower())
        # Generous bound. The measured value is ~0.04 s; anything under a couple
        # of seconds means the user sees it as instant.
        self.assertLess(stopped_after, 5.0,
                        f"abort took {stopped_after:.1f}s — too slow to feel instant")

    def test_engine_still_works_after_an_abort(self):
        """The trap: the flag is sticky. Left set, every later run dies."""
        self._arm()
        long_text = "The quick brown fox jumped over the lazy dog. " * 18
        th = threading.Thread(target=lambda: self._safe_generate(long_text))
        th.start()
        time.sleep(3.0)
        self._abort()
        th.join(timeout=60)

        # This is what _arm_kokoro() does at the start of every job.
        self._arm()
        samples, sr = self._generate("Back to normal.")
        self.assertGreater(len(samples) / sr, 0.2,
                           "no real audio after a cancel — the abort flag stayed set")

    def test_cancel_then_generate_survives_repetition(self):
        """One recovery could be luck. Do it twice."""
        for i in range(2):
            self._arm()
            th = threading.Thread(
                target=lambda: self._safe_generate(
                    "The quick brown fox jumped over the lazy dog. " * 18))
            th.start()
            time.sleep(2.0)
            self._abort()
            th.join(timeout=60)

            self._arm()
            samples, sr = self._generate("Round two.")
            self.assertGreater(len(samples) / sr, 0.2,
                               f"no audio after cancel #{i + 1}")

    def _safe_generate(self, text):
        try:
            self._generate(text)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Wiring: every path that starts work must arm the flag first
# ══════════════════════════════════════════════════════════════════════════════
class TestCancelWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with io.open(APP_PY, encoding="utf-8") as f:
            cls.src = f.read()

    def test_arm_and_abort_helpers_exist(self):
        for name in ("_arm_kokoro", "_abort_kokoro", "_reset_cancel",
                     "_is_kokoro_abort", "_restart_chatterbox_bg"):
            self.assertIn(f"def {name}(", self.src, f"{name} is missing")

    def test_no_bare_cancel_event_clear_outside_reset_cancel(self):
        """_cancel_event.clear() alone leaves the ONNX abort flag stuck on.

        Only _reset_cancel() may call it; everything else must go through
        _reset_cancel so the Fast-mode flag is cleared at the same time.
        """
        body = self.src.split("def _reset_cancel(", 1)
        self.assertEqual(len(body), 2, "_reset_cancel() not found")
        before, after = body[0], body[1]
        # Skip past _reset_cancel's own body to the next top-level def.
        rest = re.split(r"\n(?=\S)", after, maxsplit=1)
        remainder = rest[1] if len(rest) > 1 else ""
        # Word-boundary match so the Dialogue tab's own _dlg_cancel_event.clear()
        # -- a different event, and armed separately -- isn't mistaken for this.
        bare = re.compile(r"(?<!\w)_cancel_event\.clear\(\)")
        for chunk, where in ((before, "before"), (remainder, "after")):
            for line in chunk.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or '"' in stripped:
                    continue
                self.assertIsNone(
                    bare.search(stripped),
                    f"bare _cancel_event.clear() {where} _reset_cancel: "
                    "use _reset_cancel() so the Fast-mode abort flag is cleared too")

    def test_dialogue_path_arms_the_abort_flag(self):
        """Dialogue uses its own event, so it needs its own arm call."""
        i = self.src.find("_dlg_cancel_event.clear()")
        self.assertGreater(i, -1, "_dlg_cancel_event.clear() not found")
        window = self.src[i:i + 300]
        self.assertIn("_arm_kokoro()", window,
                      "Dialogue clears its flag without arming the Fast-mode "
                      "abort switch -- the next dialogue run would die instantly")

    def test_kokoro_create_translates_the_abort_error(self):
        """One funnel covers Studio, Dialogue, Audiobook and previews."""
        fn = self.src.split("def kokoro_create(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_is_kokoro_abort", fn)
        self.assertIn("GenerationCancelled", fn)

    def test_cancel_pulls_every_lever(self):
        fn = self.src.split("def cancel_generation(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_cancel_event.set()", fn)
        self.assertIn("_abort_kokoro()", fn)          # Fast
        self.assertIn("chatterbox_engine.stop(force=True)", fn)   # Natural
        self.assertIn("enhance_engine.stop(force=True)", fn)      # Enhance

    def test_second_press_logic_is_gone(self):
        """The double press only existed to opt into Natural's reload cost."""
        fn = self.src.split("def stop_audio(", 1)[1].split("\ndef ", 1)[0]
        self.assertNotIn("press Stop again", fn)
        self.assertNotIn("Natural will reload its model next time", fn)
        self.assertIn("Cancelling", fn)

    def test_cancel_reaches_enhancement(self):
        """Enhancement outlives is_generating; Cancel has to know that."""
        fn = self.src.split("def stop_audio(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_enhance_active", fn,
                      "stop_audio ignores a running enhancement")

    def test_enhance_engine_stop_supports_force(self):
        self.assertIn("def stop(self, force=False):", self.src)
        # Both engines, not just Chatterbox.
        self.assertEqual(self.src.count("def stop(self, force=False):"), 2)

    def test_generate_chunk_checks_cancel_before_stopping_the_engine(self):
        """Ordering matters: a stop() here would kill the restarted worker."""
        fn = self.src.split("def generate_chunk(", 1)[1].split("\n    def ", 1)[0]
        tail = fn.split("Reaching here means the pipe closed", 1)[1]
        cancel_at = tail.find("_cancel_event.is_set()")
        stop_at   = tail.find("self.stop()")
        self.assertGreater(cancel_at, -1)
        self.assertGreater(stop_at, -1)
        self.assertLess(cancel_at, stop_at,
                        "self.stop() runs before the cancel check — it would "
                        "race _restart_chatterbox_bg and kill the fresh worker")


if __name__ == "__main__":
    unittest.main(verbosity=2)
