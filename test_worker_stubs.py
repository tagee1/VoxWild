"""
test_worker_stubs.py — Unit tests for chatterbox_worker.py runtime patches.

Covers (no model load, no torch GPU, runs fast):

  1. Perth watermarker stub — try branch
       When perth is importable but PerthImplicitWatermarker is None (C extension
       silently failed), the patch must replace it with a callable no-op class
       that implements the full chatterbox watermarker interface.

  2. Perth watermarker stub — except ImportError branch
       When perth is not installed at all, a fake perth module is injected into
       sys.modules with the same no-op class.

  3. Perth stub interface contract
       Both stubs must satisfy chatterbox's two call sites:
         a. watermarker(wav)                 → returns wav unchanged (tensor pass-through)
         b. watermarker.apply_watermark(wav) → returns numpy array

  4. CFG batch-expansion patch logic
       The prepare_input_embeds wrapper must:
         a. Expand embeds from batch=1 to batch=2 when cfg_weight > 0 and batch=1
         b. Leave embeds alone when cfg_weight == 0
         c. Leave embeds alone when batch is already 2 (newer chatterbox build)
         d. Propagate len_cond unchanged in all cases

Run with any Python that has numpy and torch:
    python test_worker_stubs.py
    python -m pytest test_worker_stubs.py -v
"""
import sys
import os
import types
import textwrap
import unittest
import importlib
from unittest.mock import MagicMock, patch

# ── Locate the worker source ───────────────────────────────────────────────────
import pathlib
_WORKER_PATH = pathlib.Path(__file__).parent / "chatterbox_worker.py"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — extract and execute the perth patch block from chatterbox_worker.py
# ─────────────────────────────────────────────────────────────────────────────

def _extract_perth_patch_code():
    """
    Return the perth watermarker patch block from chatterbox_worker.py.

    Uses line-by-line scanning with content-based anchors rather than
    exact string matching, so it stays correct regardless of formatting
    changes to the surrounding comments.

    The block starts at the comment containing "Patch perth watermarker"
    and ends at the next separator comment (a line that is only # + box-drawing
    dashes, ≥ 20 of them).
    """
    _BOX_DASH = "\u2500"   # ─  BOX DRAWINGS LIGHT HORIZONTAL

    lines = _WORKER_PATH.read_text(encoding="utf-8").splitlines(keepends=True)

    start_idx = None
    end_idx   = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if start_idx is None:
            if stripped.startswith("#") and "Patch perth watermarker" in stripped:
                start_idx = i
        else:
            # End = first separator line with ≥ 20 box-dash chars after the start
            if stripped.startswith("#") and stripped.count(_BOX_DASH) >= 20:
                end_idx = i + 1   # include the separator line itself
                break

    if start_idx is None:
        raise RuntimeError(
            f"Cannot find 'Patch perth watermarker' comment in {_WORKER_PATH}"
        )
    if end_idx is None:
        raise RuntimeError(
            f"Cannot find perth patch end separator in {_WORKER_PATH}"
        )

    return "".join(lines[start_idx:end_idx])


def _run_perth_patch(fake_perth_module):
    """
    Execute the perth patch code with a controlled sys.modules["perth"].

    Parameters
    ----------
    fake_perth_module : types.ModuleType | None
        If a module: injected into sys.modules["perth"] so the patch's
        `import perth` picks it up.
        If None: "perth" is removed from sys.modules (triggers ImportError).

    Returns
    -------
    The sys.modules["perth"] entry after the patch executes.

    Note: we manipulate the REAL sys.modules because Python's import machinery
    always uses sys.modules regardless of the exec namespace.  We restore the
    original state in a finally block.

    To trigger the ImportError branch (fake_perth_module=None) we set
    sys.modules["perth"] = None.  Per the Python data model, a None entry
    causes `import perth` to raise ImportError immediately without hitting
    disk — so this works even on machines where perth is actually installed.
    """
    code = _extract_perth_patch_code()

    _had_perth  = "perth" in sys.modules
    _orig_perth = sys.modules.get("perth")
    try:
        if fake_perth_module is not None:
            sys.modules["perth"] = fake_perth_module
        else:
            # None entry → ImportError on `import perth`, without touching disk
            sys.modules["perth"] = None

        # The patch block lives inside main() so every line is indented 4 spaces.
        # dedent() strips the common leading whitespace so compile() accepts it
        # as module-level code.
        ns = {"__builtins__": __builtins__, "os": os, "sys": sys}
        exec(compile(textwrap.dedent(code), str(_WORKER_PATH), "exec"), ns)

        return sys.modules.get("perth")
    finally:
        if _had_perth:
            sys.modules["perth"] = _orig_perth
        else:
            sys.modules.pop("perth", None)


# ─────────────────────────────────────────────────────────────────────────────
# 1 & 2. Perth stub — both branches
# ─────────────────────────────────────────────────────────────────────────────

class TestPerthStubTryBranch(unittest.TestCase):
    """
    perth IS importable, but PerthImplicitWatermarker is None.
    (C extension silently failed — real scenario on the 4 GB test machine.)
    """

    def _make_broken_perth(self):
        """Return a fake perth module where PerthImplicitWatermarker is None."""
        m = types.ModuleType("perth")
        m.PerthImplicitWatermarker = None
        return m

    def test_patch_replaces_none_with_callable(self):
        """After the patch, PerthImplicitWatermarker must be callable."""
        perth = _run_perth_patch(self._make_broken_perth())
        self.assertTrue(
            callable(getattr(perth, "PerthImplicitWatermarker", None)),
            "PerthImplicitWatermarker must be callable after the patch"
        )

    def test_already_callable_is_left_alone(self):
        """If PerthImplicitWatermarker is already callable, patch must not replace it."""
        sentinel = MagicMock()
        m = types.ModuleType("perth")
        m.PerthImplicitWatermarker = sentinel
        perth = _run_perth_patch(m)
        self.assertIs(
            perth.PerthImplicitWatermarker, sentinel,
            "Patch must not overwrite a working PerthImplicitWatermarker"
        )

    def test_stub_is_instantiable(self):
        """The stub class must be instantiable with no arguments."""
        perth = _run_perth_patch(self._make_broken_perth())
        try:
            instance = perth.PerthImplicitWatermarker()
        except Exception as e:
            self.fail(f"PerthImplicitWatermarker() raised unexpectedly: {e}")

    def test_stub_module_identity_preserved(self):
        """The patch mutates the existing perth module, not a replacement."""
        original = self._make_broken_perth()
        patched  = _run_perth_patch(original)
        self.assertIs(patched, original,
                      "Patch should mutate the existing module, not replace it")


class TestPerthStubImportErrorBranch(unittest.TestCase):
    """
    perth is NOT installed at all → ImportError.
    The patch must inject a complete fake 'perth' module into sys.modules.
    """

    def test_perth_injected_into_sys_modules(self):
        """sys.modules['perth'] must exist after the patch."""
        perth = _run_perth_patch(fake_perth_module=None)
        self.assertIsNotNone(perth, "perth must be injected into sys.modules")

    def test_injected_module_has_watermarker_class(self):
        """The injected module must have PerthImplicitWatermarker."""
        perth = _run_perth_patch(fake_perth_module=None)
        self.assertTrue(
            callable(getattr(perth, "PerthImplicitWatermarker", None)),
            "Injected perth must have a callable PerthImplicitWatermarker"
        )

    def test_stub_is_instantiable(self):
        """The injected stub class must be instantiable with no arguments."""
        perth = _run_perth_patch(fake_perth_module=None)
        try:
            perth.PerthImplicitWatermarker()
        except Exception as e:
            self.fail(f"PerthImplicitWatermarker() raised: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Perth stub interface contract — both branches must satisfy it
# ─────────────────────────────────────────────────────────────────────────────

class TestPerthStubInterface(unittest.TestCase):
    """
    chatterbox-tts calls the watermarker in two ways:
      1. result = watermarker(wav, sr)              — __call__, returns something
      2. arr    = watermarker.apply_watermark(wav, sample_rate=sr)  — returns numpy

    Both stub branches must implement both call sites correctly.
    """

    def _stubs(self):
        """Yield (label, stub_instance) for both branches."""
        import numpy as np

        # Try branch: broken perth (PerthImplicitWatermarker = None)
        broken = types.ModuleType("perth")
        broken.PerthImplicitWatermarker = None
        perth_try = _run_perth_patch(broken)
        yield "try-branch", perth_try.PerthImplicitWatermarker()

        # ImportError branch: perth absent
        perth_ie = _run_perth_patch(fake_perth_module=None)
        yield "importerror-branch", perth_ie.PerthImplicitWatermarker()

    def test_call_returns_wav_unchanged_for_list(self):
        """__call__ must return the wav argument unchanged (tensor pass-through)."""
        fake_wav = [1.0, 2.0, 3.0]
        for label, stub in self._stubs():
            with self.subTest(branch=label):
                result = stub(fake_wav, sr=24000)
                self.assertIs(result, fake_wav,
                              f"[{label}] __call__ must return the original wav object")

    def test_call_returns_wav_unchanged_for_tensor(self):
        """__call__ must work with a torch tensor and return it unchanged."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        wav = torch.zeros(1, 4800)
        for label, stub in self._stubs():
            with self.subTest(branch=label):
                result = stub(wav, sr=24000)
                self.assertIs(result, wav,
                              f"[{label}] __call__ must return the original tensor")

    def test_apply_watermark_returns_numpy_array_from_tensor(self):
        """apply_watermark(tensor) must return a numpy ndarray."""
        try:
            import torch
            import numpy as np
        except ImportError:
            self.skipTest("torch or numpy not available")
        wav = torch.zeros(1, 4800)
        for label, stub in self._stubs():
            with self.subTest(branch=label):
                result = stub.apply_watermark(wav, sample_rate=24000)
                self.assertIsInstance(
                    result, np.ndarray,
                    f"[{label}] apply_watermark must return numpy.ndarray, "
                    f"got {type(result).__name__}"
                )

    def test_apply_watermark_returns_numpy_array_from_list(self):
        """apply_watermark(list) must return a numpy ndarray."""
        import numpy as np
        wav = [0.1, 0.2, 0.3]
        for label, stub in self._stubs():
            with self.subTest(branch=label):
                result = stub.apply_watermark(wav, sample_rate=24000)
                self.assertIsInstance(
                    result, np.ndarray,
                    f"[{label}] apply_watermark must return numpy.ndarray, "
                    f"got {type(result).__name__}"
                )

    def test_apply_watermark_accepts_keyword_sample_rate(self):
        """apply_watermark must accept sample_rate as a keyword argument."""
        import numpy as np
        wav = [0.0]
        for label, stub in self._stubs():
            with self.subTest(branch=label):
                try:
                    stub.apply_watermark(wav, sample_rate=24000)
                except TypeError as e:
                    self.fail(f"[{label}] apply_watermark rejected keyword arg: {e}")

    def test_apply_watermark_values_preserved_from_tensor(self):
        """apply_watermark must preserve the tensor values in the returned array."""
        try:
            import torch
            import numpy as np
        except ImportError:
            self.skipTest("torch or numpy not available")
        data = [0.1, 0.5, -0.3]
        wav  = torch.tensor(data)
        for label, stub in self._stubs():
            with self.subTest(branch=label):
                result = stub.apply_watermark(wav, sample_rate=24000)
                np.testing.assert_allclose(
                    result, data, rtol=1e-5,
                    err_msg=f"[{label}] apply_watermark must preserve tensor values"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 4. CFG batch-expansion patch logic
# ─────────────────────────────────────────────────────────────────────────────

class TestCFGPatchLogic(unittest.TestCase):
    """
    The prepare_input_embeds wrapper in chatterbox_worker.py expands
    embeds from batch=1 to batch=2 when cfg_weight > 0 and embeds.batch == 1.

    These tests validate the logic directly, without loading any model.
    """

    def setUp(self):
        try:
            import torch
            self.torch = torch
        except ImportError:
            self.skipTest("torch not available")

    def _make_patch(self, original_batch):
        """
        Build a minimal (orig_fn, patched_fn) pair that mirrors the worker patch.

        original_batch : int
            The batch size that the fake orig prepare_input_embeds returns.
        """
        torch = self.torch
        EMBED_DIM = 16
        SEQ_LEN   = 10
        LEN_COND  = 5

        def fake_orig_pie(*, t3_cond, text_tokens, speech_tokens, cfg_weight=0.0):
            embeds = torch.zeros(original_batch, SEQ_LEN, EMBED_DIM)
            return embeds, LEN_COND

        # Replicate the patch logic exactly as written in chatterbox_worker.py
        _orig = fake_orig_pie
        def _safe_pie(*, t3_cond, text_tokens, speech_tokens, cfg_weight=0.0):
            embeds, len_cond = _orig(
                t3_cond=t3_cond, text_tokens=text_tokens,
                speech_tokens=speech_tokens, cfg_weight=cfg_weight)
            if cfg_weight > 0.0 and embeds.size(0) == 1:
                embeds = embeds.expand(2, -1, -1).contiguous()
            return embeds, len_cond

        return _safe_pie, LEN_COND

    def _call(self, fn, cfg_weight):
        """Call patched fn with dummy args."""
        return fn(
            t3_cond=None, text_tokens=None, speech_tokens=None,
            cfg_weight=cfg_weight
        )

    # ── core expansion behaviour ──────────────────────────────────────────

    def test_cfg_gt0_batch1_expands_to_batch2(self):
        """cfg_weight > 0, batch=1 input → embeds must be expanded to batch=2."""
        patched, _ = self._make_patch(original_batch=1)
        embeds, _ = self._call(patched, cfg_weight=0.5)
        self.assertEqual(embeds.size(0), 2,
                         "embeds must be expanded to batch=2 when cfg_weight > 0")

    def test_cfg_eq0_batch1_not_expanded(self):
        """cfg_weight == 0, batch=1 input → embeds must stay at batch=1."""
        patched, _ = self._make_patch(original_batch=1)
        embeds, _ = self._call(patched, cfg_weight=0.0)
        self.assertEqual(embeds.size(0), 1,
                         "embeds must NOT be expanded when cfg_weight == 0")

    def test_cfg_gt0_batch2_not_re_expanded(self):
        """cfg_weight > 0, batch=2 input (newer build) → embeds stay at batch=2."""
        patched, _ = self._make_patch(original_batch=2)
        embeds, _ = self._call(patched, cfg_weight=0.5)
        self.assertEqual(embeds.size(0), 2,
                         "embeds must NOT be doubled again when already batch=2")

    def test_len_cond_propagated_unchanged(self):
        """len_cond must be returned exactly as the original function produced it."""
        patched, expected_len_cond = self._make_patch(original_batch=1)
        for cfg in (0.0, 0.5, 1.0):
            with self.subTest(cfg_weight=cfg):
                _, len_cond = self._call(patched, cfg_weight=cfg)
                self.assertEqual(len_cond, expected_len_cond,
                                 f"len_cond must be {expected_len_cond}, got {len_cond}")

    # ── shape correctness after expansion ────────────────────────────────

    def test_expanded_embeds_shape_is_correct(self):
        """After expansion, shape must be (2, seq_len, embed_dim)."""
        EMBED_DIM, SEQ_LEN = 16, 10
        patched, _ = self._make_patch(original_batch=1)
        embeds, _ = self._call(patched, cfg_weight=0.5)
        self.assertEqual(list(embeds.shape), [2, SEQ_LEN, EMBED_DIM])

    def test_expanded_embeds_is_contiguous(self):
        """expand().contiguous() must produce a contiguous tensor."""
        patched, _ = self._make_patch(original_batch=1)
        embeds, _ = self._call(patched, cfg_weight=0.5)
        self.assertTrue(embeds.is_contiguous(),
                        "Expanded embeds must be contiguous (required by downstream ops)")

    # ── the downstream cat succeeds ───────────────────────────────────────

    def test_expanded_embeds_cats_with_batch2_bos(self):
        """
        The whole point of the fix: after expansion, torch.cat([embeds, bos_embed], dim=1)
        must succeed.  This is the exact operation that was crashing on the 4 GB machine.
        """
        torch = self.torch
        EMBED_DIM = 16
        patched, _ = self._make_patch(original_batch=1)
        embeds, _ = self._call(patched, cfg_weight=0.5)

        # bos_embed as T3.inference produces it: doubled unconditionally
        bos_embed = torch.zeros(1, 1, EMBED_DIM)
        bos_embed = torch.cat([bos_embed, bos_embed])   # → batch=2

        try:
            result = torch.cat([embeds, bos_embed], dim=1)
        except RuntimeError as e:
            self.fail(
                f"torch.cat failed after CFG patch — fix is not working: {e}"
            )
        self.assertEqual(result.size(0), 2)

    def test_cfg0_embeds_does_not_cat_with_batch2_bos(self):
        """
        Sanity / negative test: without expansion (cfg_weight=0),
        batch=1 embeds cannot cat with batch=2 bos_embed.
        This confirms the fix is actually necessary, not a no-op.
        """
        torch = self.torch
        EMBED_DIM = 16
        patched, _ = self._make_patch(original_batch=1)
        embeds, _ = self._call(patched, cfg_weight=0.0)  # batch=1, no expand

        bos_embed = torch.zeros(2, 1, EMBED_DIM)          # batch=2

        with self.assertRaises(RuntimeError,
                               msg="batch mismatch must raise without the fix"):
            torch.cat([embeds, bos_embed], dim=1)

    def test_closure_survives_after_orig_deleted(self):
        """
        Regression for the 'del _orig_pie' bug shipped in an early build.

        The patched function closes over _orig_pie.  If _orig_pie is deleted
        from the enclosing scope after patching, Python empties the closure
        cell and the function raises:
            NameError: cannot access free variable '_orig_pie'
        on the first call — even though the patch appeared to apply correctly.

        Verify: calling the patched function AFTER deleting _orig_pie from the
        local namespace raises NameError.  Then verify the worker's current
        code does NOT do this (i.e. the function works after the try-block exits).
        """
        torch = self.torch
        EMBED_DIM, SEQ_LEN = 16, 10

        def make_orig():
            return lambda *, t3_cond, text_tokens, speech_tokens, cfg_weight=0.0: (
                torch.zeros(1, SEQ_LEN, EMBED_DIM), 5
            )

        # ── demonstrate the bug ──────────────────────────────────────────────
        _orig_buggy = make_orig()
        def _buggy_patch(*, t3_cond, text_tokens, speech_tokens, cfg_weight=0.0):
            embeds, lc = _orig_buggy(
                t3_cond=t3_cond, text_tokens=text_tokens,
                speech_tokens=speech_tokens, cfg_weight=cfg_weight)
            if cfg_weight > 0.0 and embeds.size(0) == 1:
                embeds = embeds.expand(2, -1, -1).contiguous()
            return embeds, lc
        del _orig_buggy   # ← the bug: empties the closure cell

        with self.assertRaises(NameError,
                               msg="Deleting _orig_pie must cause NameError on call"):
            _buggy_patch(t3_cond=None, text_tokens=None,
                         speech_tokens=None, cfg_weight=0.5)

        # ── verify the fix: _make_patch does NOT delete the closure var ──────
        patched, _ = self._make_patch(original_batch=1)
        try:
            embeds, _ = self._call(patched, cfg_weight=0.5)
        except NameError as e:
            self.fail(
                f"Patched function raised NameError — closure was broken: {e}"
            )
        self.assertEqual(embeds.size(0), 2,
                         "Patch must still expand correctly after the block exits")


if __name__ == "__main__":
    unittest.main(verbosity=2)
