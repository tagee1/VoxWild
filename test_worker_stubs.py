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

  4. CFG floor (E007 fix)
       The PyPI build of chatterbox-tts==0.1.7 unconditionally doubles
       bos_embed inside T3.inference, so cfg_weight=0.0 (which tells tts.py
       NOT to double text_tokens) crashes with a tensor mismatch.  The worker
       floors cfg_weight to a tiny non-zero value before calling model.generate.
       Tests verify the floor is present in the worker source.

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
# 4. CFG floor (E007 fix)
# ─────────────────────────────────────────────────────────────────────────────

class TestCFGFloor(unittest.TestCase):
    """
    The PyPI build of chatterbox-tts==0.1.7 unconditionally doubles bos_embed
    inside T3.inference (no `if cfg_weight > 0.0:` guard).  This means
    cfg_weight=0.0 — which tells tts.py NOT to double text_tokens — crashes
    with `RuntimeError: Sizes of tensors must match` on the cat of
    embeds(batch=1) with bos_embed(batch=2).

    Reproduced locally Apr 11 2026 by installing chatterbox-tts==0.1.7 into a
    fresh embeddable Python and running test_worker_local.py.

    Fix: chatterbox_worker.py floors cfg_weight to a tiny non-zero value before
    calling model.generate.  tts.py then doubles text_tokens normally and the
    CFG fusion `cond + 0.001*(cond - uncond)` is perceptually identical to
    truly disabling CFG.
    """

    # The actual generate call line is `wav = model.generate(`, distinct from
    # any in-comment mentions of model.generate() elsewhere in the worker.
    _GEN_CALL = "wav = model.generate("

    def test_floor_present_in_worker_source(self):
        """The worker must contain a cfg_weight floor before model.generate."""
        src = _WORKER_PATH.read_text(encoding="utf-8")
        self.assertIn("_safe_cfg", src,
                      "chatterbox_worker.py must define _safe_cfg as a floored cfg_weight")
        self.assertIn(self._GEN_CALL, src,
                      "worker must call wav = model.generate(...)")

        gen_idx = src.index(self._GEN_CALL)
        before_gen = src[:gen_idx]
        self.assertIn("_safe_cfg", before_gen,
                      "_safe_cfg must be defined BEFORE model.generate is called")
        self.assertRegex(before_gen, r"_safe_cfg\s*=.*0\.001",
                         "_safe_cfg must floor at 0.001")

    def test_generate_uses_floored_cfg(self):
        """model.generate must be called with cfg_weight=_safe_cfg, not raw cfg_weight."""
        src = _WORKER_PATH.read_text(encoding="utf-8")
        gen_idx = src.index(self._GEN_CALL)
        gen_block = src[gen_idx:gen_idx + 400]
        self.assertRegex(
            gen_block,
            r"cfg_weight\s*=\s*_safe_cfg",
            "model.generate must receive cfg_weight=_safe_cfg (the floored value)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
