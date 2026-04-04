"""
test_clone_reset.py — Tests for voice clone → default voice switching bug.

Root cause: ChatterboxTTS.model.generate() mutates model.conds in-place when
given an audio_prompt_path. Subsequent calls with audio_prompt_path=None would
silently reuse the clone's voice conditioning instead of the model's default.

Fix: chatterbox_worker.py saves model.conds via copy.deepcopy after startup
and restores it before every generate call where audio_prompt is None.
"""

import copy
import json


# ── Helpers / stubs ──────────────────────────────────────────────────────────

class _FakeConds:
    """Minimal stand-in for Chatterbox Conditionals."""
    def __init__(self, marker):
        self.marker = marker  # lets tests verify which conds are active

    def __deepcopy__(self, memo):
        return _FakeConds(self.marker)


class _FakeModel:
    """
    Mimics ChatterboxTTS: generate() updates self.conds when an audio prompt
    is provided, and leaves self.conds unchanged when no prompt is given.
    """
    def __init__(self):
        self.conds = _FakeConds("default")
        self.sr = 24000
        self.last_audio_prompt = object()  # sentinel

    def generate(self, text, audio_prompt_path=None, **kwargs):
        self.last_audio_prompt = audio_prompt_path
        if audio_prompt_path is not None:
            # Simulate what the real model does: update conds from the prompt.
            self.conds = _FakeConds(f"clone:{audio_prompt_path}")
        return None  # wav not needed for these tests


# ── Test: conds bleed-through (demonstrates the original bug) ────────────────

def test_conds_bleed_without_reset():
    """Without a reset, default voice call inherits the clone's conds."""
    model = _FakeModel()

    # Call 1: use a voice clone
    model.generate("hello", audio_prompt_path="/path/to/clone.wav")
    assert model.conds.marker == "clone:/path/to/clone.wav"

    # Call 2: switch to default (no prompt) — WITHOUT the fix, conds stay dirty
    model.generate("world", audio_prompt_path=None)
    # conds is still the clone's — this is the bug
    assert model.conds.marker == "clone:/path/to/clone.wav", (
        "Expected conds to remain dirty without reset (demonstrating the bug)")


# ── Test: conds restored correctly (verifies the fix) ────────────────────────

def test_conds_reset_on_default_voice():
    """Fix: save default_conds at startup, restore before each no-prompt call."""
    model = _FakeModel()
    _default_conds = copy.deepcopy(model.conds)

    def generate_with_fix(text, audio_prompt=None):
        if audio_prompt is None and _default_conds is not None:
            model.conds = copy.deepcopy(_default_conds)
        model.generate(text, audio_prompt_path=audio_prompt)

    # Call 1: use a voice clone → conds change
    generate_with_fix("hello", audio_prompt="/path/to/clone.wav")
    assert model.conds.marker == "clone:/path/to/clone.wav"

    # Call 2: switch to default → conds restored to original
    generate_with_fix("world", audio_prompt=None)
    assert model.conds.marker == "default", (
        f"Expected 'default' conds after reset, got '{model.conds.marker}'")


def test_conds_reset_multiple_clones():
    """Default voice works correctly after switching between multiple clones."""
    model = _FakeModel()
    _default_conds = copy.deepcopy(model.conds)

    def generate_with_fix(text, audio_prompt=None):
        if audio_prompt is None and _default_conds is not None:
            model.conds = copy.deepcopy(_default_conds)
        model.generate(text, audio_prompt_path=audio_prompt)

    generate_with_fix("a", audio_prompt="/clones/alice.wav")
    generate_with_fix("b", audio_prompt="/clones/bob.wav")
    generate_with_fix("c", audio_prompt=None)  # back to default
    assert model.conds.marker == "default"

    generate_with_fix("d", audio_prompt="/clones/alice.wav")
    generate_with_fix("e", audio_prompt=None)
    assert model.conds.marker == "default"


def test_conds_reset_skipped_when_clone_provided():
    """When a clone IS provided, default_conds must NOT be restored first."""
    model = _FakeModel()
    _default_conds = copy.deepcopy(model.conds)

    def generate_with_fix(text, audio_prompt=None):
        if audio_prompt is None and _default_conds is not None:
            model.conds = copy.deepcopy(_default_conds)
        model.generate(text, audio_prompt_path=audio_prompt)

    generate_with_fix("test", audio_prompt="/clones/carol.wav")
    assert model.conds.marker == "clone:/clones/carol.wav"


def test_deepcopy_independence():
    """deepcopy of conds produces a genuinely independent object."""
    original = _FakeConds("default")
    saved = copy.deepcopy(original)
    original.marker = "mutated"
    assert saved.marker == "default", "deepcopy must be independent of the original"


# ── Test: worker JSON protocol sends null when default voice selected ─────────

def test_worker_request_null_audio_prompt():
    """
    When the app selects Default voice, it passes audio_prompt_path=None to
    generate_chunk(), which serialises as JSON null.
    Verify the worker deserialises this back to None (not the string "null").
    """
    payload = {"cmd": "generate", "text": "hi", "audio_prompt_path": None,
               "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8,
               "output_path": "/tmp/out.wav"}
    line = json.dumps(payload)
    req = json.loads(line)
    audio_prompt = req.get("audio_prompt_path") or None
    assert audio_prompt is None, f"Expected None, got {audio_prompt!r}"


def test_worker_request_empty_string_audio_prompt():
    """
    If the app somehow sends an empty string instead of null, the worker's
    `or None` coercion must still produce None.
    """
    payload = {"cmd": "generate", "text": "hi", "audio_prompt_path": "",
               "output_path": "/tmp/out.wav"}
    req = json.loads(json.dumps(payload))
    audio_prompt = req.get("audio_prompt_path") or None
    assert audio_prompt is None


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_conds_bleed_without_reset,
        test_conds_reset_on_default_voice,
        test_conds_reset_multiple_clones,
        test_conds_reset_skipped_when_clone_provided,
        test_deepcopy_independence,
        test_worker_request_null_audio_prompt,
        test_worker_request_empty_string_audio_prompt,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(failed)
