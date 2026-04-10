"""
test_smoke.py — Import and integration smoke tests.

Verifies that all supporting modules load without crashing, constants are
well-formed, and key pure-logic paths work end-to-end. Does NOT launch the
CTk main window or require a GPU/TTS model.
"""
import importlib
import os
import re
import sys
import unittest


# ══════════════════════════════════════════════════════════════════════════════
# Supporting module imports
# ══════════════════════════════════════════════════════════════════════════════
class TestModuleImports(unittest.TestCase):

    def test_tts_utils_imports(self):
        import tts_utils
        self.assertTrue(hasattr(tts_utils, "chunk_text"))
        self.assertTrue(hasattr(tts_utils, "fmt_err"))
        self.assertTrue(hasattr(tts_utils, "history_card_preview"))
        self.assertTrue(hasattr(tts_utils, "history_card_voice_label"))

    def test_settings_window_imports(self):
        import settings_window
        self.assertTrue(hasattr(settings_window, "load_settings"))
        self.assertTrue(hasattr(settings_window, "save_settings"))
        self.assertTrue(hasattr(settings_window, "DEFAULT_SETTINGS"))

    def test_pronunciation_imports(self):
        import pronunciation
        self.assertTrue(hasattr(pronunciation, "load_dictionary"))
        self.assertTrue(hasattr(pronunciation, "save_dictionary"))
        self.assertTrue(hasattr(pronunciation, "apply_pronunciation"))

    def test_clone_library_imports(self):
        import clone_library
        self.assertTrue(hasattr(clone_library, "load_clone_library"))
        self.assertTrue(hasattr(clone_library, "save_clone_library"))

    def test_license_imports(self):
        import license
        self.assertTrue(hasattr(license, "load_license"))
        self.assertTrue(hasattr(license, "save_license"))
        self.assertTrue(hasattr(license, "check_startup"))
        self.assertTrue(hasattr(license, "activate_license"))


# ══════════════════════════════════════════════════════════════════════════════
# Constants and configuration
# ══════════════════════════════════════════════════════════════════════════════
class TestConstants(unittest.TestCase):

    def setUp(self):
        # Extract constants directly from app.py source without importing the
        # whole module (which would start CTk and require a display).
        self._src = open(os.path.join(os.path.dirname(__file__), "app.py"),
                         encoding="utf-8").read()

    def test_version_is_semver(self):
        m = re.search(r'^VERSION\s*=\s*["\'](\d+\.\d+\.\d+)["\']', self._src, re.MULTILINE)
        self.assertIsNotNone(m, "VERSION constant not found or not semver")

    def test_max_history_is_positive_int(self):
        m = re.search(r'^MAX_HISTORY\s*=\s*(\d+)', self._src, re.MULTILINE)
        self.assertIsNotNone(m, "MAX_HISTORY constant not found")
        self.assertGreater(int(m.group(1)), 0)

    def test_history_paths_under_appdata(self):
        """HISTORY_JSON and HISTORY_AUDIO must use _USER_DIR, not a relative path."""
        for const in ("HISTORY_JSON", "HISTORY_AUDIO", "PROFILES_FILE", "CALIBRATION_FILE"):
            idx = self._src.find(const)
            self.assertGreater(idx, 0, f"{const} not found in app.py")
            snippet = self._src[idx: idx + 120]
            self.assertIn("_USER_DIR", snippet,
                          f"{const} does not reference _USER_DIR:\n{snippet}")

    def test_github_repo_format(self):
        m = re.search(r'^GITHUB_REPO\s*=\s*["\']([^"\']+)["\']', self._src, re.MULTILINE)
        self.assertIsNotNone(m)
        self.assertIn("/", m.group(1), "GITHUB_REPO should be 'owner/repo'")


# ══════════════════════════════════════════════════════════════════════════════
# Settings defaults sanity check
# ══════════════════════════════════════════════════════════════════════════════
class TestSettingsDefaults(unittest.TestCase):

    def test_all_required_keys_present(self):
        from settings_window import DEFAULT_SETTINGS
        required = {
            "default_output_folder", "default_voice", "default_speed",
            "theme", "notify_on_completion", "notify_threshold_seconds",
            "auto_clean_text", "default_profile",
        }
        missing = required - DEFAULT_SETTINGS.keys()
        self.assertEqual(missing, set(), f"Missing settings keys: {missing}")

    def test_default_speed_in_range(self):
        from settings_window import DEFAULT_SETTINGS
        speed = DEFAULT_SETTINGS["default_speed"]
        self.assertGreaterEqual(speed, 0.5)
        self.assertLessEqual(speed, 2.0)

    def test_notify_threshold_positive(self):
        from settings_window import DEFAULT_SETTINGS
        self.assertGreater(DEFAULT_SETTINGS["notify_threshold_seconds"], 0)

    def test_load_settings_returns_dict_with_all_defaults(self):
        from settings_window import load_settings, DEFAULT_SETTINGS
        # load_settings merges defaults into whatever is on disk, so all keys
        # must be present regardless of file state.
        settings = load_settings()
        for key in DEFAULT_SETTINGS:
            self.assertIn(key, settings, f"load_settings() missing key: {key}")


# ══════════════════════════════════════════════════════════════════════════════
# Pronunciation module — file-free behaviour
# ══════════════════════════════════════════════════════════════════════════════
class TestPronunciationSmoke(unittest.TestCase):

    def test_default_entries_are_valid(self):
        import pronunciation
        entries = pronunciation._default_entries()
        self.assertIsInstance(entries, list)
        self.assertGreater(len(entries), 0)
        for e in entries:
            self.assertIn("from", e)
            self.assertIn("to", e)
            self.assertIn("case_sensitive", e)

    def test_apply_with_defaults_does_not_crash(self):
        import pronunciation
        pronunciation._dict_cache = None
        # Point to a non-existent file so it falls back to defaults
        orig = pronunciation.PRONUNCIATION_FILE
        pronunciation.PRONUNCIATION_FILE = "__no_such_file__.json"
        try:
            result = pronunciation.apply_pronunciation(
                "The AI API and URL and SQL and CLI are useful.")
            self.assertIsInstance(result, str)
        finally:
            pronunciation.PRONUNCIATION_FILE = orig
            pronunciation._dict_cache = None


# ══════════════════════════════════════════════════════════════════════════════
# License module — structure checks
# ══════════════════════════════════════════════════════════════════════════════
class TestLicenseSmoke(unittest.TestCase):

    def test_load_license_returns_dict(self):
        import license as lic
        data = lic.load_license(path="__no_such_license__.json")
        self.assertIsInstance(data, dict)

    def test_load_license_has_required_keys(self):
        import license as lic
        data = lic.load_license(path="__no_such_license__.json")
        for key in ("key", "activated", "activation_date", "launch_count"):
            self.assertIn(key, data, f"license dict missing key: {key}")

    def test_default_state_is_not_activated(self):
        import license as lic
        data = lic.load_license(path="__no_such_license__.json")
        self.assertFalse(data["activated"], "Default license should not be activated")


# ══════════════════════════════════════════════════════════════════════════════
# tts_utils end-to-end pipeline (no audio)
# ══════════════════════════════════════════════════════════════════════════════
class TestTtsUtilsPipeline(unittest.TestCase):

    def test_chunk_then_pronunciation_pipeline(self):
        """chunk_text output feeds cleanly into apply_pronunciation."""
        from tts_utils import chunk_text
        from pronunciation import apply_pronunciation
        import pronunciation
        pronunciation._dict_cache = None
        text = ("The AI researcher called the API endpoint. " * 30 +
                "Then the CLI tool ran the SQL query.")
        chunks = chunk_text(text)
        for chunk in chunks:
            result = apply_pronunciation(chunk)
            self.assertIsInstance(result, str)
            self.assertNotIn(" API ", result)
            self.assertNotIn(" AI ", result)

    def test_fmt_err_always_returns_string(self):
        from tts_utils import fmt_err
        cases = [
            Exception("something went wrong"),
            MemoryError("out of memory"),
            PermissionError("access denied"),
            ConnectionError("network unreachable"),
            FileNotFoundError("no such file"),
        ]
        for exc in cases:
            result = fmt_err(exc)
            self.assertIsInstance(result, str, f"fmt_err returned non-str for {exc!r}")
            self.assertGreater(len(result), 0)

    def test_history_card_preview_pipeline(self):
        from tts_utils import history_card_preview
        long_text = "The quick brown fox jumped over the lazy dog. " * 10
        preview = history_card_preview(long_text)
        self.assertLessEqual(len(preview), 101)   # 100 chars + "…"
        self.assertTrue(preview.endswith("…"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
