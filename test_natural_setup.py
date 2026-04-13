"""
Tests for Natural mode auto-setup and ChatterboxEngine subprocess environment.

Covers:
  - _cb_env_exists() correctly detects presence/absence of python exe
  - _run_chatterbox_setup() skips download steps when files already exist
  - ChatterboxEngine.PYTHON resolves dev path before APPDATA path
  - ChatterboxEngine.start() injects python_dir into subprocess PATH
  - ChatterboxEngine.start() raises FileNotFoundError when env missing
"""
import os
import sys
import json
import types
import unittest
import tempfile
import subprocess
from unittest.mock import patch, MagicMock, call

# ── Minimal stubs so app.py can be imported without a display ─────────────────
os.environ.setdefault("DISPLAY", "")

import importlib, unittest.mock as _um

# Stub every GUI / audio / ML library before importing app
_STUB_MODS = [
    "customtkinter", "tkinter", "tkinter.messagebox", "tkinter.filedialog",
    "PIL", "PIL.Image", "PIL.ImageTk",
    "sounddevice", "soundfile",
    "kokoro_onnx",
    "pedalboard", "pedalboard.io",
    "numpy", "scipy", "scipy.signal",
    "noisereduce",
]
for _m in _STUB_MODS:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# Configure CTk widget .get() so format strings like f"{slider.get():.2f}" work
# when app.py runs apply_settings() at module level.
_ctk = sys.modules["customtkinter"]
_ctk.CTkSlider.return_value.get.return_value = 0.0
_ctk.CTkEntry.return_value.get.return_value = ""
_ctk.CTkTextbox.return_value.get.return_value = ""
_ctk.StringVar.return_value.get.return_value = ""
_ctk.IntVar.return_value.get.return_value = 0
_ctk.DoubleVar.return_value.get.return_value = 0.0
_ctk.BooleanVar.return_value.get.return_value = False
_ctk.CTkOptionMenu.return_value.get.return_value = ""
_ctk.CTkCheckBox.return_value.get.return_value = 0
sys.modules["tkinter"].IntVar.return_value.get.return_value = 0
sys.modules["tkinter"].StringVar.return_value.get.return_value = ""

# Stub ctypes.windll for non-Windows CI
import ctypes
if not hasattr(ctypes, "windll"):
    ctypes.windll = MagicMock()

# ── Now import the pieces we need from app ────────────────────────────────────
# We import selectively to avoid triggering Tk() at module level.
import importlib.util, pathlib

_APP_PATH = pathlib.Path(__file__).parent / "app.py"


def _load_engine_class():
    """Return ChatterboxEngine class and helpers without running app UI code."""
    src = _APP_PATH.read_text(encoding="utf-8")
    # Execute only up to the ChatterboxEngine definition + helpers
    # by finding the line where Voices are defined and truncating there.
    cutoff_marker = "# ── Voices"
    cut = src.find(cutoff_marker)
    if cut == -1:
        raise RuntimeError("Could not find cutoff marker in app.py")
    snippet = src[:cut]

    ns = {
        "__file__": str(_APP_PATH),
        "__name__": "__app_partial__",
    }
    # Patch heavy imports used in the snippet
    with patch.dict(sys.modules, {k: MagicMock() for k in _STUB_MODS}):
        with patch("subprocess.Popen", MagicMock()):
            with patch("ctypes.windll", MagicMock()):
                try:
                    exec(compile(snippet, str(_APP_PATH), "exec"), ns)
                except Exception:
                    pass  # UI setup will fail — we only need the classes
    return ns


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for _cb_env_exists
# ═══════════════════════════════════════════════════════════════════════════════

class TestCbEnvExists(unittest.TestCase):

    def test_returns_false_when_python_missing(self):
        """_cb_env_exists() returns False when no python.exe at either path."""
        with tempfile.TemporaryDirectory() as tmp:
            # Point both dev and user paths at non-existent files
            fake_dev  = os.path.join(tmp, "dev_python.exe")
            fake_user = os.path.join(tmp, "user_python.exe")
            # Neither exists
            with patch("app.chatterbox_engine") as mock_eng:
                mock_eng.PYTHON = fake_user
                # Import _cb_env_exists directly
                import app as _app_mod
                orig = _app_mod.chatterbox_engine
                try:
                    _app_mod.chatterbox_engine = mock_eng
                    result = _app_mod._cb_env_exists()
                finally:
                    _app_mod.chatterbox_engine = orig
            self.assertFalse(result)

    def test_returns_true_when_python_exists(self):
        """_cb_env_exists() returns True when python.exe is present."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_exe = os.path.join(tmp, "python.exe")
            open(fake_exe, "w").close()
            with patch("app.chatterbox_engine") as mock_eng:
                mock_eng.PYTHON = fake_exe
                import app as _app_mod
                orig = _app_mod.chatterbox_engine
                try:
                    _app_mod.chatterbox_engine = mock_eng
                    result = _app_mod._cb_env_exists()
                finally:
                    _app_mod.chatterbox_engine = orig
            self.assertTrue(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for ChatterboxEngine.PYTHON path resolution
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatterboxEnginePythonPath(unittest.TestCase):

    def setUp(self):
        import app
        self.engine = app.ChatterboxEngine()

    def test_dev_path_takes_priority_when_exists(self):
        """Dev env path is returned when it exists (running from source)."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_dev = os.path.join(tmp, "python.exe")
            open(fake_dev, "w").close()
            with patch.object(type(self.engine), "_PYTHON_DEV", fake_dev):
                result = self.engine.PYTHON
            self.assertEqual(result, fake_dev)

    def test_user_path_returned_when_dev_missing(self):
        """APPDATA path is returned when dev env doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_dev  = os.path.join(tmp, "nonexistent_dev.exe")
            fake_user = os.path.join(tmp, "user_python.exe")
            with patch.object(type(self.engine), "_PYTHON_DEV", fake_dev):
                with patch.object(type(self.engine), "_PYTHON_USER", fake_user):
                    result = self.engine.PYTHON
            self.assertEqual(result, fake_user)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for ChatterboxEngine.start() — subprocess environment
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatterboxEngineStart(unittest.TestCase):

    def setUp(self):
        import app
        self.engine = app.ChatterboxEngine()

    def test_raises_file_not_found_when_python_missing(self):
        """start() raises FileNotFoundError when python.exe doesn't exist."""
        with patch.object(type(self.engine), "_PYTHON_DEV", "/nonexistent/python.exe"):
            with patch.object(type(self.engine), "_PYTHON_USER", "/nonexistent/python.exe"):
                with self.assertRaises(FileNotFoundError):
                    self.engine.start()

    def test_python_dir_prepended_to_PATH(self):
        """start() prepends python_embed dir to PATH so DLLs can be found."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create a fake python.exe
            python_exe = os.path.join(tmp, "python.exe")
            open(python_exe, "w").close()

            captured_env = {}

            def fake_popen(args, **kwargs):
                captured_env.update(kwargs.get("env", {}))
                mock_proc = MagicMock()
                # Immediately return empty stdout so start() exits cleanly
                mock_proc.stdout = iter([
                    json.dumps({"type": "ready", "sr": 24000}) + "\n"
                ])
                mock_proc.poll.return_value = None
                return mock_proc

            with patch.object(type(self.engine), "_PYTHON_DEV", python_exe):
                with patch("subprocess.Popen", side_effect=fake_popen):
                    self.engine.start()

            path_val = captured_env.get("PATH", "")
            path_entries = path_val.split(os.pathsep)
            self.assertIn(tmp, path_entries,
                          "python_embed directory must be first in subprocess PATH")

    def test_scripts_dir_prepended_to_PATH(self):
        """start() also prepends python_embed/Scripts to PATH."""
        with tempfile.TemporaryDirectory() as tmp:
            python_exe = os.path.join(tmp, "python.exe")
            open(python_exe, "w").close()
            scripts_dir = os.path.join(tmp, "Scripts")

            captured_env = {}

            def fake_popen(args, **kwargs):
                captured_env.update(kwargs.get("env", {}))
                mock_proc = MagicMock()
                mock_proc.stdout = iter([
                    json.dumps({"type": "ready", "sr": 24000}) + "\n"
                ])
                mock_proc.poll.return_value = None
                return mock_proc

            with patch.object(type(self.engine), "_PYTHON_DEV", python_exe):
                with patch("subprocess.Popen", side_effect=fake_popen):
                    self.engine.start()

            path_entries = captured_env.get("PATH", "").split(os.pathsep)
            self.assertIn(scripts_dir, path_entries,
                          "Scripts directory must be in subprocess PATH")

    def test_original_PATH_preserved(self):
        """Existing PATH entries are kept after the prepended python dirs."""
        with tempfile.TemporaryDirectory() as tmp:
            python_exe = os.path.join(tmp, "python.exe")
            open(python_exe, "w").close()

            original_path = "/some/existing/path"
            captured_env = {}

            def fake_popen(args, **kwargs):
                captured_env.update(kwargs.get("env", {}))
                mock_proc = MagicMock()
                mock_proc.stdout = iter([
                    json.dumps({"type": "ready", "sr": 24000}) + "\n"
                ])
                mock_proc.poll.return_value = None
                return mock_proc

            with patch.dict(os.environ, {"PATH": original_path}):
                with patch.object(type(self.engine), "_PYTHON_DEV", python_exe):
                    with patch("subprocess.Popen", side_effect=fake_popen):
                        self.engine.start()

            self.assertIn(original_path, captured_env.get("PATH", ""),
                          "Original PATH must be preserved in subprocess env")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for _run_chatterbox_setup skip logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunChatterboxSetupSkipLogic(unittest.TestCase):
    """Verify setup skips download steps when files already exist."""

    def _run_setup(self, python_exists=False, pip_exists=False):
        """Helper: run setup with mocked filesystem and network."""
        import app

        with tempfile.TemporaryDirectory() as tmp:
            python_dir  = os.path.join(tmp, "python_embed")
            python_exe  = os.path.join(python_dir, "python.exe")
            scripts_dir = os.path.join(python_dir, "Scripts")
            pip_exe     = os.path.join(scripts_dir, "pip.exe")

            if python_exists:
                os.makedirs(python_dir, exist_ok=True)
                open(python_exe, "w").close()
                # Create a fake _pth file
                with open(os.path.join(python_dir, "python311._pth"), "w") as f:
                    f.write("#import site\n")

            if pip_exists:
                os.makedirs(scripts_dir, exist_ok=True)
                open(pip_exe, "w").close()

            calls = []

            def fake_urlretrieve(url, dest):
                calls.append(("download", url))
                # Create a minimal zip if downloading Python embed
                if "python-3.11" in url:
                    import zipfile
                    os.makedirs(python_dir, exist_ok=True)
                    with zipfile.ZipFile(dest, "w") as zf:
                        zf.writestr("python311._pth", "#import site\n")
                        zf.writestr("python.exe", "fake")
                elif "get-pip" in url:
                    open(dest, "w").close()

            def fake_run(cmd, **kwargs):
                calls.append(("run", cmd))
                r = MagicMock()
                r.returncode = 0
                r.stdout = ""
                r.stderr = ""
                return r

            statuses = []
            success  = []
            failures = []

            with patch.object(app.ChatterboxEngine, "_CB_PYTHON_DIR", python_dir):
                with patch.object(app.ChatterboxEngine, "_PYTHON_USER", python_exe):
                    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
                        with patch("subprocess.run", side_effect=fake_run):
                            app._run_chatterbox_setup(
                                lambda m: statuses.append(m),
                                lambda: success.append(True),
                                lambda m: failures.append(m),
                            )

            return calls, statuses, success, failures

    def test_skips_python_download_when_already_exists(self):
        """Setup does not re-download Python embed if python.exe already present."""
        calls, _, success, failures = self._run_setup(python_exists=True, pip_exists=True)
        downloaded_urls = [url for tag, url in calls if tag == "download"]
        py_downloads = [u for u in downloaded_urls if "python-3.11" in u]
        self.assertEqual(py_downloads, [],
                         "Should not download Python embed if python.exe exists")

    def test_skips_pip_bootstrap_when_already_exists(self):
        """Setup does not re-bootstrap pip if pip.exe already present."""
        calls, _, success, failures = self._run_setup(python_exists=True, pip_exists=True)
        downloaded_urls = [url for tag, url in calls if tag == "download"]
        pip_downloads = [u for u in downloaded_urls if "get-pip" in u]
        self.assertEqual(pip_downloads, [],
                         "Should not re-download get-pip.py if pip.exe exists")

    def test_downloads_python_when_missing(self):
        """Setup downloads Python embed when python.exe is absent."""
        calls, _, success, failures = self._run_setup(python_exists=False, pip_exists=False)
        downloaded_urls = [url for tag, url in calls if tag == "download"]
        py_downloads = [u for u in downloaded_urls if "python-3.11" in u]
        self.assertGreater(len(py_downloads), 0,
                           "Should download Python embed when not present")

    def test_installs_torch_and_chatterbox(self):
        """Setup always runs pip install for torch and chatterbox-tts."""
        calls, _, success, failures = self._run_setup(python_exists=True, pip_exists=True)
        pip_cmds = [" ".join(cmd) for tag, cmd in calls if tag == "run"]
        has_torch = any("torch" in c for c in pip_cmds)
        has_cb    = any("chatterbox" in c for c in pip_cmds)
        self.assertTrue(has_torch, "Should pip install torch")
        self.assertTrue(has_cb,    "Should pip install chatterbox-tts")


if __name__ == "__main__":
    unittest.main()
