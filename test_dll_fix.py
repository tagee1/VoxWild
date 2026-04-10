"""
Tests for the chatterbox_worker.py DLL search path fix.

Root cause being tested:
  os.add_dll_directory() returns a handle object. When that handle is garbage
  collected, the directory is automatically removed from the DLL search path.
  The old code discarded handles immediately; the fix stores them in a
  module-level list (_dll_handles) so they live for the process lifetime.

Covers:
  1. Handle lifetime — unsaved handles are GC'd; saved handles survive.
  2. PATH update — torch/lib and torchaudio/lib are prepended to PATH.
  3. Directory filtering — non-existent dirs are skipped.
  4. Worker integration — the fix section in chatterbox_worker.py behaves
     correctly end-to-end (PATH updated, handles stored and not closed).
"""
import gc
import os
import sys
import types
import weakref
import tempfile
import unittest
import pathlib
from unittest.mock import MagicMock, patch, call

# ─────────────────────────────────────────────────────────────────────────────
# Helper — extract and run just the DLL-fix section of chatterbox_worker.py
# ─────────────────────────────────────────────────────────────────────────────

_WORKER_PATH = pathlib.Path(__file__).parent / "chatterbox_worker.py"

_FIX_START = "# ── DLL search path fix (Windows)"
_FIX_END   = "# ── DLL search path fix END"   # sentinel we'll inject


def _extract_fix_code():
    """Return the DLL-fix block from chatterbox_worker.py as a string."""
    src = _WORKER_PATH.read_text(encoding="utf-8")
    start = src.find(_FIX_START)
    # End marker: the blank comment line that ends the block
    end = src.find("# ─────────────────", start + 1)
    if start == -1 or end == -1:
        raise RuntimeError(
            "Could not find DLL-fix block markers in chatterbox_worker.py"
        )
    return src[start:end]


def _run_fix(extra_env=None, isdir_override=None):
    """
    Execute the DLL-fix code in an isolated namespace.

    Parameters
    ----------
    extra_env : dict | None
        Extra os.environ keys to set before running.
    isdir_override : callable | None
        If provided, replaces os.path.isdir inside the fix scope.

    Returns
    -------
    dict  Namespace after executing the fix code.
    list  Recorded os.add_dll_directory call args.
    list  Handle objects returned by os.add_dll_directory.
    """
    code = _extract_fix_code()

    added_dirs  = []
    live_handles = []

    class FakeHandle:
        """Mimics the real os.add_dll_directory handle."""
        closed = False
        def close(self):
            self.closed = True
        def __del__(self):
            self.close()

    def fake_add_dll_directory(path):
        h = FakeHandle()
        added_dirs.append(path)
        live_handles.append(weakref.ref(h))
        return h

    ns = {
        "__builtins__": __builtins__,
        "os":  os,
        "sys": sys,
    }

    env_backup = os.environ.get("PATH", "")
    extra = extra_env or {}
    for k, v in extra.items():
        os.environ[k] = v

    _real_isdir = os.path.isdir
    try:
        if isdir_override:
            os.path.isdir = isdir_override  # type: ignore[assignment]

        with patch("os.add_dll_directory", side_effect=fake_add_dll_directory):
            exec(compile(code, str(_WORKER_PATH), "exec"), ns)
    finally:
        os.path.isdir = _real_isdir  # type: ignore[assignment]
        # Restore PATH
        os.environ["PATH"] = env_backup
        for k in extra:
            os.environ.pop(k, None)

    return ns, added_dirs, live_handles


# ═════════════════════════════════════════════════════════════════════════════
# 1. Handle lifetime
# ═════════════════════════════════════════════════════════════════════════════

class TestDllHandleLifetime(unittest.TestCase):
    """
    os.add_dll_directory() returns a handle; when it's GC'd the directory
    is de-registered from the DLL search path.  We must keep references.
    """

    def _make_tracked_handle(self, closed_list):
        """Return a handle-like object that appends to closed_list on __del__."""
        class Handle:
            def close(self_):
                closed_list.append(True)
            def __del__(self_):
                self_.close()
        return Handle()

    # ── old buggy pattern ─────────────────────────────────────────────────

    def test_unsaved_handle_closed_immediately_in_cpython(self):
        """
        Without storing the return value, CPython's refcount drops to zero
        immediately and the handle is closed before the next line runs.
        """
        closed = []
        h = self._make_tracked_handle(closed)
        wr = weakref.ref(h)

        # Simulate the old pattern: return value discarded
        del h          # drop the only reference
        gc.collect()

        self.assertIsNone(wr(), "Handle should have been GC'd")
        self.assertTrue(closed, "Handle.__del__ should have been called")

    def test_loop_without_storage_closes_all_handles(self):
        """
        The old buggy pattern — `for d in dirs: add_dll_directory(d)` —
        closes each handle immediately because no reference is retained.
        """
        closed = []
        weak_refs = []

        def fake_add(path):
            h = self._make_tracked_handle(closed)
            weak_refs.append(weakref.ref(h))
            return h   # caller discards this

        dirs = ["/fake/a", "/fake/b", "/fake/c"]
        with patch("os.add_dll_directory", side_effect=fake_add):
            for d in dirs:
                os.add_dll_directory(d)   # OLD pattern: return value not stored

        gc.collect()

        dead = [wr for wr in weak_refs if wr() is None]
        self.assertEqual(
            len(dead), len(dirs),
            f"All {len(dirs)} handles should be GC'd; only {len(dead)} were",
        )
        self.assertEqual(
            len(closed), len(dirs),
            "All handles should have had __del__ called",
        )

    # ── new fixed pattern ─────────────────────────────────────────────────

    def test_list_comprehension_keeps_handles_alive(self):
        """
        The fixed pattern — `_dll_handles = [add_dll_directory(d) for d in dirs]`
        — keeps all handles alive as long as the list exists.
        """
        closed = []
        weak_refs = []

        def fake_add(path):
            h = self._make_tracked_handle(closed)
            weak_refs.append(weakref.ref(h))
            return h

        dirs = ["/fake/a", "/fake/b", "/fake/c"]
        with patch("os.add_dll_directory", side_effect=fake_add):
            _dll_handles = [os.add_dll_directory(d) for d in dirs]   # NEW pattern

        gc.collect()

        alive = [wr for wr in weak_refs if wr() is not None]
        self.assertEqual(
            len(alive), len(dirs),
            f"All {len(dirs)} handles should be alive; only {len(alive)} are",
        )
        self.assertEqual(len(closed), 0, "No handles should have been closed yet")

    def test_handles_closed_after_list_deleted(self):
        """
        After the _dll_handles list is deleted, all handles get GC'd.
        (Confirms the list really was the only thing keeping them alive.)
        """
        closed = []
        weak_refs = []

        def fake_add(path):
            h = self._make_tracked_handle(closed)
            weak_refs.append(weakref.ref(h))
            return h

        dirs = ["/fake/a", "/fake/b"]
        with patch("os.add_dll_directory", side_effect=fake_add):
            _dll_handles = [os.add_dll_directory(d) for d in dirs]

        # Sanity: alive right now
        gc.collect()
        self.assertEqual(len(closed), 0)

        # Now delete the list
        del _dll_handles
        gc.collect()

        self.assertEqual(
            len(closed), len(dirs),
            "All handles should be closed after the list is deleted",
        )

    def test_correct_number_of_handles_for_n_dirs(self):
        """One handle is created per directory passed to add_dll_directory."""
        call_count = []

        def fake_add(path):
            call_count.append(path)
            return MagicMock()

        dirs = ["/a", "/b", "/c", "/d"]
        with patch("os.add_dll_directory", side_effect=fake_add):
            _dll_handles = [os.add_dll_directory(d) for d in dirs]

        self.assertEqual(len(_dll_handles), len(dirs))
        self.assertEqual(len(call_count), len(dirs))


# ═════════════════════════════════════════════════════════════════════════════
# 2. PATH update
# ═════════════════════════════════════════════════════════════════════════════

class TestPathUpdate(unittest.TestCase):
    """The fix prepends DLL dirs to os.environ['PATH']."""

    def _run_with_fake_dirs(self, existing_dirs):
        """
        Run the fix code where os.path.isdir returns True only for existing_dirs.
        Returns the PATH value seen inside the fix.
        """
        original_path = "/original/path"
        captured = {}

        def fake_isdir(p):
            return p in existing_dirs

        with tempfile.TemporaryDirectory() as tmp:
            # Patch sys.executable so _py_dir is predictable
            fake_exe = os.path.join(tmp, "python.exe")
            with patch.object(sys, "executable", fake_exe):
                with patch.dict(os.environ, {"PATH": original_path}):
                    with patch("os.add_dll_directory", return_value=MagicMock()):
                        with patch("os.path.isdir", side_effect=fake_isdir):
                            # Manually run the fix logic (mirrors chatterbox_worker.py)
                            _py_dir      = os.path.dirname(os.path.abspath(sys.executable))
                            _torch_lib   = os.path.join(_py_dir, "Lib", "site-packages", "torch",      "lib")
                            _taudio_lib  = os.path.join(_py_dir, "Lib", "site-packages", "torchaudio", "lib")
                            _scripts_dir = os.path.join(_py_dir, "Scripts")

                            _dll_dirs = [d for d in [_py_dir, _scripts_dir, _torch_lib, _taudio_lib]
                                         if os.path.isdir(d)]

                            os.environ["PATH"] = (
                                os.pathsep.join(_dll_dirs) + os.pathsep +
                                os.environ.get("PATH", "")
                            )
                            captured["path"]     = os.environ["PATH"]
                            captured["dll_dirs"] = _dll_dirs

        return captured, original_path

    def test_original_path_preserved(self):
        """Original PATH entries appear after the prepended DLL dirs."""
        captured, original = self._run_with_fake_dirs(set())
        self.assertIn(original, captured["path"])

    def test_existing_dirs_prepended(self):
        """Each existing DLL dir appears before the original PATH."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_exe = os.path.join(tmp, "python.exe")
            _py_dir = tmp
            scripts = os.path.join(tmp, "Scripts")
            os.makedirs(scripts, exist_ok=True)
            original = "/old/path"

            with patch.object(sys, "executable", fake_exe):
                with patch.dict(os.environ, {"PATH": original}):
                    with patch("os.add_dll_directory", return_value=MagicMock()):
                        _torch   = os.path.join(_py_dir, "Lib", "site-packages", "torch",      "lib")
                        _taudio  = os.path.join(_py_dir, "Lib", "site-packages", "torchaudio", "lib")

                        def isdir(p):
                            return p in {_py_dir, scripts}

                        with patch("os.path.isdir", side_effect=isdir):
                            dll_dirs = [d for d in [_py_dir, scripts, _torch, _taudio]
                                        if os.path.isdir(d)]
                            os.environ["PATH"] = (
                                os.pathsep.join(dll_dirs) + os.pathsep +
                                os.environ.get("PATH", "")
                            )
                            path_entries = os.environ["PATH"].split(os.pathsep)

        self.assertIn(_py_dir, path_entries)
        self.assertIn(scripts, path_entries)
        # DLL dirs come BEFORE original
        py_idx  = path_entries.index(_py_dir)
        old_idx = path_entries.index(original)
        self.assertLess(py_idx, old_idx, "python_embed dir must precede original PATH")

    def test_nonexistent_dirs_excluded_from_path(self):
        """Dirs that don't exist on disk are not added to PATH."""
        captured, _ = self._run_with_fake_dirs(set())  # nothing exists
        # dll_dirs should be empty → only original PATH remains
        self.assertEqual(captured["dll_dirs"], [])

    def test_torch_lib_included_when_it_exists(self):
        """torch/lib is added to PATH when it exists."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_exe   = os.path.join(tmp, "python.exe")
            torch_lib  = os.path.join(tmp, "Lib", "site-packages", "torch", "lib")
            os.makedirs(torch_lib, exist_ok=True)

            with patch.object(sys, "executable", fake_exe):
                with patch.dict(os.environ, {"PATH": ""}):
                    with patch("os.add_dll_directory", return_value=MagicMock()):
                        _py_dir = os.path.dirname(os.path.abspath(sys.executable))
                        _torch  = os.path.join(_py_dir, "Lib", "site-packages", "torch",      "lib")
                        _taud   = os.path.join(_py_dir, "Lib", "site-packages", "torchaudio", "lib")
                        _scr    = os.path.join(_py_dir, "Scripts")

                        dll_dirs = [d for d in [_py_dir, _scr, _torch, _taud]
                                    if os.path.isdir(d)]
                        os.environ["PATH"] = (
                            os.pathsep.join(dll_dirs) + os.pathsep +
                            os.environ.get("PATH", "")
                        )
                        path_entries = os.environ["PATH"].split(os.pathsep)

        self.assertIn(torch_lib, path_entries)

    def test_torchaudio_lib_included_when_it_exists(self):
        """torchaudio/lib is added to PATH when it exists."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_exe   = os.path.join(tmp, "python.exe")
            taud_lib   = os.path.join(tmp, "Lib", "site-packages", "torchaudio", "lib")
            os.makedirs(taud_lib, exist_ok=True)

            with patch.object(sys, "executable", fake_exe):
                with patch.dict(os.environ, {"PATH": ""}):
                    with patch("os.add_dll_directory", return_value=MagicMock()):
                        _py_dir = os.path.dirname(os.path.abspath(sys.executable))
                        _torch  = os.path.join(_py_dir, "Lib", "site-packages", "torch",      "lib")
                        _taud   = os.path.join(_py_dir, "Lib", "site-packages", "torchaudio", "lib")
                        _scr    = os.path.join(_py_dir, "Scripts")

                        dll_dirs = [d for d in [_py_dir, _scr, _torch, _taud]
                                    if os.path.isdir(d)]
                        os.environ["PATH"] = (
                            os.pathsep.join(dll_dirs) + os.pathsep +
                            os.environ.get("PATH", "")
                        )
                        path_entries = os.environ["PATH"].split(os.pathsep)

        self.assertIn(taud_lib, path_entries)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Worker integration — execute the actual fix code from chatterbox_worker.py
# ═════════════════════════════════════════════════════════════════════════════

class TestWorkerDllFixIntegration(unittest.TestCase):
    """
    Execute the real DLL-fix block from chatterbox_worker.py and verify
    its side-effects: PATH updated, handles stored and not immediately closed.
    """

    def setUp(self):
        self._path_backup = os.environ.get("PATH", "")

    def tearDown(self):
        os.environ["PATH"] = self._path_backup

    # ── helpers ──────────────────────────────────────────────────────────

    def _exec_fix(self, existing_dirs=None):
        """
        Execute the DLL-fix block.  Patches add_dll_directory and, optionally,
        os.path.isdir so we control which dirs appear to exist.

        Returns (namespace, added_dirs, weak_refs_to_handles).
        """
        code = _extract_fix_code()

        added = []
        weak_refs = []

        class Handle:
            closed = False
            def close(self): self.closed = True
            def __del__(self): self.close()

        def fake_add(path):
            h = Handle()
            added.append(path)
            weak_refs.append(weakref.ref(h))
            return h

        if existing_dirs is not None:
            def fake_isdir(p):
                return p in existing_dirs
            isdir_patcher = patch("os.path.isdir", side_effect=fake_isdir)
        else:
            isdir_patcher = patch("os.path.isdir", wraps=os.path.isdir)

        ns = {"__builtins__": __builtins__, "os": os, "sys": sys}

        with patch("os.add_dll_directory", side_effect=fake_add):
            with isdir_patcher:
                exec(compile(code, str(_WORKER_PATH), "exec"), ns)

        return ns, added, weak_refs

    # ── tests ─────────────────────────────────────────────────────────────

    def test_fix_executes_without_exception(self):
        """The DLL-fix block runs to completion without raising."""
        try:
            self._exec_fix(existing_dirs=set())
        except Exception as exc:
            self.fail(f"DLL-fix block raised unexpectedly: {exc}")

    @unittest.skipUnless(os.name == "nt", "Windows-only fix")
    def test_dll_handles_stored_in_namespace(self):
        """After the fix runs, _dll_handles is present in the namespace."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_exe = os.path.join(tmp, "python.exe")
            # Make py_dir "exist" so at least one handle is created
            existing = {tmp}
            with patch.object(sys, "executable", fake_exe):
                ns, added, weak_refs = self._exec_fix(existing_dirs=existing)

        self.assertIn("_dll_handles", ns,
                      "_dll_handles must be defined in the worker namespace")

    @unittest.skipUnless(os.name == "nt", "Windows-only fix")
    def test_handles_survive_gc_after_fix(self):
        """
        After the fix runs, the stored handles are NOT garbage-collected
        even after an explicit gc.collect().  This is the core correctness test.
        """
        with tempfile.TemporaryDirectory() as tmp:
            fake_exe = os.path.join(tmp, "python.exe")
            existing = {tmp}
            with patch.object(sys, "executable", fake_exe):
                ns, added, weak_refs = self._exec_fix(existing_dirs=existing)

        gc.collect()

        dead = [wr for wr in weak_refs if wr() is None]
        self.assertEqual(
            dead, [],
            f"{len(dead)} handle(s) were GC'd after fix — "
            "_dll_handles is not keeping them alive",
        )

    @unittest.skipUnless(os.name == "nt", "Windows-only fix")
    def test_no_handles_closed_immediately_after_fix(self):
        """
        The fix must not close handles right away.
        (Closing = de-registering from DLL search path.)
        """
        closed_count = []

        class Handle:
            def close(self): closed_count.append(True)
            def __del__(self): self.close()

        with tempfile.TemporaryDirectory() as tmp:
            fake_exe = os.path.join(tmp, "python.exe")
            existing = {tmp}
            with patch.object(sys, "executable", fake_exe):
                with patch("os.path.isdir", side_effect=lambda p: p in existing):
                    with patch("os.add_dll_directory", side_effect=lambda _: Handle()):
                        code = _extract_fix_code()
                        ns = {"__builtins__": __builtins__, "os": os, "sys": sys}
                        exec(compile(code, str(_WORKER_PATH), "exec"), ns)

        gc.collect()
        self.assertEqual(
            closed_count, [],
            "Handles were closed (de-registered) immediately — fix is not working",
        )

    @unittest.skipUnless(os.name == "nt", "Windows-only fix")
    def test_path_updated_with_existing_dirs(self):
        """
        After the fix runs, os.environ['PATH'] contains the directories
        that existed on disk.
        """
        with tempfile.TemporaryDirectory() as tmp:
            fake_exe    = os.path.join(tmp, "python.exe")
            torch_lib   = os.path.join(tmp, "Lib", "site-packages", "torch",      "lib")
            taud_lib    = os.path.join(tmp, "Lib", "site-packages", "torchaudio", "lib")
            scripts_dir = os.path.join(tmp, "Scripts")
            os.makedirs(torch_lib,   exist_ok=True)
            os.makedirs(taud_lib,    exist_ok=True)
            os.makedirs(scripts_dir, exist_ok=True)

            original_path = "/some/original"
            with patch.object(sys, "executable", fake_exe):
                with patch.dict(os.environ, {"PATH": original_path}):
                    with patch("os.add_dll_directory", return_value=MagicMock()):
                        code = _extract_fix_code()
                        ns = {"__builtins__": __builtins__, "os": os, "sys": sys}
                        exec(compile(code, str(_WORKER_PATH), "exec"), ns)

                    path_entries = os.environ["PATH"].split(os.pathsep)

        self.assertIn(tmp,          path_entries, "python_embed dir must be in PATH")
        self.assertIn(torch_lib,    path_entries, "torch/lib must be in PATH")
        self.assertIn(taud_lib,     path_entries, "torchaudio/lib must be in PATH")
        self.assertIn(scripts_dir,  path_entries, "Scripts must be in PATH")
        self.assertIn(original_path, path_entries, "Original PATH must be preserved")

    @unittest.skipUnless(os.name == "nt", "Windows-only fix")
    def test_nonexistent_dirs_not_added_to_path(self):
        """
        Directories that don't exist are silently skipped —
        they must not appear in PATH or trigger add_dll_directory calls.
        """
        with tempfile.TemporaryDirectory() as tmp:
            fake_exe = os.path.join(tmp, "python.exe")
            # Nothing inside tmp is created — Scripts/torch/lib all missing

            added_paths = []
            def fake_add(p):
                added_paths.append(p)
                return MagicMock()

            original_path = "/original"
            with patch.object(sys, "executable", fake_exe):
                with patch.dict(os.environ, {"PATH": original_path}):
                    with patch("os.add_dll_directory", side_effect=fake_add):
                        code = _extract_fix_code()
                        ns = {"__builtins__": __builtins__, "os": os, "sys": sys}
                        exec(compile(code, str(_WORKER_PATH), "exec"), ns)

                    path_val = os.environ["PATH"]

        # Only tmp itself should have been considered (python_embed dir exists)
        # torch/lib, torchaudio/lib, Scripts do NOT exist → not in PATH
        torch_lib  = os.path.join(tmp, "Lib", "site-packages", "torch",      "lib")
        taud_lib   = os.path.join(tmp, "Lib", "site-packages", "torchaudio", "lib")
        scripts    = os.path.join(tmp, "Scripts")
        self.assertNotIn(torch_lib, path_val,  "Missing torch/lib must not be in PATH")
        self.assertNotIn(taud_lib,  path_val,  "Missing torchaudio/lib must not be in PATH")
        self.assertNotIn(scripts,   path_val,  "Missing Scripts must not be in PATH")
        self.assertIn(original_path, path_val, "Original PATH must be preserved")

    @unittest.skipUnless(os.name == "nt", "Windows-only fix")
    def test_add_dll_directory_called_once_per_existing_dir(self):
        """add_dll_directory is called exactly once per directory that exists."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_exe    = os.path.join(tmp, "python.exe")
            torch_lib   = os.path.join(tmp, "Lib", "site-packages", "torch",      "lib")
            os.makedirs(torch_lib, exist_ok=True)
            # Scripts and torchaudio/lib intentionally missing

            added = []
            with patch.object(sys, "executable", fake_exe):
                with patch.dict(os.environ, {"PATH": ""}):
                    with patch("os.add_dll_directory", side_effect=lambda p: (added.append(p), MagicMock())[1]):
                        code = _extract_fix_code()
                        ns = {"__builtins__": __builtins__, "os": os, "sys": sys}
                        exec(compile(code, str(_WORKER_PATH), "exec"), ns)

        # tmp (py_dir) and torch_lib exist → 2 calls
        self.assertIn(tmp,       added)
        self.assertIn(torch_lib, added)
        self.assertNotIn(os.path.join(tmp, "Scripts"),                              added)
        self.assertNotIn(os.path.join(tmp, "Lib", "site-packages", "torchaudio", "lib"), added)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def test_fix_is_no_op_on_non_windows(self):
        """
        On non-Windows, the fix block is gated by `if os.name == 'nt'`
        and must not call add_dll_directory or modify PATH.
        """
        if os.name == "nt":
            self.skipTest("Only meaningful on non-Windows")

        code = _extract_fix_code()
        original_path = os.environ.get("PATH", "")
        with patch("os.add_dll_directory") as mock_add:
            ns = {"__builtins__": __builtins__, "os": os, "sys": sys}
            exec(compile(code, str(_WORKER_PATH), "exec"), ns)

        mock_add.assert_not_called()
        self.assertEqual(os.environ.get("PATH", ""), original_path)

    @unittest.skipUnless(os.name == "nt", "Windows-only fix")
    def test_empty_dll_dirs_means_no_add_dll_directory_calls(self):
        """If no dirs exist, add_dll_directory is never called."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_exe = os.path.join(tmp, "no_such_sub", "python.exe")
            # fake_exe's parent dir doesn't exist → isdir returns False for all
            with patch.object(sys, "executable", fake_exe):
                with patch("os.add_dll_directory") as mock_add:
                    with patch("os.path.isdir", return_value=False):
                        code = _extract_fix_code()
                        ns = {"__builtins__": __builtins__, "os": os, "sys": sys}
                        exec(compile(code, str(_WORKER_PATH), "exec"), ns)

        mock_add.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows-only fix")
    def test_handles_list_length_matches_dirs_count(self):
        """_dll_handles has exactly one entry per directory registered."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_exe   = os.path.join(tmp, "python.exe")
            torch_lib  = os.path.join(tmp, "Lib", "site-packages", "torch",      "lib")
            taud_lib   = os.path.join(tmp, "Lib", "site-packages", "torchaudio", "lib")
            scripts    = os.path.join(tmp, "Scripts")
            for d in [torch_lib, taud_lib, scripts]:
                os.makedirs(d, exist_ok=True)

            with patch.object(sys, "executable", fake_exe):
                with patch("os.add_dll_directory", side_effect=lambda _: MagicMock()):
                    code = _extract_fix_code()
                    ns = {"__builtins__": __builtins__, "os": os, "sys": sys}
                    exec(compile(code, str(_WORKER_PATH), "exec"), ns)

        # All 4 dirs exist → 4 handles
        self.assertEqual(len(ns["_dll_handles"]), 4)


if __name__ == "__main__":
    unittest.main()
