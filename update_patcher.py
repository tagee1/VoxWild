"""
update_patcher.py — In-app update mechanism for VoxWild.

Downloads a patch zip from a GitHub release, extracts it, and spawns a
tiny batch script that waits for the app to exit, copies the new files
over the install directory, and relaunches the app.

Why this exists:
  - Avoids forcing users to download a full 377MB installer every update
  - Avoids SmartScreen prompts on updates (we're writing files from an
    already-trusted process, not executing a newly downloaded .exe)
  - Typical patch is ~20MB

The patch zip contains files (relative paths match the install layout):
  VoxWild.exe
  _internal/chatterbox_worker.py
  _internal/enhance_worker.py
  _internal/icon.ico
  _internal/logo.png
  _internal/theme.json
  ... etc
  manifest.json  (version + per-file sha256)
"""

import hashlib
import json
import os
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def _install_dir() -> Path:
    """Directory containing VoxWild.exe (the install root)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    # Dev fallback — running from source
    return Path(__file__).parent


def _ssl_context():
    """Use certifi CA bundle so HTTPS works in the frozen app."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _log(path: Path, msg: str):
    """Append to patch diagnostic log."""
    try:
        from datetime import datetime
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def fetch_patch_url(github_repo: str, target_version: str) -> str | None:
    """Look up the patch zip asset URL in a GitHub release.

    Returns None if the release or the patch asset doesn't exist
    (older releases only shipped the full installer).
    """
    api = f"https://api.github.com/repos/{github_repo}/releases/tags/v{target_version}"
    req = urllib.request.Request(api, headers={"User-Agent": "VoxWild-Patcher"})
    with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
        data = json.loads(resp.read())
    want_name = f"VoxWild-Patch-{target_version}.zip"
    for asset in data.get("assets", []):
        if asset.get("name") == want_name:
            return asset.get("browser_download_url")
    return None


def download_patch(url: str, dest: Path, progress=None) -> None:
    """Download the patch zip. Calls progress(bytes_read, total) periodically."""
    req = urllib.request.Request(url, headers={"User-Agent": "VoxWild-Patcher"})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if progress:
                    try:
                        progress(read, total)
                    except Exception:
                        pass


def verify_patch(zip_path: Path) -> tuple[bool, str]:
    """Verify the manifest.json inside the zip matches every file's SHA256.

    Returns (ok, message). On failure we do NOT apply the patch.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            try:
                manifest = json.loads(zf.read("manifest.json"))
            except KeyError:
                return False, "Patch is missing manifest.json"
            for arcname, info in manifest.get("files", {}).items():
                expected = info.get("sha256")
                size     = info.get("size")
                try:
                    data = zf.read(arcname)
                except KeyError:
                    return False, f"Patch is missing file: {arcname}"
                if size is not None and len(data) != size:
                    return False, f"Size mismatch for {arcname}"
                if expected:
                    actual = hashlib.sha256(data).hexdigest()
                    if actual != expected:
                        return False, f"SHA256 mismatch for {arcname}"
        return True, "ok"
    except zipfile.BadZipFile:
        return False, "Patch is not a valid zip"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def _write_apply_script(script_path: Path, extract_dir: Path,
                       install_dir: Path, exe_path: Path, log_file: Path) -> None:
    """Write a .bat file that waits for VoxWild to exit, copies files, restarts.

    Why a .bat and not a Python script: the Python runtime is inside the
    install dir being updated. Using cmd.exe keeps the updater independent
    of any file we're replacing.
    """
    # Use robocopy for reliable file replacement even if target is locked.
    # Batch quoting: double percent signs are for batch's own use; real percent
    # signs in paths need escaping too. We keep paths simple via variables.
    script = f"""@echo off
setlocal
set "LOG={log_file}"
echo [%%DATE%% %%TIME%%] Patch apply starting >> "%%LOG%%"

REM Wait for VoxWild.exe to fully exit (up to 30 seconds)
set /a TRIES=0
:wait_loop
tasklist /fi "imagename eq VoxWild.exe" 2^>nul | find /i "VoxWild.exe" ^>nul
if errorlevel 1 goto ready
set /a TRIES+=1
if %%TRIES%% gtr 30 (
  echo [%%DATE%% %%TIME%%] ERROR: VoxWild.exe did not exit in time >> "%%LOG%%"
  exit /b 1
)
timeout /t 1 /nobreak ^>nul
goto wait_loop

:ready
echo [%%DATE%% %%TIME%%] Copying files >> "%%LOG%%"

REM Mirror the extracted directory over the install directory.
REM /E = include subdirs (empty ones too), /IS /IT = overwrite same/older,
REM /NFL /NDL /NJH /NJS = less log noise, /R:3 /W:1 = retry locked files briefly
robocopy "{extract_dir}" "{install_dir}" /E /IS /IT /NFL /NDL /NJH /NJS /R:3 /W:1 ^>^> "%%LOG%%" 2^>^&1

if errorlevel 8 (
  echo [%%DATE%% %%TIME%%] ERROR: robocopy returned %%errorlevel%% >> "%%LOG%%"
  exit /b 1
)

echo [%%DATE%% %%TIME%%] Copy complete, launching app >> "%%LOG%%"
start "" "{exe_path}"

REM Clean up temp dirs
rmdir /s /q "{extract_dir}" 2^>nul

REM Delete this script
(goto) 2^>nul ^& del "%%~f0"
"""
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)


def apply_patch(zip_path: Path, status_cb=None) -> tuple[bool, str]:
    """Extract patch and spawn the apply script. Caller should exit immediately after.

    Returns (ok, message). On success, caller must exit the process so the
    spawned script can replace VoxWild.exe.
    """
    install_dir = _install_dir()
    log_file    = install_dir.parent / "patch_apply.log"

    _log(log_file, f"apply_patch start: zip={zip_path}  install={install_dir}")

    # 1. Verify integrity
    if status_cb: status_cb("Verifying patch...")
    ok, msg = verify_patch(zip_path)
    if not ok:
        _log(log_file, f"VERIFY FAILED: {msg}")
        return False, msg

    # 2. Extract to a temp dir (atomic once fully written)
    if status_cb: status_cb("Extracting...")
    extract_dir = Path(tempfile.mkdtemp(prefix="voxwild_patch_"))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            # Drop manifest.json from extraction; it's not part of the install
            members = [n for n in zf.namelist() if n != "manifest.json"]
            zf.extractall(extract_dir, members)
    except Exception as e:
        _log(log_file, f"EXTRACT FAILED: {e}")
        return False, f"Failed to extract patch: {e}"

    # 3. Build the apply script
    if status_cb: status_cb("Preparing installer...")
    script_path = Path(tempfile.gettempdir()) / f"voxwild_apply_{os.getpid()}.bat"
    exe_path    = install_dir / "VoxWild.exe"
    try:
        _write_apply_script(script_path, extract_dir, install_dir, exe_path, log_file)
    except Exception as e:
        _log(log_file, f"SCRIPT WRITE FAILED: {e}")
        return False, f"Could not prepare updater: {e}"

    # 4. Spawn the script detached — it runs after we exit
    if status_cb: status_cb("Restarting to finish update...")
    try:
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP so it survives our exit
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        subprocess.Popen(
            ["cmd.exe", "/c", str(script_path)],
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    except Exception as e:
        _log(log_file, f"SPAWN FAILED: {e}")
        return False, f"Could not start updater: {e}"

    _log(log_file, "apply_patch: updater spawned, caller should exit")
    return True, "Update will finish after restart"


def cleanup_old_patches(tmp_dir: Path | None = None) -> None:
    """Remove leftover patch temp dirs from previous update cycles."""
    try:
        import glob
        import shutil
        for d in glob.glob(str(Path(tempfile.gettempdir()) / "voxwild_patch_*")):
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        for f in glob.glob(str(Path(tempfile.gettempdir()) / "voxwild_apply_*.bat")):
            try:
                os.unlink(f)
            except Exception:
                pass
    except Exception:
        pass
