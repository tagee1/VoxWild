"""
update_patcher.py — In-app update mechanism for VoxWild.

Downloads a patch zip from a GitHub release, copies most files in-process
(while the app is running), and spawns a tiny batch script to swap the
exe after the app exits.

Architecture:
  Phase A (Python, in-process):
    - Download + verify patch zip
    - Copy ALL files except VoxWild.exe to install dir via shutil.copy2
    - Stage the new exe as VoxWild_update.exe

  Phase B (batch, ~10 lines, after app exits):
    - Wait 2s for process to fully release file locks
    - Delete old VoxWild.exe (retry 3x)
    - Rename VoxWild_update.exe → VoxWild.exe
    - Launch the new exe

Why Phase A works while the app runs:
  On Windows, only the running .exe process image is locked. Worker .py
  files, theme.json, icon.ico, logo.png etc. are loaded into memory at
  startup and the file handles are released — shutil.copy2 succeeds.
"""

import hashlib
import json
import os
import shutil
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
    return Path(__file__).parent


def _ssl_context():
    """Use certifi CA bundle so HTTPS works in the frozen app."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _log(path: Path, msg: str):
    try:
        from datetime import datetime
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


# ── Download + verify (unchanged from previous version) ─────────────────────

def fetch_patch_url(github_repo: str, target_version: str) -> str | None:
    """Look up the patch zip asset URL in a GitHub release."""
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
    """Verify the manifest.json inside the zip matches every file's SHA256."""
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


# ── Phase A: in-process file copy ────────────────────────────────────────────

def apply_patch(zip_path: Path, status_cb=None) -> tuple[bool, str]:
    """Copy patch files in-process, stage new exe, spawn swap batch.

    Caller must exit the process immediately after this returns True
    so the batch can replace VoxWild.exe.
    """
    install_dir = _install_dir()
    log_file    = install_dir.parent / "patch_apply.log"
    exe_name    = "VoxWild.exe"
    staged_name = "VoxWild_update.exe"

    _log(log_file, f"=== apply_patch start ===")
    _log(log_file, f"  zip={zip_path}")
    _log(log_file, f"  install={install_dir}")

    # 1. Verify
    if status_cb: status_cb("Verifying patch...")
    ok, msg = verify_patch(zip_path)
    if not ok:
        _log(log_file, f"VERIFY FAILED: {msg}")
        return False, msg

    # 2. Extract to temp
    if status_cb: status_cb("Extracting...")
    extract_dir = Path(tempfile.mkdtemp(prefix="voxwild_patch_"))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [n for n in zf.namelist() if n != "manifest.json"]
            zf.extractall(extract_dir, members)
    except Exception as e:
        _log(log_file, f"EXTRACT FAILED: {e}")
        return False, f"Failed to extract patch: {e}"

    _log(log_file, f"  extracted {len(members)} files to {extract_dir}")

    # 3. Phase A — copy everything except the exe, IN PROCESS
    #    Files that can't be copied (locked DLLs etc.) get deferred to the
    #    batch script alongside VoxWild.exe.
    if status_cb: status_cb("Copying files...")
    copied = 0
    deferred = []  # (src, rel) pairs for files that need post-exit copy
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            src = Path(root) / fname
            rel = src.relative_to(extract_dir)
            rel_str = str(rel).replace("\\", "/")

            if rel_str == exe_name:
                # Stage the new exe — don't try to overwrite the running one
                dst = install_dir / staged_name
                try:
                    if dst.exists():
                        dst.unlink()
                    shutil.copy2(src, dst)
                    _log(log_file, f"  staged: {rel_str} -> {staged_name}")
                except Exception as e:
                    _log(log_file, f"  STAGE FAILED: {rel_str}: {e}")
                    return False, f"Failed to stage {exe_name}: {e}"
                continue

            # Regular file — try to copy directly
            dst = install_dir / rel
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
            except PermissionError:
                # File is locked (loaded DLL, open handle) — defer to batch
                _log(log_file, f"  DEFERRED (locked): {rel_str}")
                deferred.append((str(src), str(install_dir / rel)))
            except Exception as e:
                _log(log_file, f"  COPY FAILED: {rel_str}: {e}")
                return False, f"Failed to copy {rel_str}: {e}"

    _log(log_file, f"  copied {copied} files, {len(deferred)} deferred")

    # Verify staged exe exists
    staged = install_dir / staged_name
    if not staged.exists():
        _log(log_file, "  ERROR: staged exe not found after extraction")
        return False, "Patch zip did not contain VoxWild.exe"

    # 4. Write the swap batch (handles exe + any deferred locked files)
    if status_cb: status_cb("Preparing restart...")
    script_path = Path(tempfile.gettempdir()) / f"voxwild_swap_{os.getpid()}.bat"
    exe_path = install_dir / exe_name
    try:
        _write_swap_script(script_path, exe_path, staged, log_file,
                          deferred_files=deferred, install_dir=install_dir)
    except Exception as e:
        _log(log_file, f"SCRIPT WRITE FAILED: {e}")
        return False, f"Could not prepare updater: {e}"

    # 5. Spawn batch hidden
    if status_cb: status_cb("Restarting to finish update...")
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        subprocess.Popen(
            ["cmd.exe", "/c", str(script_path)],
            startupinfo=si,
            creationflags=CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    except Exception as e:
        _log(log_file, f"SPAWN FAILED: {e}")
        return False, f"Could not start updater: {e}"

    # 6. Clean up temp extract dir (only if no deferred files need it)
    if not deferred:
        try:
            shutil.rmtree(extract_dir, ignore_errors=True)
        except Exception:
            pass
    else:
        _log(log_file, f"  keeping extract dir for {len(deferred)} deferred file(s)")

    _log(log_file, "apply_patch: swap batch spawned, caller should exit now")
    return True, "Update will finish after restart"


# ── Phase B: exe swap batch ──────────────────────────────────────────────────

def _write_swap_script(script_path: Path, exe_path: Path,
                      staged_path: Path, log_file: Path,
                      deferred_files=None, install_dir=None) -> None:
    """Write a batch that swaps the exe + copies any deferred locked files.

    Deferred files are ones that couldn't be copied in Phase A because
    they were locked (loaded DLLs like vcomp140.dll). After the app exits
    and releases all handles, we copy them here.
    """
    # Build deferred copy commands
    deferred_cmds = ""
    if deferred_files:
        deferred_cmds = "\nREM Copy deferred locked files\n"
        for src_str, dst_str in deferred_files:
            deferred_cmds += f'copy /y "{src_str}" "{dst_str}" >> "!LOG!" 2>&1\n'
        deferred_cmds += f'echo [%DATE% %TIME%] Copied {len(deferred_files)} deferred file(s) >> "!LOG!"\n'

    script = f"""@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "LOG={log_file}"
set "EXE={exe_path}"
set "STAGED={staged_path}"

echo [%DATE% %TIME%] Swap script started >> "!LOG!"

REM Wait for app to exit
ping -n 3 127.0.0.1 >nul
{deferred_cmds}
REM Try to delete old exe (retry up to 5 times)
set /a TRIES=0
:retry_del
del /f /q "!EXE!" >nul 2>&1
if not exist "!EXE!" goto del_ok
set /a TRIES+=1
if !TRIES! geq 5 (
    echo [%DATE% %TIME%] ERROR: could not delete exe after 5 tries >> "!LOG!"
    start "" "!EXE!"
    exit /b 1
)
echo [%DATE% %TIME%] Retry !TRIES!/5 >> "!LOG!"
ping -n 2 127.0.0.1 >nul
goto retry_del

:del_ok
echo [%DATE% %TIME%] Old exe deleted >> "!LOG!"

REM Rename staged exe
move /y "!STAGED!" "!EXE!" >> "!LOG!" 2>&1
if errorlevel 1 (
    echo [%DATE% %TIME%] ERROR: rename failed >> "!LOG!"
    exit /b 1
)

echo [%DATE% %TIME%] Swap complete, launching >> "!LOG!"
start "" "!EXE!"

REM Self-delete
(goto) 2>nul & del "%~f0"
"""
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)


# ── Startup recovery ─────────────────────────────────────────────────────────

def check_interrupted_update() -> bool:
    """Return True if a staged VoxWild_update.exe exists (previous update
    completed Phase A but Phase B failed)."""
    staged = _install_dir() / "VoxWild_update.exe"
    return staged.exists()


def retry_exe_swap() -> tuple[bool, str]:
    """Try to swap VoxWild_update.exe → VoxWild.exe from within the running app.

    On Windows, renaming a running exe IS allowed (unlike deleting or
    overwriting). So we rename the current exe to .old, rename staged to
    the real name, then the user restarts to pick up the new exe.
    """
    d = _install_dir()
    staged = d / "VoxWild_update.exe"
    exe    = d / "VoxWild.exe"
    backup = d / "VoxWild.exe.old"

    if not staged.exists():
        return False, "No staged update found"

    try:
        # Remove old backup if exists
        if backup.exists():
            backup.unlink()
        # Rename running exe → .old (allowed on Windows)
        exe.rename(backup)
        # Move staged into place
        staged.rename(exe)
        # Clean up backup (best effort)
        try:
            backup.unlink()
        except OSError:
            pass  # still locked, cleanup on next launch
        return True, "Update applied. Restart to use the new version."
    except Exception as e:
        return False, f"Retry failed: {e}"


# ── Cleanup ──────────────────────────────────────────────────────────────────

def cleanup_old_patches() -> None:
    """Remove leftover temp dirs and batch scripts from previous updates.

    Only removes items older than 1 hour — avoids deleting files that a
    currently-running batch script still needs (deferred DLL copies).
    """
    try:
        import glob
        import time
        cutoff = time.time() - 3600  # 1 hour ago
        for d in glob.glob(str(Path(tempfile.gettempdir()) / "voxwild_patch_*")):
            try:
                if os.path.getmtime(d) < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        for f in glob.glob(str(Path(tempfile.gettempdir()) / "voxwild_swap_*.bat")):
            try:
                if os.path.getmtime(f) < cutoff:
                    os.unlink(f)
            except Exception:
                pass
    except Exception:
        pass
