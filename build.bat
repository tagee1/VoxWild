@echo off
REM build.bat — Build TTS Studio installer in one step
REM
REM Requirements:
REM   - PyInstaller installed in the active Python env (pip install pyinstaller)
REM   - Inno Setup 6 installed at default path (or update ISCC_PATH below)
REM   - Run from C:\tts-app

setlocal enabledelayedexpansion

set "APP_DIR=%~dp0"
set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

echo.
echo ============================================================
echo  TTS Studio Build Script
echo ============================================================
echo.

REM ── Step 1: Clean previous build ─────────────────────────────────────────────
echo [1/3] Cleaning previous build...
if exist "%APP_DIR%dist" rmdir /s /q "%APP_DIR%dist"
if exist "%APP_DIR%build" rmdir /s /q "%APP_DIR%build"
if exist "%APP_DIR%installer_output" rmdir /s /q "%APP_DIR%installer_output"
echo       Done.

REM ── Step 2: PyInstaller ───────────────────────────────────────────────────────
echo.
echo [2/3] Running PyInstaller...
cd /d "%APP_DIR%"
pyinstaller app.spec --noconfirm
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller failed. Check output above.
    exit /b 1
)
echo       PyInstaller complete.

REM ── Verify the exe was produced ───────────────────────────────────────────────
if not exist "%APP_DIR%dist\TTS Studio\TTS Studio.exe" (
    echo.
    echo ERROR: Expected exe not found at dist\TTS Studio\TTS Studio.exe
    exit /b 1
)

REM ── Step 3: Inno Setup ────────────────────────────────────────────────────────
echo.
echo [3/3] Running Inno Setup...
if not exist "%ISCC_PATH%" (
    echo.
    echo WARNING: Inno Setup not found at:
    echo   %ISCC_PATH%
    echo   Download from: https://jrsoftware.org/isinfo.php
    echo   Then re-run this script.
    exit /b 1
)

"%ISCC_PATH%" "%APP_DIR%installer.iss"
if errorlevel 1 (
    echo.
    echo ERROR: Inno Setup compilation failed. Check output above.
    exit /b 1
)

REM ── Report sizes ─────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo  Build complete!
echo ============================================================
echo.
echo  Installer: installer_output\TTS-Studio-Setup.exe
echo.

REM Show installer size
for %%F in ("%APP_DIR%installer_output\TTS-Studio-Setup.exe") do (
    set /a "SIZE_MB=%%~zF / 1048576"
    echo  Size: !SIZE_MB! MB
)

echo.
echo  Next steps:
echo    1. Test the installer on a clean Windows machine
echo    2. Sign with: signtool sign /tr http://timestamp.sectigo.com /td sha256 /fd sha256 /a installer_output\TTS-Studio-Setup.exe
echo    3. Upload to Lemon Squeezy / Gumroad
echo.

endlocal
