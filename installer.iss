; installer.iss — Inno Setup script for VoxWild
;
; Prerequisites:
;   1. Run PyInstaller:  pyinstaller app.spec
;      Output must be in:  dist\VoxWild\
;   2. Install Inno Setup 6:  https://jrsoftware.org/isinfo.php
;   3. Compile:  iscc installer.iss
;      Output: installer_output\VoxWild-Setup.exe
;
; Signing (after you have a code signing cert):
;   signtool sign /tr http://timestamp.sectigo.com /td sha256 /fd sha256 ^
;     /a installer_output\VoxWild-Setup.exe

#define MyAppName      "VoxWild"
#define MyAppVersion   "1.1.0"
#define MyAppPublisher "Cookie Studios"
#define MyAppURL       "https://cookiestudios.gumroad.com/l/TTSStudioPro"
#define MyAppSupportURL "mailto:cookiestudios.dev@gmail.com"
#define MyAppUpdatesURL "https://github.com/tagee1/voxwild/releases/latest"
#define MyAppExeName   "VoxWild.exe"
#define MyBuildDir     "dist\VoxWild"

[Setup]
AppId={{B3F2A1C4-7D8E-4F0A-9B2C-5E6D3A1F8C90}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppSupportURL}
AppUpdatesURL={#MyAppUpdatesURL}

; Install to Program Files by default (requires UAC elevation)
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes

; Output
OutputDir=installer_output
OutputBaseFilename=VoxWild-Setup
SetupIconFile=icon.ico

; Compression (LZMA2 is best ratio for large binaries)
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes

; Minimum Windows 10
MinVersion=10.0

; Require 64-bit Windows
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Show a license page during installation
LicenseFile=EULA.rtf

; Require admin for install (so it goes into Program Files)
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog

; Uninstall
Uninstallable=yes
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
CreateUninstallRegKey=yes


[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"


[Tasks]
Name: "desktopicon";   Description: "{cm:CreateDesktopIcon}";   GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startmenuicon"; Description: "Create a Start Menu shortcut"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce


[Files]
; ── Main application (PyInstaller one-dir build) ────────────────────────────
Source: "{#MyBuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; ── Legal docs ──────────────────────────────────────────────────────────────
Source: "CREDITS.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "PRIVACY.txt"; DestDir: "{app}"; Flags: ignoreversion

; ── worker scripts (alongside the exe, found via _res()) ─────────────────────
; Already included above via recursesubdirs since they're in the PyInstaller output.
; chatterbox_worker.py — Natural mode (chatterbox_env / python_embed)
; enhance_worker.py   — AI Enhancement (python_embed)


[Icons]
Name: "{group}\{#MyAppName}";          Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}";   Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon


[Run]
; Launch app after install (optional — user can uncheck)
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent


[UninstallRun]
; Nothing special needed — user data lives in %APPDATA%\TTS Studio (legacy path preserved for upgrades), not here


[Registry]
; File association, version info, etc. — add here if needed


[Code]
// Silently remove any existing installation before installing the new version.
// /VERYSILENT + SW_HIDE means no uninstaller window appears — user only sees
// the new installer, making upgrades feel like a single seamless install.
function InitializeSetup(): Boolean;
var
  UninstExe: String;
  ResultCode: Integer;
begin
  Result := True;
  if RegQueryStringValue(HKLM, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{B3F2A1C4-7D8E-4F0A-9B2C-5E6D3A1F8C90}_is1',
                         'UninstallString', UninstExe) then
  begin
    Exec(RemoveQuotes(UninstExe), '/VERYSILENT /NORESTART', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;
