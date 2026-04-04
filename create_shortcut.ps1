$WshShell = New-Object -ComObject WScript.Shell
$Desktop = [System.Environment]::GetFolderPath("Desktop")
$Shortcut = $WshShell.CreateShortcut("$Desktop\TTS Studio.lnk")
$Shortcut.TargetPath = "C:\tts-app\launch.vbs"
$Shortcut.WorkingDirectory = "C:\tts-app"
$Shortcut.Description = "TTS Studio"
$Shortcut.Save()
Write-Host "Shortcut created at $Desktop\TTS Studio.lnk"
