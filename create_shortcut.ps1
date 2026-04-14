$WshShell = New-Object -ComObject WScript.Shell
$Desktop = [System.Environment]::GetFolderPath("Desktop")
$Shortcut = $WshShell.CreateShortcut("$Desktop\VoxWild.lnk")
$Shortcut.TargetPath = "C:\tts-app\launch.vbs"
$Shortcut.WorkingDirectory = "C:\tts-app"
$Shortcut.Description = "VoxWild"
$Shortcut.Save()
Write-Host "Shortcut created at $Desktop\VoxWild.lnk"
