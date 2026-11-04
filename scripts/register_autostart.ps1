<#
.SYNOPSIS
  Starts JARVIS (the supervisor) when you log on: a Task Scheduler logon task with your normal rights.
  Not a Windows service - services cannot see the desktop, audio devices or keyboard/mouse input.
#>
[CmdletBinding()]
param(
    [string]$Python = '',
    [string]$WorkDir = (Split-Path -Parent $PSScriptRoot),
    [string]$TaskName = 'JARVIS'
)

$ErrorActionPreference = 'Stop'
if (-not $Python) { $Python = Join-Path $WorkDir '.venv\Scripts\pythonw.exe' }
if (-not (Test-Path $Python)) { throw "Python not found: $Python (run scripts\install.ps1 first)" }

$user = "$env:USERDOMAIN\$env:USERNAME"
$action = New-ScheduledTaskAction -Execute $Python -Argument '-m jarvis.supervisor' -WorkingDirectory $WorkDir
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$trigger.Delay = 'PT15S'  # let audio devices and the network come up first
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
    -Description 'JARVIS voice agent and UBG fighter (supervisor). Remove with scripts\unregister_autostart.ps1.' -Force | Out-Null
Write-Host "Logon task '$TaskName' registered for $user."
