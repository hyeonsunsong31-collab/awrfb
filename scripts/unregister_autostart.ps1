<#
.SYNOPSIS
  Removes the JARVIS logon task (JARVIS no longer starts when you log on).
#>
[CmdletBinding()]
param([string]$TaskName = 'JARVIS')

$ErrorActionPreference = 'Stop'
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Logon task '$TaskName' removed."
} else {
    Write-Host "No logon task named '$TaskName'."
}
