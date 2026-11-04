<#
.SYNOPSIS
  (Run as administrator - install.ps1 does this with one UAC prompt.) Copies the default safety policy to
  %ProgramData%\Jarvis\safety.yaml and makes it read-only for normal users, so JARVIS - which runs with
  your normal rights - can read its rules but never loosen them.
#>
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$Source)

$ErrorActionPreference = 'Stop'
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error 'This script must run as administrator.'
    exit 1
}
if (-not (Test-Path $Source)) { Write-Error "policy source not found: $Source"; exit 2 }

$dir = Join-Path $env:ProgramData 'Jarvis'
$dest = Join-Path $dir 'safety.yaml'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
if (-not (Test-Path $dest)) { Copy-Item -LiteralPath $Source -Destination $dest }

# Well-known SIDs (names are localized, e.g. on Korean Windows): SYSTEM, Administrators, Users.
$system = '*S-1-5-18'
$admins = '*S-1-5-32-544'
$users = '*S-1-5-32-545'
& icacls $dir /inheritance:r /grant:r "${system}:(OI)(CI)F" "${admins}:(OI)(CI)F" "${users}:(OI)(CI)RX" | Out-Null
& icacls $dest /inheritance:r /grant:r "${system}:F" "${admins}:F" "${users}:R" | Out-Null
& icacls $dest /setowner $admins | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "icacls failed ($LASTEXITCODE)"; exit 3 }
Write-Host "Safety policy installed read-only at $dest"
exit 0
