<#
.SYNOPSIS
  Installs JARVIS on this Windows PC. Safe to re-run: every step checks what is already there.

.DESCRIPTION
  1. winget: uv, Node.js LTS, FFmpeg (and Ollama with -WithOllama)
  2. Python 3.12 venv in .venv with the Windows, voice, GPU, ML, web and OCR extras
     (torch/torchvision/torchaudio from the PyTorch CUDA index; onnxruntime-gpu as the ONNX runtime)
  3. Builds the holographic HUD (Electron)
  4. Downloads the pinned voice models (SHA-256 verified) and, optionally, exports the zero-shot detector
  5. ONE admin prompt: writes the read-only safety policy to %ProgramData%\Jarvis\safety.yaml
  6. Registers the logon task (your normal user rights - not a service)

  Your Roblox password is never asked for or stored. Your NVIDIA API key is set afterwards with
  `jarvis api-key` (Windows Credential Manager) - never put it in a file or a chat.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
#>
[CmdletBinding()]
param(
    [switch]$WithOllama,
    [switch]$SkipModels,
    [switch]$SkipHud,
    [switch]$SkipDetector,
    [switch]$NoAutostart,
    [switch]$NoPolicy,
    [string]$CudaIndex = 'https://download.pytorch.org/whl/cu128',
    [string]$PythonVersion = '3.12'
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

function Step([string]$text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Warn([string]$text) { Write-Host "    $text" -ForegroundColor Yellow }
function Have([string]$cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}
function Invoke-Checked([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe $($arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

# ---------------------------------------------------------------------------------------- 0. checks
Step 'Checking Windows'
$build = [Environment]::OSVersion.Version.Build
if ($build -lt 19041) { throw "Windows 10 version 2004 (build 19041) or newer is required (this is build $build)." }
if (-not [Environment]::Is64BitOperatingSystem) { throw 'A 64-bit Windows is required.' }
Write-Host "    Windows build $build, 64-bit: ok"

# ---------------------------------------------------------------------------------------- 1. tools
Step 'Installing tools with winget (uv, Node.js LTS, FFmpeg)'
if (-not (Have 'winget')) { throw 'winget is missing - install "App Installer" from the Microsoft Store first.' }
$packages = @(
    @{ Id = 'astral-sh.uv'; Cmd = 'uv' },
    @{ Id = 'OpenJS.NodeJS.LTS'; Cmd = 'node' },
    @{ Id = 'Gyan.FFmpeg'; Cmd = 'ffmpeg' }
)
if ($WithOllama) { $packages += @{ Id = 'Ollama.Ollama'; Cmd = 'ollama' } }
foreach ($p in $packages) {
    if (Have $p.Cmd) { Write-Host "    $($p.Cmd): already installed"; continue }
    winget install --id $p.Id -e --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { Warn "winget could not install $($p.Id) (exit $LASTEXITCODE) - install it manually" }
}
Refresh-Path
foreach ($c in @('uv', 'node', 'npm', 'ffmpeg')) {
    if (-not (Have $c)) { throw "$c is still not on PATH - open a new terminal and run this script again." }
}

# ---------------------------------------------------------------------------------------- 2. python
Step "Python $PythonVersion environment (.venv)"
Invoke-Checked 'uv' @('python', 'install', $PythonVersion)
if (-not (Test-Path '.venv\Scripts\python.exe')) { Invoke-Checked 'uv' @('venv', '--python', $PythonVersion, '.venv') }
$Py = Join-Path $Repo '.venv\Scripts\python.exe'
$PyW = Join-Path $Repo '.venv\Scripts\pythonw.exe'
$env:VIRTUAL_ENV = Join-Path $Repo '.venv'

Step 'PyTorch with CUDA (from the PyTorch index; PyPI only has CPU builds for Windows)'
Invoke-Checked 'uv' @('pip', 'install', '--python', $Py, 'torch', 'torchvision', 'torchaudio', '--index-url', $CudaIndex)

Step 'JARVIS and its extras'
Invoke-Checked 'uv' @('pip', 'install', '--python', $Py, '-e', '.[win,voice,gpu,speaker,ml,detect,web,ocr,kb]')

$vramMb = 0
if (Have 'nvidia-smi') {
    $first = (& nvidia-smi '--query-gpu=memory.total' '--format=csv,noheader,nounits' | Select-Object -First 1)
    [int]::TryParse("$first".Trim(), [ref]$vramMb) | Out-Null
}
if ($vramMb -ge 7500) {
    Step "Qwen3-TTS voice engine ($vramMb MB VRAM)"
    & 'uv' pip install --python $Py -e '.[tts-qwen]'
    if ($LASTEXITCODE -ne 0) { Warn 'qwen-tts did not install - JARVIS uses the Supertonic voice instead' }
}

Step 'ONNX Runtime with CUDA'
# Several packages pull the CPU-only "onnxruntime"; it shadows onnxruntime-gpu. Keep only the GPU build.
$prev = $ErrorActionPreference
$ErrorActionPreference = 'Continue'  # uv reports on stderr; Windows PowerShell 5.1 would treat that as an error
& 'uv' pip uninstall --python $Py onnxruntime *> $null
$ErrorActionPreference = $prev
Invoke-Checked 'uv' @('pip', 'install', '--python', $Py, '--reinstall', 'onnxruntime-gpu')
& $Py -c "import onnxruntime as o; p=o.get_available_providers(); print('    ONNX Runtime providers:', p); import sys; sys.exit(0 if 'CUDAExecutionProvider' in p else 3)"
if ($LASTEXITCODE -ne 0) { Warn 'CUDAExecutionProvider is not available - check the NVIDIA driver (jarvis doctor).' }

if (Have 'ollama') {
    Step 'Local models for offline use (Ollama)'
    $local = @('qwen3:4b')
    if ($vramMb -ge 11500) { $local += @('qwen3:8b', 'qwen2.5vl:7b') }  # 12 GB+: bigger chat model + local vision
    foreach ($m in $local) {
        & ollama pull $m
        if ($LASTEXITCODE -ne 0) { Warn "ollama could not pull $m (is the Ollama app running?) - JARVIS uses the cloud only" }
    }
}

# ---------------------------------------------------------------------------------------- 3. HUD
if (-not $SkipHud) {
    Step 'Holographic HUD (Electron)'
    Push-Location (Join-Path $Repo 'hud')
    try {
        Invoke-Checked 'npm.cmd' @('ci', '--no-audit', '--no-fund')
        Invoke-Checked 'npm.cmd' @('run', 'build')
    } finally { Pop-Location }
}

# ---------------------------------------------------------------------------------------- 4. models
if (-not $SkipModels) {
    Step 'Voice models (wake word, VAD, Whisper, Supertonic, speaker model)'
    & $Py -m jarvis voice-models --all
    if ($LASTEXITCODE -ne 0) { Warn 'some models did not download - re-run: jarvis voice-models --all' }
}
if (-not $SkipDetector) {
    Step 'Zero-shot fighter detector (YOLO-World -> ONNX)'
    & $Py (Join-Path $Repo 'scripts\export_detector.py')
    if ($LASTEXITCODE -ne 0) { Warn 'detector export failed - the fighter still works HUD-only; re-run scripts\export_detector.py later' }
}

# ---------------------------------------------------------------------------------------- 5. safety policy
if (-not $NoPolicy) {
    Step 'Safety policy (one admin prompt)'
    $dest = Join-Path $env:ProgramData 'Jarvis\safety.yaml'
    if (Test-Path $dest) {
        Write-Host "    $dest already exists - kept (edit it as administrator if you must)"
    } else {
        $script = Join-Path $PSScriptRoot 'install_safety_policy.ps1'
        $src = Join-Path $Repo 'configs\safety.default.yaml'
        try {
            $proc = Start-Process -FilePath 'powershell.exe' -Verb RunAs -Wait -PassThru -ArgumentList @(
                '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$script`"", '-Source', "`"$src`"")
            if ($proc.ExitCode -ne 0) { Warn "the safety policy was not installed (exit $($proc.ExitCode)); JARVIS uses the repo copy" }
        } catch {
            Warn "admin prompt declined or failed ($($_.Exception.Message)); JARVIS uses the repo copy - re-run this script to protect it"
        }
    }
}

# ---------------------------------------------------------------------------------------- 6. autostart
if (-not $NoAutostart) {
    Step 'Start at logon (Task Scheduler)'
    try {
        & (Join-Path $PSScriptRoot 'register_autostart.ps1') -Python $PyW -WorkDir $Repo
    } catch {
        Warn "could not register the logon task ($($_.Exception.Message)); start JARVIS with: jarvis start"
    }
}

Step 'Checking the result'
& $Py -m jarvis doctor

Write-Host "`nJARVIS is installed." -ForegroundColor Green
Write-Host 'Next:'
Write-Host '  .venv\Scripts\jarvis setup      # API key, voiceprint, voice, Roblox checklist, calibration'
Write-Host '  .venv\Scripts\jarvis start      # or log off and on again'
