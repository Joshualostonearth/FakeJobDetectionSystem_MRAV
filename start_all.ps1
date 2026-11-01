<#
start_all.ps1 - Windows equivalent of start_all.sh.

  .\start_all.ps1              start everything, including the public call link
  .\start_all.ps1 -NoTunnel    skip the Cloudflare tunnel (company answers on this PC)
  .\start_all.ps1 -NoVoice     start only the Streamlit app (image/text -> M4 verdict + XAI);
                               no Ollama, no voice services

The verdict (M4) never needs Ollama; Ollama is started only for the voice agent.

Services:
  Ollama          :11434  local LLM for the voice agent (reused if already running)
  WhisperLive     :9090   speech-to-text for the company's answers
  call_bridge.py  :5000   question generation, TTS, answer processing
  server.js       :3000   WebRTC call page + signalling
  cloudflared             public HTTPS link to :3000 for the "company"
  Streamlit       :8501   the app (runs from .venv-m4, or $env:M4_VENV)

Logs go to logs\<service>.log. Ctrl+C stops everything this script started.
If script execution is blocked: powershell -ExecutionPolicy Bypass -File .\start_all.ps1 -NoVoice
#>
param([switch]$NoVoice, [switch]$NoTunnel)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$AppVenv = if ($env:M4_VENV) { $env:M4_VENV } else { Join-Path $Root ".venv-m4" }
$AppPy = Join-Path $AppVenv "Scripts\python.exe"
$VoicePy = Join-Path $Root "whisper_Test\.venv\Scripts\python.exe"
$LogDir = Join-Path $Root "logs"
$UseTunnel = -not ($NoTunnel -or $NoVoice)
$Started = New-Object System.Collections.ArrayList

function Test-Port([int]$Port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { $c.Connect("127.0.0.1", $Port); return $true } catch { return $false } finally { $c.Close() }
}

function Start-Service2([string]$Name, [int]$Port, [int]$Timeout, [string]$Exe, [string[]]$Arguments, [string]$Dir = $Root) {
    if (Test-Port $Port) { throw "Port $Port is already in use - is $Name already running? Stop it first." }
    Write-Host ("  starting {0,-12}" -f "$Name...") -NoNewline
    $log = Join-Path $LogDir "$Name.log"
    $p = Start-Process -FilePath $Exe -ArgumentList $Arguments -WorkingDirectory $Dir -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    [void]$Started.Add(@{ Name = $Name; Proc = $p })
    $waited = 0
    while (-not (Test-Port $Port)) {
        if ($p.HasExited) {
            Write-Host " x exited. Last lines of $log.err:"
            Get-Content "$log.err" -Tail 15 -ErrorAction SilentlyContinue | ForEach-Object { "    $_" }
            throw "$Name failed to start"
        }
        if ($waited -ge $Timeout) { throw "$Name not listening on :$Port after ${Timeout}s. See $log" }
        Start-Sleep -Seconds 1; $waited++
    }
    Write-Host " ok :$Port"
}

# ── Preflight ──────────────────────────────────────────────────────────
if (-not (Test-Path $AppPy)) { throw "App virtualenv not found at $AppVenv (see README: py -3.12 -m venv .venv-m4; pip install -r requirements.txt)" }
if (-not $env:M4_MODEL_DIR -and -not (Test-Path (Join-Path $Root "models\mravplus_bundle.joblib"))) {
    throw "models\mravplus_bundle.joblib not found (see README: Model files)"
}
if (-not $NoVoice) {
    if (-not (Test-Path $VoicePy)) { throw "Voice virtualenv not found at whisper_Test\.venv (or run with -NoVoice)" }
    foreach ($cmd in @("node", "ollama")) {
        if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "'$cmd' is not installed (or run with -NoVoice)" }
    }
    if (-not (Test-Path (Join-Path $Root "webrtc-call\server\node_modules"))) { throw "Run 'npm install' in webrtc-call\server first" }
    if ($UseTunnel -and -not (Get-Command cloudflared -ErrorAction SilentlyContinue)) { throw "cloudflared is not installed (or run with -NoTunnel)" }
}
New-Item -ItemType Directory -Force $LogDir | Out-Null

Write-Host "Starting Fake Job Detection services (logs in logs\)"
$TunnelUrl = $null
try {
    if (-not $NoVoice) {
        if (Test-Port 11434) { Write-Host "  ollama        ok already running" }
        else { Start-Service2 "ollama" 11434 30 "ollama" @("serve") }
        Start-Service2 "whisper" 9090 300 $VoicePy @("-u", "whisper_Test\server.py")
        Start-Service2 "call_bridge" 5000 120 $VoicePy @("-u", "whisper_Test\call_bridge.py")
        Start-Service2 "webrtc" 3000 30 "node" @("server.js") (Join-Path $Root "webrtc-call\server")
        if ($UseTunnel) {
            $tlog = Join-Path $LogDir "tunnel.log"
            $tp = Start-Process cloudflared -ArgumentList @("tunnel", "--no-autoupdate", "--url", "http://localhost:3000") `
                -PassThru -WindowStyle Hidden -RedirectStandardOutput "$tlog.out" -RedirectStandardError $tlog
            [void]$Started.Add(@{ Name = "tunnel"; Proc = $tp })
            for ($i = 0; $i -lt 60 -and -not $TunnelUrl; $i++) {
                Start-Sleep -Seconds 1
                $m = Select-String -Path $tlog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($m) { $TunnelUrl = $m.Matches[0].Value }
            }
        }
    }
    Start-Service2 "streamlit" 8501 180 $AppPy @("-m", "streamlit", "run", "app.py", "--server.headless", "true")

    Write-Host ""
    Write-Host "  App (you):            http://localhost:8501"
    if (-not $NoVoice) { Write-Host "  Call page (you):      http://localhost:3000" }
    if ($TunnelUrl) { Write-Host "  Call page (company):  $TunnelUrl  (send this link; it changes every run)" }
    Write-Host "  Logs: logs\<service>.log     Stop everything: Ctrl+C"

    while ($true) {
        foreach ($s in $Started) {
            if ($s.Proc.HasExited) { throw "$($s.Name) stopped unexpectedly. See logs\$($s.Name).log(.err)" }
        }
        Start-Sleep -Seconds 2
    }
}
finally {
    Write-Host "Stopping services..."
    foreach ($s in $Started) {
        if (-not $s.Proc.HasExited) {
            & taskkill /PID $s.Proc.Id /T /F 2>$null | Out-Null
            Write-Host "  stopped $($s.Name)"
        }
    }
}
