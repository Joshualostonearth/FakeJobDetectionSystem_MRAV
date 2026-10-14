#!/usr/bin/env bash
#
# start_all.sh — start every service the Fake Job Detection app needs.
#
#   ./start_all.sh              start everything, including the public call link
#   ./start_all.sh --no-tunnel  skip the Cloudflare tunnel (company answers on this PC)
#
# Services:
#   Ollama            :11434  local LLM (reused if already running, e.g. systemd)
#   WhisperLive       :9090   speech-to-text for the company's answers
#   call_bridge.py    :5000   question generation, TTS, answer processing
#   server.js         :3000   WebRTC call page + signalling
#   cloudflared               public HTTPS link to :3000 for the "company"
#   Streamlit         :8501   the app
#
# Logs go to logs/<service>.log. Ctrl+C stops everything this script started.

set -uo pipefail
set -m  # each background job gets its own process group, so we can stop its children too

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/whisper_Test/.venv"
LOG_DIR="$ROOT/logs"
USE_TUNNEL=1

for arg in "$@"; do
    case "$arg" in
        --no-tunnel) USE_TUNNEL=0 ;;
        -h|--help) sed -n '3,17p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)"; exit 1 ;;
    esac
done

PIDS=()
NAMES=()

cleanup() {
    local rc=$?
    trap - INT TERM EXIT
    echo
    echo "Stopping services..."
    for i in "${!PIDS[@]}"; do
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then
            kill -TERM -- "-${PIDS[$i]}" 2>/dev/null || kill -TERM "${PIDS[$i]}" 2>/dev/null
            echo "  stopped ${NAMES[$i]}"
        fi
    done
    wait 2>/dev/null
    exit "$rc"
}
trap cleanup INT TERM EXIT

port_in_use() {
    ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN
}

port_owner() {
    ss -ltnp "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}

# start NAME PORT TIMEOUT_SECONDS COMMAND...
# Runs COMMAND in the background and waits until PORT is listening.
start() {
    local name="$1" port="$2" timeout="$3"; shift 3
    local log="$LOG_DIR/$name.log"

    if port_in_use "$port"; then
        local pid; pid="$(port_owner "$port")"
        echo "✗ Port $port is already in use${pid:+ (pid $pid)} — is $name already running?"
        echo "  Stop it first${pid:+ (kill $pid)}, then rerun this script."
        exit 1
    fi

    printf "  starting %-12s" "$name..."
    "$@" >"$log" 2>&1 &
    local pid=$!
    PIDS+=("$pid"); NAMES+=("$name")

    local waited=0
    until port_in_use "$port"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo " ✗ exited. Last lines of $log:"
            tail -n 15 "$log" | sed 's/^/    /'
            exit 1
        fi
        if (( waited >= timeout )); then
            echo " ✗ not listening on :$port after ${timeout}s. See $log"
            exit 1
        fi
        sleep 1; ((waited++))
    done
    echo " ✓ :$port"
}

# ── Preflight ──────────────────────────────────────────────────────────
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "✗ Virtualenv not found at $VENV"; exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

for cmd in node ollama; do
    command -v "$cmd" >/dev/null || { echo "✗ '$cmd' is not installed"; exit 1; }
done
if (( USE_TUNNEL )) && ! command -v cloudflared >/dev/null; then
    echo "✗ cloudflared is not installed (or run with --no-tunnel)"; exit 1
fi

mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1

echo "Starting Fake Job Detection services (logs in logs/)"

# ── Ollama ─────────────────────────────────────────────────────────────
if curl -s -m 2 http://localhost:11434/api/tags >/dev/null; then
    echo "  ollama        ✓ already running"
else
    start ollama 11434 30 ollama serve
fi

# ── WhisperLive (speech-to-text) ───────────────────────────────────────
# First run downloads model weights, so allow extra time.
start whisper 9090 300 python -u whisper_Test/server.py

# ── Call bridge ────────────────────────────────────────────────────────
start call_bridge 5000 120 python -u whisper_Test/call_bridge.py

# ── WebRTC call page ───────────────────────────────────────────────────
start webrtc 3000 30 bash -c 'cd webrtc-call/server && exec node server.js'

# ── Cloudflare tunnel ──────────────────────────────────────────────────
TUNNEL_URL=""
if (( USE_TUNNEL )); then
    printf "  starting %-12s" "tunnel..."
    cloudflared tunnel --no-autoupdate --url http://localhost:3000 >"$LOG_DIR/tunnel.log" 2>&1 &
    PIDS+=("$!"); NAMES+=("tunnel")
    for _ in $(seq 1 60); do
        TUNNEL_URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/tunnel.log" | head -1)"
        [[ -n "$TUNNEL_URL" ]] && break
        sleep 1
    done
    if [[ -n "$TUNNEL_URL" ]]; then
        echo " ✓"
    else
        echo " ✗ no URL after 60s (see logs/tunnel.log). Continuing without it."
    fi
fi

# ── Streamlit ──────────────────────────────────────────────────────────
start streamlit 8501 120 streamlit run app.py --server.headless true

# ── Summary ────────────────────────────────────────────────────────────
cat <<EOF

══════════════════════════════════════════════════════════════
  All services running.

  App (you):            http://localhost:8501
  Call page (you):      http://localhost:3000
EOF
if [[ -n "$TUNNEL_URL" ]]; then
    echo "  Call page (company): $TUNNEL_URL"
    echo "                       ↑ send this link; it changes every run"
fi
cat <<EOF

  Before analysing an offer, open the call page on both sides
  and click Join on each.

  Logs: logs/<service>.log     Stop everything: Ctrl+C
══════════════════════════════════════════════════════════════
EOF

# Exit (and clean up) as soon as any service dies.
while true; do
    for i in "${!PIDS[@]}"; do
        if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            echo "✗ ${NAMES[$i]} stopped unexpectedly. Last lines of logs/${NAMES[$i]}.log:"
            tail -n 15 "$LOG_DIR/${NAMES[$i]}.log" | sed 's/^/    /'
            exit 1
        fi
    done
    sleep 2
done
