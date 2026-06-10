#!/bin/bash
# Run this on the REMOTE box (e.g. inside the vast.ai container, after the
# repo is cloned and requirements.txt is installed). It launches the Django
# frontend and the reverie backend in tmux so both stay alive across ssh
# disconnects.
#
# Two modes, chosen automatically:
#
#   * If you are NOT already inside a tmux session, the script creates a
#     fresh detached session called "gen_agents" with two windows:
#       window 0 "frontend" -> environment/frontend_server, runserver on :8000
#       window 1 "backend"  -> reverie/backend_server, reverie.py (interactive)
#     Attach with: tmux attach -t gen_agents
#
#   * If you ARE inside a tmux session (e.g. vast.ai drops you into a default
#     session called "main"), the script opens TWO NEW WINDOWS in the current
#     session named "gen-frontend" and "gen-backend". This avoids the
#     "sessions should be nested with care" warning and the awkward
#     Ctrl-b Ctrl-b prefix collision that nesting causes.
#
# To see the frontend locally, run on your laptop:
#   ./ssh2tunnel.sh "ssh -p PORT USER@HOST ..."
# then execute the command it prints. The Django dev server is then
# reachable at http://localhost:8000/simulator_home.
#
# Re-running this script while the windows or session already exist is a no-op
# (it prints how to clean up and exits).
#
# Logging (on by default, one set of files per launch, under logs/):
#   logs/frontend_console_<timestamp>.log       full Django window transcript
#   logs/backend_console_<timestamp>.log        full reverie.py window transcript
#                                               (the print_run_prompts blocks)
#   logs/backend_prompt_pairs_<timestamp>.jsonl exact model input/output pairs,
#                                               one JSON record per LLM call
#                                               including every retry (written
#                                               by the harness layer; enabled
#                                               via REVERIE_PROMPT_LOG)
#   logs/backend_prompt_pairs_<timestamp>.txt   the same records, pretty-printed
#                                               for humans (banner-delimited
#                                               blocks; read this one in vim)
# Console transcripts are captured with `tmux pipe-pane`, so they survive
# scrollback loss and ssh disconnects.
#
# Note: the backend steps the simulation on its own; the browser view at
# http://localhost:8000/simulator_home is a read-only observer. "run N" makes
# progress whether or not a browser tab is open or focused, and refreshing
# the page just re-attaches at the live step.

set -euo pipefail

SESSION="gen_agents"
WIN_FRONTEND="gen-frontend"
WIN_BACKEND="gen-backend"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
FRONTEND_LOG="${LOG_DIR}/frontend_console_${STAMP}.log"
BACKEND_LOG="${LOG_DIR}/backend_console_${STAMP}.log"
PROMPT_LOG="${LOG_DIR}/backend_prompt_pairs_${STAMP}.jsonl"

if ! command -v tmux >/dev/null 2>&1; then
  echo "error: tmux is not installed. Install it first (apt-get install -y tmux)." >&2
  exit 1
fi

# Build a per-window prefix that sources .env if present, so each window
# inherits API keys regardless of the tmux server's parent environment.
ENV_PREFIX=""
if [ -f "${REPO_ROOT}/.env" ]; then
  ENV_PREFIX="set -a; . ${REPO_ROOT}/.env; set +a; "
fi

# REVERIE_PROMPT_LOG is exported after the .env sourcing so the per-launch
# path always wins.
FRONTEND_CMD="${ENV_PREFIX}cd ${REPO_ROOT}/environment/frontend_server && python manage.py runserver 127.0.0.1:8000"
BACKEND_CMD="${ENV_PREFIX}export REVERIE_PROMPT_LOG=${PROMPT_LOG}; cd ${REPO_ROOT}/reverie/backend_server && python reverie.py"

# Mirror a window's full output to a log file. Started before the command is
# sent so the transcript captures it from the first line.
start_window_log() {  # $1 = tmux target window, $2 = log file
  tmux pipe-pane -t "$1" -o "cat >> '$2'"
}

# ---------------------------------------------------------------------------
# Mode A: already inside a tmux session -> add windows to the current one.
# ---------------------------------------------------------------------------
if [ -n "${TMUX:-}" ]; then
  current_session=$(tmux display-message -p '#S')

  for w in "$WIN_FRONTEND" "$WIN_BACKEND"; do
    if tmux list-windows -t "$current_session" -F '#W' | grep -qx "$w"; then
      echo "Window '$w' already exists in session '$current_session'."
      echo "Clean up first:"
      echo "    tmux kill-window -t ${current_session}:${WIN_FRONTEND}"
      echo "    tmux kill-window -t ${current_session}:${WIN_BACKEND}"
      exit 0
    fi
  done

  tmux new-window -t "$current_session" -n "$WIN_FRONTEND"
  start_window_log "${current_session}:${WIN_FRONTEND}" "$FRONTEND_LOG"
  tmux send-keys  -t "${current_session}:${WIN_FRONTEND}" "$FRONTEND_CMD" C-m

  tmux new-window -t "$current_session" -n "$WIN_BACKEND"
  start_window_log "${current_session}:${WIN_BACKEND}" "$BACKEND_LOG"
  tmux send-keys  -t "${current_session}:${WIN_BACKEND}" "$BACKEND_CMD" C-m

  echo "Opened two windows in session '${current_session}':"
  echo "  ${WIN_FRONTEND}: Django dev server on 127.0.0.1:8000"
  echo "  ${WIN_BACKEND}:  reverie.py REPL (use 'run N', 'save', 'fin')"
  echo ""
  echo "Logs for this launch:"
  echo "  console (frontend): ${FRONTEND_LOG}"
  echo "  console (backend):  ${BACKEND_LOG}"
  echo "  prompt pairs:       ${PROMPT_LOG}"
  echo ""
  echo "Switch:        Ctrl-b n / Ctrl-b p (cycle)   Ctrl-b w (picker)"
  echo "Clean up:"
  echo "    tmux kill-window -t ${current_session}:${WIN_FRONTEND}"
  echo "    tmux kill-window -t ${current_session}:${WIN_BACKEND}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Mode B: not inside tmux -> create a detached 'gen_agents' session.
# ---------------------------------------------------------------------------
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "Attach with:  tmux attach -t $SESSION"
  echo "Kill with:    tmux kill-session -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n frontend
start_window_log "${SESSION}:frontend" "$FRONTEND_LOG"
tmux send-keys   -t "${SESSION}:frontend" "$FRONTEND_CMD" C-m

tmux new-window  -t "$SESSION" -n backend
start_window_log "${SESSION}:backend" "$BACKEND_LOG"
tmux send-keys   -t "${SESSION}:backend" "$BACKEND_CMD" C-m

echo "Started tmux session '$SESSION'."
echo "  frontend window: Django dev server on 127.0.0.1:8000"
echo "  backend  window: reverie.py REPL (use 'run N', 'save', 'fin')"
echo ""
echo "Logs for this launch:"
echo "  console (frontend): ${FRONTEND_LOG}"
echo "  console (backend):  ${BACKEND_LOG}"
echo "  prompt pairs:       ${PROMPT_LOG}"
echo ""
echo "Attach:        tmux attach -t $SESSION"
echo "Switch window: Ctrl-b n (next) / Ctrl-b p (prev)"
echo "Detach:        Ctrl-b d"
echo "Kill session:  tmux kill-session -t $SESSION"
