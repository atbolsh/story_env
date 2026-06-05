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

set -euo pipefail

SESSION="gen_agents"
WIN_FRONTEND="gen-frontend"
WIN_BACKEND="gen-backend"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

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

FRONTEND_CMD="${ENV_PREFIX}cd ${REPO_ROOT}/environment/frontend_server && python manage.py runserver 127.0.0.1:8000"
BACKEND_CMD="${ENV_PREFIX}cd ${REPO_ROOT}/reverie/backend_server && python reverie.py"

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
  tmux send-keys  -t "${current_session}:${WIN_FRONTEND}" "$FRONTEND_CMD" C-m

  tmux new-window -t "$current_session" -n "$WIN_BACKEND"
  tmux send-keys  -t "${current_session}:${WIN_BACKEND}" "$BACKEND_CMD" C-m

  echo "Opened two windows in session '${current_session}':"
  echo "  ${WIN_FRONTEND}: Django dev server on 127.0.0.1:8000"
  echo "  ${WIN_BACKEND}:  reverie.py REPL (use 'run N', 'save', 'fin')"
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
tmux send-keys   -t "${SESSION}:frontend" "$FRONTEND_CMD" C-m

tmux new-window  -t "$SESSION" -n backend
tmux send-keys   -t "${SESSION}:backend" "$BACKEND_CMD" C-m

echo "Started tmux session '$SESSION'."
echo "  frontend window: Django dev server on 127.0.0.1:8000"
echo "  backend  window: reverie.py REPL (use 'run N', 'save', 'fin')"
echo ""
echo "Attach:        tmux attach -t $SESSION"
echo "Switch window: Ctrl-b n (next) / Ctrl-b p (prev)"
echo "Detach:        Ctrl-b d"
echo "Kill session:  tmux kill-session -t $SESSION"
