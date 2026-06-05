#!/bin/bash
# Run this on the REMOTE box (e.g. inside the vast.ai container, after you've
# cloned the repo and installed requirements.txt). It launches the Django
# frontend and the reverie backend in a single tmux session so both stay
# alive across ssh disconnects.
#
# Layout:
#   tmux session "gen_agents"
#     window 0 "frontend" -> environment/frontend_server, runserver on :8000
#     window 1 "backend"  -> reverie/backend_server, reverie.py (interactive)
#
# Once running, attach with:
#   tmux attach -t gen_agents
# and switch windows with Ctrl-b n / Ctrl-b p. Detach with Ctrl-b d.
#
# To see the frontend locally, run on your laptop:
#   ./ssh2tunnel.sh "ssh -p PORT user@host ..."
# then execute the command it prints (or `eval` it). The Django dev server
# will then be reachable at http://localhost:8000/simulator_home.
#
# Re-running this script is a no-op if the session already exists.

set -euo pipefail

SESSION="gen_agents"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

if ! command -v tmux >/dev/null 2>&1; then
  echo "error: tmux is not installed. Install it first (apt-get install -y tmux)." >&2
  exit 1
fi

# Make API keys available to both windows. Both processes inherit them via
# the tmux session's environment.
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1090,SC1091
  . "${REPO_ROOT}/.env"
  set +a
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

# Bind Django to loopback only; the tunnel does the rest.
FRONTEND_CMD="cd ${REPO_ROOT}/environment/frontend_server && python manage.py runserver 127.0.0.1:8000"
BACKEND_CMD="cd ${REPO_ROOT}/reverie/backend_server && python reverie.py"

tmux new-session  -d -s "$SESSION" -n frontend
tmux send-keys    -t "${SESSION}:frontend" "$FRONTEND_CMD" C-m

tmux new-window   -t "$SESSION" -n backend
tmux send-keys    -t "${SESSION}:backend" "$BACKEND_CMD" C-m

echo "Started tmux session '$SESSION'."
echo "  frontend window: Django dev server on 127.0.0.1:8000"
echo "  backend  window: reverie.py REPL (use 'run N', 'save', 'fin')"
echo ""
echo "Attach:        tmux attach -t $SESSION"
echo "Switch window: Ctrl-b n (next) / Ctrl-b p (prev)"
echo "Detach:        Ctrl-b d"
