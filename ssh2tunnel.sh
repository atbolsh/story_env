#!/bin/bash
# Usage: ./ssh2tunnel.sh "ssh -p PORT user@host [-L ...]" [extra_port ...]
#
# Echoes the input ssh command with one or more -L port forwards appended,
# so that when you connect to a remote GPU box (e.g. vast.ai) you can also
# reach servers running on it from your local browser.
#
# By default we forward :8000 -> remote localhost:8000, which is where the
# Django frontend (environment/frontend_server) binds. With this tunnel
# running, http://localhost:8000/simulator_home in your local browser hits
# the remote frontend, which in turn drives the remote reverie backend via
# the shared filesystem.
#
# Extra ports may be passed as additional arguments. Each is either:
#   - a bare port (8001)            -> forwarded as 8001:localhost:8001
#   - a full local:host:remote spec -> passed straight through to ssh -L
#
# Example:
#   ./ssh2tunnel.sh "ssh -p 40230 root@174.78.228.101 -L 8080:localhost:8080"
#   # -> ssh -p 40230 root@174.78.228.101 -L 8080:localhost:8080 \
#   #        -L 8000:localhost:8000
#
#   ./ssh2tunnel.sh "ssh -p 40230 root@174.78.228.101" 8001 9000:localhost:9000
#   # -> ssh -p 40230 root@174.78.228.101 \
#   #        -L 8000:localhost:8000 -L 8001:localhost:8001 -L 9000:localhost:9000
#
# To run the result directly: eval "$(./ssh2tunnel.sh '...')"

ssh_cmd="$1"
shift

forwards=("-L" "8000:localhost:8000")
for p in "$@"; do
  if [[ "$p" == *:* ]]; then
    forwards+=("-L" "$p")
  else
    forwards+=("-L" "${p}:localhost:${p}")
  fi
done

echo "${ssh_cmd} ${forwards[*]}"
