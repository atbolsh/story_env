#!/usr/bin/env bash
#
# nams_db.sh -- per-persona Neo4j (Community Edition) lifecycle for the
# generative_agents NAMS harnesses.
#
# Why per-persona containers: NAMS long-term memory is a graph, and each
# character's semantic model must be HIS OWN -- never mutually visible to
# another character. Neo4j Community Edition supports only one database per
# instance, so isolation is achieved at the *instance* level: one Community
# container + one named volume per NAMS persona, each on its own bolt port.
# This keeps the whole stack on the free GPLv3 Community Edition (no Neo4j
# Enterprise license flag, no account) and still gives clean per-persona
# .dump save files.
#
# This script is the single entry point for everything except starting /
# stopping the remote server host itself (that's your job). It:
#   * launches per-persona containers (`up`)
#   * stops / removes them (`down`)
#   * saves a per-persona .dump non-destructively (`save`)
#   * saves + stops a persona's container, ready for you to spin the server
#     down (`teardown`)
#   * reinstates a persona from a .dump file (`load`)
#   * lists registered personas + container status + dump files (`list`)
#
# When only ONE persona in a simulation is on NAMS (e.g. midnight_test.sh
# with just Klaus Mueller), you do NOT need this script at all -- the single
# `docker compose up -d neo4j` instance is enough and isolation is trivially
# satisfied. Use this script only when TWO OR MORE personas are NAMS-backed
# in the same run.
#
# Save-file format: Neo4j's native `neo4j-admin database dump` output -- a
# binary archive of the store files. One file per persona, named
#   nams_dumps/<sanitized_persona>__<UTC-timestamp>.dump
# Restored with `neo4j-admin database load --overwrite-destination`. This is
# the documented Neo4j backup/restore format; it is not human-readable and is
# only meaningful to neo4j-admin on the same (or newer) Neo4j major version.
#
# Usage:
#   scripts/nams_db.sh up "Klaus Mueller" ["Isabella Rodriguez" ...]
#   scripts/nams_db.sh down "Klaus Mueller" ["Isabella Rodriguez" ...] [--purge]
#   scripts/nams_db.sh save "Klaus Mueller" [output_dir]
#   scripts/nams_db.sh teardown "Klaus Mueller" [output_dir] [--purge]
#   scripts/nams_db.sh load "Klaus Mueller" <dump_file>
#   scripts/nams_db.sh list
#
# Env overrides:
#   NEO4J_IMAGE       (default neo4j:5.20)
#   NEO4J_PASSWORD    (default password)  -- must match docker-compose.yml
#   NAMS_BOLT_BASE    (default 7688)      -- first per-persona bolt port
#   NAMS_HTTP_BASE    (default 8474)      -- first per-persona browser port
#   NAMS_REGISTRY     (default ./nams_databases.json)
#   NAMS_DUMPS_DIR    (default ./nams_dumps)

set -euo pipefail

# ---- defaults ---------------------------------------------------------------

NEO4J_IMAGE="${NEO4J_IMAGE:-neo4j:5.20}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"
NAMS_BOLT_BASE="${NAMS_BOLT_BASE:-7688}"
NAMS_HTTP_BASE="${NAMS_HTTP_BASE:-8474}"
NAMS_REGISTRY="${NAMS_REGISTRY:-$(pwd)/nams_databases.json}"
NAMS_DUMPS_DIR="${NAMS_DUMPS_DIR:-$(pwd)/nams_dumps}"

# docker-compose single-instance bolt port (used by `list` to also report the
# shared instance, and as a guard so `up` never reuses 7687).
COMPOSE_BOLT_PORT=7687

# ---- helpers ----------------------------------------------------------------

log() { printf '[nams_db] %s\n' "$*" >&2; }
die() { printf '[nams_db] ERROR: %s\n' "$*" >&2; exit 1; }

# Sanitize a persona name into a docker-safe slug: lowercase, [a-z0-9_].
# "Klaus Mueller" -> "klaus_mueller"; "Isabella Rodriguez" -> "isabella_rodriguez".
sanitize() {
  local name="$1"
  local slug
  slug="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')"
  slug="${slug// /_}"
  slug="$(printf '%s' "$slug" | tr -cd 'a-z0-9_')"
  [[ -n "$slug" ]] || die "persona name '$name' sanitizes to empty slug"
  printf '%s' "$slug"
}

container_name_for() { printf 'nams_%s' "$(sanitize "$1")"; }
volume_name_for()    { printf 'nams_%s_data' "$(sanitize "$1")"; }

# Read a JSON field from the registry for a persona. Echoes empty string if
# the persona or field is missing. Requires python3 for JSON parsing (the
# repo already requires python3.10+ for the harness itself).
reg_get() {
  local persona="$1" field="$2"
  [[ -f "$NAMS_REGISTRY" ]] || { printf ''; return; }
  python3 - "$NAMS_REGISTRY" "$persona" "$field" <<'PY'
import json, sys
path, persona, field = sys.argv[1:4]
try:
    with open(path) as f:
        reg = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(0)
entry = reg.get(persona) or {}
val = entry.get(field)
if val is not None:
    print(val)
PY
}

# Write/update a persona's entry in the registry.
reg_set() {
  local persona="$1" container="$2" port="$3" http_port="$4"
  python3 - "$NAMS_REGISTRY" "$persona" "$container" "$port" "$http_port" <<'PY'
import json, os, sys
path, persona, container, port, http_port = sys.argv[1:6]
try:
    with open(path) as f:
        reg = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    reg = {}
reg[persona] = {
    "container": container,
    "port": int(port),
    "http_port": int(http_port),
    "host": "localhost",
}
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "w") as f:
    json.dump(reg, f, indent=2, sort_keys=True)
print(f"[nams_db] registry updated: {path}")
PY
}

reg_remove() {
  local persona="$1"
  [[ -f "$NAMS_REGISTRY" ]] || return 0
  python3 - "$NAMS_REGISTRY" "$persona" <<'PY'
import json, os, sys
path, persona = sys.argv[1:3]
try:
    with open(path) as f:
        reg = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(0)
if persona in reg:
    del reg[persona]
    with open(path, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)
    print(f"[nams_db] registry entry removed: {persona}")
PY
}

# Pick the next free bolt port >= NAMS_BOLT_BASE, skipping any port already
# claimed in the registry OR by the compose single instance.
next_port() {
  local used
  used="$(python3 - "$NAMS_REGISTRY" "$COMPOSE_BOLT_PORT" <<'PY'
import json, sys
path, compose_port = sys.argv[1:3]
used = {int(compose_port)}
try:
    with open(path) as f:
        reg = json.load(f)
    for v in reg.values():
        p = v.get("port")
        if p is not None: used.add(int(p))
except (FileNotFoundError, json.JSONDecodeError):
    pass
print(" ".join(str(p) for p in sorted(used)))
PY
)"
  local base="$NAMS_BOLT_BASE"
  while printf '%s\n' "$used" | tr ' ' '\n' | grep -qx "$base"; do
    base=$((base + 1))
  done
  printf '%s' "$base"
}

# Wait for bolt to answer on a port (max ~60s). Non-fatal if it times out --
# the caller will surface a clearer error.
wait_for_bolt() {
  local port="$1"
  local i=0
  while (( i < 60 )); do
    if python3 - <<PY
import socket, sys
s = socket.socket(); s.settimeout(2)
try:
    s.connect(("localhost", $port)); s.close(); sys.exit(0)
except Exception:
    sys.exit(1)
PY
    then
      return 0
    fi
    sleep 1; i=$((i + 1))
  done
  return 1
}

# ---- subcommands ------------------------------------------------------------

cmd_up() {
  [[ $# -ge 1 ]] || die "up needs at least one persona name"
  local personas=("$@")
  for name in "${personas[@]}"; do
    local cont vol port http_port
    cont="$(container_name_for "$name")"
    vol="$(volume_name_for "$name")"
    # If already running, skip.
    if docker ps --format '{{.Names}}' | grep -qx "$cont"; then
      log "already running: $name -> container $cont"
      continue
    fi
    # If the container exists but is stopped, start it.
    if docker ps -a --format '{{.Names}}' | grep -qx "$cont"; then
      log "starting existing stopped container: $cont"
      docker start "$cont" >/dev/null
    else
      port="$(next_port)"
      http_port=$((NAMS_HTTP_BASE + (port - NAMS_BOLT_BASE)))
      log "creating container $cont for '$name' on bolt :$port http :$http_port"
      docker run -d \
        --name "$cont" \
        -p "${port}:7687" \
        -p "${http_port}:7474" \
        -e "NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}" \
        -e 'NEO4J_PLUGINS=["apoc"]' \
        -e APOC_PROCEDURE_ENABLE=true \
        -e NEO4J_server_memory_heap_initial__size=512m \
        -e NEO4J_server_memory_heap_max__size=2G \
        -v "${vol}:/data" \
        "$NEO4J_IMAGE" >/dev/null
      reg_set "$name" "$cont" "$port" "$http_port"
    fi
    # Wait for bolt, regardless of whether we created or just started.
    local p; p="$(reg_get "$name" port)"
    if [[ -n "$p" ]]; then
      if wait_for_bolt "$p"; then
        log "bolt ready for $name on :$p"
      else
        log "WARNING: bolt on :$p did not answer in 60s -- check 'docker logs $cont'"
      fi
    fi
  done
}

cmd_down() {
  [[ $# -ge 1 ]] || die "down needs at least one persona name"
  local purge=0
  local personas=()
  for a in "$@"; do
    if [[ "$a" == "--purge" ]]; then purge=1; else personas+=("$a"); fi
  done
  [[ ${#personas[@]} -ge 1 ]] || die "down needs at least one persona name"
  for name in "${personas[@]}"; do
    local cont vol
    cont="$(container_name_for "$name")"
    vol="$(volume_name_for "$name")"
    if docker ps -a --format '{{.Names}}' | grep -qx "$cont"; then
      log "stopping + removing container $cont"
      docker stop "$cont" >/dev/null 2>&1 || true
      docker rm "$cont" >/dev/null 2>&1 || true
    else
      log "no container for '$name' ($cont) -- nothing to stop"
    fi
    if (( purge )); then
      if docker volume ls --format '{{.Name}}' | grep -qx "$vol"; then
        log "--purge: removing volume $vol (DATA LOSS for $name)"
        docker volume rm "$vol" >/dev/null
      fi
      reg_remove "$name"
    fi
  done
}

# Dump a stopped container's neo4j db to a host-side .dump file via a
# sidecar container that mounts the same data volume + an output dir.
# Args: persona_name out_dir
# Echoes the path of the produced .dump file.
do_dump() {
  local name="$1" out_dir="$2"
  local cont vol slug ts out_name out_path
  cont="$(container_name_for "$name")"
  vol="$(volume_name_for "$name")"
  slug="$(sanitize "$name")"
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  out_name="${slug}__${ts}.dump"
  mkdir -p "$out_dir"
  out_path="${out_dir}/${out_name}"
  docker volume inspect "$vol" >/dev/null 2>&1 || die "no volume $vol for '$name' -- has it ever been `up`?"
  log "stopping $cont for consistent dump"
  docker stop "$cont" >/dev/null
  log "dumping via sidecar: $vol -> $out_path"
  docker run --rm \
    -v "${vol}:/data" \
    -v "${out_dir}:/out" \
    "$NEO4J_IMAGE" \
    neo4j-admin database dump --database=neo4j \
      --to="/out/${out_name}" --overwrite-destination
  # Restart the container so the persona is back online after a `save`.
  log "restarting $cont"
  docker start "$cont" >/dev/null
  # Sanity: the dump file must be non-empty.
  [[ -s "$out_path" ]] || die "dump produced empty file $out_path"
  log "dump ok: $out_path ($(du -h "$out_path" | cut -f1))"
  printf '%s' "$out_path"
}

cmd_save() {
  [[ $# -ge 1 ]] || die "save needs a persona name"
  local name="$1" out_dir="${2:-$NAMS_DUMPS_DIR}"
  do_dump "$name" "$out_dir"
}

cmd_teardown() {
  [[ $# -ge 1 ]] || die "teardown needs a persona name"
  local name="$1" out_dir="${2:-$NAMS_DUMPS_DIR}" purge=0
  shift
  for a in "$@"; do
    if [[ "$a" == "--purge" ]]; then purge=1; fi
  done
  local dump_path
  dump_path="$(do_dump "$name" "$out_dir")"
  log "teardown: dump verified at $dump_path"
  log "teardown: stopping + removing container (volume retained unless --purge)"
  local cont vol
  cont="$(container_name_for "$name")"
  vol="$(volume_name_for "$name")"
  docker stop "$cont" >/dev/null 2>&1 || true
  docker rm "$cont" >/dev/null 2>&1 || true
  if (( purge )); then
    if docker volume ls --format '{{.Name}}' | grep -qx "$vol"; then
      log "--purge: removing volume $vol"
      docker volume rm "$vol" >/dev/null
    fi
    reg_remove "$name"
  fi
  log "teardown complete for '$name'. Safe to spin the server down now."
  log "Restore later with: scripts/nams_db.sh load '$name' '$dump_path'"
}

cmd_load() {
  [[ $# -ge 2 ]] || die "load needs <persona> <dump_file>"
  local name="$1" dump_file="$2"
  [[ -f "$dump_file" ]] || die "dump file not found: $dump_file"
  local cont vol slug dump_in_container
  cont="$(container_name_for "$name")"
  vol="$(volume_name_for "$name")"
  slug="$(sanitize "$name")"
  # Ensure the persona's container exists (create if needed) and is stopped
  # for the load. We bring it up via cmd_up, then stop it.
  if ! docker ps -a --format '{{.Names}}' | grep -qx "$cont"; then
    log "no container for '$name' -- creating via `up` first"
    cmd_up "$name"
  fi
  log "stopping $cont for load"
  docker stop "$cont" >/dev/null
  dump_in_container="/out/$(basename "$dump_file")"
  log "loading $dump_file -> $vol via sidecar"
  docker run --rm \
    -v "${vol}:/data" \
    -v "$(dirname "$(abspath "$dump_file")"):/out:ro" \
    "$NEO4J_IMAGE" \
    neo4j-admin database load --database=neo4j \
      --from="$dump_in_container" --overwrite-destination
  log "restarting $cont"
  docker start "$cont" >/dev/null
  local p; p="$(reg_get "$name" port)"
  if [[ -n "$p" ]] && wait_for_bolt "$p"; then
    log "bolt ready for $name on :$p -- load complete"
  else
    log "WARNING: bolt did not come back after load -- check 'docker logs $cont'"
  fi
}

abspath() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    *)  printf '%s/%s' "$(pwd)" "$1" ;;
  esac
}

cmd_list() {
  echo "=== NAMS per-persona registry ($NAMS_REGISTRY) ==="
  if [[ -f "$NAMS_REGISTRY" ]]; then
    python3 - "$NAMS_REGISTRY" <<'PY'
import json, subprocess, sys
path = sys.argv[1]
with open(path) as f:
    reg = json.load(f)
if not reg:
    print("  (empty)")
for name, e in sorted(reg.items()):
    cont = e.get("container", "?")
    port = e.get("port", "?")
    http_port = e.get("http_port", "?")
    state = "?"
    try:
        state = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", cont],
            capture_output=True, text=True, check=False
        ).stdout.strip() or "absent"
    except Exception:
        pass
    print(f"  {name:<24} container={cont:<32} bolt=:{port}  http=:{http_port}  state={state}")
PY
  else
    echo "  (no registry -- use `up` to launch per-persona containers)"
  fi
  echo
  echo "=== docker-compose single instance (port $COMPOSE_BOLT_PORT) ==="
  if docker ps --format '{{.Names}}\t{{.Ports}}' | grep -q "$COMPOSE_BOLT_PORT"; then
    docker ps --format '  {{.Names}}\t{{.Ports}}' | grep "$COMPOSE_BOLT_PORT" || true
  else
    echo "  (not running -- fine if all NAMS personas are in the registry above)"
  fi
  echo
  echo "=== dump files ($NAMS_DUMPS_DIR) ==="
  if [[ -d "$NAMS_DUMPS_DIR" ]]; then
    local found=0
    while IFS= read -r f; do
      printf '  %s\t%s\n' "$(du -h "$f" | cut -f1)" "$f"
      found=1
    done < <(find "$NAMS_DUMPS_DIR" -name '*.dump' -type f 2>/dev/null | sort)
    (( found )) || echo "  (none)"
  else
    echo "  (no dumps dir yet)"
  fi
}

# ---- dispatch ---------------------------------------------------------------

cmd="${1:-}"; shift || true
case "$cmd" in
  up)        cmd_up "$@" ;;
  down)      cmd_down "$@" ;;
  save)      cmd_save "$@" ;;
  teardown)  cmd_teardown "$@" ;;
  load)      cmd_load "$@" ;;
  list)      cmd_list "$@" ;;
  ""|-h|--help|help)
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *) die "unknown subcommand '$cmd' (try: $0 --help)" ;;
esac
