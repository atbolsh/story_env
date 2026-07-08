#!/bin/bash
# midnight_test.sh -- unattended overnight NAMS mixed-harness benchmark.
#
# Runs ONE FULL IN-GAME DAY (sec_per_step=10 -> 86400/10 = 8640 steps) of
# base_the_ville_isabella_maria_klaus, twice, on the Gemma 4 E4B harness:
#
#     1. gemma4-e4b_klaus-nams-spacy-only
#        Isabella + Maria run the legacy JSON memory; Klaus (the gentrification
#        scholar) runs on NAMS with the spaCy + GLiNER + GLiREL-only entity
#        extraction pipeline (no LLM in the extractor -- deterministic).
#
#     2. gemma4-e4b_klaus-nams-llm-extraction
#        Same mixed setup, but Klaus's NAMS extractor additionally pipes raw
#        short-term messages through the Gemma 4 E4B chat LLM for richer
#        fact extraction.
#
# Both runs share one local Neo4j (docker compose up -d neo4j). Each run
# starts by *translating* Klaus's JSON bootstrap_memory into the graph
# (reverie.py --import-nams-only --force-import), so the profile is fresh
# in Neo4j before the sim steps begin; then it launches the headless sim.
#
# Every LLM call -- cognitive-module calls for all three personas AND the
# NAMS extraction-LLM calls for Klaus in run 2 -- is logged with full
# request/response context to backend_prompt_pairs.jsonl (+ .txt), exactly
# as in the interactive midnight runs, via REVERIE_PROMPT_LOG.
#
# Usage, on the remote box, from the repo root:
#
#     bash midnight_test.sh
#
# The script immediately re-launches itself under nohup and returns your
# shell -- it is safe to disconnect ssh. Follow along with:
#
#     tail -f logs/midnight_orchestrator_<stamp>.log
#
# For each run it:
#   1. ensures the local Neo4j is up (docker compose up -d neo4j) and bolt
#      answers,
#   2. translates Klaus's JSON profile into Neo4j
#      (reverie.py --import-nams-only --force-import) -- fail-fast if the
#      import barfs, so we don't burn a 9-hour run on a broken graph,
#   3. opens a tmux session with the frontend (Django, for browser
#      observation) and backend (headless reverie.py --steps 8640) windows,
#   4. polls temp_storage/curr_step.json until the day completes, the run
#      crashes, the backend dies/stalls, or MAX_RUN_SECONDS elapses,
#   5. on completion/crash/timeout sends Ctrl-C (which the headless
#      reverie.py turns into a partial save), kills the tmux session, and
#      moves on.
#
# In the morning you should find:
#   logs/midnight_<run>_<stamp>/          one per run, containing
#       frontend_console.log              Django window transcript
#       backend_console.log               reverie.py window transcript
#       backend_prompt_pairs.jsonl        exact prompt/response pairs (every
#                                         LLM call, including NAMS extraction)
#       backend_prompt_pairs.txt          same, human-readable
#       import_console.log                the JSON->NAMS translation transcript
#       klaus_memories_<run>.dump         Klaus's NAMS memory graph, saved via
#                                         scripts/nams_baremetal_db.sh save
#                                         (Neo4j native .dump binary archive;
#                                         reload with nams_baremetal_db.sh load)
#       db_save.log                       save/dump transcript
#   logs/midnight_db_wipe_<stamp>.log     wipe-between-runs transcript
#   logs/midnight_summary_<stamp>.txt     one status line per run
#   environment/frontend_server/storage/midnight_<run>_<stamp>/
#                                        the saved sims (forkable/replayable)
#
# Cross-run isolation: the two runs share one local Neo4j instance, but
# (a) each run starts with --force-import, which wipes Klaus's session and
# re-imports his JSON bootstrap, and (b) between runs the orchestrator calls
# scripts/nams_baremetal_db.sh wipe to drop the neo4j db files and recreate
# an empty db, so run 2 starts on a graph that has zero leftover nodes from
# run 1. Each run's accumulated Klaus memories are saved to that run's logs
# dir as klaus_memories_<run>.dump BEFORE the wipe, so both runs' memories
# survive for offline analysis.
#
# Secrets: API keys are sourced from ${REPO_ROOT}/.env, never hardcoded here.
#
# Shakedown overrides (shorter runs for a smoke test):
#     MIDNIGHT_STEPS=50 MIDNIGHT_RUNS="gemma4-e4b_klaus-nams-spacy-only" bash midnight_test.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

STAMP="${MIDNIGHT_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
export MIDNIGHT_STAMP="$STAMP"

# ----------------------------------------------------------------- settings
FORK_SIM="${MIDNIGHT_FORK:-base_the_ville_isabella_maria_klaus}"
# Each run spec is "<run_label>|<nams-extraction-mode>".
#   no-llm      -> spaCy + GLiNER + GLiREL only
#   harness-llm -> + Gemma 4 E4B chat LLM as the NAMS extraction LLM stage
# The LLM harness for *every* persona (JSON and NAMS alike) is gemma4-e4b;
# only Klaus runs on NAMS memory, via --nams-personas.
RUNS=(${MIDNIGHT_RUNS:-gemma4-e4b_klaus-nams-spacy-only|no-llm gemma4-e4b_klaus-nams-llm-extraction|harness-llm})
NAMS_PERSONAS="${MIDNIGHT_NAMS_PERSONAS:-Klaus Mueller}"
BASE_HARNESS="${MIDNIGHT_HARNESS:-gemma4-e4b}"
STEPS="${MIDNIGHT_STEPS:-8640}"        # one in-game day at sec_per_step=10
MAX_RUN_SECONDS=$((9 * 3600))          # per-run wall-clock budget before we cut it
STALL_SECONDS="${MIDNIGHT_STALL_SECONDS:-1200}" # no curr_step progress => hung
POLL_SECONDS=60
IMPORT_TIMEOUT=900                     # JSON->NAMS translation budget (no LLM)
FIN_TIMEOUT=600                        # wait after Ctrl-C for the partial save

LOG_ROOT="${REPO_ROOT}/logs"
TEMP_STORAGE="${REPO_ROOT}/environment/frontend_server/temp_storage"
SUMMARY="${LOG_ROOT}/midnight_summary_${STAMP}.txt"

# ------------------------------------------------- self-detach under nohup
# EARLY_WATCH_SECONDS: after detaching, the launcher follows the orchestrator
# log in the FOREGROUND for this long so early failures (bad import, env, bolt)
# surface in your terminal immediately instead of being discovered the next
# morning. Set to 0 to skip watching and return the shell instantly.
EARLY_WATCH_SECONDS="${MIDNIGHT_EARLY_WATCH_SECONDS:-240}"
if [ -z "${MIDNIGHT_DETACHED:-}" ]; then
  mkdir -p "$LOG_ROOT"
  ORCH_LOG="${LOG_ROOT}/midnight_orchestrator_${STAMP}.log"
  # setsid (when available) puts the orchestrator in its own session so a
  # Ctrl-C on the foreground watcher below can't reach it -- the run keeps
  # going in the background no matter how you stop watching. stdin from
  # /dev/null so it never blocks on a read.
  if command -v setsid >/dev/null 2>&1; then
    MIDNIGHT_DETACHED=1 setsid bash "$0" "$@" >"$ORCH_LOG" 2>&1 </dev/null &
  else
    MIDNIGHT_DETACHED=1 nohup bash "$0" "$@" >"$ORCH_LOG" 2>&1 </dev/null &
  fi
  orch_pid=$!
  echo "Detached orchestrator (pid ${orch_pid}). Safe to disconnect."
  echo "Follow along:  tail -f ${ORCH_LOG}"
  echo "Summary file:  ${SUMMARY}"

  if [ "$EARLY_WATCH_SECONDS" -le 0 ] 2>/dev/null; then
    exit 0
  fi

  echo
  echo "Watching early startup (up to ${EARLY_WATCH_SECONDS}s) so import/env/bolt"
  echo "failures show up here right away. Ctrl-C stops watching only -- the run"
  echo "continues in the background."
  echo "----------------------------------------------------------------------"
  trap 'echo; echo ">>> Stopped watching; run continues (pid '"${orch_pid}"'). Follow: tail -f '"${ORCH_LOG}"'"; exit 0' INT

  early_deadline=$(( $(date +%s) + EARLY_WATCH_SECONDS ))
  last_size=0
  early_done=0
  while [ "$(date +%s)" -lt "$early_deadline" ]; do
    if [ -f "$ORCH_LOG" ]; then
      cur_size=$(wc -c < "$ORCH_LOG" 2>/dev/null || echo 0)
      if [ "$cur_size" -gt "$last_size" ]; then
        tail -c +"$((last_size + 1))" "$ORCH_LOG"
        last_size=$cur_size
      fi
      # Decisive FAILURE markers -> surface loudly and stop watching.
      if grep -qE "IMPORT FAILED|: FAILED \(|bolt never answered|Traceback \(most recent|^error:|invalid option" "$ORCH_LOG" 2>/dev/null; then
        echo "----------------------------------------------------------------------"
        echo ">>> EARLY FAILURE detected (see above). This stage aborted."
        echo ">>> Full log: ${ORCH_LOG}"
        echo ">>> Summary:  ${SUMMARY}"
        early_done=1
        break
      fi
      # Decisive SUCCESS markers -> the sim is actually stepping; safe to leave.
      if grep -qE "headless sim launched|step [0-9]+/" "$ORCH_LOG" 2>/dev/null; then
        echo "----------------------------------------------------------------------"
        echo ">>> Startup healthy: import passed and the sim is stepping."
        echo ">>> Detaching watcher; run continues. Follow: tail -f ${ORCH_LOG}"
        early_done=1
        break
      fi
    fi
    # Orchestrator already gone (finished or died)?
    if ! kill -0 "$orch_pid" 2>/dev/null; then
      if [ -f "$ORCH_LOG" ]; then
        cur_size=$(wc -c < "$ORCH_LOG" 2>/dev/null || echo 0)
        [ "$cur_size" -gt "$last_size" ] && tail -c +"$((last_size + 1))" "$ORCH_LOG"
      fi
      echo "----------------------------------------------------------------------"
      echo ">>> Orchestrator exited early. See ${ORCH_LOG} and ${SUMMARY}."
      early_done=1
      break
    fi
    sleep 2
  done
  if [ "$early_done" -eq 0 ]; then
    echo "----------------------------------------------------------------------"
    echo ">>> Still starting after ${EARLY_WATCH_SECONDS}s (likely a first-run"
    echo ">>> model download). Detaching watcher; run continues in background."
    echo ">>> Follow: tail -f ${ORCH_LOG}"
  fi
  exit 0
fi

# --------------------------------------------------------------- preflight
log() { echo "[$(date +%H:%M:%S)] $*"; }

if ! command -v tmux >/dev/null 2>&1; then
  echo "error: tmux is not installed (apt-get install -y tmux)." >&2
  exit 1
fi
if [ ! -d "${REPO_ROOT}/environment/frontend_server/storage/${FORK_SIM}" ]; then
  echo "error: fork sim ${FORK_SIM} not found in storage/." >&2
  exit 1
fi
# Docker is OPTIONAL. The preferred path is `docker compose up -d neo4j` for
# the single shared instance, but on hosts where the docker daemon isn't
# available (e.g. inside a container running Neo4j bare-metal), we just rely
# on whatever Neo4j is already listening on localhost:7687 -- the bolt
# connectivity check below is what actually gates us, not the daemon.
HAVE_DOCKER=0
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  HAVE_DOCKER=1
fi

# Source .env for OPENAI_API_KEY etc. (the headless backend + the tmux windows
# all re-source it via ENV_PREFIX).
ENV_PREFIX=""
if [ -f "${REPO_ROOT}/.env" ]; then
  ENV_PREFIX="set -a; . ${REPO_ROOT}/.env; set +a; "
  set -a; . "${REPO_ROOT}/.env"; set +a
fi

mkdir -p "$LOG_ROOT"
echo "midnight_test ${STAMP}: ${STEPS} steps each, harness=${BASE_HARNESS}, "
echo "  NAMS personas=${NAMS_PERSONAS}, runs: ${RUNS[*]}" > "$SUMMARY"

# Bring up the local Neo4j (idempotent) and wait for bolt to answer. The
# NAMS SDK connects to bolt://localhost:7687; without this, both the
# JSON->NAMS translation and the NAMS personas at runtime will fail.
# If docker is available, use the compose single instance; otherwise assume
# a bare-metal/external Neo4j is already running on :7687 (the bolt check
# below gates either path).
if [ "$HAVE_DOCKER" -eq 1 ]; then
  log "ensuring local Neo4j is up (docker compose up -d neo4j)..."
  docker compose up -d neo4j >/dev/null 2>&1 || true
else
  log "no docker daemon detected; assuming bare-metal Neo4j on localhost:7687."
fi
bolt_ok=0
for _ in $(seq 1 60); do
  if python3 -c "import neo4j,os; d=neo4j.GraphDatabase.driver(os.environ.get('NEO4J_URI','bolt://localhost:7687'),auth=(os.environ.get('NEO4J_USER','neo4j'),os.environ.get('NEO4J_PASSWORD','password'))); s=d.session(); s.run('RETURN 1').consume(); d.close()" 2>/dev/null; then
    bolt_ok=1; break
  fi
  sleep 5
done
if [ "$bolt_ok" -ne 1 ]; then
  log "error: Neo4j bolt never answered on localhost:7687."
  if [ "$HAVE_DOCKER" -eq 1 ]; then
    log "  Check: docker compose ps neo4j ; docker compose logs neo4j"
  else
    log "  Check: neo4j status ; tail /var/log/neo4j/neo4j.log"
  fi
  echo "ALL RUNS: FAILED (Neo4j bolt unreachable)" >> "$SUMMARY"
  exit 1
fi
log "Neo4j bolt is up."

# ----------------------------------------------------------------- helpers
wait_for_pattern() {  # $1=file $2=grep-pattern $3=timeout-seconds
  local file=$1 pattern=$2 timeout=$3 waited=0
  while ! grep -q "$pattern" "$file" 2>/dev/null; do
    sleep 2
    waited=$((waited + 2))
    [ "$waited" -ge "$timeout" ] && return 1
  done
  return 0
}

current_step() {
  python3 -c "import json; print(json.load(open('${TEMP_STORAGE}/curr_step.json'))['step'])" \
    2>/dev/null || echo "-1"
}

backend_running() {  # $1=session
  local pane_pid
  pane_pid=$(tmux display-message -p -t "$1:backend" '#{pane_pid}' 2>/dev/null) || return 1
  pgrep -P "$pane_pid" -f "reverie.py" >/dev/null 2>&1
}

kill_session() {  # $1=session
  tmux kill-session -t "$1" 2>/dev/null
  sleep 5  # let the Django port and GPU memory free up
}

# Save the NAMS memory graph to the run's logs dir, then (between runs) wipe
# the DB so the next run starts on a clean graph on top of --force-import.
# Bare-metal path: uses scripts/nams_baremetal_db.sh against the running
# local Neo4j. Docker path: not wired here (rely on --force-import for
# cleanliness; use scripts/nams_db.sh manually to save per-persona containers).
BAREMETAL_DB_SCRIPT="${REPO_ROOT}/scripts/nams_baremetal_db.sh"

save_memories() {  # $1=run_dir $2=run_label
  local run_dir=$1 run_label=$2
  local dump="${run_dir}/klaus_memories_${run_label}.dump"
  if [ "$HAVE_DOCKER" -eq 0 ] && [ -x "$BAREMETAL_DB_SCRIPT" ]; then
    log "  saving Klaus's NAMS memories -> ${dump}"
    if "$BAREMETAL_DB_SCRIPT" save "$dump" >>"${run_dir}/db_save.log" 2>&1; then
      log "  save ok: $(du -h "$dump" 2>/dev/null | cut -f1)"
    else
      log "  WARNING: save failed (see ${run_dir}/db_save.log); continuing."
    fi
  else
    log "  (skip bare-metal save; HAVE_DOCKER=${HAVE_DOCKER})"
  fi
}

wipe_db_between_runs() {
  if [ "$HAVE_DOCKER" -eq 0 ] && [ -x "$BAREMETAL_DB_SCRIPT" ]; then
    log "  wiping NAMS DB between runs (clean slate for the next run)"
    "$BAREMETAL_DB_SCRIPT" wipe >>"${LOG_ROOT}/midnight_db_wipe_${STAMP}.log" 2>&1 \
      || log "  WARNING: wipe failed (see ${LOG_ROOT}/midnight_db_wipe_${STAMP}.log)"
  else
    log "  (skip bare-metal wipe; --force-import will clean Klaus's session)"
  fi
}

# Translate the NAMS personas' JSON bootstrap_memory into the local Neo4j
# *before* the sim runs. Uses reverie.py --import-nams-only --force-import,
# which wipes each persona's existing session first so the import is clean
# and idempotent (the two runs get independent Klaus graphs). Fail-fast: if
# the import barfs, we abort the run instead of launching a 9-hour sim on a
# broken graph. No LLM is loaded for the import (the importer bypasses the
# NERS pipeline and writes facts directly), so this is fast.
translate_profiles() {  # $1=run_dir
  local run_dir=$1
  local import_log="${run_dir}/import_console.log"
  : > "$import_log"
  log "  translating ${NAMS_PERSONAS} JSON bootstrap -> Neo4j (--force-import)..."
  # Run inline (not in tmux) so the orchestrator sees the exit code and can
  # fail-fast. The import loads the sentence-transformers embedder but no
  # chat LLM, so it's cheap and synchronous.
  #
  # NB: do NOT prepend ${ENV_PREFIX} here. ENV_PREFIX embeds ';'-separated
  # commands ("set -a; . .env; set +a; ") which only work when *typed* into an
  # interactive shell (the tmux send-keys path). Expanded inline as a command
  # prefix, bash word-splits it but does not honor the ';' separators, so it
  # runs `set` with a literal `-a;` arg and dies ("set: -;: invalid option").
  # The orchestrator already sourced .env into its own environment above, so
  # this python3 subprocess inherits those vars without ENV_PREFIX.
  if ! PYTHONPATH=${REPO_ROOT}/shared:${REPO_ROOT}/reverie/backend_server \
        python3 "${REPO_ROOT}/reverie/backend_server/reverie.py" \
          --import-nams-only \
          --fork "${FORK_SIM}" \
          --nams-personas "${NAMS_PERSONAS}" \
          --embedder "BAAI/bge-small-en-v1.5" \
          --force-import \
        >>"$import_log" 2>&1; then
    log "  IMPORT FAILED; see ${import_log}. Aborting run."
    return 1
  fi
  # Sanity-check: the import log should mention at least one persona.
  if ! grep -q "import-nams-only" "$import_log"; then
    log "  IMPORT produced no output; see ${import_log}. Aborting run."
    return 1
  fi
  log "  translation done."
  return 0
}

# ------------------------------------------------------------------ one run
# $1=run_label  $2=extraction_mode
run_one() {
  local run_label=$1 extraction=$2
  local run_name="midnight_${run_label}_${STAMP}"
  local run_dir="${LOG_ROOT}/${run_name}"
  local sim_code="$run_name"
  local session="${run_name//[.:]/-}"  # tmux session names can't contain . or :
  local fe_log="${run_dir}/frontend_console.log"
  local be_log="${run_dir}/backend_console.log"
  local pair_log="${run_dir}/backend_prompt_pairs.jsonl"
  local t_start t_now step status="FAILED"

  mkdir -p "$run_dir"
  log "=== ${run_label} (extraction=${extraction}): starting ==="

  # 1. Translate the NAMS personas' JSON profile into Neo4j first.
  if ! translate_profiles "$run_dir"; then
    echo "${run_label}: FAILED (JSON->NAMS import; see ${run_dir}/import_console.log)" >> "$SUMMARY"
    return 1
  fi

  # 2. tmux: frontend (Django observer) + backend (headless reverie.py).
  if tmux has-session -t "$session" 2>/dev/null; then
    log "${run_label}: session ${session} already exists?! killing it."
    kill_session "$session"
  fi

  tmux new-session -d -s "$session" -n frontend
  tmux pipe-pane -t "${session}:frontend" -o "cat >> '${fe_log}'"
  tmux send-keys -t "${session}:frontend" \
    "${ENV_PREFIX}cd ${REPO_ROOT}/environment/frontend_server && python3 manage.py runserver 127.0.0.1:8000" C-m

  tmux new-window -t "$session" -n backend
  tmux pipe-pane -t "${session}:backend" -o "cat >> '${be_log}'"
  # Headless: one command, no interactive prompts to walk. REVERIE_PROMPT_LOG
  # captures every LLM call (cognitive modules + NAMS extraction LLM) with
  # full request/response context, exactly as in the interactive midnight runs.
  tmux send-keys -t "${session}:backend" \
    "${ENV_PREFIX}export REVERIE_PROMPT_LOG=${pair_log}; \
     export REVERIE_HARNESS=${BASE_HARNESS}; \
     export REVERIE_NAMS_PERSONAS=\"${NAMS_PERSONAS}\"; \
     export REVERIE_NAMS_EXTRACTION=${extraction}; \
     cd ${REPO_ROOT}/reverie/backend_server && \
     PYTHONPATH=${REPO_ROOT}/shared:. python3 reverie.py \
       --harness ${BASE_HARNESS} \
       --fork ${FORK_SIM} \
       --target ${sim_code} \
       --steps ${STEPS} \
       --nams-personas \"${NAMS_PERSONAS}\" \
       --nams-extraction ${extraction}" C-m
  log "${run_label}: headless sim launched (sim ${sim_code}); polling every ${POLL_SECONDS}s."

  # 3. Poll until the day completes, the run crashes, the backend dies/stalls,
  # or the wall-clock budget runs out. The headless reverie.py saves on clean
  # completion, on crash (try/except in _run_cli), and on Ctrl-C
  # (KeyboardInterrupt -> partial save), so we don't need to send "fin".
  t_start=$(date +%s)
  local last_step=-1 last_progress_t=$t_start
  while :; do
    sleep "$POLL_SECONDS"
    step=$(current_step)
    t_now=$(( $(date +%s) - t_start ))
    log "${run_label}: step ${step}/${STEPS} (${t_now}s elapsed)"

    if [ "$step" -ge "$STEPS" ] 2>/dev/null; then
      status="COMPLETED"
      break
    fi
    if ! backend_running "$session"; then
      # Backend exited. The headless reverie.py prints a distinct banner on
      # clean completion vs crash (both save, both exit the process), so we
      # grep the log to tell them apart.
      if grep -q "\[reverie\] COMPLETED:" "$be_log" 2>/dev/null; then
        status="COMPLETED"
      else
        status="CRASHED at step ${step}"
        log "${run_label}: backend process died; see ${be_log}"
      fi
      break
    fi

    if [ "$step" -gt "$last_step" ] 2>/dev/null; then
      last_step=$step
      last_progress_t=$(date +%s)
    elif [ "$(( $(date +%s) - last_progress_t ))" -ge "$STALL_SECONDS" ]; then
      status="STALLED at step ${step}"
      log "${run_label}: no curr_step progress for ${STALL_SECONDS}s; stopping. See ${be_log}."
      tmux send-keys -t "${session}:backend" C-c
      sleep 15
      break
    fi

    if [ "$t_now" -ge "$MAX_RUN_SECONDS" ]; then
      status="TIMED OUT at step ${step}"
      log "${run_label}: budget exhausted; interrupting (Ctrl-C -> partial save)."
      tmux send-keys -t "${session}:backend" C-c
      sleep 15
      break
    fi
  done

  # 4. Give the backend a moment to finish its (partial) save after Ctrl-C,
  # then tear down the tmux session.
  if backend_running "$session"; then
    local waited=0
    while backend_running "$session"; do
      sleep 5
      waited=$((waited + 5))
      [ "$waited" -ge "$FIN_TIMEOUT" ] && break
    done
  fi
  kill_session "$session"

  t_now=$(( $(date +%s) - t_start ))
  log "=== ${run_label}: ${status} (${t_now}s) ==="
  echo "${run_label}: ${status}, ${t_now}s, sim ${sim_code}, logs ${run_dir}" >> "$SUMMARY"
  [ "$status" = "COMPLETED" ]
}

# --------------------------------------------------------------------- main
run_idx=0
total_runs=${#RUNS[@]}
for spec in "${RUNS[@]}"; do
  run_idx=$((run_idx + 1))
  run_label="${spec%%|*}"
  extraction="${spec##*|}"
  run_one "$run_label" "$extraction" \
    || log "${run_label}: run did not complete cleanly; moving on."
  # Save Klaus's memories to this run's logs dir (distinct filename per run).
  save_memories "${LOG_ROOT}/midnight_${run_label}_${STAMP}" "$run_label"
  # Wipe the DB between runs so the next run starts on a clean graph. The
  # last run's memories are already saved above; no wipe after the final run.
  if [ "$run_idx" -lt "$total_runs" ]; then
    wipe_db_between_runs
  fi
done

log "All runs done. Summary:"
cat "$SUMMARY"
