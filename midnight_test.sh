#!/bin/bash
# midnight_test.sh -- unattended overnight benchmark of all three harnesses.
#
# Runs ONE FULL IN-GAME DAY (sec_per_step=10 -> 86400/10 = 8640 steps) of
# base_the_ville_isabella_maria_klaus on each harness, sequentially:
#
#     gemma4-e2b  ->  gemma4-e4b  ->  legacy-gpt  ->  qwen3-0.6b  ->
#     gemma4-e2b-thinking  ->  gemma4-e4b-thinking  ->  qwen3-0.6b-thinking
#
# The trailing *-thinking runs exercise the reasoning-channel harness variants
# (same code, enable_thinking=True): the reasoning is logged but stripped from
# memories/conversations/JSON, so they're directly comparable to the runs above.
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
# For each harness it:
#   1. opens a dedicated tmux session with the usual two windows
#      (frontend: Django runserver; backend: reverie.py),
#   2. answers reverie.py's interactive prompts (harness, fork sim, new sim),
#   3. issues "run 8640" and polls temp_storage/curr_step.json until the day
#      completes (or MAX_RUN_SECONDS elapses, in which case it Ctrl-C's the
#      run and saves whatever progress was made),
#   4. sends "fin" (which saves the sim into storage/<sim_code>), waits for
#      the backend process to exit, and kills the tmux session,
#   5. moves on to the next harness.
#
# In the morning you should find:
#   logs/midnight_<harness>_<stamp>/     one per harness, containing
#       frontend_console.log             Django window transcript
#       backend_console.log              reverie.py window transcript
#       backend_prompt_pairs.jsonl       exact prompt/response pairs
#       backend_prompt_pairs.txt         same, human-readable
#   logs/midnight_summary_<stamp>.txt    one status line per run
#   environment/frontend_server/storage/midnight_<harness>_<stamp>/
#                                        the saved sims (forkable/replayable)
#
# Secrets: API keys are sourced from ${REPO_ROOT}/.env, never hardcoded here.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

STAMP="${MIDNIGHT_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
export MIDNIGHT_STAMP="$STAMP"

# ----------------------------------------------------------------- settings
# STEPS and the harness list can be overridden from the environment for
# shorter shakedown runs, e.g.:
#     MIDNIGHT_STEPS=50 MIDNIGHT_HARNESSES="gemma4-e2b" bash midnight_test.sh
FORK_SIM="base_the_ville_isabella_maria_klaus"
HARNESSES=(${MIDNIGHT_HARNESSES:-gemma4-e2b gemma4-e4b legacy-gpt qwen3-0.6b gemma4-e2b-thinking gemma4-e4b-thinking qwen3-0.6b-thinking})
STEPS="${MIDNIGHT_STEPS:-8640}" # one in-game day at sec_per_step=10
MAX_RUN_SECONDS=$((3 * 3600))   # per-run wall-clock budget before we cut it
STALL_SECONDS="${MIDNIGHT_STALL_SECONDS:-1200}" # no curr_step progress for this
                                # long => assume a hung run and stop early (a
                                # backstop for hangs that never surface a prompt)
POLL_SECONDS=60                 # how often to check progress
PROMPT_TIMEOUT=600              # max wait for reverie.py's interactive prompts
FIN_TIMEOUT=600                 # max wait for save+exit after "fin"

LOG_ROOT="${REPO_ROOT}/logs"
TEMP_STORAGE="${REPO_ROOT}/environment/frontend_server/temp_storage"
SUMMARY="${LOG_ROOT}/midnight_summary_${STAMP}.txt"

# ------------------------------------------------- self-detach under nohup
# So that the overnight orchestration survives the ssh session that launched
# it. The detached copy re-runs this script with MIDNIGHT_DETACHED set.
if [ -z "${MIDNIGHT_DETACHED:-}" ]; then
  mkdir -p "$LOG_ROOT"
  ORCH_LOG="${LOG_ROOT}/midnight_orchestrator_${STAMP}.log"
  MIDNIGHT_DETACHED=1 nohup bash "$0" "$@" >"$ORCH_LOG" 2>&1 &
  echo "Detached orchestrator (pid $!). Safe to disconnect."
  echo "Follow along:  tail -f ${ORCH_LOG}"
  echo "Summary file:  ${SUMMARY}"
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

# Sourcing .env here only to *check* key presence for legacy-gpt; the tmux
# windows source it themselves via ENV_PREFIX.
ENV_PREFIX=""
if [ -f "${REPO_ROOT}/.env" ]; then
  ENV_PREFIX="set -a; . ${REPO_ROOT}/.env; set +a; "
  set -a; . "${REPO_ROOT}/.env"; set +a
fi

mkdir -p "$LOG_ROOT"
echo "midnight_test ${STAMP}: ${STEPS} steps each on: ${HARNESSES[*]}" > "$SUMMARY"

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

backend_running() {  # $1=session  -- is reverie.py still alive in its pane?
  local pane_pid
  pane_pid=$(tmux display-message -p -t "$1:backend" '#{pane_pid}' 2>/dev/null) || return 1
  pgrep -P "$pane_pid" -f "reverie.py" >/dev/null 2>&1
}

kill_session() {  # $1=session
  tmux kill-session -t "$1" 2>/dev/null
  sleep 5  # let the Django port and GPU memory free up
}

# ------------------------------------------------------------------ one run
run_one() {  # $1=harness
  local harness=$1
  local run_name="midnight_${harness}_${STAMP}"
  local run_dir="${LOG_ROOT}/${run_name}"
  local sim_code="$run_name"
  # tmux uses '.' as the window.pane separator and ':' as the session:window
  # separator in target specs, so a session name containing either is invalid
  # (this is why the qwen3-0.6b runs failed: "0.6b" has a dot). Sanitize the
  # session name only -- run_name / sim_code / run_dir keep their real value
  # (dots are fine in directory names and simulation codes).
  local session="${run_name//[.:]/-}"
  local fe_log="${run_dir}/frontend_console.log"
  local be_log="${run_dir}/backend_console.log"
  local pair_log="${run_dir}/backend_prompt_pairs.jsonl"
  local t_start t_now step status="FAILED"

  mkdir -p "$run_dir"
  log "=== ${harness}: starting (sim ${sim_code}) ==="

  if tmux has-session -t "$session" 2>/dev/null; then
    log "${harness}: session ${session} already exists?! killing it."
    kill_session "$session"
  fi

  tmux new-session -d -s "$session" -n frontend
  tmux pipe-pane -t "${session}:frontend" -o "cat >> '${fe_log}'"
  tmux send-keys -t "${session}:frontend" \
    "${ENV_PREFIX}cd ${REPO_ROOT}/environment/frontend_server && python3 manage.py runserver 127.0.0.1:8000" C-m

  tmux new-window -t "$session" -n backend
  tmux pipe-pane -t "${session}:backend" -o "cat >> '${be_log}'"
  tmux send-keys -t "${session}:backend" \
    "${ENV_PREFIX}export REVERIE_PROMPT_LOG=${pair_log}; cd ${REPO_ROOT}/reverie/backend_server && python3 reverie.py" C-m

  # Walk reverie.py's interactive prompts.
  if ! wait_for_pattern "$be_log" "Select model harness" "$PROMPT_TIMEOUT"; then
    log "${harness}: never saw the harness prompt; aborting run."
    echo "${harness}: FAILED (no harness prompt; see ${be_log})" >> "$SUMMARY"
    kill_session "$session"; return 1
  fi
  tmux send-keys -t "${session}:backend" "$harness" C-m

  if ! wait_for_pattern "$be_log" "Enter the name of the forked simulation" "$PROMPT_TIMEOUT"; then
    log "${harness}: never saw the fork prompt; aborting run."
    echo "${harness}: FAILED (no fork prompt; see ${be_log})" >> "$SUMMARY"
    kill_session "$session"; return 1
  fi
  tmux send-keys -t "${session}:backend" "$FORK_SIM" C-m

  if ! wait_for_pattern "$be_log" "Enter the name of the new simulation" "$PROMPT_TIMEOUT"; then
    log "${harness}: never saw the new-sim prompt; aborting run."
    echo "${harness}: FAILED (no new-sim prompt; see ${be_log})" >> "$SUMMARY"
    kill_session "$session"; return 1
  fi
  tmux send-keys -t "${session}:backend" "$sim_code" C-m

  if ! wait_for_pattern "$be_log" "Enter option" "$PROMPT_TIMEOUT"; then
    log "${harness}: fork never finished loading; aborting run."
    echo "${harness}: FAILED (fork never loaded; see ${be_log})" >> "$SUMMARY"
    kill_session "$session"; return 1
  fi
  tmux send-keys -t "${session}:backend" "run ${STEPS}" C-m
  log "${harness}: 'run ${STEPS}' issued; polling every ${POLL_SECONDS}s."

  # Poll until the day completes, the run crashes, the backend dies/stalls, or
  # the wall-clock budget runs out.
  #
  # Crash detection matters here: reverie.py's "run" loop is wrapped in a
  # bare-except that prints a traceback + "Error." and drops back to the
  # "Enter option:" prompt, leaving the *process alive*. That used to be
  # mislabeled "TIMED OUT" -- the step counter froze but backend_running()
  # stayed true, so we'd idle away the entire MAX_RUN_SECONDS budget before
  # cutting it. We now notice the run returned to the prompt (a new
  # "Enter option:" appears while step < STEPS) and report it as the crash it
  # is, immediately. A separate STALL_SECONDS backstop catches genuine hangs
  # that never surface a new prompt.
  t_start=$(date +%s)
  local last_step=-1 last_progress_t=$t_start base_prompts now_prompts
  # Count of "Enter option:" prompts already emitted before we issued "run".
  base_prompts=$(grep -o "Enter option:" "$be_log" 2>/dev/null | wc -l | tr -d ' ')
  while :; do
    sleep "$POLL_SECONDS"
    step=$(current_step)
    t_now=$(( $(date +%s) - t_start ))
    log "${harness}: step ${step}/${STEPS} (${t_now}s elapsed)"

    if [ "$step" -ge "$STEPS" ] 2>/dev/null; then
      status="COMPLETED"
      break
    fi
    if ! backend_running "$session"; then
      status="CRASHED at step ${step}"
      log "${harness}: backend process died; see ${be_log}"
      break
    fi

    # A new "Enter option:" prompt while we're short of STEPS means the run
    # aborted back to the interactive loop -- i.e. reverie's bare-except fired.
    now_prompts=$(grep -o "Enter option:" "$be_log" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${now_prompts:-0}" -gt "${base_prompts:-0}" ] 2>/dev/null; then
      status="CRASHED at step ${step}"
      log "${harness}: run returned to the prompt early (bare-except); see ${be_log}."
      break
    fi

    # Progress / stall tracking. A live-but-not-advancing sim (e.g. a hung
    # model call that never raises) shouldn't burn the whole budget.
    if [ "$step" -gt "$last_step" ] 2>/dev/null; then
      last_step=$step
      last_progress_t=$(date +%s)
    elif [ "$(( $(date +%s) - last_progress_t ))" -ge "$STALL_SECONDS" ]; then
      status="STALLED at step ${step}"
      log "${harness}: no curr_step progress for ${STALL_SECONDS}s; stopping. See ${be_log}."
      tmux send-keys -t "${session}:backend" C-c
      sleep 15
      break
    fi

    if [ "$t_now" -ge "$MAX_RUN_SECONDS" ]; then
      status="TIMED OUT at step ${step}"
      log "${harness}: budget exhausted; interrupting the run."
      tmux send-keys -t "${session}:backend" C-c
      sleep 15  # let the bare-except land us back at the "Enter option" prompt
      break
    fi
  done

  # Save and shut down (skip "fin" if the process is already gone).
  if backend_running "$session"; then
    log "${harness}: sending 'fin' to save."
    tmux send-keys -t "${session}:backend" "fin" C-m
    local waited=0
    while backend_running "$session"; do
      sleep 5
      waited=$((waited + 5))
      if [ "$waited" -ge "$FIN_TIMEOUT" ]; then
        log "${harness}: backend did not exit after fin (${FIN_TIMEOUT}s); killing anyway."
        status="${status} (fin hung)"
        break
      fi
    done
  fi
  kill_session "$session"

  t_now=$(( $(date +%s) - t_start ))
  log "=== ${harness}: ${status} (${t_now}s) ==="
  echo "${harness}: ${status}, ${t_now}s, sim ${sim_code}, logs ${run_dir}" >> "$SUMMARY"
  [ "$status" = "COMPLETED" ]
}

# --------------------------------------------------------------------- main
for harness in "${HARNESSES[@]}"; do
  if [ "$harness" = "legacy-gpt" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    log "legacy-gpt: OPENAI_API_KEY not set (check .env); skipping."
    echo "legacy-gpt: SKIPPED (no OPENAI_API_KEY)" >> "$SUMMARY"
    continue
  fi
  run_one "$harness" || log "${harness}: run did not complete cleanly; moving on."
done

log "All runs done. Summary:"
cat "$SUMMARY"
