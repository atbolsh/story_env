# Midnight run analysis — 2026-06-10_20-56-07

Analysis of the three `midnight_test.sh` runs (one full in-game day each,
8640 steps, `base_the_ville_isabella_maria_klaus`):

| run | summary-file status | what actually happened |
|---|---|---|
| gemma4-e2b | TIMED OUT at step 8256 | **Crashed** at 22:10 UTC (step 8256, 95.6% of the day) on a `KeyError: 'bedroom'`, then sat idle at the `Enter option:` prompt for ~1.8 h until the orchestrator budget expired. Pace before the crash was good. |
| gemma4-e4b | COMPLETED, 10036 s (2.8 h) | Finished the full day. Survived a burst of 5 CUDA OOMs at 02:37 caused by a degenerate 542 KB prompt (see below). |
| legacy-gpt | COMPLETED, 3433 s (0.95 h) | Clean run; one transient `ServiceUnavailableError` from the OpenAI API. |

All numbers below come from the scripts in this folder
(`analyze_prompt_pairs.py`, `scan_console_logs.py`, `categorize_and_compare.py`),
run against `logs/midnight_*_2026-06-10_20-56-07/backend_prompt_pairs.jsonl`
and the console logs.

## 1. Hard mistakes, especially JSON format errors

**JSON format errors are essentially zero for both Gemma models.**
Re-running the harness's own extraction logic over every `chat_json` record:

| run | chat_json calls | parse failures | missing `"output"` key |
|---|---|---|---|
| gemma4-e2b | 1785 | 0 | 0 |
| gemma4-e4b | 1822 | 0 | 0 |
| legacy-gpt (JSON-wrapped chat) | ~1800 | 0 | 0 |

The strict JSON system prompt + conservative sampling (T=0.4) in
`harnesses/gemma4.py::_safe_json` is doing its job. One footnote: 3 of e2b's
12 `focal_points` responses parse only with the new brace-balanced extractor;
the legacy `rfind('}')` heuristic alone would have failed them. The extractor
upgrade is earning its keep.

**Validation failures (the safe_* loop retrying because the *content* didn't
validate) are comparable across all three harnesses** — and legacy GPT is not
the best of the three:

| run | call sites | retried ≥1 | hit repeat ceiling (fail-safe used) |
|---|---|---|---|
| gemma4-e2b | 3507 | 5.6% | 4.4% |
| gemma4-e4b | 3492 | 3.5% | **2.2%** |
| legacy-gpt | 3433 | 5.0% | 4.3% |

Per-template highlights (give-up rate of call sites):

- `event_triple`: e2b **13.9%** vs e4b 1.0% vs gpt 2.0%. e2b is the outlier —
  it mangles triples for object-events (e.g. emitted `(kitchen sink, be used, -)`).
- `conv_summary`: bad everywhere — e2b 19.5%, e4b 17.1%, gpt **36.6%**.
- `insight_evidence`: gpt **69%** give-ups vs e2b 6%, e4b 0%. GPT-3.5 tends to
  drop the required `insight (because of 1, 5, 3)` citation format entirely;
  both Gemmas keep it.
- `schedule_revision`: 31–44% give-ups on all three; this prompt is just hard.

**The two run-threatening hard mistakes were both Gemma's:**

1. **e2b killed its run** with an out-of-set choice. The `action_arena` prompt
   said `MUST pick one of {main room, bathroom}` and the model answered
   `bedroom`. `run_gpt_prompt` does not validate set membership for this
   prompt, so the bad value flowed into
   `spatial_memory.get_str_accessible_arena_game_objects()` →
   `KeyError: 'bedroom'` → the bare except in `reverie.py` dropped to the
   command prompt and the day never finished. (Console log lines ~337150+.)
2. **e4b hit 5 consecutive CUDA OOMs** (tried to allocate 35 GiB on an 80 GiB
   GPU) at 02:37 on one `schedule_revision` call whose prompt had ballooned to
   **542,816 characters**. All 5 retries OOM'd, the fail-safe kicked in, and
   the run recovered and completed.

## 2. Soft quality differences

Side-by-side samples are in `side_by_side_samples.txt` (regenerate with
`categorize_and_compare.py --dump-samples`).

- **Bread-and-butter calls are indistinguishable across all three**: poignancy
  ratings, emoji conversion, object/sector selection, hourly schedule
  continuations, wake-up hours all look correct and sensible from e2b, e4b,
  and gpt alike.
- **Each Gemma model has one characteristic degeneration that compounds
  through sim state** (counted over user prompts, i.e. content already saved
  into memory/schedules and fed back):
  - **e2b: "conversing about conversing about …" echo.** 585 of 4125 records
    (14%) contain the doubled prefix. The model copies the "conversing about"
    framing into action descriptions, which get stored and re-prefixed.
  - **e4b: parenthetical self-nesting of task descriptions** — `task (task
    (task (task)))` — growing roughly exponentially across schedule
    decompositions. 94 records affected; the worst case is the 542 KB prompt
    that OOM'd the GPU. Five prompts exceeded 50 KB.
  - **legacy-gpt: zero instances of either pattern.**
  Neither degeneration is a one-off sampling fluke; both are systematic and
  would get worse over multi-day runs. Worth a sanitizer at the sim level
  (cap description length / strip nested parentheticals before saving).
- **Groundedness vs confabulation**: on `relationship` summaries where
  retrieval returned nothing about the other persona, both Gemmas reliably
  answer "there is no information to summarize a relationship". GPT-3.5
  sometimes does the same, but sometimes invents a full friendship narrative
  ("They have planned to have lunch together…"). The Gemma behavior is more
  honest; the GPT behavior arguably produces livelier (if less grounded)
  social dynamics.
- **e2b-specific weaknesses**: insight extraction cites nearly every statement
  number indiscriminately (`because of 0, 1, 2, …, 18`); schedule revisions
  sometimes degenerate into meta-commentary about the schedule instead of
  schedule lines. e4b's outputs on the same templates are clean and
  well-formed, generally on par with or better than GPT-3.5.
- **e4b lapses observed**: occasional inverted triple semantics
  (`(common room sofa, occupy, Maria Lopez)`), and one conversation summary
  that described the pair found in the retrieved statements rather than the
  pair named in the question.

## 3. Timing

Yes — every JSONL record carries an ISO-8601 `ts` (1-second resolution), and
since calls are sequential, the delta between consecutive records is a good
per-call latency proxy.

| run | calls/min | completion mean / p90 | day wall-clock |
|---|---|---|---|
| legacy-gpt | 71 | 0.9 s / 2 s | 57 min |
| gemma4-e2b | 56 | 1.1 s / 2 s | ~77 min pace (crashed at 95.6%) |
| gemma4-e4b | 23 | 4.3 s / 9 s (max 60 s) | 167 min |

- **e2b is up to par**: ~1.3× the OpenAI-API wall clock for the full day, with
  per-call latencies in the same band as the API.
- **e4b is ~3× slower than the API run.** Its completions average 4.3 s with a
  long tail (26 inter-call gaps ≥ 60 s, all on long schedule-decomposition
  generations). The sum of per-call latencies ≈ total wall time, so the run is
  almost entirely generation-bound — batching or a faster inference path
  (e.g. vLLM) would translate ~1:1 into wall-clock savings.
- `time_gemma_prompts.py` in this folder benchmarks a representative prompt
  mix directly against the harness (no frontend/backend servers). **Remote box
  only** — it loads model weights and needs the requirements.txt environment.

## Suggested follow-ups

1. ~~Validate set membership in `run_gpt_prompt` for arena/sector choices.~~
   **Done** (2026-06-12): `match_option()` in `run_gpt_prompt.py` (exact
   match after strip+lower, no fuzzy matching) is now enforced inside the
   validate/clean-up closures of `action_sector`, `action_arena`, and
   `action_game_object`, so out-of-set answers make `safe_generate_response`
   retry. Clean-up canonicalizes casing to the offered option; fail-safes are
   now guaranteed-legal values (living-area sector / first legal arena).
   Replayed against the midnight logs: the fatal `bedroom` answer is the only
   sector/arena violation across all three runs and is now caught.
2. ~~Sanitize action/task descriptions before they are saved.~~ **Done**
   (2026-06-12): `sanitize_action_description` in `global_methods.py`,
   applied at the four save chokepoints (`task_decomp` composition,
   `summarize_conversation` clean-up, `new_decomp_schedule` clean-up, and the
   `inserted_act` composition in `plan.py`). Collapses the `conversing about`
   echo and degenerate parenthetical self-nesting, and caps description
   length at 350 chars. Verified against the midnight logs: the 542 KB OOM
   prompt's schedule content reduces to ~5 KB. This also largely defuses
   follow-up 3 (the OOM path), since descriptions can no longer compound.
3. Cap prompt size in `gemma4.llm_request` (truncate or refuse > N chars)
   so a degenerate prompt can never OOM the GPU. (Belt-and-suspenders after
   item 2; harness left untouched for now.)
4. `midnight_test.sh` could detect "no step progress for N polls while the
   backend is alive" and cut the run early instead of burning the full budget
   (e2b idled ~1.8 h after its crash).
