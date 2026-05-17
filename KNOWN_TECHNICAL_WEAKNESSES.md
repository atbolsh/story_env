# Known Technical Weaknesses

This document is an audit of the original `generative_agents` codebase
(Park et al., 2023, the "Smallville" paper at <https://arxiv.org/abs/2304.03442>),
focused on issues that would matter for **long-running** simulations beyond
the published 25-agent / 2-day demo. It is not exhaustive, but every entry below
was confirmed by reading the source.

Items are grouped by category and rated:

- **[BLOCKER]** — would prevent or silently corrupt a multi-day run.
- **[SCALING]** — works fine for the demo but degrades super-linearly.
- **[CORRECTNESS]** — produces wrong but non-fatal behavior.
- **[CLEANUP]** — code-quality / maintainability concern.

All file paths are relative to the repository root.

---

## 1. Memory growth & retrieval

### 1.1 [BLOCKER] `AssociativeMemory` has no eviction policy
*File:* `reverie/backend_server/persona/memory_structures/associative_memory.py`

`add_event`, `add_thought`, `add_chat` each unconditionally insert into
`self.id_to_node`, `self.seq_*`, `self.kw_to_*`, and `self.embeddings`. There is
no expiration handling (the `expiration` field on `ConceptNode` is read but
never actioned), no LRU eviction, no decay, no size cap. Over weeks of game
time this becomes:

- the JSON `nodes.json` grows without bound and re-serializes the whole tree
  on every `save()`;
- `embeddings.json` becomes a multi-MB dict (each text-embedding-ada-002 vector
  is 1536 floats);
- every retrieval pass (`new_retrieve`, `generate_focal_points`,
  `retrieve_relevant_*`) iterates every node.

### 1.2 [BLOCKER] `f_daily_schedule` accumulates a stray `["sleeping", …]` entry every action
*File:* `reverie/backend_server/persona/cognitive_modules/plan.py:611-613`

```
if 1440 - x_emergency > 0: 
    print ("x_emergency__AAA", x_emergency)
persona.scratch.f_daily_schedule += [["sleeping", 1440 - x_emergency]]
```

The append is **not** gated by the `if`. Every call to `_determine_action`
appends a sleeping entry to the schedule, with possibly negative duration
(when `x_emergency >= 1440`). In a multi-day simulation, `f_daily_schedule`
grows by O(actions/day) of garbage entries, polluting later
`get_f_daily_schedule_index` lookups and inflating the persona's saved state.

### 1.3 [SCALING] Path-A keyword retrieval has no top-k or scoring
*File:* `reverie/backend_server/persona/memory_structures/associative_memory.py:305-326`
*Consumer:* `run_gpt_prompt.py:1244` (`run_gpt_prompt_decide_to_talk`),
`run_gpt_prompt.py:1344` (`run_gpt_prompt_decide_to_react`).

`retrieve_relevant_events` / `retrieve_relevant_thoughts` return the **entire
set** of nodes matching any keyword from the (s, p, o) triple. `decide_to_talk`
then concatenates **all** node descriptions into the prompt's `Context:` block,
unweighted, no dedup beyond `set()`. After many days of simulation the prompt
balloons with hundreds of near-duplicate "Isabella was pouring coffee"
sentences. This is partially saved by issue 1.4, but should be rewritten to
share scoring/cap logic with `new_retrieve`.

### 1.4 [CORRECTNESS] `retrieve_relevant_*` has a case-sensitivity bug that quietly drops most queries
*File:* `reverie/backend_server/persona/memory_structures/associative_memory.py`

`add_event` (line 178) lowercases keywords before storing, but the lookup
methods compare against raw `s_content` / `p_content` / `o_content` without
lowercasing:

```
def retrieve_relevant_events(self, s_content, p_content, o_content): 
    contents = [s_content, p_content, o_content]
    ret = []
    for i in contents: 
        if i in self.kw_to_event: 
            ret += self.kw_to_event[i]
    ret = set(ret)
    return ret
```

Because `event.subject` is normally a colon-separated address like
`"the Ville:Hobbs Cafe:cafe:coffee pot"` (rarely a key), and the predicate
isn't stored in keywords at all, Path-A retrieval silently returns the
empty set most of the time. Worse, in `retrieve_relevant_thoughts` the gate
checks `if i in self.kw_to_thought` but the access uses `self.kw_to_thought[i.lower()]` —
which would `KeyError` if a mixed-case `i` ever did make it into the dict.
This is fragile dead-on-arrival behavior masked by an outer `try/except` in
the caller.

### 1.5 [SCALING] `kw_strength_event` / `kw_strength_thought` only ever increase
*File:* `reverie/backend_server/persona/memory_structures/associative_memory.py:186-192, 230-236`

Keyword salience counters monotonically increase with no decay. After a
long run, every keyword becomes "salient enough" and the reflection
focal-point selection (`generate_focal_points`) loses its bias signal.
A multiplicative decay per game-day (or eviction tied to 1.1) is needed.

### 1.6 [SCALING] `new_retrieve` is O(N · n_focal_pts) per call
*File:* `reverie/backend_server/persona/cognitive_modules/retrieve.py:199-271`

`new_retrieve` iterates **every** event + thought node, computes a fresh
embedding for the focal point, and computes cosine similarity against every
node's stored embedding. With ~10k nodes this is fine; with ~1M nodes (years
of sim time, 25 agents) you need an ANN index (FAISS / hnswlib / pgvector).
Same goes for `generate_focal_points` (`reflect.py:21-35`).

### 1.7 [SCALING] Path-finding is O(V²) per query
*File:* `reverie/backend_server/path_finder.py:96-161` (`path_finder_v2`)

`make_step` rescans the entire grid (`O(V)`) on each BFS wave, and the outer
loop runs `O(V)` waves in the worst case → `O(V²)`. For a 140×100 maze and
25 agents pathing every step, this is the dominant CPU cost. Replace with a
proper queue-based BFS (`O(V)`).

There is also a silent failure mode: the `except_handle = 150` cap in
`path_finder_v2` aborts after 150 BFS waves. In that case `m[end] == 0`,
the `while k > 1` loop never executes, and `path_finder` returns `[end]` —
i.e. the persona **teleports** to the destination. Should raise instead.

---

## 2. Crash recovery & state durability

### 2.1 [BLOCKER] No autosave; manual `save` / `fin` is the only commit
*File:* `reverie/backend_server/reverie.py:438-468`

The simulation persists only when the user types `save` or `fin`. Any crash
or Ctrl-C between saves erases all in-memory state since the last save:
new memories, decomposed schedules, reflections, conversations. For a
multi-day run this is catastrophic. Add an autosave every N steps and on
SIGINT/SIGTERM.

### 2.2 [BLOCKER] `Scratch.save()` crashes if `act_start_time` is `None`
*File:* `reverie/backend_server/persona/memory_structures/scratch.py:286-287`

```
scratch["act_start_time"] = (self.act_start_time
                                 .strftime("%B %d, %Y, %H:%M:%S"))
```

This is unconditional. If any save fires before the first action is set
(e.g., immediately after `__init__`, or after a state where the persona
finished an action and hasn't been replanned yet), it raises
`AttributeError: 'NoneType' object has no attribute 'strftime'`. Should mirror
the load path's `if self.act_start_time` guard.

### 2.3 [BLOCKER] `exit` and `start path tester mode` `rmtree` the simulation folder
*File:* `reverie/backend_server/reverie.py:454, 461`

```
shutil.rmtree(sim_folder) 
```

No confirmation, no backup. A typo at the prompt destroys the run. Add a
confirmation dialog or rename instead of delete.

### 2.4 [CORRECTNESS] Backend↔frontend file handshake is non-atomic
*Files:* `reverie/backend_server/reverie.py:405-407` and
`environment/frontend_server/translator/views.py:262-263`

Both sides do `with open(path, "w") as f: f.write(json.dumps(...))`, which
truncates the file to zero bytes before writing. A reader that polls between
truncate and write sees an empty file, falls into a bare `except: pass`
(line 329 of `reverie.py`), and uses stale `env_retrieved=True` from the
previous iteration with an undefined `new_env` → silent corruption.

Use atomic `os.replace(tmp, path)` after writing to a `.tmp` file.

### 2.5 [BLOCKER] Backend hangs forever if frontend tab closes
*File:* `reverie/backend_server/reverie.py:316-321`

The main loop polls for `environment/<step>.json`. If the browser tab is
closed mid-run, no new env file ever appears, and the backend
`time.sleep(0.1)`s forever with no timeout, no warning, no retry. Add a
heartbeat / max-stall-time / health endpoint.

### 2.6 [BLOCKER] `execute.py` silently breaks when GPT picks a non-existent address
*File:* `reverie/backend_server/persona/cognitive_modules/execute.py:91-94`

```
if plan not in maze.address_tiles: 
    maze.address_tiles["Johnson Park:park:park garden"] #ERRORRRRRRR
else: 
    target_tiles = maze.address_tiles[plan]
```

The "fallback" merely fetches a value and discards it; `target_tiles`
remains undefined and the next line crashes with `NameError`. GPT
hallucinations of location names (very common at scale) take the whole
simulation down. Either fall back to the persona's living area or
re-prompt with the list of valid addresses.

### 2.7 [SCALING] `movement/N.json` files are never cleaned up
*File:* `reverie/backend_server/reverie.py:405-407`

Every step writes a new file in `storage/<sim>/movement/`. After 10k steps
that's 10k files (a few KB each). Inode pressure aside, simulation startup
and the frontend's listing operations slow down. Either compress periodically
or roll the files into a SQLite/lmdb store.

### 2.8 [CORRECTNESS] `act_check_finished` requires exact second match
*File:* `reverie/backend_server/persona/memory_structures/scratch.py:556`

```
if end_time.strftime("%H:%M:%S") == self.curr_time.strftime("%H:%M:%S"): 
    return True
```

If `sec_per_step` is changed (or floor/ceiling rounding ever shifts the
second), the action's end-time second never coincides with `curr_time`'s
second and the action runs forever. Use `>=` on a comparable type.

---

## 3. Robustness of GPT calls

### 3.1 [BLOCKER] Bare `except:` swallows real errors as "TOKEN LIMIT EXCEEDED" / `False`
*Files:* `gpt_structure.py:53-56, 79-81, 117-118, 161-162, 187-188, 233-235`,
`run_gpt_prompt.py` (many sites), `cognitive_modules/converse.py:33-38`,
`cognitive_modules/reflect.py:48-55`.

Almost every GPT call is wrapped in `try / except: pass` or
`try / except: return False`. Network errors, model deprecations, rate
limits, JSON parse failures all surface as either silent fail-safe values or
the literal string `"TOKEN LIMIT EXCEEDED"`. We already hit this once: the
GPT-3 `text-davinci-002/003` deprecation showed up as bogus
"token limit exceeded" output for weeks.

Switch to typed exception handling and a structured logger so failures are
diagnosable. Every silenced exception should at minimum write to a debug
file with traceback, persona name, prompt template, and prompt input.

### 3.2 [CORRECTNESS] `safe_generate_response` returns the fail-safe transparently
*File:* `reverie/backend_server/persona/prompt_template/gpt_structure.py:266-293`

After 5 retries the function returns `fail_safe_response` (e.g. integer `4`
for poignancy, the string `"yes"` for decide-to-talk, a hardcoded sleep
schedule for daily plan). The caller has no way to tell whether the value
came from GPT or the fail-safe. Long simulations therefore drift toward
"every poignancy is 4" / "every conversation happens" without warning.
Return `(value, source)` and let callers handle degraded mode.

### 3.3 [CORRECTNESS] No retry/backoff for OpenAI rate limits
*File:* `reverie/backend_server/persona/prompt_template/gpt_structure.py:16-17, 218-235`

`temp_sleep(0.1)` is the only inter-call delay. `RateLimitError`
(429) is caught by the bare `except` and immediately returns the fail-safe.
Add exponential backoff (`tenacity` or hand-rolled) on
`openai.error.RateLimitError`, `APIConnectionError`, `Timeout`.

### 3.4 [SCALING] No prompt token-counting → silent context-window overflows
*Files:* all of `run_gpt_prompt.py`

The codebase concatenates retrieved memory descriptions into prompts with
zero token accounting. `gpt-3.5-turbo` is 16k tokens; `decide_to_talk` with
hundreds of memory nodes (1.3) can blow the limit, in which case OpenAI
returns 400 → caught by bare `except` → fail-safe used. Add
`tiktoken`-based budgeting and graceful truncation that drops lowest-scoring
nodes first.

### 3.5 [SCALING] Same description embedded N times across N personas
*File:* `reverie/backend_server/persona/cognitive_modules/perceive.py:141-144`

The `embeddings` cache lives on each `AssociativeMemory` instance. If 25
personas all perceive "the coffee pot is brewing" within the same step,
that's 25 separate `get_embedding` calls (≈ 25× the cost). Promote to a
shared LRU cache keyed by description string.

### 3.6 [SCALING] Every new event triggers an embedding **and** a poignancy GPT call
*File:* `reverie/backend_server/persona/cognitive_modules/perceive.py:135-150`

Two API round-trips per perceived novel event. With 25 agents at vision
radius 8, this is dozens of calls per step. Two amortizations help:

- Batch embeddings via OpenAI's batch endpoint (the API supports up to
  2048 inputs per request).
- Replace per-event poignancy calls with a heuristic (e.g., a small
  classifier on the embedding) and fall back to GPT only for ambiguous
  cases.

### 3.7 [CORRECTNESS] JSON wrapper logic in `ChatGPT_safe_generate_response` is brittle
*File:* `reverie/backend_server/persona/prompt_template/gpt_structure.py:123-164`

The post-processor does
`json.loads(curr_gpt_response[:rfind('}')+1])["output"]`. This breaks if
the model returns:
- a JSON array instead of an object;
- a nested object whose first key is not `"output"`;
- prose with `}` anywhere (e.g., a code block).
Currently any of these silently fall through to the bare `except` →
fail-safe. Use an LLM with `response_format={"type":"json_object"}` (or
function calling) on a 1.0+ SDK.

### 3.8 [CLEANUP] Hardcoded `text-davinci-002` / `text-davinci-003` engine strings everywhere
*File:* `reverie/backend_server/persona/prompt_template/run_gpt_prompt.py` (75+ sites)

Currently silently mapped to `gpt-3.5-turbo` by the legacy-engine map I
added to `gpt_structure.py:200-203`. Should be centralized in a config
(e.g., `MODEL_FOR_TASK = {"poignancy": ..., "wake_up_hour": ..., ...}`)
so model upgrades are a one-line change and per-task model selection is
possible.

### 3.9 [CLEANUP] Pinned `openai==0.27.0`
*File:* `requirements.txt:31`

The OpenAI Python SDK was rewritten in 1.0 (Nov 2023). Module-level
`openai.api_key = …`, `openai.ChatCompletion.create`, and
`openai.Completion.create` are all gone in modern SDKs. A migration is
required eventually; the longer it's deferred the more drift accumulates.

### 3.10 [CLEANUP] `stream=False` everywhere
First-token latency for `gpt-3.5-turbo` on long prompts is ~2-5s. Streaming
would let the simulator overlap GPT decoding with embedding I/O and reduce
wall-clock per step.

---

## 4. Concurrency & throughput

### 4.1 [SCALING] Personas processed strictly sequentially each step
*File:* `reverie/backend_server/reverie.py:378-392`

```
for persona_name, persona in self.personas.items(): 
    next_tile, pronunciatio, description = persona.move(...)
```

With 25 agents and ~10 GPT calls/agent on a planning step, a single step
takes 5-15 minutes. Each `persona.move()` is independent (modulo the maze
event-tile updates, which are mutex-friendly), so you can run them in a
`ThreadPoolExecutor` to get an N×–10× speedup, bottlenecked only by OpenAI
concurrency limits.

### 4.2 [SCALING] No batching across personas for similar prompts
Many prompts (poignancy, action-event triple, pronunciatio emoji) are
small and identical-shape across personas. They could be issued as a
single ChatCompletion with a `n=N` sample-batched request, or via OpenAI's
async batch API for non-realtime workloads (e.g. overnight reflection).

### 4.3 [CORRECTNESS] `chatting_with_buffer` decremented in non-chat steps only
*File:* `reverie/backend_server/persona/cognitive_modules/plan.py:1002-1005`

```
for persona_name, buffer_count in curr_persona_chat_buffer.items():
    if persona_name != persona.scratch.chatting_with: 
        persona.scratch.chatting_with_buffer[persona_name] -= 1
```

So a persona that is currently chatting with `Klaus` keeps the buffer for
**other** people decrementing while they chat, then never decrements
Klaus's own buffer (set to 800 in `_chat_react`). The 800-tick cooldown
will expire on Klaus immediately when the chat ends, since the entry will
be missing from the dict (or stuck at 800). The intent appears to be
"prevent re-chatting with Klaus for 800 ticks after the chat ends" but
the implementation doesn't enforce that.

---

## 5. Game-logic correctness

### 5.1 [CORRECTNESS] Reflection produces a `{"this is blank": "node_1"}` thought on parser failure
*File:* `reverie/backend_server/persona/cognitive_modules/reflect.py:54-55`

```
except: 
    return {"this is blank": "node_1"} 
```

When `run_gpt_prompt_insight_and_guidance` returns malformed output, the
reflection module silently writes a thought with description `"this is blank"`
into the persona's memory. Over time this accumulates as junk (and then
gets retrieved during `new_retrieve` because of its embedding). Should
either retry or skip the reflection.

### 5.2 [CORRECTNESS] `<random>` action with empty `target_tiles` crashes
*File:* `reverie/backend_server/persona/cognitive_modules/execute.py:80-83`

```
plan = ":".join(plan.split(":")[:-1])
target_tiles = maze.address_tiles[plan]
target_tiles = random.sample(list(target_tiles), 1)
```

If the GPT-supplied `plan` doesn't exist in `address_tiles`, `KeyError`.
If it exists but the address has zero tiles (shouldn't happen but is
possible if maze data is partial), `random.sample` raises `ValueError`.

### 5.3 [CORRECTNESS] Loaded scratch may have field mismatch with newer code
*File:* `reverie/backend_server/persona/memory_structures/scratch.py:161-234`

The constructor reads ~30 fields from `scratch.json` with bare `[...]`
access. If a future code version adds a new persona field (e.g.
`personality_modifier`), reloading an old save crashes. Add `.get(..., default)`
for forward compatibility.

### 5.4 [CORRECTNESS] Generate-action-sector fallback is `living_area.split(":")[1]`
*File:* `reverie/backend_server/persona/prompt_template/run_gpt_prompt.py:619`

When GPT's chosen sector isn't in the persona's accessible sector list,
the fallback is the persona's living-area sector. That means a non-resident
visitor (e.g. Isabella visiting Klaus's dorm) defaults home, which can make
agents endlessly oscillate during a buggy action choice.

### 5.5 [CORRECTNESS] `idle` event filtering is a string `in` check
*Files:* `cognitive_modules/retrieve.py:226`, `reflect.py:26`,
`perceive.py:16`

`if "idle" not in i.embedding_key` — but legitimate descriptions like
"Maria avoided idle gossip" would be filtered out. Should compare against
a marker token, not substring.

### 5.6 [CORRECTNESS] `f_daily_schedule_hourly_org` is a copy-by-slice (`[:]`) but inner lists are shared
*File:* `reverie/backend_server/persona/cognitive_modules/plan.py:496-497`

```
persona.scratch.f_daily_schedule_hourly_org = (persona.scratch
                                                 .f_daily_schedule[:])
```

Outer list is copied but inner `[task, duration]` lists are shared. If
later code mutates a duration in `f_daily_schedule` it would also mutate
the "original" snapshot. This currently doesn't bite because the mutating
patterns happen via slice-replacement (`f_daily_schedule[i:i+1] = ...`),
not in-place item edits, but it's a footgun if anyone refactors.

---

## 6. Code quality / observability

### 6.1 [CLEANUP] Pervasive debug `print` statements in production paths
Examples:
- `reflect.py:149-150` (the `importance_trigger_curr` spam you saw).
- `plan.py:457` `print ("WE ARE HERE!!!", new_daily_req)`.
- `plan.py:598-603` `"DEBUG LJSDLFSKJF"` and the schedule dump.
- `run_gpt_prompt.py:1881, 1952, 2012, ...` `"asdhfapsh8p9hfaiafdsi;ldfj as DEBUG N"`.
- `execute.py:45-46` `"aldhfoaf/????"`.
- `cognitive_modules/converse.py:117-122` `"July 23 N"` block.

Replace with a real logger gated on `utils.debug` and named per-module.

### 6.2 [CLEANUP] Dead branch `defunct_run_gpt_prompt.py` parallel to live one
*File:* `reverie/backend_server/persona/prompt_template/defunct_run_gpt_prompt.py`

Three thousand lines of older versions kept "just in case." Easy to update
the wrong file by accident. Extract anything still useful, delete the rest.

### 6.3 [CLEANUP] `selenium` imported but unused
*File:* `reverie/backend_server/reverie.py:31`

`from selenium import webdriver` — no usage in the file. Drag-along
dependency.

### 6.4 [CLEANUP] Bare `except:` (vs `except Exception:`) in many places
Catches `KeyboardInterrupt` and `SystemExit`, making Ctrl-C unreliable
during long blocks of GPT calls. Prefer narrow exception types.

### 6.5 [CLEANUP] `requirements.txt` pins ancient versions of tangentially-related libs
`Django==2.2` (released 2019, EOL April 2022; security-unsupported),
`numpy==1.25.2` (fine), `pandas==2.0.3` (fine), `nltk==3.6.5` (old),
`openai==0.27.0` (see 3.9). The Django pin is the spookiest — it has
known CVEs. Plan a migration.

---

## 7. Frontend / UX

### 7.1 [CORRECTNESS] Camera defaults far from any persona
*File:* `environment/frontend_server/templates/home/main_script.html:268-272`

The hidden `player` sprite the camera follows is created at pixel
`(800, 288)` (tile 25,9), an empty corner of the map. None of the demo
personas spawn anywhere near there. New users always think "the
simulation is broken" until they pan with arrow keys. Default to centering
on persona[0]'s spawn instead.

### 7.2 [CORRECTNESS] `home` view consumes `curr_step.json` on every load
*File:* `environment/frontend_server/translator/views.py:120` (`os.remove(f_curr_step)`)

A page refresh after starting the backend produces "Please start the
backend first" because the file was removed by the previous load. The
backend has to be restarted to regenerate it. This makes debugging the
frontend miserable. Move to a non-destructive read.

### 7.3 [CORRECTNESS] `simulator_home` has no error display when handshake stalls
If movement files stop being produced (issue 2.5 / 2.6), the frontend
just freezes silently. Add a "backend not responding" toast tied to a
heartbeat from the polling loop.

### 7.4 [CLEANUP] Phaser `pixelArt: true` plus a tiny `width/height`
The viewport is hardcoded to `1500 × 800`. Doesn't scale to mobile or
large monitors; no zoom controls.

---

## 8. Cost discipline

### 8.1 [SCALING] No model selection per prompt
Every prompt today goes to `gpt-3.5-turbo`. Some (e.g. poignancy,
emoji selection) are easy and could use cheaper/faster models; others
(daily plan, reflection) might want the full `gpt-4`. Add a per-task
model setting in a central config (see 3.8).

### 8.2 [SCALING] Reflection generates ~3 GPT calls × persona × every-N-steps
At 25 agents and a reflection frequency of every ~2-4 game-hours per
agent, that's ~75-150 GPT calls **just** on reflection per simulated
day, on top of the per-step perception/poignancy calls. Budget this
explicitly and consider running reflection in a daily batch overnight
rather than during the live loop.

### 8.3 [CLEANUP] No cost telemetry
The codebase doesn't track tokens-in / tokens-out / dollars per
simulation. For multi-day runs this is essential. Wrap the GPT call
sites in a single helper that increments per-prompt-template counters.

---

## 9. Quick-fix priority list

If you're going to spend a fixed budget hardening this for
multi-day use, I'd attack in this order:

1. **2.1 autosave** — without this, every other fix is ephemeral.
2. **2.6 execute fallback** — single most common cause of crashes.
3. **3.1 / 3.2 logging + typed exceptions** — required to debug
   anything else.
4. **2.2 scratch save guard** — pairs with autosave.
5. **1.2 schedule append bug** — silent state corruption every step.
6. **1.1 / 1.5 / 1.6 memory eviction + ANN index** — biggest
   long-term scaling win.
7. **2.4 atomic file handshake** — one-line fix that prevents
   subtle corruption.
8. **4.1 parallel persona processing** — biggest wall-clock win.
9. **3.4 token budgeting** — prevents silent prompt overflows.
10. **1.4 retrieval case bug + 1.3 unification with `new_retrieve`**.

Items in §6 (cleanup) and §7 (frontend) can wait, but §3.9 (OpenAI 1.0
SDK migration) blocks any future model upgrade and should be planned
in parallel.
