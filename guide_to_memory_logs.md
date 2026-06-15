# Guide to memory in the prompt logs

How the generative-agents memory stream actually shows up in the prompt logs
(`backend_prompt_pairs.{jsonl,txt}`), where to find it, and the gotchas that
make it easy to miss. Written against the logs under
`logs/midnight_*_2026-06-10_20-56-07/`.

---

## 1. There are TWO retrieval mechanisms, not one

The paper describes a single "retrieval" step (recency + importance +
relevance), but the code has **two distinct retrieval paths**, and only one of
them is cosine-similarity based:

| Path | Function | How it scores | Used by |
|---|---|---|---|
| **Cosine** | `new_retrieve` | recency + importance + **cosine relevance** (embedding `cos_sim`) | reflection, daily-plan revision, conversation |
| **Keyword** | `retrieve_relevant_events` / `retrieve_relevant_thoughts` | set-intersection on the perceived event's subject/predicate/object **keywords** (no embeddings) | reaction decisions (`decide_to_talk` / `decide_to_react`) |

The cosine path:

```199:249:reverie/backend_server/persona/cognitive_modules/retrieve.py
def new_retrieve(persona, focal_points, n_count=30): 
```

The keyword path (note: pure dict lookups on `kw_to_event` / `kw_to_thought`,
no embeddings, no `cos_sim`):

```305:326:reverie/backend_server/persona/memory_structures/associative_memory.py
  def retrieve_relevant_thoughts(self, s_content, p_content, o_content): 
    contents = [s_content, p_content, o_content]

    ret = []
    for i in contents: 
      if i in self.kw_to_thought: 
        ret += self.kw_to_thought[i.lower()]

    ret = set(ret)
    return ret
```

**Why this matters for reading logs:** the location/action-selection prompts
(the many "choose the location" / "fill in the correct location" prompts)
inject *no* retrieved memory at all — they run off the persona's scratch state
and spatial tree. Retrieved *statement* memory only appears in the prompts
listed below.

---

## 2. Where cosine-retrieved statement memories are injected

All four sites call `new_retrieve`. The retrieved nodes are flattened into the
prompt as a plain numbered or bulleted list — there is no label like "relevant
memories," and the Gemma harness rewraps everything under a generic
`Continue the user's text directly` system prompt, which is why they're easy to
scroll past.

### 2a. Reflection — "insight" prompt  (most common)

`reflect.run_reflect` retrieves nodes per focal point, then asks for high-level
insights.

```113:121:reverie/backend_server/persona/cognitive_modules/reflect.py
  retrieved = new_retrieve(persona, focal_points)

  # For each of the focal points, generate thoughts and save it in the 
  # agent's memory. 
  for focal_pt, nodes in retrieved.items(): 
```

- **Builder:** `run_gpt_prompt_insight_and_guidance` → template
  `v2/insight_and_evidence_v1.txt`
- **Log signature:** `high-level insights can you infer`
- **Looks like:** an `Input:` block of numbered statements `0..N`, then
  `What 5 high-level insights can you infer from the above statements?` — the
  numbered list *is* the `new_retrieve` output. (Example: e4b run, call #395.)

### 2b. Daily-plan revision — `revise_identity` (new day only)

```418:431:reverie/backend_server/persona/cognitive_modules/plan.py
  retrieved = new_retrieve(persona, focal_points)

  statements = "[Statements]\n"
  for key, val in retrieved.items():
    for i in val: 
      statements += f"{i.created.strftime('%A %B %d -- %H:%M %p')}: {i.embedding_key}\n"
```

- **Builder:** inline `chat_single_request` (no template file).
- **Log signature:** `should remember as they plan` (and a timestamped
  `[Statements]` block). Note: did **not** fire in the 2026-06-10 e4b run (only
  runs on a day rollover).

### 2c. Conversation — relationship summary + utterance (the live chat path)

`converse.agent_chat_v2` (the path actually used; `agent_chat_v1` is commented
out in `plan.py:281`) retrieves twice per turn:

```131:146:reverie/backend_server/persona/cognitive_modules/converse.py
    focal_points = [f"{target_persona.scratch.name}"]
    retrieved = new_retrieve(init_persona, focal_points, 50)
    relationship = generate_summarize_agent_relationship(init_persona, target_persona, retrieved)
    ...
    retrieved = new_retrieve(init_persona, focal_points, 15)
    utt, end = generate_one_utterance(maze, init_persona, target_persona, retrieved, curr_chat)
```

- **Relationship summary** — top 50 nodes → template
  `v3_ChatGPT/summarize_chat_relationship_v2.txt`.
  - **Log signature:** `[Statements]` followed by
    `Based on the statements above, summarize ... relationship` (e4b call #687).
- **Utterance generation** — top 15 nodes → template
  `v3_ChatGPT/iterative_convo_v1.txt`.
  - **Log signature:** `Here is the memory that is in <name>'s head:` followed
    by a `- bullet` list (e4b call #688).

### 2d. Interview / "analysis" mode (interactive only)

`open_convo_session("analysis")` (reverie.py REPL) → `new_retrieve(...,50)` →
`generate_summarize_ideas` → template `v3_ChatGPT/summarize_ideas_v1.txt`
(`Summarize the Statements that are most relevant to the interviewer's line`).
Only fires when you drive an interview by hand; absent from unattended runs.

---

## 3. Where keyword-retrieved statement memories are injected

These are the reaction decisions. Memory enters via the `context` variable
(`retrieved["events"]` reformatted with the predicate replaced by "was", then
`retrieved["thoughts"]` appended) and lands on the prompt's `Context:` line.

- `decide_to_talk` — `run_gpt_prompt_decide_to_talk`, template
  `v2/decide_to_talk_v2.txt`.
  - **Log signature:** `initiate a conversation with`
  - The `Context:` line right before `Question: Would X initiate a
    conversation with Y?` holds the retrieved statement(s).
- `decide_to_react` — `run_gpt_prompt_decide_to_react`, template
  `v2/decide_to_react_v1.txt`.
  - **Log signature:** `Of the following three options, what should` (this also
    matches the two hard-coded `Jane`/`Sam` few-shot examples in every prompt;
    the **real** instance is the last one, after the final `---`).
  - The `Context:` line after the Sam few-shot holds the retrieved
    statement(s). The `Jane`/`Liz`/`Sam`/`Sarah` lines are static template
    text, **not** memory.

Caveats: keyword retrieval usually returns just **one** statement (often the
perceived event echoed back), unlike the 15–50 node cosine lists. And because
it is keyword-based, there is **no cosine score** for these — they are *not* in
the new memory-retrieval log (section 5).

---

## 4. Are these ALL the places statement memories are injected?

Yes. Exhaustively, statement (memory-stream) injection happens at exactly:

- **Cosine (`new_retrieve`):** reflection insight (2a), `revise_identity`
  (2b), conversation relationship summary + utterance (2c), interview analysis
  mode (2d).
- **Keyword (`retrieve_relevant_*`):** `decide_to_talk`, `decide_to_react`
  (section 3).

Everything else that references a persona's state in a prompt uses the
**scratch** profile (the `Name / Age / Innate traits / Learned traits /
Currently / Lifestyle / Daily plan requirement` block, via
`scratch.get_str_iss()`), the spatial-memory tree, or the schedule — none of
which are memory-stream retrievals. (`agent_chat_v1` and
`generate_agent_chat_summarize_ideas` also call `new_retrieve` but are dead
code in this build.)

### grep cheat-sheet

```bash
# cosine path
grep -n "high-level insights can you infer"        backend_prompt_pairs.txt  # reflection
grep -n "should remember as they plan"             backend_prompt_pairs.txt  # revise_identity
grep -n "Based on the statements above, summarize" backend_prompt_pairs.txt  # convo relationship
grep -n "is the memory that is in"                 backend_prompt_pairs.txt  # convo utterance
grep -n "most relevant to the interviewer"         backend_prompt_pairs.txt  # interview mode

# keyword path
grep -n "initiate a conversation with"                backend_prompt_pairs.txt  # decide_to_talk
grep -n "Of the following three options, what should" backend_prompt_pairs.txt  # decide_to_react
```

---

## 5. The new `memory_retrieval_log`

The prompt-pair log shows the statements *after* they are flattened into a
prompt, but not *why* those statements were chosen (their similarity scores) or
what they were scored against. The memory-retrieval log fills that gap.

- **Module:** `reverie/backend_server/harnesses/memory_retrieval_log.py`,
  hooked into `new_retrieve` (the cosine path only).
- **What it records, per retrieval:** the **seed statement** (the focal point —
  the answer to "cosine similarity *to what?*") and, for each returned node,
  both views of the relevance, so there's no ambiguity about which number
  drove the ranking:
  - `cosine_raw` — the **true** cosine similarity to the seed (raw `cos_sim`
    output, ~[0,1], comparable across retrievals). This is *interpretable* but
    is **not** the value the ranking multiplies in.
  - `relevance_norm` / `recency_norm` / `importance_norm` — each component
    **min-max normalized within this retrieval's candidate set**
    (`normalize_dict_floats`: lowest→0, highest→1). These are what the score
    uses.
  - `w_recency` / `w_relevance` / `w_importance` — the weighted contributions
    (`weight * *_norm * gw`); they **sum to exactly `score`**.
  - `score` — the final combined value that ranked the node.
  - record-level `weights` — the persona `recency_w`/`relevance_w`/
    `importance_w` and the `gw` global multipliers, so the formula is
    self-documenting.

  **Important:** the ranking does **not** use the raw cosine; it uses the
  per-retrieval min-max-normalized relevance (then weighted ×`relevance_w`×`gw[1]`,
  with `gw=[0.5, 3, 2]`). The same min-max is applied to recency and importance.
  So `cosine_raw` is the genuine similarity, while `relevance_norm` /
  `w_relevance` are what actually decided the order.
- **Where the file lands:** it auto-derives from `REVERIE_PROMPT_LOG` — same
  directory, filename `memory_retrieval_log_<harness>_<stamp>.jsonl` (+ a
  human-readable `.txt` sibling), so the keyword, the LLM under test, and the
  date are all in the name. No change to `run_remote_servers.sh` /
  `midnight_test.sh` is needed; it appears next to `backend_prompt_pairs.*`
  automatically. An explicit `REVERIE_MEMORY_RETRIEVAL_LOG=<path>` overrides.
- **`.txt` format:** one banner block per retrieval — caller, seed, the score
  formula, and a table `rank | cos_raw | w_rec | w_rel | w_imp | score |
  statement` (the three `w_*` columns sum to `score`; full raw+normalized
  values are in the JSONL).

Note the keyword reaction path (section 3) has no cosine score and is
deliberately **not** logged here.

---

## 6. Finding: the reaction prompts never get a usable answer

Empirically, across all three 2026-06-10 runs, **0%** of `decide_to_talk` /
`decide_to_react` calls produced an answer the harness accepted; every decision
fell back to its hard-coded fail-safe (`decide_to_talk` → `"yes"` = always
proceed to chat once the non-LLM gating passes; `decide_to_react` → `"3"` = no
wait/react). With `temperature=0` the 5 retries are byte-identical, so each
decision burns 5 model calls and then defaults.

| run | decide_to_talk (accepted / calls) | decide_to_react |
|---|---|---|
| gemma4-e2b | 0 / 40 | 0 / 15 |
| gemma4-e4b | 0 / 40 | 0 / 25 |
| legacy-gpt | 0 / 50 | 0 / 60 |

Two distinct root causes:

1. **20-token cap truncates the chain-of-thought.** Both templates pre-fill a
   `Reasoning:` / `Let's think step by step.` lead-in, but `max_tokens=20`, so
   e4b and legacy get guillotined mid-reasoning before reaching the answer line
   (e.g. `'Reasoning: Isabella is currently restocking supplies ... Maria is
   studying'` <EOF>). This is the "cut off before finishing reasoning" effect.
2. **Validator/template delimiter mismatch.** gemma4-e2b often *skipped*
   reasoning and answered directly (`Answer in "yes" or "no": no`), but the
   validator splits on `"Answer in yes or no:"` (no quotes) while the template
   primes the quoted form `Answer in "yes" or "no":`. The split never matches,
   so even a correct yes/no is discarded.

```1406:1415:reverie/backend_server/persona/prompt_template/run_gpt_prompt.py
  def __func_validate(gpt_response, prompt=""): 
    try: 
      if gpt_response.split("Answer in yes or no:")[-1].strip().lower() in ["yes", "no"]: 
        return True
      return False     
```

Net effect: the reaction subsystem ran entirely on its fail-safe defaults
(always-willing-to-talk, never-wait).

### Suggested fix (NOT yet implemented)

For `decide_to_talk` / `decide_to_react`: raise `max_tokens` well above 20
(and/or drop the `Reasoning:` pre-fill so the model emits the answer first),
**and** align the validator's split delimiter with the wording the template
actually elicits (the quoted `Answer in "yes" or "no":`, and likewise check the
`Answer: Option` form for `decide_to_react`). Left as a recommendation here per
request; not changed in code.
