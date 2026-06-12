#!/usr/bin/env python3
"""Standalone Gemma 4 latency benchmark -- REMOTE BOX ONLY.

Do NOT run this on a local dev machine: it loads Gemma 4 weights through the
gemma4 harness and needs the full requirements.txt environment (torch,
transformers, GPU). It does *not* start the Django frontend or reverie.py --
it drives the harness object directly, exactly the way the simulation does.

It times a small set of prompts representative of the midnight-run call mix:

  short completion   wake-up-hour style, max_tokens=5      (most common call)
  medium completion  event-triple / object-choice style    (~15-30 tokens)
  long completion    schedule-decomposition style, max_tokens=1000
  chat_json          poignancy rating and emoji conversion (JSON-wrapped)

Each prompt is run once as warm-up (excluded; the first call also triggers
weight loading), then --repeats times. Reports wall seconds per call and
approximate output tokens/sec (tokenizing the returned text).

Usage, from the repo root on the remote box:

    python3 log_analysis_helper_scripts/time_gemma_prompts.py --model e4b
    python3 log_analysis_helper_scripts/time_gemma_prompts.py --model e2b --repeats 5
"""

import argparse
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "reverie", "backend_server"))

MODEL_IDS = {
    "e2b": "google/gemma-4-E2B-it",
    "e4b": "google/gemma-4-E4B-it",
}

PERSONA_HEADER = """\
Name: Isabella Rodriguez
Age: 34
Innate traits: friendly, outgoing, hospitable
Learned traits: Isabella Rodriguez is a cafe owner of Hobbs Cafe who loves to make people feel welcome.
Currently: Isabella Rodriguez is planning on having a Valentine's Day party at Hobbs Cafe on February 14th, 2023 at 5pm.
Lifestyle: Isabella Rodriguez goes to bed around 11pm, awakes up around 6am.
Daily plan requirement: Isabella Rodriguez opens Hobbs Cafe at 8am everyday, and works at the counter until 8pm.
Current Date: Monday February 13
"""

COMPLETION_CASES = [
    # (label, prompt, gpt_parameter) -- mirrors run_gpt_prompt.py params
    ("short_completion_wakeup",
     PERSONA_HEADER + "\n\nIn general, Isabella Rodriguez goes to bed around "
     "11pm, awakes up around 6am.\nIsabella's wake up hour:",
     {"max_tokens": 5, "temperature": 0.8, "top_p": 1, "stop": ["\n"]}),

    ("medium_completion_object",
     "Current activity: cooking\n"
     "Objects available: {stove, sink, fridge, counter}\n"
     "Pick ONE most relevant object from the objects available: stove\n---\n"
     "Current activity: study\n"
     "Objects available: {desk, computer, chair, bookshelf}\n"
     "Pick ONE most relevant object from the objects available: desk\n---\n"
     "Current activity: tidying the counter area\n"
     "Objects available: {refrigerator, cafe customer seating, kitchen sink, "
     "behind the cafe counter, piano}\n"
     "Pick ONE most relevant object from the objects available:",
     {"max_tokens": 15, "temperature": 0, "top_p": 1, "stop": ["---"]}),

    ("long_completion_decomp",
     PERSONA_HEADER + "\n"
     "Today is Monday February 13. From 08:00am ~ 09:00am, Isabella Rodriguez "
     "is planning on opening Hobbs Cafe and preparing for the day. In 5 min "
     "increments, list the subtasks Isabella does when Isabella is opening "
     "Hobbs Cafe and preparing for the day (total duration in minutes 60):\n"
     "1) Isabella is",
     {"max_tokens": 1000, "temperature": 0, "top_p": 1, "stop": None}),
]

JSON_CASES = [
    # (label, prompt, example_output, special_instruction)
    ("chat_json_poignancy",
     PERSONA_HEADER + "\n\nOn the scale of 1 to 10, where 1 is purely mundane "
     "and 10 is extremely poignant, rate the likely poignancy of the "
     "following event for Isabella Rodriguez.\n\n"
     "Event: setting up decorations for the Valentine's Day party\n"
     "Rate (return a number between 1 to 10):",
     "5",
     "The output should ONLY contain ONE integer value on the scale of 1 to 10."),

    ("chat_json_emoji",
     "Convert an action description to an emoji (important: use two or less "
     "emojis).\n\nAction description: brewing coffee for the morning rush\nEmoji:",
     "\U0001F6C1\U0001F9D6",
     "The value for the output must ONLY contain the emojis."),
]


def time_call(fn, repeats):
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return times, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODEL_IDS), default="e4b")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    from harnesses import gemma4  # heavy imports happen lazily inside
    h = gemma4.build(MODEL_IDS[args.model])

    print(f"benchmarking {MODEL_IDS[args.model]} (repeats={args.repeats}, "
          f"first warm-up call per case excluded)")

    # Warm-up / weight load.
    t0 = time.perf_counter()
    h.llm_request("Say hi.", {"max_tokens": 5, "temperature": 0, "top_p": 1,
                              "stop": ["\n"]})
    print(f"warm-up + weight load: {time.perf_counter() - t0:.1f}s\n")

    def n_tokens(text):
        try:
            return len(h._processor(text=text)["input_ids"][0])
        except Exception:
            return None

    rows = []
    for label, prompt, params in COMPLETION_CASES:
        time_call(lambda: h.llm_request(prompt, params), 1)  # per-case warm-up
        times, out = time_call(lambda: h.llm_request(prompt, params),
                               args.repeats)
        toks = n_tokens(out or "")
        rows.append((label, times, out, toks))

    for label, prompt, example, instruction in JSON_CASES:
        fn = lambda: h.safe_chat_response_json(
            prompt, example, instruction, repeat=1,
            func_validate=lambda r, prompt="": True,
            func_clean_up=lambda r, prompt="": r)
        time_call(fn, 1)
        times, out = time_call(fn, args.repeats)
        toks = n_tokens(str(out) if out is not False else "")
        rows.append((label, times, str(out), toks))

    print(f"{'case':>26} {'mean':>7} {'min':>7} {'max':>7} {'~tok/s':>7}  output (head)")
    for label, times, out, toks in rows:
        mean = statistics.mean(times)
        tps = f"{toks / mean:.1f}" if (toks and mean > 0) else "n/a"
        head = (out or "").replace("\n", "\\n")[:60]
        print(f"{label:>26} {mean:6.2f}s {min(times):6.2f}s "
              f"{max(times):6.2f}s {tps:>7}  {head}")


if __name__ == "__main__":
    main()
