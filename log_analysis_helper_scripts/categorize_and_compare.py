#!/usr/bin/env python3
"""Per-prompt-template comparison across harness runs.

Pure-stdlib; safe to run on any machine (no torch, no model loads).

Buckets every record in each backend_prompt_pairs.jsonl by which
run_gpt_prompt template produced it (keyword fingerprints over the user
message), then reports per template per run:

  * call count, retry rate, gave-up rate (hit the safe_* repeat ceiling)
  * JSON-parse health, including legacy-gpt's JSON-wrapped "chat" calls
    (detected by the 'Output the response to the prompt above in json'
    wrapper) using the same rfind('}') logic legacy_gpt.py uses
  * median per-call latency (timestamp delta, 1s resolution)

With --dump-samples FILE it also writes N sample prompt/response pairs per
(template, run) to FILE for human side-by-side quality reading.

Usage:
    python3 categorize_and_compare.py logs/midnight_*_2026-06-10_20-56-07 \
        --dump-samples /tmp/samples.txt --samples-per-bucket 3
"""

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime

# Ordered (category, list-of-substrings) rules; first match wins. Substrings
# are distinctive phrases from persona/prompt_template/v2,v3 template files.
RULES = [
    ("wake_up_hour", ["wake up hour"]),
    ("daily_plan_broad", ["Today is", "Here is", "broad-stroke", "daily plan"]),
    ("hourly_schedule", ["Hourly schedule format"]),
    ("task_decomp", ["in 5 min increments", "minutes left"]),
    ("action_sector", ["area in {", "MUST be one of", "Answer: {"]),
    ("action_arena", ["MUST pick one of"]),
    ("action_object", ["pick ONE most relevant object", "Objects available:"]),
    ("pronunciatio", ["emoji", "Emoji"]),
    ("event_triple", ["(subject, predicate, object)"]),
    ("obj_event", ["What is the "]),
    ("schedule_revision", ["originally planned schedule"]),
    ("conversation_utterance", ["utterance"]),
    ("conv_summary", ["conversing about"]),
    ("poignancy", ["poignancy"]),
    ("focal_points", ["Given only the information above",
                      "questions we can answer"]),
    ("insight_evidence", ["insights", "What insights can you infer",
                          "high-level insights"]),
    ("memo_on_convo", ["memo"]),
    ("relationship", ["What do they feel or know about each other"]),
    ("keyword_thought", ["thoughts", "feelings about"]),
    ("first_daily_plan", ["start her day", "start his day", "start their day"]),
    ("whisper_inner_thought", ["inner thought"]),
    ("safety_or_misc", []),
]

JSON_WRAPPER = "Output the response to the prompt above in json"
REPEAT_CEILING = {"chat_json": 3, "completion": 5, "chat": 3}


def categorize(user_text):
    for cat, subs in RULES:
        for s in subs:
            if s in user_text:
                return cat
    return "uncategorized"


def user_content(rec):
    msgs = (rec.get("request") or {}).get("messages") or []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def is_json_call(rec, user):
    return rec.get("kind") == "chat_json" or (
        rec.get("kind") == "chat" and JSON_WRAPPER in user)


def legacy_style_parse_ok(raw):
    """legacy_gpt.py's parse: trim to last '}', json.loads, take ['output']."""
    end_index = raw.rfind("}") + 1
    try:
        obj = json.loads(raw[:end_index])
        return isinstance(obj, dict) and "output" in obj
    except Exception:
        return False


def request_key(rec):
    msgs = (rec.get("request") or {}).get("messages")
    return rec.get("kind"), json.dumps(msgs, sort_keys=True, default=str)


def parse_ts(s):
    return datetime.fromisoformat(s)


def load(jsonl_path):
    recs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    return recs


def analyze_run(jsonl_path):
    recs = load(jsonl_path)
    # group consecutive identical requests into call sites
    groups = []  # (first_rec, size, latencies, category, json_fail_count)
    i, n = 0, len(recs)
    prev_t = None
    lat = []  # per-record latency, aligned with recs
    for r in recs:
        t = parse_ts(r["ts"])
        lat.append(None if prev_t is None else (t - prev_t).total_seconds())
        prev_t = t
    while i < n:
        k = request_key(recs[i])
        j = i + 1
        while j < n and request_key(recs[j]) == k:
            j += 1
        groups.append((i, j - i))
        i = j

    stats = defaultdict(lambda: {
        "sites": 0, "calls": 0, "retried": 0, "gave_up": 0,
        "json_calls": 0, "json_parse_fail": 0, "lats": [], "examples": []})
    for start, size in groups:
        r0 = recs[start]
        user = user_content(r0)
        cat = categorize(user)
        st = stats[cat]
        st["sites"] += 1
        st["calls"] += size
        if size > 1:
            st["retried"] += 1
        if size >= REPEAT_CEILING.get(r0.get("kind"), 99):
            st["gave_up"] += 1
        for idx in range(start, start + size):
            r = recs[idx]
            u = user if idx == start else user_content(r)
            if is_json_call(r, u):
                st["json_calls"] += 1
                raw = str(((r.get("response") or {}).get("raw")) or "")
                if not legacy_style_parse_ok(raw):
                    st["json_parse_fail"] += 1
            if lat[idx] is not None:
                st["lats"].append(lat[idx])
        st["examples"].append(start)
    return recs, stats


def fmt_pct(a, b):
    return f"{100.0*a/b:5.1f}%" if b else "  n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--dump-samples", metavar="FILE", default=None)
    ap.add_argument("--samples-per-bucket", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    dump_f = open(args.dump_samples, "w", encoding="utf-8") if args.dump_samples else None

    for p in args.paths:
        jsonl = p if p.endswith(".jsonl") else os.path.join(p, "backend_prompt_pairs.jsonl")
        if not os.path.exists(jsonl):
            print(f"== {p}: no jsonl, skipping", file=sys.stderr)
            continue
        recs, stats = analyze_run(jsonl)
        run_name = os.path.basename(os.path.dirname(jsonl)) or jsonl
        print(f"\n{'='*100}\n== {run_name}  ({len(recs)} records)\n{'='*100}")
        header = (f"{'template':>24} {'sites':>6} {'calls':>6} {'retry%':>7} "
                  f"{'giveup%':>8} {'jsonN':>6} {'jsonFail%':>9} {'medLat':>7}")
        print(header)
        for cat in sorted(stats, key=lambda c: -stats[c]["sites"]):
            st = stats[cat]
            med = (f"{statistics.median(st['lats']):.0f}s"
                   if st["lats"] else "n/a")
            print(f"{cat:>24} {st['sites']:>6} {st['calls']:>6} "
                  f"{fmt_pct(st['retried'], st['sites']):>7} "
                  f"{fmt_pct(st['gave_up'], st['sites']):>8} "
                  f"{st['json_calls']:>6} "
                  f"{fmt_pct(st['json_parse_fail'], st['json_calls']):>9} "
                  f"{med:>7}")

        if dump_f:
            dump_f.write(f"\n{'#'*100}\n# RUN: {run_name}\n{'#'*100}\n")
            for cat in sorted(stats):
                st = stats[cat]
                picks = random.sample(
                    st["examples"],
                    min(args.samples_per_bucket, len(st["examples"])))
                for start in sorted(picks):
                    r = recs[start]
                    u = user_content(r)
                    resp = (r.get("response") or {})
                    dump_f.write(
                        f"\n----- [{cat}] {run_name} ts={r['ts']} "
                        f"kind={r['kind']} -----\n")
                    dump_f.write("--- prompt (tail) ---\n")
                    dump_f.write(u[-1200:] + "\n")
                    dump_f.write("--- response (returned) ---\n")
                    dump_f.write(str(resp.get("returned"))[:1200] + "\n")
                    if r.get("error"):
                        dump_f.write(f"--- ERROR: {r['error'][:300]}\n")
    if dump_f:
        dump_f.close()
        print(f"\nsamples written to {args.dump_samples}")


if __name__ == "__main__":
    main()
