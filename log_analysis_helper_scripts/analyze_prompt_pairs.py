#!/usr/bin/env python3
"""Analyze backend_prompt_pairs.jsonl logs from a midnight_test.sh run.

Pure-stdlib; safe to run on any machine (no torch, no model loads).

For each run directory (or bare .jsonl path) given on the command line:

  * record counts by `kind`, run wall-time span
  * hard errors: records with an `error` field (the call raised)
  * JSON-path health (`kind == "chat_json"`): re-runs the same extraction
    logic the gemma4 harness uses (brace-balanced extractor, then the
    rfind('}') fallback) and reports how many responses fail to parse as
    JSON or lack the required "output" key
  * retry analysis: consecutive records with an identical request are the
    safe_* loops retrying after a parse/validation failure; groups that hit
    the repeat ceiling (3 for chat_json, 5 for completion) mean the loop
    gave up and the caller's fail-safe value was used
  * timing: per-call latency approximated by the timestamp delta from the
    previous record (records are written immediately after each call
    returns, and calls are sequential), reported per `kind`

Usage:
    python3 analyze_prompt_pairs.py logs/midnight_*_2026-06-10_20-56-07
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime

# Repeat ceilings in the harness safe_* loops (see harnesses/gemma4.py and
# harnesses/legacy_gpt.py): chat_json retries 3x, completion retries 5x.
REPEAT_CEILING = {"chat_json": 3, "completion": 5, "chat": 3}


def extract_first_json_object(s):
    """Mirror of harnesses/gemma4.py::_extract_first_json_object."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def json_parse_status(raw):
    """Classify a chat_json raw response the same way _safe_json would.

    Returns one of:
      "ok"            parsed, has "output" key
      "no_output_key" parsed JSON but no "output" key
      "parse_fail"    could not be parsed as JSON at all
    """
    obj_str = extract_first_json_object(raw)
    if obj_str is None:
        end_index = raw.rfind("}") + 1
        obj_str = raw[:end_index] if end_index > 0 else raw
    try:
        parsed = json.loads(obj_str)
    except Exception:
        return "parse_fail"
    if not isinstance(parsed, dict) or "output" not in parsed:
        return "no_output_key"
    return "ok"


def request_key(rec):
    """Stable fingerprint of a request, used to detect retry runs."""
    msgs = (rec.get("request") or {}).get("messages")
    return rec.get("kind"), json.dumps(msgs, sort_keys=True, default=str)


def parse_ts(s):
    return datetime.fromisoformat(s)


def load_records(jsonl_path):
    records = []
    bad_lines = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                bad_lines += 1
    return records, bad_lines


def pct(n, d):
    return f"{100.0 * n / d:.2f}%" if d else "n/a"


def quantiles(vals):
    if not vals:
        return "n/a"
    vals = sorted(vals)
    med = statistics.median(vals)
    p90 = vals[min(len(vals) - 1, int(0.9 * len(vals)))]
    return (f"n={len(vals)} min={vals[0]:.0f}s med={med:.0f}s "
            f"p90={p90:.0f}s max={vals[-1]:.0f}s mean={statistics.mean(vals):.1f}s")


def analyze(jsonl_path):
    records, bad_lines = load_records(jsonl_path)
    if not records:
        print(f"  (no records in {jsonl_path})")
        return

    n = len(records)
    kinds = Counter(r.get("kind") for r in records)
    t0 = parse_ts(records[0]["ts"])
    t1 = parse_ts(records[-1]["ts"])
    span_s = (t1 - t0).total_seconds()

    print(f"  records: {n}  (+{bad_lines} unparseable lines)")
    print(f"  span: {records[0]['ts']} .. {records[-1]['ts']}  "
          f"({span_s:.0f}s = {span_s/3600:.2f}h)")
    print(f"  overall rate: {n / span_s * 60:.1f} calls/min" if span_s else "")
    print(f"  kinds: {dict(kinds)}")

    # ---- hard errors (call raised) -------------------------------------
    errors = [r for r in records if r.get("error")]
    print(f"\n  raised-exception records: {len(errors)} ({pct(len(errors), n)})")
    for r in errors[:5]:
        print(f"    {r['ts']}  [{r['kind']}]  {r['error'][:120]}")

    # ---- JSON path health ----------------------------------------------
    cj = [r for r in records if r.get("kind") == "chat_json"]
    if cj:
        status = Counter()
        parse_fail_examples = []
        for r in cj:
            raw = ((r.get("response") or {}).get("raw")) or ""
            st = json_parse_status(raw)
            status[st] += 1
            if st != "ok" and len(parse_fail_examples) < 5:
                parse_fail_examples.append((r["ts"], st, raw[:160].replace("\n", "\\n")))
        print(f"\n  chat_json records: {len(cj)}")
        for st in ("ok", "no_output_key", "parse_fail"):
            print(f"    {st:>14}: {status.get(st, 0):>6}  ({pct(status.get(st, 0), len(cj))})")
        for ts, st, ex in parse_fail_examples:
            print(f"    e.g. {ts} [{st}] {ex}")

    # ---- retry analysis --------------------------------------------------
    # Consecutive identical requests = the safe_* loop retrying.
    groups = []  # (kind, size)
    i = 0
    while i < n:
        k = request_key(records[i])
        j = i + 1
        while j < n and request_key(records[j]) == k:
            j += 1
        groups.append((records[i].get("kind"), j - i))
        i = j

    retried = [(k, sz) for k, sz in groups if sz > 1]
    gave_up = [(k, sz) for k, sz in retried
               if sz >= REPEAT_CEILING.get(k, 99)]
    n_first_try = sum(1 for _, sz in groups if sz == 1)
    print(f"\n  unique call sites (request groups): {len(groups)}")
    print(f"    first-try success: {n_first_try} ({pct(n_first_try, len(groups))})")
    print(f"    retried at least once: {len(retried)} ({pct(len(retried), len(groups))})")
    retry_by_kind = Counter(k for k, _ in retried)
    if retry_by_kind:
        print(f"    retries by kind: {dict(retry_by_kind)}")
    print(f"    hit repeat ceiling (fail-safe likely used): {len(gave_up)} "
          f"({pct(len(gave_up), len(groups))})")
    giveup_by_kind = Counter(k for k, _ in gave_up)
    if giveup_by_kind:
        print(f"    give-ups by kind: {dict(giveup_by_kind)}")

    # ---- timing ----------------------------------------------------------
    # Delta from previous record ~= this call's latency (+small sim overhead).
    deltas_by_kind = defaultdict(list)
    big_gaps = []
    prev = t0
    for r in records[1:]:
        t = parse_ts(r["ts"])
        d = (t - prev).total_seconds()
        prev = t
        deltas_by_kind[r.get("kind")].append(d)
        if d >= 60:
            big_gaps.append((r["ts"], r.get("kind"), d))
    print("\n  per-call latency (ts delta from previous record, 1s resolution):")
    for k in sorted(deltas_by_kind, key=lambda k: -len(deltas_by_kind[k])):
        print(f"    {k:>12}: {quantiles(deltas_by_kind[k])}")
    if big_gaps:
        print(f"  gaps >= 60s: {len(big_gaps)}")
        for ts, k, d in big_gaps[:5]:
            print(f"    {ts}  [{k}]  {d:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+",
                    help="run directories (containing backend_prompt_pairs.jsonl) "
                         "or .jsonl files")
    args = ap.parse_args()
    for p in args.paths:
        jsonl = p if p.endswith(".jsonl") else os.path.join(p, "backend_prompt_pairs.jsonl")
        if not os.path.exists(jsonl):
            print(f"== {p}: no jsonl found, skipping", file=sys.stderr)
            continue
        print(f"\n{'=' * 76}\n== {jsonl}\n{'=' * 76}")
        analyze(jsonl)


if __name__ == "__main__":
    main()
