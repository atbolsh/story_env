#!/usr/bin/env python3
"""Scan backend_console.log files for hard-failure markers.

Pure-stdlib; safe to run anywhere.

Counts occurrences of the failure strings the reverie backend / harnesses
print when something actually goes wrong (as opposed to the prompt noise the
console is mostly full of):

  FAIL SAFE TRIGGERED    a safe_* retry loop exhausted its retries
  Gemma4 ERROR           gemma4 chat_request raised
  ChatGPT ERROR          legacy chat path raised
  TOKEN LIMIT EXCEEDED   legacy/gemma completion call raised
  Traceback              any python traceback printed to the console
  Error / ERROR          generic, reported separately as "other"

Usage:
    python3 scan_console_logs.py logs/midnight_*_2026-06-10_20-56-07
"""

import argparse
import os
import re
import sys
from collections import Counter

MARKERS = [
    ("FAIL SAFE TRIGGERED", re.compile(r"FAIL SAFE TRIGGERED")),
    ("Gemma4 ERROR", re.compile(r"Gemma4 ERROR")),
    ("ChatGPT ERROR", re.compile(r"ChatGPT ERROR")),
    ("TOKEN LIMIT EXCEEDED", re.compile(r"TOKEN LIMIT EXCEEDED")),
    ("Traceback", re.compile(r"^Traceback \(most recent call last\)")),
]


def scan(path, context=2):
    counts = Counter()
    samples = {}  # marker -> list of (lineno, surrounding lines)
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        for name, rx in MARKERS:
            if rx.search(line):
                counts[name] += 1
                if len(samples.setdefault(name, [])) < 3:
                    lo = max(0, i - context)
                    hi = min(len(lines), i + context + 1)
                    snippet = "".join(lines[lo:hi]).rstrip()
                    samples[name].append((i + 1, snippet))
    return counts, samples, len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+",
                    help="run dirs (containing backend_console.log) or log files")
    ap.add_argument("--show-samples", action="store_true",
                    help="print up to 3 context snippets per marker")
    args = ap.parse_args()

    for p in args.paths:
        log = p if p.endswith(".log") else os.path.join(p, "backend_console.log")
        if not os.path.exists(log):
            print(f"== {p}: no console log found, skipping", file=sys.stderr)
            continue
        counts, samples, nlines = scan(log)
        print(f"\n== {log}  ({nlines} lines)")
        if not counts:
            print("   no failure markers found")
        for name, _ in MARKERS:
            if counts.get(name):
                print(f"   {name}: {counts[name]}")
        if args.show_samples:
            for name, snips in samples.items():
                for lineno, snippet in snips:
                    print(f"\n   --- {name} @ line {lineno} ---")
                    for ln in snippet.splitlines():
                        print(f"   | {ln}")


if __name__ == "__main__":
    main()
