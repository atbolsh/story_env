#!/usr/bin/env python3
"""
nams_connect_diagnostic.py -- standalone probe for the NAMS MemoryClient hang.

Runs three probes and compares them:

  1. bolt pre-check       -- direct neo4j-driver connectivity (bypasses SDK).
  2. asyncio.run path     -- build MemoryClient, __aenter__, get_stats on a
                             fresh main-thread event loop (the control).
  3. async_bridge path    -- the SAME calls via harnesses.nams.async_bridge.run
                             (the persistent background loop the sim uses),
                             with a hard timeout. This is the suspect: probe 2
                             completes in <1s but the sim hangs for 20+ min
                             on this path.

If probe 3 times out where probe 2 succeeded, the bug is in the async bridge
(not the SDK, not Neo4j).

Run from the repo root:
    PYTHONPATH=shared:reverie/backend_server python3 scripts/nams_connect_diagnostic.py
"""
import asyncio
import datetime
import logging
import os
import sys
import time

for name in ("neo4j_agent_memory", "neo4j"):
  logging.getLogger(name).setLevel(logging.INFO)  # DEBUG is very noisy; bump to DEBUG if needed

sys.path.insert(0, os.path.join(os.getcwd(), "reverie", "backend_server"))
sys.path.insert(0, os.path.join(os.getcwd(), "shared"))

from harnesses.nams.nams_memory import build_memory_settings, NAMS_EXTRACTION_NO_LLM
from harnesses.nams import async_bridge

PERSONA = "Klaus Mueller"
EMBEDDER = "BAAI/bge-small-en-v1.5"
BRIDGE_TIMEOUT = 30  # probe 2 finishes in <1s; give the bridge 30s, plenty.


def log(msg):
  print(f"[diag {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def bolt_precheck():
  log("PROBE 1: bolt pre-check (direct neo4j driver, bypasses SDK)...")
  t0 = time.time()
  from neo4j import GraphDatabase
  uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
  user = os.environ.get("NEO4J_USER", "neo4j")
  pw = os.environ.get("NEO4J_PASSWORD", "password")
  driver = GraphDatabase.driver(uri, auth=(user, pw))
  try:
    with driver.session() as s:
      rec = s.run("RETURN 1 AS x").single()
      log(f"PROBE 1 ok in {time.time() - t0:.1f}s: RETURN 1 -> {rec['x']}")
  finally:
    driver.close()


async def _build_and_enter():
  """Build a MemoryClient and enter its context. Used by both probes."""
  settings = build_memory_settings(
    embedder_name=EMBEDDER,
    extraction_mode=NAMS_EXTRACTION_NO_LLM,
    llm_harness=None,
    persona_name=PERSONA,
  )
  from neo4j_agent_memory import MemoryClient
  client = MemoryClient(settings)
  await client.__aenter__()
  return client


def probe_asyncio_run():
  """Control: run __aenter__ + get_stats on a fresh main-thread loop."""
  log("PROBE 2 (control): __aenter__ + get_stats via asyncio.run (fresh loop)...")
  t0 = time.time()

  async def _go():
    client = await _build_and_enter()
    try:
      if hasattr(client, "get_stats"):
        await client.get_stats()
      log(f"PROBE 2 ok in {time.time() - t0:.1f}s: connected + get_stats done.")
    finally:
      await client.__aexit__(None, None, None)
  try:
    asyncio.run(_go())
  except Exception as e:
    log(f"PROBE 2 FAILED in {time.time() - t0:.1f}s: {type(e).__name__}: {e}")


def probe_bridge_enter():
  """Suspect: __aenter__ via the persistent background loop (what the sim does)."""
  log(f"PROBE 3 (suspect): __aenter__ via async_bridge.run "
      f"(persistent background loop) with {BRIDGE_TIMEOUT}s timeout...")
  t0 = time.time()
  try:
    client = async_bridge.run(_build_and_enter(), timeout=BRIDGE_TIMEOUT)
    log(f"PROBE 3 ok in {time.time() - t0:.1f}s: __aenter__ returned via bridge.")
    # If enter worked, try the actual graph_exists-style cypher read the sim does.
    log("PROBE 3b: client.query.cypher (the graph_exists query) via bridge...")
    t1 = time.time()

    async def _q():
      return await client.query.cypher(
        "OPTIONAL MATCH (c:Conversation {session_id: $sid}) "
        "OPTIONAL MATCH (e:Entity {name: $sid}) "
        "OPTIONAL MATCH (f:Fact) WHERE f.subject = $sid OR f.object = $sid "
        "RETURN count(c) + count(e) + count(f) AS c LIMIT 1",
        {"sid": PERSONA},
      )
    try:
      rows = async_bridge.run(_q(), timeout=BRIDGE_TIMEOUT)
      log(f"PROBE 3b ok in {time.time() - t1:.1f}s: rows = {rows}")
    except Exception as e:
      log(f"PROBE 3b FAILED in {time.time() - t1:.1f}s: {type(e).__name__}: {e}")
    # Close via the bridge too.
    try:
      async_bridge.run(client.__aexit__(None, None, None), timeout=BRIDGE_TIMEOUT)
    except Exception:
      pass
  except Exception as e:
    log(f"PROBE 3 FAILED/TIMED OUT in {time.time() - t0:.1f}s: "
        f"{type(e).__name__}: {e}")
    log("  -> The bug is in async_bridge.run (persistent background loop), "
        "NOT the SDK: probe 2 ran the same __aenter__ in <1s on a fresh loop.")


def main():
  log(f"NAMS connect diagnostic: persona={PERSONA!r} embedder={EMBEDDER!r}")
  bolt_precheck()
  probe_asyncio_run()
  probe_bridge_enter()
  async_bridge.shutdown()
  log("done.")


if __name__ == "__main__":
  main()
