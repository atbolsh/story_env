#!/usr/bin/env python3
"""
nams_connect_diagnostic.py -- standalone probe for the NAMS MemoryClient
__aenter__ hang.

Reproduces exactly what NamsMemory._ensure_client does (build MemorySettings,
construct MemoryClient, call __aenter__) but with:
  * a hard timeout (asyncio.wait_for) so it can never hang forever,
  * granular flushed prints around every step,
  * a direct bolt-connectivity pre-check (rules out Neo4j itself),
  * the SDK's own logger cranked to DEBUG so any internal download / query
    shows up on stderr.

Run from the repo root:
    PYTHONPATH=shared:reverie/backend_server python3 scripts/nams_connect_diagnostic.py

If it times out, the last printed line tells you which sub-step is stuck.
"""
import asyncio
import datetime
import logging
import os
import sys
import time

# Crank the NAMS SDK + neo4j driver loggers to DEBUG so downloads / queries
# surface on stderr instead of being silent.
for name in ("neo4j_agent_memory", "neo4j", "urllib3", "sentence_transformers",
             "transformers", "gliner", "spacy"):
  logging.getLogger(name).setLevel(logging.DEBUG)
  logging.getLogger(name).addHandler(logging.StreamHandler(sys.stderr))

# Make sure the harness path resolves.
sys.path.insert(0, os.path.join(os.getcwd(), "reverie", "backend_server"))
sys.path.insert(0, os.path.join(os.getcwd(), "shared"))

from harnesses.nams.nams_memory import build_memory_settings, NAMS_EXTRACTION_NO_LLM

PERSONA = "Klaus Mueller"
EMBEDDER = "BAAI/bge-small-en-v1.5"
TIMEOUT_S = 120  # hard cap; the sim path hangs indefinitely, this won't


def log(msg):
  print(f"[diag {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def bolt_precheck():
  """Direct neo4j-driver connectivity check, bypassing the NAMS SDK entirely.
  If this hangs too, the problem is Neo4j; if it's fast, the problem is in the
  SDK's __aenter__ (extraction pipeline init)."""
  log("bolt pre-check: connecting directly with the neo4j driver (bypasses SDK)...")
  t0 = time.time()
  from neo4j import GraphDatabase
  uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
  user = os.environ.get("NEO4J_USER", "neo4j")
  pw = os.environ.get("NEO4J_PASSWORD", "password")
  driver = GraphDatabase.driver(uri, auth=(user, pw))
  try:
    with driver.session() as s:
      rec = s.run("RETURN 1 AS x").single()
      log(f"bolt pre-check ok in {time.time() - t0:.1f}s: RETURN 1 -> {rec['x']}")
  finally:
    driver.close()


async def connect_with_probes():
  log("building MemorySettings (no-llm, same as the sim path)...")
  settings = build_memory_settings(
    embedder_name=EMBEDDER,
    extraction_mode=NAMS_EXTRACTION_NO_LLM,
    llm_harness=None,
    persona_name=PERSONA,
  )
  log("MemorySettings built.")

  log("constructing MemoryClient(settings) (cheap; no I/O yet)...")
  from neo4j_agent_memory import MemoryClient
  client = MemoryClient(settings)
  log("MemoryClient constructed.")

  log("calling client.__aenter__() -- this is where the sim hangs. "
      "It initializes the bolt driver pool + the extraction pipeline "
      "(spaCy + GLiNER + GLiREL models). Hard timeout = "
      f"{TIMEOUT_S}s. Watch stderr for SDK DEBUG logs / download bars.")
  t0 = time.time()
  try:
    await asyncio.wait_for(client.__aenter__(), timeout=TIMEOUT_S)
  except asyncio.TimeoutError:
    elapsed = time.time() - t0
    log(f"__aenter__ TIMED OUT after {elapsed:.1f}s. The SDK's context-entry "
        "is stuck -- not a download (no tqdm bars), so likely a Neo4j schema "
        "op (vector-index create/constraint) or an async deadlock inside the "
        "SDK. Check the DEBUG logs above for the last query / download it "
        "attempted.")
    return
  log(f"__aenter__ returned in {time.time() - t0:.1f}s -- connected + "
      "pipeline ready.")

  log("sanity: calling client.is_connected / get_stats...")
  try:
    connected = await asyncio.wait_for(_is_connected(client), timeout=30)
    log(f"is_connected -> {connected}")
  except Exception as e:
    log(f"is_connected probe failed: {type(e).__name__}: {e}")

  try:
    await client.__aexit__(None, None, None)
    log("client closed cleanly.")
  except Exception as e:
    log(f"__aexit__ failed: {type(e).__name__}: {e}")


async def _is_connected(client):
  if hasattr(client, "is_connected"):
    return client.is_connected
  if hasattr(client, "get_stats"):
    await client.get_stats()
    return True
  return "no is_connected attr"


def main():
  log(f"NAMS connect diagnostic: persona={PERSONA!r} embedder={EMBEDDER!r} "
      f"timeout={TIMEOUT_S}s")
  bolt_precheck()
  try:
    asyncio.run(connect_with_probes())
  except Exception as e:
    log(f"top-level error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
  log("done.")


if __name__ == "__main__":
  main()
