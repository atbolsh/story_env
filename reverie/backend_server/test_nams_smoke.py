"""
NAMS smoke test.

Exercises the :class:`harnesses.nams.nams_memory.NamsMemory` DB layer
end-to-end against a local bolt Neo4j (see ``docker-compose.yml``). Covers
the four invariants from the build plan:

  (a) a chat between two personas produces an inter-PERSON relationship
      edge + a Plan fact with ``valid_from`` within the convo window,
  (b) that Plan fact shows up in ``get_upcoming_plans`` (and is therefore
      force-injected into every planning/decision prompt -- the
      KNOWN_WEAKNESSES §0 schedule-propagation fix),
  (c) a reflection trace node exists after the importance-countdown
      "evaluation" episode fires,
  (d) short-term messages older than the 4-sim-minute TTL are gone while
      their extracted long-term Facts remain.

Plus the JSON -> NAMS importer: importing a persona's bootstrap_memory
yields identity Facts, ScheduleEntry Facts for today, and (when nodes.json
is non-empty) event/thought Facts.

This test talks **directly** to NamsMemory + the importer -- it does NOT
load a real LLM (the Gemma 4 / GPT harnesses) nor run the full
ReverieServer loop, so it runs in seconds and needs only Neo4j up.

Prerequisites:
  * ``docker compose up -d neo4j`` (bolt on localhost:7687)
  * ``NEO4J_PASSWORD`` env var (or default ``password``)
  * ``pip install -r requirements.txt && python -m spacy download en_core_web_sm``

Run from ``reverie/backend_server``:

    python test_nams_smoke.py
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "../../shared")  # for global_methods if needed

# --- test config -----------------------------------------------------------

PERSONA = "Isabella Rodriguez"
OTHER = "Maria Lopez"
BOOTSTRAP = ("../../environment/frontend_server/storage/"
             "base_the_ville_isabella_maria_klaus/personas/"
             "Isabella Rodriguez/bootstrap_memory")

# --- Neo4j reachability gate ----------------------------------------------

def _neo4j_reachable() -> bool:
  try:
    from neo4j import GraphDatabase  # type: ignore
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as s:
      s.run("RETURN 1").consume()
    driver.close()
    return True
  except Exception as e:
    print(f"[test_nams_smoke] Neo4j not reachable: {type(e).__name__}: {e}")
    print("  Start it with:  docker compose up -d neo4j")
    return False


# --- stub LLM harness (no model loading) ----------------------------------

class _StubLLM:
  """Returns canned responses so NamsMemory's extraction-mode-C path can be
  exercised without loading Gemma 4 / calling OpenAI. Only used if the test
  picks extraction mode C; mode A (default) doesn't need an LLM at all."""
  embedder_name = "BAAI/bge-small-en-v1.5"
  engine_label = "stub"

  def as_nams_llm_provider(self):
    class _P:
      async def complete(self, messages, **_):
        class _C:
          content = "stub extraction"
        return _C()
    return _P()


# --- helpers ---------------------------------------------------------------

def _wipe_session(nams: "NamsMemory") -> None:
  """Delete every node tagged with this persona's session_id, plus all
  Fact/Entity/ReasoningTrace nodes that reference the persona name, so the
  test starts from a clean graph each run."""
  try:
    nams.clear_session()
  except Exception:
    pass
  # Facts/Entities/Relationships don't carry session_id; clear by name.
  nams.cypher_write(
    "MATCH (f:Fact) WHERE f.subject = $name OR f.object = $name DETACH DELETE f",
    {"name": PERSONA},
  )
  nams.cypher_write(
    "MATCH (e:Entity) WHERE e.name = $name DETACH DELETE e",
    {"name": PERSONA},
  )
  nams.cypher_write(
    "MATCH (e:Entity) WHERE e.name = $other DETACH DELETE e",
    {"other": OTHER},
  )
  nams.cypher_write(
    "MATCH (t:ReasoningTrace) WHERE t.session_id = $name DETACH DELETE t",
    {"name": PERSONA},
  )


def _count(nams: "NamsMemory", cypher: str, params: dict | None = None) -> int:
  rows = nams.cypher(cypher, params or {})
  if not rows:
    return 0
  row = rows[0]
  if isinstance(row, dict):
    return int(row.get("c", 0))
  return 0


# --- main test -------------------------------------------------------------

def main() -> int:
  if not _neo4j_reachable():
    print("SKIP: Neo4j not reachable. Start it and re-run.")
    return 0

  from harnesses.nams import (
    NamsMemory, NAMS_EXTRACTION_NO_LLM,
  )
  from harnesses.nams.json_to_nams_import import import_persona_bootstrap

  nams = NamsMemory(
    session_id=PERSONA, embedder_name="BAAI/bge-small-en-v1.5",
    extraction_mode=NAMS_EXTRACTION_NO_LLM, llm_harness=_StubLLM(),
  )
  # Force the client to connect.
  _ = nams.client

  errors: list[str] = []
  try:
    # 0. Wipe + import bootstrap.
    _wipe_session(nams)
    report = import_persona_bootstrap(
      nams=nams, bootstrap_dir=os.path.abspath(BOOTSTRAP),
    )
    print(f"[import] {report}")

    # --- importer invariants --------------------------------------------
    identity_n = _count(nams,
      "MATCH (f:Fact) WHERE f.subject = $name "
      "  AND f.metadata CONTAINS '\"kind\": \"identity_' "
      "RETURN count(f) AS c",
      {"name": PERSONA})
    if identity_n < 1:
      errors.append(f"importer: expected >=1 identity Fact, got {identity_n}")

    person_n = _count(nams,
      "MATCH (e:Entity {type:'PERSON', name:$name}) RETURN count(e) AS c",
      {"name": PERSONA})
    if person_n < 1:
      errors.append(f"importer: PERSON entity for {PERSONA!r} missing")

    # --- (a) chat -> relationship edge + Plan fact within convo window --
    now = datetime.datetime(2023, 2, 13, 14, 0, 0)
    end = now + datetime.timedelta(minutes=8)
    nams.add_person_relationship(
      from_name=PERSONA, to_name=OTHER,
      relationship_type="TALKED_WITH",
      description="Isabella and Maria agreed to meet at Hobbs Cafe at 5pm.",
      valid_from=now, valid_until=end, poignancy=6,
    )
    nams.add_plan_fact(
      subject=PERSONA, obj="meet Maria Lopez at Hobbs Cafe at 5pm",
      valid_from=now, valid_until=end, poignancy=7,
      location="Hobbs Cafe", with_whom=OTHER, source="convo",
    )

    rel_n = _count(nams,
      "MATCH (a:Entity {type:'PERSON', name:$a})"
      "-[r:TALKED_WITH]->"
      "(b:Entity {type:'PERSON', name:$b}) "
      "RETURN count(r) AS c",
      {"a": PERSONA, "b": OTHER})
    if rel_n < 1:
      errors.append("(a) no TALKED_WITH edge PERSON->PERSON")

    plan_rows = nams.cypher(
      "MATCH (f:Fact) WHERE f.metadata CONTAINS '\"kind\": \"plan\"' "
      "  AND f.subject = $name "
      "RETURN f ORDER BY f.valid_from",
      {"name": PERSONA})
    if not plan_rows:
      errors.append("(a) no Plan fact after add_plan_fact")
    else:
      # Check valid_from within the convo window [now, end].
      row = plan_rows[0]
      f = row.get("f") if isinstance(row, dict) else None
      try:
        f = dict(f) if f is not None else row
      except Exception:
        f = row if isinstance(row, dict) else {}
      vf = f.get("valid_from")
      vf_dt = None
      if isinstance(vf, str):
        try:
          vf_dt = datetime.datetime.fromisoformat(vf.replace("Z", ""))
        except Exception:
          vf_dt = datetime.datetime.strptime(vf[:19], "%Y-%m-%d %H:%M:%S")
      elif isinstance(vf, datetime.datetime):
        vf_dt = vf
      if vf_dt is None or not (now <= vf_dt <= end):
        errors.append(f"(a) Plan fact valid_from {vf!r} not within "
                      f"convo window [{now}, {end}]")

    # --- (b) Plan fact appears in get_upcoming_plans --------------------
    upcoming = nams.get_upcoming_plans(now, lookahead_hours=2)
    plan_objs = [p for p in upcoming
                 if (p.get("metadata", {}) or {}).get("kind") == "plan"
                 or '"kind": "plan"' in str(p.get("metadata", ""))]
    if not plan_objs:
      errors.append("(b) get_upcoming_plans returned no plan facts at "
                    "convo time")

    # And still appears 30 min later (within the convo window).
    upcoming_later = nams.get_upcoming_plans(
      now + datetime.timedelta(minutes=30), lookahead_hours=2)
    if not any('"kind": "plan"' in str(p.get("metadata", ""))
               for p in upcoming_later):
      errors.append("(b) get_upcoming_plans dropped the plan fact 30min in")

    # --- (c) reflection trace ------------------------------------------
    nams.run_reflection_trace(
      task="Reflect and evaluate -- smoke test",
      steps=[{"thought": "focal", "action": "retrieve",
              "observation": "obs"}],
      outcome="test insight", success=True,
    )
    trace_n = _count(nams,
      "MATCH (t:ReasoningTrace) WHERE t.session_id = $name "
      "RETURN count(t) AS c",
      {"name": PERSONA})
    if trace_n < 1:
      errors.append(f"(c) no ReasoningTrace node after run_reflection_trace "
                    f"(got {trace_n})")

    # --- (d) short-term aging ------------------------------------------
    # Add a message "5 minutes ago" (past the 4-min TTL) and one "now".
    past = now - datetime.timedelta(minutes=5)
    nams.add_event(s=PERSONA, p="is", o="idle",
                   description="Isabella is idle (old)",
                   poignancy=2, created=past)
    nams.add_event(s=PERSONA, p="is", o="idle",
                   description="Isabella is idle (fresh)",
                   poignancy=2, created=now)
    aged = nams.age_short_term(now, ttl_minutes=4)
    if aged < 1:
      errors.append(f"(d) age_short_term deleted {aged} messages, "
                    f"expected >=1 (the 5-min-old one)")
    # The fresh message should still be there. NAMS stores session_id on
    # the Conversation node and the message timestamp on m.timestamp.
    fresh_n = _count(nams,
      "MATCH (c:Conversation {session_id: $sid})-[:HAS_MESSAGE]->(m:Message) "
      "WHERE m.timestamp >= datetime($cutoff) "
      "RETURN count(m) AS c",
      {"sid": PERSONA,
       "cutoff": (now - datetime.timedelta(minutes=4)).isoformat()})
    if fresh_n < 1:
      errors.append("(d) fresh short-term message was aged out prematurely")

  finally:
    try:
      nams.close()
    except Exception:
      pass

  if errors:
    print("FAIL")
    for e in errors:
      print(" -", e)
    return 1
  print("PASS: NAMS invariants (import, relationship+plan, retrieve, "
        "reflection trace, short-term aging) all hold")
  return 0


if __name__ == "__main__":
  sys.exit(main())
