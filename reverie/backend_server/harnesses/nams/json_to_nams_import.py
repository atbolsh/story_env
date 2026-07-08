"""
One-way translator from the legacy JSON bootstrap memory into NAMS.

Run once when a *-nams harness boots a sim that was forked from a JSON
bootstrap (so the persona already has a populated ``bootstrap_memory/``).
Gated by ``NamsMemory.graph_exists()`` so re-runs against an already-imported
graph are no-ops.

For each persona's ``bootstrap_memory/``:

  * ``spatial_memory.json``    -> LOCATION entities + CONTAINS relationships
                                  (one Entity per sector/arena/object, a
                                  typed edge from sector down to object).
  * ``scratch.json``           -> PERSON entity (the persona) + permanent
                                  identity Facts (innate / learned / currently
                                  / lifestyle / daily_plan_req).
  * ``scratch.f_daily_schedule`` -> temporal ScheduleEntry Facts (valid_from =
                                  today start + cumulative minutes).
  * ``associative_memory/nodes.json``:
      * ``event``  -> Fact(subject, predicate, obj, kind='event',
                           salience=poignancy, valid_from=created)
      * ``chat``   -> one Conversation + Messages then immediate
                      extract+clear (or, if ``filling`` is the transcript,
                      a single Fact with the transcript as ``obj``).
      * ``thought``-> Fact(kind='thought', salience=poignancy).
  * ``kw_strength.json``       -> bumps ``metadata.salience`` on the
                                  matching Fact/Entity (best-effort match by
                                  keyword against ``keywords``).
  * ``embeddings.json``        -> discarded (NAMS re-embeds everything).
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Any, Optional

from .nams_memory import NamsMemory


# --- helpers ---------------------------------------------------------------

def _parse_created(s: str) -> datetime.datetime:
  """Legacy nodes store created as 'YYYY-MM-DD HH:MM:SS' (no tz)."""
  if not s:
    return datetime.datetime.now()
  try:
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
  except ValueError:
    try:
      return datetime.datetime.fromisoformat(s)
    except Exception:
      return datetime.datetime.now()


def _sim_day_start(scratch: dict) -> datetime.datetime:
  """Best-effort start-of-day for schedule valid_from. Falls back to the
  earliest event created date in nodes.json, then to 'today'."""
  pass


def _kw_salience(kw_strength: dict, keywords: list[str]) -> Optional[int]:
  """Pick the highest kw_strength count among the node's keywords, capped
  to 10 (poignancy scale). Returns None if no kw_strength data."""
  if not kw_strength:
    return None
  best = 0
  for k in keywords or []:
    if not k:
      continue
    kl = k.lower()
    for table in ("kw_strength_event", "kw_strength_thought"):
      best = max(best, int(kw_strength.get(table, {}).get(kl, 0)))
  return min(10, best) if best > 0 else None


def _salience_for_node(node: dict, kw_strength: dict) -> int:
  """Combine the legacy poignancy score and kw_strength count into a single
  1-10 salience. kw_strength is a coarse popularity signal; the per-event
  poignancy is the more direct 'how salient is this' signal, so it wins when
  they disagree."""
  p = int(node.get("poignancy") or 0)
  kw = _kw_salience(kw_strength, node.get("keywords") or [])
  if kw is not None and kw > p:
    return kw
  return max(1, min(10, p or 1))


# --- per-file importers ----------------------------------------------------

def _import_spatial(nams: NamsMemory, spatial: dict) -> int:
  """spatial_memory.json: {ville: {sector: {arena: [object, ...]}}}
  -> LOCATION Entity per sector/arena/object + CONTAINS edges."""
  count = 0
  for ville, sectors in (spatial or {}).items():
    # The Ville itself is the root location entity.
    for sector, arenas in (sectors or {}).items():
      for arena, objects in (arenas or {}).items():
        # arena-level entity
        for obj in (objects or []):
          # We model the leaf objects as LOCATION entities with a CONTAINS
          # edge from arena -> object. Skips the ville/sector nesting edges
          # to keep the graph shallow; the arena string already encodes them.
          #
          # Create the entities through the SDK's add_entity so they get the
          # id UUID + embedding NAMS's readers require (a raw MERGE (:Entity)
          # here produced id-less nodes that crashed reflection's entity
          # search with KeyError 'id'). The CONTAINS edge is then written by
          # name via cypher_write -- it matches the SDK-created nodes.
          arena_name = f"{sector}:{arena}"
          obj_name = f"{sector}:{arena}:{obj}"
          try:
            nams.add_entity(arena_name, "LOCATION")
            nams.add_entity(obj_name, "LOCATION")
            nams.cypher_write(
              "MATCH (arena:Entity {name: $arena}) "
              "MATCH (obj:Entity {name: $obj}) "
              "MERGE (arena)-[:CONTAINS]->(obj)",
              {"arena": arena_name, "obj": obj_name},
            )
            count += 1
          except Exception as e:
            print(f"[import] spatial edge failed for {sector}:{arena}:{obj}: "
                  f"{type(e).__name__}: {e}")
  return count


def _import_identity(nams: NamsMemory, scratch: dict) -> int:
  """scratch.json identity fields -> PERSON entity + permanent Facts."""
  name = scratch.get("name") or "Unknown Persona"
  # PERSON entity for the persona itself. Via the SDK's add_entity so it
  # carries the id UUID + embedding NAMS readers require (not a raw MERGE,
  # which produced the id-less node that crashed reflection).
  try:
    nams.add_entity(name, "PERSON")
  except Exception as e:
    print(f"[import] PERSON merge failed for {name!r}: {type(e).__name__}: {e}")

  identity_facts = [
    ("innate",       "is",            scratch.get("innate", "")),
    ("learned",      "is characterized by", scratch.get("learned", "")),
    ("currently",    "is currently",  scratch.get("currently", "")),
    ("lifestyle",    "lives by",      scratch.get("lifestyle", "")),
    ("daily_plan_req", "needs to",    scratch.get("daily_plan_req", "")),
  ]
  count = 0
  for kind, pred, val in identity_facts:
    if not val:
      continue
    try:
      nams.add_fact(
        subject=name, predicate=pred, obj=str(val),
        poignancy=10, kind=f"identity_{kind}",
        valid_from=None, valid_until=None,
        metadata={"permanent": True},
      )
      count += 1
    except Exception as e:
      print(f"[import] identity fact {kind} failed: {type(e).__name__}: {e}")
  return count


def _import_schedule(nams: NamsMemory, scratch: dict,
                     day_start: datetime.datetime) -> int:
  """f_daily_schedule (list of [task, duration_minutes]) -> temporal
  ScheduleEntry Facts, valid_from = day_start + cumulative minutes."""
  sched = scratch.get("f_daily_schedule") or []
  if not sched:
    return 0
  name = scratch.get("name") or "Unknown Persona"
  day_label = day_start.strftime("%A %B %d")
  cursor = day_start
  order = 0
  count = 0
  for entry in sched:
    if not entry or len(entry) < 2:
      continue
    task, duration = entry[0], int(entry[1])
    try:
      nams.add_schedule_entry(
        subject=name, description=str(task),
        start=cursor, duration_minutes=duration,
        poignancy=5, day=day_label, order=order,
      )
      cursor = cursor + datetime.timedelta(minutes=duration)
      order += 1
      count += 1
    except Exception as e:
      print(f"[import] schedule entry {order} failed: {type(e).__name__}: {e}")
  return count


def _import_nodes(nams: NamsMemory, nodes: dict, kw_strength: dict,
                  scratch: dict) -> int:
  """associative_memory/nodes.json -> Facts / Conversations.

  Per the plan:
    * event  -> Fact(kind='event')
    * chat   -> if ``filling`` looks like a [[speaker, utterance], ...]
                transcript, add a single summary Fact whose ``obj`` is the
                joined transcript (preserves the conversational content as a
                long-term fact). We deliberately do NOT re-stage the
                transcript as short-term messages + extract+clear here,
                because at import time there is no live conversation to
                close; the transcript *is* the long-term memory already.
    * thought -> Fact(kind='thought')
  """
  name = scratch.get("name") or "Unknown Persona"
  count = 0
  for node_id, node in (nodes or {}).items():
    ntype = node.get("type")
    created = _parse_created(node.get("created"))
    s = node.get("subject") or name
    p = node.get("predicate") or "is"
    o = node.get("object") or ""
    desc = node.get("description") or ""
    salience = _salience_for_node(node, kw_strength)
    try:
      if ntype == "event":
        nams.add_fact(
          subject=s, predicate=p, obj=o or desc,
          poignancy=salience, kind="event",
          valid_from=created, valid_until=None,
          metadata={"description": desc,
                    "keywords": list(node.get("keywords") or []),
                    "source_node_id": node_id},
        )
        count += 1
      elif ntype == "thought":
        nams.add_fact(
          subject=s, predicate="thought", obj=desc or o,
          poignancy=salience, kind="thought",
          valid_from=created, valid_until=None,
          metadata={"keywords": list(node.get("keywords") or []),
                    "source_node_id": node_id},
        )
        count += 1
      elif ntype == "chat":
        filling = node.get("filling") or []
        transcript_bits = []
        is_transcript = False
        for row in filling:
          if isinstance(row, list) and len(row) >= 2 and isinstance(row[0], str):
            transcript_bits.append(f"{row[0]}: {row[1]}")
            is_transcript = True
          elif isinstance(row, str):
            # ``filling`` may instead be a list of node_ids pointing at the
            # events/thoughts that fed this thought; for chat nodes that is
            # unusual, but we degrade gracefully by treating it as a
            # description prefix.
            transcript_bits.append(row)
        body = desc
        if is_transcript and transcript_bits:
          body = desc + "\n" + "\n".join(transcript_bits) if desc else "\n".join(transcript_bits)
        nams.add_fact(
          subject=s, predicate=p, obj=body or o,
          poignancy=salience, kind="chat",
          valid_from=created, valid_until=None,
          metadata={"with_whom": o,
                    "keywords": list(node.get("keywords") or []),
                    "source_node_id": node_id},
        )
        count += 1
    except Exception as e:
      print(f"[import] node {node_id} ({ntype}) failed: "
            f"{type(e).__name__}: {e}")
  return count


# --- top-level entry -------------------------------------------------------

def import_persona_bootstrap(*, nams: NamsMemory,
                             bootstrap_dir: str,
                             scratch: Optional[dict] = None,
                             spatial: Optional[dict] = None,
                             nodes: Optional[dict] = None,
                             kw_strength: Optional[dict] = None) -> dict:
  """Import one persona's bootstrap_memory/ into NAMS.

  Either pass the directory (and the files will be loaded) or pass the
  already-parsed dicts (useful for tests). Returns a small report dict.

  Idempotent in spirit: callers should gate on ``nams.graph_exists()`` so a
  graph that's already populated is never re-imported. This function itself
  does not check (it just writes), so the caller controls the gate.
  """
  if scratch is None:
    scratch = json.load(open(os.path.join(bootstrap_dir, "scratch.json")))
  if spatial is None:
    spatial = json.load(open(os.path.join(bootstrap_dir, "spatial_memory.json")))
  if nodes is None:
    nodes = json.load(open(os.path.join(bootstrap_dir,
                                        "associative_memory", "nodes.json")))
  if kw_strength is None:
    kw_path = os.path.join(bootstrap_dir, "associative_memory",
                           "kw_strength.json")
    kw_strength = json.load(open(kw_path)) if os.path.exists(kw_path) else {}

  # Determine day_start for schedule valid_from: earliest event created date
  # if any, else scratch.curr_time if present, else today.
  day_start = None
  if scratch.get("curr_time"):
    try:
      day_start = _parse_created(scratch["curr_time"])
      day_start = day_start.replace(hour=0, minute=0, second=0, microsecond=0)
    except Exception:
      day_start = None
  if day_start is None:
    earliest = None
    for node in (nodes or {}).values():
      c = node.get("created")
      if not c:
        continue
      try:
        dt = _parse_created(c)
        if earliest is None or dt < earliest:
          earliest = dt
      except Exception:
        continue
    if earliest is not None:
      day_start = earliest.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
      day_start = datetime.datetime.now().replace(hour=0, minute=0,
                                                  second=0, microsecond=0)

  report = {
    "persona": nams.session_id,
    "spatial_edges": _import_spatial(nams, spatial),
    "identity_facts": _import_identity(nams, scratch),
    "schedule_entries": _import_schedule(nams, scratch, day_start),
    "nodes": _import_nodes(nams, nodes, kw_strength, scratch),
  }
  return report


def import_sim_bootstrap(*, personas_bootstrap_root: str,
                         build_nams_for_persona,
                         skip_if_exists: bool = True) -> dict:
  """Import every persona under ``personas_bootstrap_root``.

  ``build_nams_for_persona(persona_name) -> NamsMemory`` is a caller-supplied
  factory so this function doesn't need to know how the per-persona
  MemoryClient is constructed (it depends on the active harness's embedder +
  extraction mode, which the caller owns).

  Returns ``{persona_name: report}``.
  """
  out = {}
  if not os.path.isdir(personas_bootstrap_root):
    return out
  for name in sorted(os.listdir(personas_bootstrap_root)):
    pdir = os.path.join(personas_bootstrap_root, name, "bootstrap_memory")
    if not os.path.isdir(pdir):
      continue
    nams = build_nams_for_persona(name)
    try:
      if skip_if_exists and nams.graph_exists():
        out[name] = {"persona": name, "skipped": True}
        continue
      out[name] = import_persona_bootstrap(
        nams=nams, bootstrap_dir=pdir,
      )
    finally:
      try:
        nams.close()
      except Exception:
        pass
  return out
