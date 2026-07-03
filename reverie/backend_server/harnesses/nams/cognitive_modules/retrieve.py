"""
NAMS-aware Retrieve module.

Two entry points, matching the legacy contract:

  * ``retrieve(persona, perceived)`` -- per-event retrieval that feeds into
    ``plan``'s ``_choose_retrieved``. Delegates to the legacy keyword lookup
    on the in-memory cache (which the a_mem adapter keeps populated as
    events are perceived).

  * ``new_retrieve(persona, focal_points, n_count)`` -- focal-point retrieval
    used by ``plan`` (long-term planning) and ``converse``. Queries NAMS
    ``get_context_for_planning``, which:

        - calls NAMS ``client.get_context`` for semantic similarity over
          long-term facts/entities and recent short-term messages,
        - force-injects ``get_upcoming_plans`` (temporal Plan + active
          ScheduleEntry facts whose validity window overlaps [now, now+2h])
          so future commitments always ride along -- the structural fix for
          KNOWN_WEAKNESSES §0 (chat outcomes not propagating to plans),
        - re-ranks with the classic recency * relevance * importance using
          ``metadata.salience`` as the importance term.

    Returns ``{focal_pt: [ConceptNode-like, ...]}`` shaped exactly like the
    legacy ``new_retrieve`` so the existing prompt templates
    (``revise_identity``, ``_long_term_planning``, ``agent_chat_v2``) consume
    it unchanged.
"""
from __future__ import annotations

import datetime
from typing import Any

from persona.cognitive_modules.retrieve import retrieve as _legacy_retrieve


class _NamsRetrievedNode:
  """Lightweight ConceptNode-compatible object wrapping a NAMS fact/message
  dict. Exposes the attributes the prompt templates read: ``created``,
  ``embedding_key``, ``description``, ``subject``, ``predicate``, ``object``,
  ``poignancy``, ``type``, ``node_id``.
  """

  def __init__(self, d: dict):
    self._d = d
    self.node_id = d.get("id") or d.get("uid") or repr(d)
    self.type = (d.get("metadata", {}) or {}).get("kind", "event") if isinstance(d.get("metadata"), dict) else d.get("type", "event")
    self.subject = d.get("subject", "")
    self.predicate = d.get("predicate", "")
    self.object = d.get("object", "")
    self.description = d.get("description") or d.get("content") or ""
    self.embedding_key = self.description or self.object
    self.poignancy = int((d.get("metadata", {}) or {}).get("salience", 5)) if isinstance(d.get("metadata"), dict) else int(d.get("salience", 5))
    self.keywords = set()
    self.filling = []
    self.last_accessed = datetime.datetime.now()
    c = d.get("valid_from") or d.get("created_at") or d.get("created")
    if isinstance(c, str):
      try:
        self.created = datetime.datetime.fromisoformat(c.replace("Z", ""))
      except Exception:
        try:
          self.created = datetime.datetime.strptime(c[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
          self.created = datetime.datetime.now()
    elif isinstance(c, datetime.datetime):
      self.created = c
    else:
      self.created = datetime.datetime.now()
    self.expiration = None
    self.depth = 0

  def spo_summary(self):
    return (self.subject, self.predicate, self.object)


def retrieve(persona, perceived):
  """Per-event retrieval. Delegates to the legacy keyword lookup on the
  in-memory cache (the a_mem adapter keeps it populated as events are
  perceived). Returns the legacy ``{event.description: {curr_event, events,
  thoughts}}`` shape."""
  return _legacy_retrieve(persona, perceived)


def new_retrieve(persona, focal_points, n_count: int = 30) -> dict:
  """Focal-point retrieval via NAMS. See module docstring."""
  now = persona.scratch.curr_time or datetime.datetime.now()
  scratch = persona.scratch
  ctx = persona.nams.get_context_for_planning(
    focal_points=list(focal_points),
    now=now,
    max_items=n_count,
    recency_w=getattr(scratch, "recency_w", 1.0),
    relevance_w=getattr(scratch, "relevance_w", 1.0),
    importance_w=getattr(scratch, "importance_w", 1.0),
    recency_decay=getattr(scratch, "recency_decay", 0.99),
    lookahead_hours=2,
  )
  out: dict = {}
  for fp, info in ctx.items():
    nodes: list = []
    # The plans block is the key schedule-propagation vector: upcoming Plan
    # facts + active ScheduleEntry facts whose window overlaps [now, now+2h].
    for p in info.get("plans", []):
      nodes.append(_NamsRetrievedNode(p))
    # If the SDK's get_context returned structured items, wrap them too.
    for it in info.get("items", []):
      nodes.append(_NamsRetrievedNode(it))
    # The context_str is kept for logging/debugging but not returned as a
    # node; the prompt templates build their own [Statements] block from the
    # node list, and the plans are already represented as nodes above.
    out[fp] = nodes
  return out
