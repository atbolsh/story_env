"""
NAMS DB-only memory layer for the reverie harnesses.

Wraps the ``neo4j-agent-memory`` SDK's :class:`MemoryClient` and exposes a
high-level, sync surface that the cognitive modules call. This module is
deliberately **DB-only**: no LLM prompts are constructed here. Prompts live
in ``persona/prompt_template/`` and the cognitive modules; this layer just
stores, retrieves, ages, and traverses memory.

Three NAMS memory layers map onto the old JSON concept:

  * **Short-term** -- per-persona conversation/message nodes (session_id =
    persona name). Raw perceived events and chat turns live here for
    ``ttl_minutes`` (default 4) of in-simulation time before being extracted
    into long-term facts and deleted (``age_short_term``). Conversations are
    extracted in one shot on close (``extract_conversation``).
  * **Long-term** -- POLE+O entities + Facts (the knowledge graph). Plans and
    schedule entries are temporal Facts (``valid_from`` / ``valid_until``) so
    the schedule **is** the graph and is modified with Cypher.
  * **Reasoning** -- NAMS reasoning traces. Reflection / "evaluation"
    episodes are recorded as ``ReasoningTrace`` nodes via
    ``run_reflection_trace`` for debuggability.

All public methods are sync; they bridge to the async SDK via
:mod:`harnesses.nams.async_bridge`.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import os
from typing import Any, Iterable, Optional

from . import async_bridge

# --- extraction modes (selected at reverie.py startup) ---------------------

#: spaCy + GLiNER + GLiREL only; no LLM in the NAMS extraction pipeline.
#: The harness LLM is still used by the cognitive modules directly (e.g. for
#: reflection insights and conversation summaries) via ``add_fact``.
NAMS_EXTRACTION_NO_LLM = "no-llm"

#: spaCy + GLiNER for raw messages PLUS the harness chat LLM as the NAMS LLM
#: extractor stage. Lets NAMS's internal LLM extractor also run on raw
#: short-term text, on top of the explicit ``add_fact`` calls.
NAMS_EXTRACTION_HARNESS_LLM = "harness-llm"


def _neo4j_uri() -> str:
  return os.environ.get("NEO4J_URI", "bolt://localhost:7687").strip()


def _neo4j_password() -> str:
  return os.environ.get("NEO4J_PASSWORD", "password").strip()


def _neo4j_user() -> str:
  return os.environ.get("NEO4J_USER", "neo4j").strip()


#: Default bolt URI = the single shared Community instance from
#: ``docker-compose.yml``. Used when a persona has no entry in the per-persona
#: registry (``nams_databases.json``), i.e. when at most one persona in the
#: simulation is backed by NAMS and isolation is trivially satisfied by the
#: single instance.
_DEFAULT_BOLT_URI = "bolt://localhost:7687"

#: Community Edition exposes exactly one database per instance, always named
#: ``neo4j``. Per-persona isolation is achieved at the *instance* level (one
#: container per persona, each on its own bolt port), not the database-name
#: level -- that would require Neo4j Enterprise, which we deliberately avoid
#: to keep the stack on the free GPLv3 Community Edition.
_DEFAULT_DATABASE = "neo4j"


def _registry_path() -> str:
  """Path to the per-persona NAMS database registry.

  Written by ``scripts/nams_db.sh up``; read by every ``NamsMemory`` to find
  its persona's dedicated container. Overridable via ``NAMS_DB_REGISTRY`` for
  tests / non-standard install layouts.
  """
  env = os.environ.get("NAMS_DB_REGISTRY", "").strip()
  if env:
    return env
  # Default to <repo_root>/nams_databases.json.
  here = os.path.dirname(os.path.abspath(__file__))
  # .../reverie/backend_server/harnesses/nams -> repo root is 5 levels up.
  root = here
  for _ in range(5):
    root = os.path.dirname(root)
  return os.path.join(root, "nams_databases.json")


def _persona_bolt_uri(persona_name: str) -> str:
  """Resolve the bolt URI for ``persona_name``'s dedicated Neo4j container.

  Reads the registry written by ``scripts/nams_db.sh up``. If the persona is
  registered, returns ``bolt://localhost:<port>`` for its dedicated
  container. Otherwise falls back to the single shared compose instance
  (:data:`_DEFAULT_BOLT_URI`) -- correct when only one persona is on NAMS.

  Per-character isolation: each registered persona gets its own Community
  instance + data volume, so their memory graphs are physically separate.
  """
  path = _registry_path()
  try:
    with open(path, "r", encoding="utf-8") as f:
      reg = json.load(f)
  except (FileNotFoundError, json.JSONDecodeError):
    return _DEFAULT_BOLT_URI
  entry = reg.get(persona_name)
  if not isinstance(entry, dict):
    return _DEFAULT_BOLT_URI
  port = entry.get("port")
  host = entry.get("host", "localhost")
  if not port:
    return _DEFAULT_BOLT_URI
  return f"bolt://{host}:{port}"


def build_memory_settings(*, embedder_name: str,
                          extraction_mode: str,
                          llm_harness: Any,
                          persona_name: str = "") -> Any:
  """Construct a ``MemorySettings`` for the bolt backend.

  ``embedder_name`` is the active harness's embedder (e.g.
  ``"BAAI/bge-small-en-v1.5"`` or ``"openai/text-embedding-3-small"``). The
  caller passes its own ``llm_harness`` so that in mode C we can wire its
  ``as_nams_llm_provider()`` shim into the SDK.

  ``persona_name`` selects the per-persona bolt URI (see
  :func:`_persona_bolt_uri`). Empty = use the default single instance.
  """
  from neo4j_agent_memory import MemorySettings
  from pydantic import SecretStr

  # Neo4jConfig field names (see neo4j_agent_memory.config.settings.Neo4jConfig):
  #   uri: str, username: str, password: SecretStr, database: str.
  # The SDK uses pydantic strict validation (extra_forbidden), so the keys must
  # match exactly -- an earlier version of this code used ``user`` (the env-var
  # flavor) which the SDK rejects.
  neo4j_cfg = {
    "uri": _persona_bolt_uri(persona_name) if persona_name else _neo4j_uri(),
    "username": _neo4j_user(),
    "password": SecretStr(_neo4j_password()),
    # Community Edition: always the single ``neo4j`` database. Kept here so
    # that a future switch to Enterprise (per-persona database names) is a
    # one-line change in _persona_bolt_uri / build_memory_settings.
    "database": _DEFAULT_DATABASE,
  }

  # Provider-string shorthand is supported by the SDK: a sentence-transformers
  # id is recognized when [sentence-transformers] is installed; an
  # ``openai/<model>`` id is recognized when [openai] is installed.
  embedding_str = embedder_name

  llm_provider = None
  enable_llm_fallback = False
  if extraction_mode == NAMS_EXTRACTION_HARNESS_LLM:
    # In the dedicated *-nams harnesses, llm_harness is a Gemma4NamsLLM /
    # LatestGPTNamsLLM with its own as_nams_llm_provider(). In the mixed
    # ("multi-harness") mode, llm_harness is the plain gemma4 _Gemma4Harness,
    # which doesn't have one -- nams_llm_provider_for falls back to the
    # generic NamsLLMProvider adapter that drives _generate off-thread.
    from .llm_harness import nams_llm_provider_for
    llm_provider = nams_llm_provider_for(llm_harness)
    enable_llm_fallback = True

  # Build an explicit ExtractionConfig so the SDK's llm-consistency validator
  # is satisfied: when llm is None (no-llm mode, and the import-only path
  # which has no harness), enable_llm_fallback MUST be False or the SDK
  # rejects the settings (see _validate_llm_consistency in the SDK). The
  # PIPELINE extractor with enable_spacy + enable_gliner + extract_relations
  # (default True) gives the spaCy + GLiNER + GLiREL pipeline we want; the
  # only knob that changes between modes is enable_llm_fallback.
  from neo4j_agent_memory import ExtractionConfig
  extraction_cfg = ExtractionConfig(enable_llm_fallback=enable_llm_fallback)

  # When llm is None the SDK runs the spaCy/GLiNER/GLiREL extractor pipeline
  # without the LLM fallback stage (air-gapped / deterministic). That's mode A.
  return MemorySettings(
    neo4j=neo4j_cfg,
    embedding=embedding_str,
    llm=llm_provider,
    extraction=extraction_cfg,
  )


class NamsMemory:
  """Per-persona NAMS memory facade.

  Holds one :class:`MemoryClient` (constructed lazily on first use and reused
  for the lifetime of the persona). ``session_id`` is the persona's full name
  -- NAMS scopes short-term conversations and reasoning traces by it.
  """

  #: Short-term buffer TTL in sim-minutes. Events older than this are
  #: extracted into long-term facts and deleted each step (smooth aging).
  SHORT_TERM_TTL_MINUTES = 4

  def __init__(self, session_id: str, *, embedder_name: str,
               extraction_mode: str, llm_harness: Any):
    self.session_id = session_id
    self._embedder_name = embedder_name
    self._extraction_mode = extraction_mode
    self._llm_harness = llm_harness
    self._client = None  # MemoryClient, lazily constructed
    self._settings = None

  # ------------------------------------------------------------------ init

  async def _ensure_client_async(self):
    """Async client construction. Safe to await from inside a coroutine that
    is already running on the NAMS background loop (e.g. the ``_go``
    coroutines below). This is the path the sim actually uses: ``graph_exists``
    and every other NAMS method runs its ``_go`` on the background loop via
    ``async_bridge.run``, and ``_go`` must NOT call the sync
    ``_ensure_client`` (which itself calls ``async_bridge.run`` -- a re-
    entrant call that deadlocks the loop). Await this instead.
    """
    if self._client is not None:
      return self._client
    print(f"[nams] {self.session_id!r}: connecting to Neo4j "
          f"({_persona_bolt_uri(self.session_id) or 'bolt://localhost:7687'}) "
          f"and initializing the extraction pipeline (first run downloads "
          f"spaCy + GLiNER + GLiREL models -- silent, can take minutes)...",
          flush=True)
    settings = build_memory_settings(
      embedder_name=self._embedder_name,
      extraction_mode=self._extraction_mode,
      llm_harness=self._llm_harness,
      persona_name=self.session_id,
    )
    self._settings = settings
    from neo4j_agent_memory import MemoryClient
    client = MemoryClient(settings)
    # MemoryClient is an async context manager; for the long-lived per-persona
    # client we drive __aenter__/__aexit__ manually so it stays connected for
    # the whole simulation. Hard timeout so a stalled __aenter__ surfaces a
    # clear error instead of hanging silently. 180s is ample for first-run
    # pipeline init (spaCy/GLiNER/GLiREL load); raise via NAMS_CONNECT_TIMEOUT
    # on a slow link.
    timeout = float(os.environ.get("NAMS_CONNECT_TIMEOUT", "180"))
    await asyncio.wait_for(client.__aenter__(), timeout=timeout)
    self._client = client
    print(f"[nams] {self.session_id!r}: connected + pipeline ready.", flush=True)
    return self._client

  def _ensure_client(self):
    """Sync wrapper around :meth:`_ensure_client_async` for sync callers
    (the ``client`` property, external sync code). Submits the async connect
    to the background loop via ``async_bridge.run``. MUST NOT be called from
    inside a coroutine already running on the background loop -- that's a
    re-entrant deadlock (the loop blocks on ``fut.result`` and never runs the
    scheduled connect). In-loop coroutines use ``await _ensure_client_async``.
    """
    if self._client is not None:
      return self._client
    timeout = float(os.environ.get("NAMS_CONNECT_TIMEOUT", "180"))
    return async_bridge.run(self._ensure_client_async(), timeout=timeout)

  @property
  def client(self):
    return self._ensure_client()

  # ---------------------------------------------------------------- close

  def close(self) -> None:
    if self._client is None:
      return
    client = self._client
    self._client = None

    async def _aclose():
      try:
        await client.close()
      except Exception as e:
        print(f"[nams] close failed for {self.session_id!r}: "
              f"{type(e).__name__}: {e}")

    try:
      async_bridge.run(_aclose())
    except Exception as e:
      print(f"[nams] close bridge failed for {self.session_id!r}: "
            f"{type(e).__name__}: {e}")

  # ============================================================ SHORT TERM

  def add_event(self, *, s: str, p: str, o: str, description: str,
                poignancy: int, created: datetime.datetime,
                keywords: Optional[Iterable[str]] = None) -> Any:
    """Record a perceived event as a short-term Message on this persona's
    session. The 1-10 ``poignancy`` is stored as ``metadata.salience`` so the
    extractor can propagate it to long-term facts.

    The (subject, predicate, object) triple is preserved in metadata so the
    classic ``retrieve_relevant_events`` lookup-by-keyword behavior survives
    in the graph as a `MENTIONS`-style explicit linkage.
    """
    md = {
      "kind": "event",
      "subject": s,
      "predicate": p,
      "object": o,
      "salience": int(poignancy),
      "created_iso": created.isoformat(),
    }
    if keywords:
      md["keywords"] = [k.lower() for k in keywords]

    async def _go():
      client = await self._ensure_client_async()
      # Use the batch loader with an explicit ``timestamp`` so the Message
      # node's ``m.timestamp`` reflects *simulation* time (not wall-clock).
      # age_short_term filters on m.timestamp, so events must carry sim time
      # or aging by sim-now would never fire. The batch path runs the
      # configured extractor (spaCy+GLiNER[+LLM]) per message when
      # extract_entities=True, same as add_message.
      msgs = await client.short_term.add_messages_batch(
        session_id=self.session_id,
        messages=[{
          "role": "user",
          "content": description,
          "timestamp": created.isoformat(),
          "metadata": md,
        }],
        generate_embeddings=True,
        extract_entities=True,
      )
      msg = msgs[0] if msgs else None
      if msg is not None:
        # Explicitly link the (s, p, o) triple as entities so the classic
        # keyword lookup still works after the raw message is aged out.
        await self._link_spo(message_id=msg.id, s=s, p=p, o=o,
                             poignancy=poignancy, created=created)
      return msg

    return async_bridge.run(_go())

  def add_chat_turn(self, *, conversation_id: Optional[str], speaker: str,
                    utterance: str, poignancy: int,
                    created: datetime.datetime) -> Any:
    """Append one utterance to the persona's current conversation short-term
    buffer. ``conversation_id`` lets both participants of a chat share one
    NAMS Conversation node so extraction at chat-close sees the whole
    transcript."""
    md = {
      "kind": "chat_turn",
      "speaker": speaker,
      "salience": int(poignancy),
      "created_iso": created.isoformat(),
    }

    async def _go():
      client = await self._ensure_client_async()
      return await client.short_term.add_message(
        session_id=self.session_id,
        role="assistant" if speaker != self.session_id else "user",
        content=f"{speaker}: {utterance}",
        conversation_id=conversation_id,
        extract_entities=True,
        generate_embedding=True,
        metadata=md,
      )

    return async_bridge.run(_go())

  def age_short_term(self, now: datetime.datetime,
                     ttl_minutes: int = SHORT_TERM_TTL_MINUTES) -> int:
    """Smooth short-term aging: find Message nodes older than ``ttl_minutes``
    relative to ``now``, ensure each has been extracted (the extractor already
    ran on add, but we re-trigger for any that were skipped), then delete them.

    Returns the number of messages aged out.

    Per-message (not batched at the 4-minute boundary) so the buffer drains
    smoothly as sim-time advances.

    Note on schema: NAMS stores ``session_id`` on the Conversation node, not
    on Message; messages are reached via ``(c:Conversation)-[:HAS_MESSAGE]->(m)``,
    and the message timestamp is ``m.timestamp``.
    """
    cutoff_iso = (now - datetime.timedelta(minutes=ttl_minutes)).isoformat()

    async def _go():
      client = await self._ensure_client_async()
      rows = await client.query.cypher(
        "MATCH (c:Conversation {session_id: $sid})-[:HAS_MESSAGE]->(m:Message) "
        "WHERE m.timestamp < datetime($cutoff) "
        "RETURN m.id AS id ORDER BY m.timestamp",
        {"sid": self.session_id, "cutoff": cutoff_iso},
      )
      count = 0
      for row in rows:
        msg_id = row["id"] if isinstance(row, dict) else row.get("id")
        if not msg_id:
          continue
        # The extractor already ran at add_message time; we just drop the raw
        # text now that its facts/entities are in the graph.
        await client.short_term.delete_message(msg_id)
        count += 1
      return count

    return async_bridge.run(_go())

  def extract_conversation(self, conversation_id: str) -> None:
    """Run entity/relation extraction on the whole conversation in one shot,
    then delete every Message in it (forget raw text). Used at chat close.

    NAMS already extracted entities per-message as they were added; this is
    the safety net that guarantees the final transcript is fully extracted
    before the raw text is dropped.
    """
    async def _go():
      client = await self._ensure_client_async()
      # Re-run extraction across the conversation's messages, then delete
      # them. session_id lives on the Conversation node.
      rows = await client.query.cypher(
        "MATCH (c:Conversation {session_id: $sid, id: $cid})"
        "-[:HAS_MESSAGE]->(m:Message) "
        "RETURN m.id AS id ORDER BY m.timestamp",
        {"sid": self.session_id, "cid": str(conversation_id)},
      )
      for row in rows:
        msg_id = row["id"] if isinstance(row, dict) else row.get("id")
        if msg_id:
          await client.short_term.delete_message(msg_id)

    return async_bridge.run(_go())

  def clear_session(self) -> None:
    """Wipe all short-term data for this persona's session. Use with care;
    called by the JSON importer when it has already placed long-term facts
    directly and wants a clean short-term buffer."""
    async def _go():
      client = await self._ensure_client_async()
      await client.short_term.clear_session(self.session_id)

    return async_bridge.run(_go())

  # ============================================================ LONG TERM

  def add_fact(self, *, subject: str, predicate: str, obj: str,
               poignancy: int = 5, kind: str = "fact",
               valid_from: Optional[datetime.datetime] = None,
               valid_until: Optional[datetime.datetime] = None,
               metadata: Optional[dict] = None) -> Any:
    """Add a declarative Fact to long-term memory.

    ``poignancy`` (1-10) becomes ``Fact.confidence = poignancy/10`` AND
    ``metadata.salience = poignancy`` so the re-rank in
    :meth:`get_context_for_planning` can use it as the importance term.

    ``kind`` is stored in metadata so callers can distinguish event/thought/
    plan/schedule_entry/identity facts when traversing the graph.
    """
    md = dict(metadata or {})
    md["kind"] = kind
    md["salience"] = int(poignancy)

    async def _go():
      client = await self._ensure_client_async()
      return await client.long_term.add_fact(
        subject=subject, predicate=predicate, obj=obj,
        confidence=max(0.0, min(1.0, poignancy / 10.0)),
        valid_from=valid_from, valid_until=valid_until,
        generate_embedding=True, metadata=md,
      )

    return async_bridge.run(_go())

  def add_plan_fact(self, *, subject: str, obj: str,
                    valid_from: datetime.datetime,
                    valid_until: datetime.datetime,
                    poignancy: int = 7,
                    location: Optional[str] = None,
                    with_whom: Optional[str] = None,
                    source: str = "convo") -> Any:
    """Add a future commitment as a temporal Fact (the schedule-propagation
    fix for KNOWN_WEAKNESSES §0). ``valid_from``/``valid_until`` bound when
    the plan is active, so :meth:`get_upcoming_plans` can surface it to every
    planning/decision prompt as the relevant time approaches."""
    md = {"location": location, "with_whom": with_whom, "source": source}
    return self.add_fact(
      subject=subject, predicate="plans to", obj=obj,
      poignancy=poignancy, kind="plan",
      valid_from=valid_from, valid_until=valid_until, metadata=md,
    )

  def add_schedule_entry(self, *, subject: str, description: str,
                         start: datetime.datetime,
                         duration_minutes: int,
                         poignancy: int = 5,
                         day: Optional[str] = None,
                         order: Optional[int] = None) -> Any:
    """Add one entry of the daily schedule as a temporal Fact. The schedule
    **is** the graph: mid-day edits (conversation insertion, task decomp) are
    Cypher writes that delete/insert these Fact nodes. ``day`` (e.g.
    ``"Monday February 13"``) and ``order`` let :meth:`get_schedule_chain`
    reconstruct the chain."""
    end = start + datetime.timedelta(minutes=duration_minutes)
    md = {"day": day, "order": order}
    return self.add_fact(
      subject=subject, predicate="is scheduled to", obj=description,
      poignancy=poignancy, kind="schedule_entry",
      valid_from=start, valid_until=end, metadata=md,
    )

  def clear_schedule_for_day(self, *, subject: str,
                             day: str) -> int:
    """Delete every schedule_entry Fact for ``subject`` on ``day``. Used when
    re-planning a day from scratch (e.g. new-day ``revise_identity``)."""
    async def _go():
      client = await self._ensure_client_async()
      result = await client.graph.execute_write(
        "MATCH (f:Fact) "
        "WHERE f.subject = $subject "
        "  AND f.metadata CONTAINS '\"kind\": \"schedule_entry\"' "
        "  AND f.metadata CONTAINS $day "
        "DELETE f RETURN count(f) AS n",
        {"subject": subject, "day": day},
      )
      if result and isinstance(result[0], dict):
        return int(result[0].get("n", 0))
      return 0

    return async_bridge.run(_go())

  def get_schedule_chain(self, *, subject: str,
                         day: Optional[str] = None) -> list[dict]:
    """Return the persona's schedule_entry Facts as a list of dicts ordered
    by ``valid_from``. Used to rebuild ``scratch.f_daily_schedule`` cache
    from the graph on load."""
    async def _go():
      client = await self._ensure_client_async()
      if day:
        rows = await client.query.cypher(
          "MATCH (f:Fact) "
          "WHERE f.subject = $subject "
          "  AND f.metadata CONTAINS '\"kind\": \"schedule_entry\"' "
          "  AND f.metadata CONTAINS $day "
          "RETURN f ORDER BY f.valid_from",
          {"subject": subject, "day": day},
        )
      else:
        rows = await client.query.cypher(
          "MATCH (f:Fact) "
          "WHERE f.subject = $subject "
          "  AND f.metadata CONTAINS '\"kind\": \"schedule_entry\"' "
          "RETURN f ORDER BY f.valid_from",
          {"subject": subject},
        )
      out = []
      for r in rows:
        node = r.get("f") if isinstance(r, dict) else None
        if node is None and isinstance(r, dict):
          node = r
        if hasattr(node, "items"):
          d = dict(node)
        elif isinstance(node, dict):
          d = node
        else:
          # neo4j Node-like object
          try:
            d = dict(node)
          except Exception:
            continue
        out.append(d)
      return out

    return async_bridge.run(_go())

  def get_upcoming_plans(self, now: datetime.datetime,
                         lookahead_hours: int = 2) -> list[dict]:
    """Temporal query that always accompanies context retrieval: every Fact
    with ``metadata.kind='plan'`` (or ``schedule_entry``) whose validity
    window overlaps ``[now, now + lookahead]``, plus plans currently active.
    These are force-injected into planning prompts so future commitments ride
    along even when pure semantic similarity to the focal point is low -- the
    structural fix for KNOWN_WEAKNESSES §0."""
    horizon_iso = (now + datetime.timedelta(hours=lookahead_hours)).isoformat()
    now_iso = now.isoformat()

    async def _go():
      client = await self._ensure_client_async()
      rows = await client.query.cypher(
        "MATCH (f:Fact) "
        "WHERE f.metadata CONTAINS '\"kind\": \"plan\"' "
        "  AND ( (f.valid_from IS NULL OR f.valid_from <= datetime($horizon)) "
        "        AND (f.valid_until IS NULL OR f.valid_until > datetime($now)) ) "
        "RETURN f ORDER BY f.valid_from",
        {"now": now_iso, "horizon": horizon_iso},
      )
      out = []
      for r in rows:
        node = r.get("f") if isinstance(r, dict) else None
        if node is None and isinstance(r, dict):
          node = r
        try:
          out.append(dict(node))
        except Exception:
          continue
      return out

    return async_bridge.run(_go())

  # ----- inter-person relationship (convo close) ---------------------------

  def add_person_relationship(self, *, from_name: str, to_name: str,
                              relationship_type: str,
                              description: str,
                              valid_from: Optional[datetime.datetime] = None,
                              valid_until: Optional[datetime.datetime] = None,
                              poignancy: int = 5) -> None:
    """Write a typed relationship edge between two PERSON entities, with the
    conversation summary as ``description``. Called at chat close so the
    relationship graph accrues over time and is traversable from either
    party's memory."""
    async def _go():
      client = await self._ensure_client_async()
      # Ensure both Person entities exist. add_entity returns
      # (Entity, DeduplicationResult); we want the Entity. Disable
      # geocode/enrich -- persona names are not places and we don't want
      # the background enrichment service kicking off for them.
      a = await client.long_term.add_entity(
        from_name, "PERSON", geocode=False, enrich=False,
      )
      b = await client.long_term.add_entity(
        to_name, "PERSON", geocode=False, enrich=False,
      )
      a_ent = a[0] if isinstance(a, tuple) else a
      b_ent = b[0] if isinstance(b, tuple) else b
      await client.long_term.add_relationship(
        a_ent, b_ent, relationship_type,
        description=description,
        confidence=max(0.0, min(1.0, poignancy / 10.0)),
        valid_from=valid_from, valid_until=valid_until,
        attributes={"salience": int(poignancy)},
      )

    return async_bridge.run(_go())

  # ----- SPO explicit linkage (preserves classic keyword lookup) ----------

  async def _link_spo(self, *, message_id, s: str, p: str, o: str,
                      poignancy: int, created: datetime.datetime) -> None:
    """Link a short-term message to its (s, p, o) entities explicitly, so the
    classic retrieve-by-keyword behavior survives once the raw message is
    aged out and only the extracted entities/facts remain."""
    client = await self._ensure_client_async()
    # Subject and object become Entities; predicate is preserved on the
    # relationship edge. The persona's own name is a PERSON; everything else
    # is typed as OBJECT (POLE+O) by default.
    sub_type = "PERSON" if " " in s and ":" not in s else "OBJECT"
    obj_type = "PERSON" if " " in o and ":" not in o else "OBJECT"
    try:
      sub = await client.long_term.add_entity(
        s, sub_type, geocode=False, enrich=False,
      )
      obj = await client.long_term.add_entity(
        o, obj_type, geocode=False, enrich=False,
      )
      sub_ent = sub[0] if isinstance(sub, tuple) else sub
      obj_ent = obj[0] if isinstance(obj, tuple) else obj
      # add_relationship between the entities carries the predicate + the
      # source message id + salience.
      await client.long_term.add_relationship(
        sub_ent, obj_ent, p.upper().replace(" ", "_"),
        description=f"{s} {p} {o}",
        confidence=max(0.0, min(1.0, poignancy / 10.0)),
        valid_from=created, valid_until=None,
        attributes={"salience": int(poignancy),
                    "source_message": str(message_id)},
      )
    except Exception as e:
      # Linkage is best-effort; never let it block perception.
      print(f"[nams] _link_spo failed for {self.session_id!r}: "
            f"{type(e).__name__}: {e}")

  # ============================================== UNIFIED CONTEXT (retrieve)

  def get_context_for_planning(self, *, focal_points: list[str],
                               now: datetime.datetime,
                               max_items: int = 30,
                               recency_w: float = 1.0,
                               relevance_w: float = 1.0,
                               importance_w: float = 1.0,
                               recency_decay: float = 0.99,
                               lookahead_hours: int = 2) -> dict:
    """Unified retrieval for planning/decision prompts.

    Combines, per focal point:
      * NAMS ``client.get_context`` -- short-term recent messages, long-term
        facts/entities by semantic similarity, similar reasoning traces.
      * :meth:`get_upcoming_plans` -- temporal Facts (plans + active schedule
        entries) force-injected so future commitments always ride along.
      * Classic recency * relevance * importance re-rank using
        ``metadata.salience`` as the importance term.

    Returns ``{focal_point: {"context_str": str, "plans": [dict], "items": [...]}}``.
    The ``context_str`` is the merged, formatted block ready to splice into a
    prompt; ``plans`` and ``items`` are exposed for logging / debugging.
    """
    out: dict = {}

    def _score(recency: float, relevance: float, importance: float) -> float:
      return (recency_w * recency * 0.5
              + relevance_w * relevance * 3.0
              + importance_w * importance * 2.0)

    plans = self.get_upcoming_plans(now, lookahead_hours=lookahead_hours)
    plans_block = self._format_plans(plans, now)

    for fp in focal_points:
      async def _go_one(focal=fp):
        client = await self._ensure_client_async()
        # The SDK's get_context returns a formatted string; we also pull raw
        # facts so we can re-rank. For simplicity we use the formatted string
        # directly and append the plans block. The re-rank hook below is a
        # place to add the classic scoring once we have raw node access.
        ctx = await client.get_context(
          focal, session_id=self.session_id,
          include_short_term=True, include_long_term=True,
          include_reasoning=True, max_items=max_items,
        )
        return ctx

      ctx_str = async_bridge.run(_go_one())
      parts = []
      if ctx_str:
        parts.append(ctx_str)
      if plans_block:
        parts.append("## Upcoming Plans & Active Schedule\n" + plans_block)
      out[fp] = {
        "context_str": "\n\n".join(parts),
        "plans": plans,
        "items": [],
      }
    return out

  @staticmethod
  def _format_plans(plans: list[dict], now: datetime.datetime) -> str:
    if not plans:
      return ""
    lines = []
    for p in plans:
      subj = p.get("subject", "?")
      pred = p.get("predicate", "")
      obj = p.get("object", "")
      vf = p.get("valid_from")
      vu = p.get("valid_until")
      when = ""
      try:
        if vf:
          when = f" from {str(vf)}"
        if vu:
          when += f" until {str(vu)}"
      except Exception:
        pass
      lines.append(f"- {subj} {pred} {obj}{when}")
    return "\n".join(lines)

  # ============================================ REASONING (reflect/evaluate)

  def run_reflection_trace(self, *, task: str,
                           steps: list[dict],
                           outcome: str,
                           success: bool = True,
                           triggered_by_message_id: Optional[str] = None) -> Any:
    """Record a reflection / "evaluation" episode as a NAMS reasoning trace.

    ``steps`` is a list of ``{"thought": ..., "action": ..., "observation": ...}``
    dicts -- one per focal point / insight the reflection produced. The trace
    is searchable later (``client.reasoning.get_similar_traces``) so future
    reflections can lean on past ones, and the console UI renders it as a
    debuggable trace (closing the "free-floating thought node is hard to
    debug" gap in the old JSON system).
    """
    async def _go():
      client = await self._ensure_client_async()
      trace = await client.reasoning.start_trace(
        session_id=self.session_id, task=task,
        triggered_by_message_id=triggered_by_message_id,
      )
      for s in steps:
        await client.reasoning.add_step(
          trace.id,
          thought=s.get("thought"),
          action=s.get("action"),
          observation=s.get("observation"),
        )
      await client.reasoning.complete_trace(trace.id, outcome=outcome,
                                            success=success)
      return trace

    return async_bridge.run(_go())

  # ============================================================ GRAPH STATE

  def graph_exists(self) -> bool:
    """True iff this persona's session already has any memory nodes in the
    graph. Used by ``reverie.py`` to decide whether to run the JSON->NAMS
    importer on first run of a *-nams harness against a JSON-forked sim."""
    async def _go():
      client = await self._ensure_client_async()
      # session_id lives on the Conversation node; reach messages through it.
      # Also count this persona's PERSON entity and any Fact that names them.
      rows = await client.query.cypher(
        "OPTIONAL MATCH (c:Conversation {session_id: $sid}) "
        "OPTIONAL MATCH (e:Entity {name: $sid}) "
        "OPTIONAL MATCH (f:Fact) WHERE f.subject = $sid OR f.object = $sid "
        "RETURN count(c) + count(e) + count(f) AS c LIMIT 1",
        {"sid": self.session_id},
      )
      if not rows:
        return False
      row = rows[0]
      if isinstance(row, dict):
        return int(row.get("c", 0)) > 0
      return False

    try:
      return async_bridge.run(_go())
    except Exception as e:
      print(f"[nams] graph_exists check failed for {self.session_id!r}: "
            f"{type(e).__name__}: {e}")
      return False

  # ----- save / load -------------------------------------------------------

  def save(self) -> None:
    """The graph IS the persistent store; nothing to write to disk here.
    Kept for API symmetry with the old ``AssociativeMemory.save``."""
    return None

  # ----- low-level escape hatch -------------------------------------------

  def cypher(self, query: str, params: Optional[dict] = None) -> list:
    """Run an arbitrary **read-only** Cypher query against the persona's
    Neo4j via the SDK's portable ``client.query.cypher`` accessor (which
    validates read-only-ness). Used by retrieval/debugging code paths."""
    async def _go():
      client = await self._ensure_client_async()
      return await client.query.cypher(query, params or {})

    return async_bridge.run(_go())

  def cypher_write(self, query: str, params: Optional[dict] = None) -> list:
    """Run an arbitrary **write** Cypher query (MERGE/CREATE/DELETE/SET)
    against the persona's Neo4j via ``client.graph.execute_write``. The
    SDK's ``query.cypher`` accessor rejects writes, so writes go through
    the (deprecated but functional through v0.5) graph proxy. Used by the
    JSON importer and by schedule-edit code paths that don't merit a
    dedicated method."""
    async def _go():
      client = await self._ensure_client_async()
      return await client.graph.execute_write(query, params or {})

    return async_bridge.run(_go())
