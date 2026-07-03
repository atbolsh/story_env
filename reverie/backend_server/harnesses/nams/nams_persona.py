"""
NAMS-backed Persona.

Subclass of the legacy :class:`persona.persona.Persona` that swaps the JSON
``AssociativeMemory`` for a NAMS-backed adapter while keeping the rest of the
Persona contract (scratch, s_mem, cognitive-module entry points) intact.

Key pieces:

  * :class:`NamsAssociativeMemory` -- subclasses the legacy
    ``AssociativeMemory`` and overrides ``add_event`` / ``add_chat`` /
    ``add_thought`` to write through to NAMS (short-term Messages for events
    and chat turns; long-term Facts for thoughts). The in-memory cache
    (``seq_event`` / ``seq_thought`` / ``id_to_node`` / ``embeddings`` /
    ``kw_to_*``) is still maintained by the legacy base class so the
    existing cognitive modules' read patterns keep working. ``save`` is a
    no-op: the graph IS the long-term store; only ``scratch.json`` +
    ``spatial_memory.json`` are persisted to disk.

  * :class:`NamsPersona` -- overrides ``__init__`` to skip JSON
    ``AssociativeMemory`` loading and build a per-persona
    :class:`harnesses.nams.nams_memory.NamsMemory` instead; overrides
    ``save`` to write only scratch + spatial; overrides the cognitive-module
    entry points (``perceive`` / ``retrieve`` / ``plan`` / ``reflect`` /
    ``execute`` / ``open_convo_session``) to call the NAMS-aware modules in
    ``harnesses.nams.cognitive_modules``.
"""
from __future__ import annotations

import datetime
import os
import sys

from persona.memory_structures.spatial_memory import MemoryTree
from persona.memory_structures.scratch import Scratch

# The legacy AssociativeMemory loads embeddings.json/nodes.json in __init__.
# For NAMS we never load those (the graph is the store), so we construct the
# base class with a sentinel and then clear its caches. We import it lazily
# inside NamsAssociativeMemory so a missing embeddings.json at the bootstrap
# path doesn't break import.


class NamsAssociativeMemory:
  """NAMS-backed replacement for the legacy ``AssociativeMemory``.

  Maintains the same in-memory cache surface the cognitive modules read
  (``seq_event``, ``seq_thought``, ``seq_chat``, ``id_to_node``,
  ``embeddings``, ``kw_to_*``, ``kw_strength_*``) but writes through to NAMS
  on every add. Reads (``retrieve_relevant_*``, ``get_summarized_latest_events``,
  ``get_last_chat``) operate on the in-memory cache as before -- the cache is
  the per-step working set, populated as events are perceived.

  Construction is lightweight: no JSON files are loaded. The cache starts
  empty and grows during the sim. On save, nothing is written to disk (the
  graph is the durable store).
  """

  def __init__(self, nams_memory):
    # Defer the legacy import to here so the package can be imported even
    # when a bootstrap path is missing.
    from persona.memory_structures.associative_memory import (
      AssociativeMemory, ConceptNode,
    )
    self._AssociativeMemory = AssociativeMemory
    self._ConceptNode = ConceptNode
    self.nams = nams_memory

    # In-memory cache -- mirrors the legacy AssociativeMemory's bookkeeping.
    self.id_to_node = dict()
    self.seq_event = []
    self.seq_thought = []
    self.seq_chat = []
    self.kw_to_event = dict()
    self.kw_to_thought = dict()
    self.kw_to_chat = dict()
    self.kw_strength_event = dict()
    self.kw_strength_thought = dict()
    self.embeddings = dict()

  # ------------------------------------------------------------------ save

  def save(self, out_json) -> None:
    """No-op. The graph is the long-term store; only scratch.json +
    spatial_memory.json are persisted (by NamsPersona.save)."""
    return None

  # ------------------------------------------------------------------ adds

  def _record_node(self, node):
    """Insert ``node`` into the in-memory caches exactly as the legacy
    AssociativeMemory.add_* does, without re-running the NAMS write."""
    self.id_to_node[node.node_id] = node
    if node.type == "event":
      self.seq_event[0:0] = [node]
      for kw in [k.lower() for k in node.keywords]:
        self.kw_to_event.setdefault(kw, [])[0:0] = [node]
    elif node.type == "thought":
      self.seq_thought[0:0] = [node]
      for kw in [k.lower() for k in node.keywords]:
        self.kw_to_thought.setdefault(kw, [])[0:0] = [node]
    elif node.type == "chat":
      self.seq_chat[0:0] = [node]
      for kw in [k.lower() for k in node.keywords]:
        self.kw_to_chat.setdefault(kw, [])[0:0] = [node]

  def add_event(self, created, expiration, s, p, o,
                description, keywords, poignancy,
                embedding_pair, filling):
    node_count = len(self.id_to_node) + 1
    type_count = len(self.seq_event) + 1
    node_id = f"node_{node_count}"
    # Match the legacy description-cleanup so the in-memory node's
    # description matches what the legacy code would have stored.
    if "(" in description:
      description = (" ".join(description.split()[:3])
                     + " " + description.split("(")[-1][:-1])
    node = self._ConceptNode(node_id, node_count, type_count, "event", 0,
                             created, expiration, s, p, o,
                             description, embedding_pair[0],
                             poignancy, keywords, filling)
    self._record_node(node)
    self.embeddings[embedding_pair[0]] = embedding_pair[1]
    # kw_strength bookkeeping (matches legacy).
    if f"{p} {o}" != "is idle":
      for kw in [k.lower() for k in keywords]:
        self.kw_strength_event[kw] = self.kw_strength_event.get(kw, 0) + 1
    # NAMS write-through: perceived events go to short-term memory so they
    # age out into long-term facts via the extractor.
    try:
      self.nams.add_event(
        s=s, p=p, o=o, description=description or embedding_pair[0],
        poignancy=int(poignancy or 1), created=created or datetime.datetime.now(),
        keywords=[k.lower() for k in keywords],
      )
    except Exception as e:
      print(f"[nams.a_mem] add_event write-through failed: "
            f"{type(e).__name__}: {e}")
    return node

  def add_chat(self, created, expiration, s, p, o,
               description, keywords, poignancy,
               embedding_pair, filling):
    node_count = len(self.id_to_node) + 1
    type_count = len(self.seq_chat) + 1
    node_id = f"node_{node_count}"
    node = self._ConceptNode(node_id, node_count, type_count, "chat", 0,
                             created, expiration, s, p, o,
                             description, embedding_pair[0],
                             poignancy, keywords, filling)
    self._record_node(node)
    self.embeddings[embedding_pair[0]] = embedding_pair[1]
    # NAMS write-through: stage the transcript as short-term chat turns so
    # the extractor sees them. The conversation-close hook (converse module)
    # runs extract + clear at chat end.
    try:
      convo_id = f"chat_{s}_{o}_{(created or datetime.datetime.now()).strftime('%Y%m%d%H%M%S')}"
      if isinstance(filling, list):
        for row in filling:
          if isinstance(row, (list, tuple)) and len(row) >= 2:
            self.nams.add_chat_turn(
              conversation_id=convo_id, speaker=str(row[0]),
              utterance=str(row[1]), poignancy=int(poignancy or 5),
              created=created or datetime.datetime.now(),
            )
      else:
        self.nams.add_chat_turn(
          conversation_id=convo_id, speaker=str(s),
          utterance=str(description), poignancy=int(poignancy or 5),
          created=created or datetime.datetime.now(),
        )
    except Exception as e:
      print(f"[nams.a_mem] add_chat write-through failed: "
            f"{type(e).__name__}: {e}")
    return node

  def add_thought(self, created, expiration, s, p, o,
                  description, keywords, poignancy,
                  embedding_pair, filling):
    node_count = len(self.id_to_node) + 1
    type_count = len(self.seq_thought) + 1
    node_id = f"node_{node_count}"
    depth = 1
    try:
      if filling:
        depth += max([self.id_to_node[i].depth for i in filling if i in self.id_to_node],
                     default=0)
    except Exception:
      pass
    node = self._ConceptNode(node_id, node_count, type_count, "thought", depth,
                             created, expiration, s, p, o,
                             description, embedding_pair[0], poignancy,
                             keywords, filling)
    self._record_node(node)
    self.embeddings[embedding_pair[0]] = embedding_pair[1]
    if f"{p} {o}" != "is idle":
      for kw in [k.lower() for k in keywords]:
        self.kw_strength_thought[kw] = self.kw_strength_thought.get(kw, 0) + 1
    # NAMS write-through: thoughts go directly to long-term facts.
    try:
      self.nams.add_fact(
        subject=s, predicate="thought", obj=description or o,
        poignancy=int(poignancy or 5), kind="thought",
        valid_from=created, valid_until=expiration,
        metadata={"keywords": [k.lower() for k in keywords]},
      )
    except Exception as e:
      print(f"[nams.a_mem] add_thought write-through failed: "
            f"{type(e).__name__}: {e}")
    return node

  # --------------------------------------------------------- legacy reads

  def get_summarized_latest_events(self, retention):
    ret_set = set()
    for e_node in self.seq_event[:retention]:
      ret_set.add(e_node.spo_summary())
    return ret_set

  def get_str_seq_events(self):
    ret_str = ""
    for count, event in enumerate(self.seq_event):
      ret_str += (f'Event {len(self.seq_event) - count}: '
                  f'{event.spo_summary()} -- {event.description}\n')
    return ret_str

  def get_str_seq_thoughts(self):
    ret_str = ""
    for count, event in enumerate(self.seq_thought):
      ret_str += (f'Thought {len(self.seq_thought) - count}: '
                  f'{event.spo_summary()} -- {event.description}\n')
    return ret_str

  def get_str_seq_chats(self):
    ret_str = ""
    for count, event in enumerate(self.seq_chat):
      ret_str += f"with {event.object} ({event.description})\n"
      ret_str += f'{event.created.strftime("%B %d, %Y, %H:%M:%S")}\n'
      for row in event.filling:
        ret_str += f"{row[0]}: {row[1]}\n"
    return ret_str

  def retrieve_relevant_thoughts(self, s_content, p_content, o_content):
    ret = []
    for i in (s_content, p_content, o_content):
      if i and str(i).lower() in self.kw_to_thought:
        ret += self.kw_to_thought[str(i).lower()]
    return set(ret)

  def retrieve_relevant_events(self, s_content, p_content, o_content):
    ret = []
    for i in (s_content, p_content, o_content):
      if i and str(i).lower() in self.kw_to_event:
        ret += self.kw_to_event[str(i).lower()]
    return set(ret)

  def get_last_chat(self, target_persona_name):
    if target_persona_name and target_persona_name.lower() in self.kw_to_chat:
      return self.kw_to_chat[target_persona_name.lower()][0]
    return False


# =========================================================================


class NamsPersona:
  """NAMS-backed persona. Drop-in replacement for ``persona.persona.Persona``
  when a ``*-nams`` harness is active.

  Constructed by ``reverie.py`` (which owns the harness + extraction mode
  selection) -- the per-persona ``NamsMemory`` is built once and held on
  ``self.nams`` for the lifetime of the persona.
  """

  def __init__(self, name, folder_mem_saved, *,
               nams_memory, llm_harness):
    self.name = name
    self.llm = llm_harness
    self.nams = nams_memory

    # Spatial memory stays JSON-backed (it's the world layout, not episodic).
    f_s_mem_saved = f"{folder_mem_saved}/bootstrap_memory/spatial_memory.json"
    self.s_mem = MemoryTree(f_s_mem_saved)

    # Scratch stays JSON-backed (transient state: curr_time, act_*, chatting_*,
    # importance_trigger_curr, recency_w/relevance_w/importance_w + decay,
    # daily_req cache, f_daily_schedule cache).
    scratch_saved = f"{folder_mem_saved}/bootstrap_memory/scratch.json"
    self.scratch = Scratch(scratch_saved)

    # Associative memory is NAMS-backed. The in-memory cache starts empty
    # and is repopulated as events are perceived; long-term facts live in
    # the graph. f_daily_schedule cache is rebuilt from the graph's
    # ScheduleEntry chain so a restarted sim picks up where it left off.
    self.a_mem = NamsAssociativeMemory(nams_memory)
    self._rebuild_schedule_cache_from_graph()

  # ---------------------------------------------------------- save / load

  def save(self, save_folder) -> None:
    """Persist only scratch.json + spatial_memory.json. The graph is the
    long-term store; nothing else needs to be written to disk."""
    f_s_mem = f"{save_folder}/spatial_memory.json"
    self.s_mem.save(f_s_mem)
    f_scratch = f"{save_folder}/scratch.json"
    self.scratch.save(f_scratch)

  def _rebuild_schedule_cache_from_graph(self) -> None:
    """On load, rebuild ``scratch.f_daily_schedule`` from the graph's
    ScheduleEntry chain for today so a restarted sim's planning reads the
    persisted schedule. Best-effort: on any error, leave the existing
    scratch schedule intact."""
    try:
      day_label = (self.scratch.curr_time or datetime.datetime.now()).strftime("%A %B %d")
      chain = self.nams.get_schedule_chain(subject=self.name, day=day_label)
      if not chain:
        return
      new_schedule = []
      for entry in chain:
        md = entry.get("metadata") or {}
        if isinstance(md, str):
          import json as _json
          try:
            md = _json.loads(md)
          except Exception:
            md = {}
        if md.get("kind") != "schedule_entry":
          continue
        vf = entry.get("valid_from")
        vu = entry.get("valid_until")
        duration = 60
        if vf and vu:
          try:
            start_dt = _to_dt(vf)
            end_dt = _to_dt(vu)
            duration = max(1, int((end_dt - start_dt).total_seconds() // 60))
          except Exception:
            pass
        new_schedule.append([entry.get("object") or entry.get("description") or "", duration])
      if new_schedule:
        self.scratch.f_daily_schedule = new_schedule
    except Exception as e:
      print(f"[nams.persona] rebuild schedule cache failed for "
            f"{self.name!r}: {type(e).__name__}: {e}")

  # ------------------------------------------------ cognitive entry points

  def perceive(self, maze):
    from harnesses.nams.cognitive_modules.perceive import perceive as _perceive
    return _perceive(self, maze)

  def retrieve(self, perceived):
    from harnesses.nams.cognitive_modules.retrieve import retrieve as _retrieve
    return _retrieve(self, perceived)

  def plan(self, maze, personas, new_day, retrieved):
    from harnesses.nams.cognitive_modules.plan import plan as _plan
    return _plan(self, maze, personas, new_day, retrieved)

  def execute(self, maze, personas, plan):
    from harnesses.nams.cognitive_modules.execute import execute as _execute
    return _execute(self, maze, personas, plan)

  def reflect(self):
    from harnesses.nams.cognitive_modules.reflect import reflect as _reflect
    _reflect(self)

  def move(self, maze, personas, curr_tile, curr_time):
    """Main cognitive sequence. Mirrors ``Persona.move`` but dispatches to
    the NAMS-aware cognitive modules."""
    self.scratch.curr_tile = curr_tile
    new_day = False
    if not self.scratch.curr_time:
      new_day = "First day"
    elif (self.scratch.curr_time.strftime('%A %B %d')
          != curr_time.strftime('%A %B %d')):
      new_day = "New day"
    self.scratch.curr_time = curr_time

    perceived = self.perceive(maze)
    retrieved = self.retrieve(perceived)
    plan = self.plan(maze, personas, new_day, retrieved)
    self.reflect()
    return self.execute(maze, personas, plan)

  def open_convo_session(self, convo_mode):
    from persona.cognitive_modules.converse import open_convo_session as _legacy
    _legacy(self, convo_mode)


def _to_dt(v):
  if isinstance(v, datetime.datetime):
    return v
  if isinstance(v, str):
    try:
      return datetime.datetime.fromisoformat(v.replace("Z", ""))
    except Exception:
      return datetime.datetime.strptime(v[:19], "%Y-%m-%d %H:%M:%S")
  raise ValueError(f"cannot parse datetime from {v!r}")
