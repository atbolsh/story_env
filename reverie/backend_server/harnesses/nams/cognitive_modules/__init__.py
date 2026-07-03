"""NAMS-aware cognitive modules.

These are thin, LLM-agnostic wrappers around the existing
``persona/cognitive_modules/*`` functions. They reuse the legacy modules
(no copy-paste) and layer on the NAMS-specific behavior:

  * ``perceive``  -- legacy perceive, but ``persona.a_mem`` is a
                     :class:`harnesses.nams.nams_persona.NamsAssociativeMemory`
                     that write-throughs to NAMS short-term memory. Also runs
                     the smooth short-term aging each step and checks the
                     importance-trigger countdown for reflection.
  * ``retrieve``  -- ``new_retrieve`` queries NAMS ``get_context_for_planning``
                     (which force-injects upcoming plans) and wraps the
                     results as ConceptNode-like objects; ``retrieve``
                     (per-event) delegates to the legacy keyword lookup.
  * ``plan``      -- legacy plan orchestration, but ``_long_term_planning``
                     mirrors the generated schedule into the graph as
                     ScheduleEntry facts (the schedule IS the graph), and
                     ``_chat_react`` adds a Plan fact for any future
                     commitment parsed from the conversation summary.
  * ``reflect``   -- keeps the ``importance_trigger_curr`` countdown in
                     scratch; when it hits 0, runs a NAMS reasoning trace
                     (``run_reflection_trace``) instead of writing
                     free-floating thought nodes.
  * ``converse``  -- legacy ``agent_chat_v2`` for the live conversation; on
                     chat close, extracts entities/relations, writes an
                     inter-PERSON relationship edge with the convo summary,
                     and forgets the raw text.
  * ``execute``   -- legacy pathing/address logic, unchanged.
"""
from . import perceive, retrieve, plan, reflect, converse, execute

__all__ = ["perceive", "retrieve", "plan", "reflect", "converse", "execute"]
