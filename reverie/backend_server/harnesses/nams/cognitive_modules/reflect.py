"""
NAMS-aware Reflect module.

Keeps the legacy ``importance_trigger_curr`` countdown in scratch (decremented
by each perceived event's poignancy in ``perceive``). When it hits 0, runs a
NAMS **reasoning trace** (``NamsMemory.run_reflection_trace``) capturing the
"reflect and evaluate" episode -- focal points, retrieved nodes, generated
insights, and the final outcome -- as a structured, debuggable trace node
instead of the legacy free-floating thought nodes. The trace is searchable
later so future reflections can lean on past ones.

Chat-end block: when ``curr_time + 10s == chatting_end_time``, the legacy
reflect generates a planning thought and a memo thought from the conversation
transcript. We keep those LLM calls (they reuse the active harness) and store
the results as long-term Facts (kind='thought') via the a_mem adapter, then
fire the NAMS conversation-close hook (extract + inter-PERSON relationship
edge + forget raw text) defined in :mod:`harnesses.nams.cognitive_modules.converse`.
"""
from __future__ import annotations

import datetime

from persona.cognitive_modules.reflect import (
  generate_focal_points,
  generate_insights_and_evidence,
  generate_action_event_triple,
  generate_poig_score,
  generate_planning_thought_on_convo,
  generate_memo_on_convo,
)
from persona.prompt_template.gpt_structure import get_embedding

from . import retrieve as nams_retrieve
from . import converse as nams_converse


def reflection_trigger(persona) -> bool:
  """Same countdown as the legacy trigger, but the 'any events yet' check
  uses the a_mem adapter's in-memory cache (which perceive populates)."""
  if (persona.scratch.importance_trigger_curr <= 0
      and [] != persona.a_mem.seq_event + persona.a_mem.seq_thought):
    return True
  return False


def reset_reflection_counter(persona) -> None:
  persona.scratch.importance_trigger_curr = persona.scratch.importance_trigger_max
  persona.scratch.importance_ele_n = 0


def run_reflect(persona) -> None:
  """NAMS reflection: focal points -> retrieve -> insights -> reasoning trace.

  Reuses the legacy LLM prompts (``generate_focal_points``,
  ``generate_insights_and_evidence``) which route to the active harness. The
  difference from legacy ``run_reflect`` is that the insights are recorded
  as a NAMS reasoning trace (one ``add_step`` per insight) rather than as
  free-floating thought ConceptNodes.
  """
  focal_points = generate_focal_points(persona, 3)
  retrieved = nams_retrieve.new_retrieve(persona, focal_points)

  steps = []
  all_insights = []
  for focal_pt, nodes in retrieved.items():
    node_labels = [i.embedding_key for i in nodes]
    insights = generate_insights_and_evidence(persona, nodes, 5)
    insight_texts = list(insights.keys()) if isinstance(insights, dict) else []
    all_insights.extend(insight_texts)
    steps.append({
      "thought": f"Focal point: {focal_pt}",
      "action": "retrieve + generate insights",
      "observation": "; ".join(node_labels[:5]) + " => " + " | ".join(insight_texts[:3]),
    })

  outcome = " ".join(all_insights)[:2000] if all_insights else "no insights"
  try:
    persona.nams.run_reflection_trace(
      task=f"Reflect and evaluate -- {persona.name}",
      steps=steps,
      outcome=outcome,
      success=True,
    )
  except Exception as e:
    print(f"[nams.reflect] run_reflection_trace failed for {persona.name!r}: "
          f"{type(e).__name__}: {e}")


def reflect(persona) -> None:
  """Main reflection entry point. See module docstring."""
  if reflection_trigger(persona):
    run_reflect(persona)
    reset_reflection_counter(persona)

  # Chat-end block: fires once per persona as the conversation window closes.
  # We keep the legacy planning/memo thought generation (reuses the active
  # harness's LLM) and store the thoughts as long-term Facts, then trigger
  # the NAMS conversation-close hook (extract + relationship edge + forget).
  if persona.scratch.chatting_end_time:
    if persona.scratch.curr_time + datetime.timedelta(seconds=10) == persona.scratch.chatting_end_time:
      all_utt = ""
      if persona.scratch.chat:
        for row in persona.scratch.chat:
          all_utt += f"{row[0]}: {row[1]}\n"

      try:
        last_chat = persona.a_mem.get_last_chat(persona.scratch.chatting_with)
        evidence = [last_chat.node_id] if last_chat else []
      except Exception:
        evidence = []

      planning_thought = generate_planning_thought_on_convo(persona, all_utt)
      planning_thought = f"For {persona.scratch.name}'s planning: {planning_thought}"
      _store_thought_fact(persona, planning_thought, evidence)

      memo_thought = generate_memo_on_convo(persona, all_utt)
      memo_thought = f"{persona.scratch.name} {memo_thought}"
      _store_thought_fact(persona, memo_thought, evidence)

      # NAMS conversation-close: extract entities/relations, write an
      # inter-PERSON relationship edge with the convo summary, forget raw.
      try:
        nams_converse.on_conversation_close(persona)
      except Exception as e:
        print(f"[nams.reflect] on_conversation_close failed for "
              f"{persona.name!r}: {type(e).__name__}: {e}")


def _store_thought_fact(persona, thought_text: str, evidence) -> None:
  """Store a conversation-derived thought as a long-term Fact (kind='thought')
  via the a_mem adapter, which write-throughs to NAMS."""
  created = persona.scratch.curr_time
  expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
  try:
    s, p, o = generate_action_event_triple(thought_text, persona)
  except Exception:
    s, p, o = persona.name, "thought", thought_text
  keywords = set([s, p, o])
  try:
    thought_poignancy = generate_poig_score(persona, "thought", thought_text)
  except Exception:
    thought_poignancy = 5
  thought_embedding_pair = (thought_text, get_embedding(thought_text))
  try:
    persona.a_mem.add_thought(created, expiration, s, p, o,
                              thought_text, keywords, thought_poignancy,
                              thought_embedding_pair, evidence)
  except Exception as e:
    print(f"[nams.reflect] add_thought failed for {persona.name!r}: "
          f"{type(e).__name__}: {e}")
