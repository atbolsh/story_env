"""
NAMS-aware Plan module.

Reuses the legacy plan orchestration helpers (``_long_term_planning``,
``_determine_action``, ``_choose_retrieved``, ``_should_react``,
``_chat_react``, ``_wait_react``, ``_create_react``) unchanged -- no
copy-paste. The NAMS-specific behavior is layered on top:

  * **Schedule-as-graph.** After ``_long_term_planning`` generates
    ``scratch.f_daily_schedule`` on a new day, we mirror it into the graph
    as temporal ScheduleEntry Facts (``valid_from``/``valid_until`` bounded
    to today's time windows). The schedule **is** the graph; mid-day edits
    (chat insertion, task decomp) are reflected by re-mirroring the chain.
    ``_determine_action`` reads the schedule from the graph (rebuilt into
    the scratch cache on load) so decisions are consistent with whatever the
    graph currently says.

  * **Chat-react writes a ScheduleEntry.** When ``_chat_react`` inserts a
    conversation into the persona's day, we add a ScheduleEntry Fact for the
    conversation window so retrieve force-injects it as a near-future plan.
    The inter-PERSON relationship edge and any future-commitment Plan fact
    are added at conversation close (see
    :mod:`harnesses.nams.cognitive_modules.converse`).

  * **Future plans ride along.** ``retrieve`` already unions
    ``get_upcoming_plans`` into every focal-point context, so the schedule-
    propagation gap (KNOWN_WEAKNESSES §0) is closed at retrieval time; this
    module just keeps the graph's schedule chain in sync with scratch.
"""
from __future__ import annotations

import datetime

from persona.cognitive_modules.plan import (
  _long_term_planning as _legacy_long_term_planning,
  _determine_action as _legacy_determine_action,
  _choose_retrieved,
  _should_react,
  _chat_react as _legacy_chat_react,
  _wait_react,
)


def _mirror_schedule_to_graph(persona) -> None:
  """Replace the graph's schedule_entry chain for today with the current
  ``scratch.f_daily_schedule``. Called after _long_term_planning and after
  any schedule mutation (chat insertion, task decomp)."""
  scratch = persona.scratch
  name = scratch.name
  day_label = scratch.curr_time.strftime("%A %B %d")
  day_start = scratch.curr_time.replace(hour=0, minute=0, second=0,
                                        microsecond=0)
  try:
    persona.nams.clear_schedule_for_day(subject=name, day=day_label)
  except Exception as e:
    print(f"[nams.plan] clear_schedule_for_day failed: {type(e).__name__}: {e}")
  cursor = day_start
  order = 0
  for entry in (scratch.f_daily_schedule or []):
    if not entry or len(entry) < 2:
      continue
    task, duration = entry[0], int(entry[1])
    try:
      persona.nams.add_schedule_entry(
        subject=name, description=str(task),
        start=cursor, duration_minutes=duration,
        poignancy=5, day=day_label, order=order,
      )
    except Exception as e:
      print(f"[nams.plan] add_schedule_entry failed: {type(e).__name__}: {e}")
    cursor = cursor + datetime.timedelta(minutes=duration)
    order += 1


def _nams_chat_react(maze, persona, focused_event, reaction_mode, personas):
  """Legacy _chat_react, plus a ScheduleEntry fact for the conversation
  window so retrieve force-injects it as a near-future plan."""
  _legacy_chat_react(maze, persona, focused_event, reaction_mode, personas)
  # Re-mirror the (now chat-inserted) schedule to the graph. The legacy
  # _chat_react rewrites scratch.f_daily_schedule via _create_react, so a
  # fresh mirror captures the conversation as a schedule entry.
  try:
    _mirror_schedule_to_graph(persona)
  except Exception as e:
    print(f"[nams.plan] post-chat_react mirror failed: {type(e).__name__}: {e}")


def plan(persona, maze, personas, new_day, retrieved):
  """NAMS-aware plan. See module docstring."""
  # PART 1: long-term planning on a new day, then mirror to graph.
  if new_day:
    _legacy_long_term_planning(persona, new_day)
    try:
      _mirror_schedule_to_graph(persona)
    except Exception as e:
      print(f"[nams.plan] post-long_term mirror failed: {type(e).__name__}: {e}")

  # PART 2: if the current action has expired, decide the next one.
  if persona.scratch.act_check_finished():
    _legacy_determine_action(persona, maze)

  # PART 3: react to a perceived event that needs a response.
  focused_event = False
  if retrieved.keys():
    focused_event = _choose_retrieved(persona, retrieved)
  if focused_event:
    reaction_mode = _should_react(persona, focused_event, personas)
    if reaction_mode:
      if reaction_mode[:9] == "chat with":
        _nams_chat_react(maze, persona, focused_event, reaction_mode, personas)
      elif reaction_mode[:4] == "wait":
        _wait_react(persona, reaction_mode)

  # PART 4: chat-related state clean up (verbatim from legacy plan).
  if persona.scratch.act_event[1] != "chat with":
    persona.scratch.chatting_with = None
    persona.scratch.chat = None
    persona.scratch.chatting_end_time = None
  curr_persona_chat_buffer = persona.scratch.chatting_with_buffer
  for persona_name, buffer_count in curr_persona_chat_buffer.items():
    if persona_name != persona.scratch.chatting_with:
      persona.scratch.chatting_with_buffer[persona_name] -= 1

  return persona.scratch.act_address
