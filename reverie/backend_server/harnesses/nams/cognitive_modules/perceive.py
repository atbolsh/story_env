"""
NAMS-aware Perceive module.

Reuses the legacy ``persona.cognitive_modules.perceive.perceive`` unchanged
-- the NAMS write-through happens inside
:class:`harnesses.nams.nams_persona.NamsAssociativeMemory.add_event` /
``add_chat``, which the legacy perceive calls.

This wrapper additionally:

  1. Runs the smooth short-term aging step (``nams.age_short_term``) so raw
     messages older than ``SHORT_TERM_TTL_MINUTES`` (default 4) of in-sim time
     are extracted into long-term facts and deleted each step, rather than
     batching at the 4-minute boundary.
  2. Returns the perceived events exactly as the legacy perceive does (a list
     of ``ConceptNode`` from the in-memory cache), so the downstream
     ``retrieve`` / ``plan`` modules keep working.
"""
from __future__ import annotations

import datetime
import math
from operator import itemgetter

from persona.cognitive_modules.perceive import (
  generate_poig_score as _legacy_generate_poig_score,
  perceive as _legacy_perceive,
)


def perceive(persona, maze):
  """NAMS-aware perceive. See module docstring."""
  # Age out short-term messages whose in-sim age exceeds the TTL, smoothly
  # per step. Run BEFORE perceiving new events so the buffer drains as sim
  # time advances.
  try:
    now = persona.scratch.curr_time or datetime.datetime.now()
    persona.nams.age_short_term(now, ttl_minutes=persona.nams.SHORT_TERM_TTL_MINUTES)
  except Exception as e:
    print(f"[nams.perceive] age_short_term failed for {persona.name!r}: "
          f"{type(e).__name__}: {e}")

  # Legacy perceive writes through to NAMS via the a_mem adapter. It also
  # decrements scratch.importance_trigger_curr per perceived event; the
  # reflect module checks that countdown.
  return _legacy_perceive(persona, maze)
