"""
NAMS-aware Execute module.

The pathing / address / object-selection logic is unchanged -- it does not
touch long-term memory, only scratch + the maze. We delegate to the legacy
``execute`` unchanged.
"""
from __future__ import annotations

from persona.cognitive_modules.execute import execute as _legacy_execute


def execute(persona, maze, personas, plan):
  return _legacy_execute(persona, maze, personas, plan)
