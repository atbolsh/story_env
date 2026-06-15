"""
LLM harness registry.

This package decouples the reverie backend from a single LLM vendor. Each
harness module exposes the same vendor-neutral function surface (see
``base.py``), and ``get_active()`` returns the currently selected one.

Selection is driven by the ``REVERIE_HARNESS`` environment variable, which
``reverie.py``'s ``__main__`` sets from an interactive prompt before any
heavy LLM call happens. Resolution and weight loading are both lazy --
importing this package is cheap; loading e.g. Gemma 4's weights only
happens on the first ``get_active()`` call.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict

DEFAULT_HARNESS = "legacy-gpt"

_AVAILABLE: Dict[str, str] = {
  "legacy-gpt":  "OpenAI GPT-3.5 / GPT-4 via the openai==0.28 SDK (default).",
  "gemma4-e2b":  "Local Gemma 4 E2B (instruction-tuned) via transformers.",
  "gemma4-e4b":  "Local Gemma 4 E4B (instruction-tuned) via transformers.",
  "qwen3-0.6b":  "Local Qwen3-0.6B (non-thinking mode) via transformers.",
  "claude":      "Anthropic Claude (scaffolded; not implemented).",
  "latest-gpt":  "Latest OpenAI chat models (scaffolded; not implemented).",
}


def available_names() -> Dict[str, str]:
  """Return a mapping of harness name -> short description."""
  return dict(_AVAILABLE)


# Cached active harness module/object. Resolved on first ``get_active()`` call
# so that ``import harnesses`` stays cheap and the env var can be set after
# import time (but before any LLM call).
_active: Any = None
_active_name: str = ""


def _resolve(name: str) -> Any:
  if name == "legacy-gpt":
    from . import legacy_gpt
    return legacy_gpt
  if name == "gemma4-e2b":
    from . import gemma4
    return gemma4.build("google/gemma-4-E2B-it")
  if name == "gemma4-e4b":
    from . import gemma4
    return gemma4.build("google/gemma-4-E4B-it")
  if name == "qwen3-0.6b":
    from . import qwen
    return qwen.build("Qwen/Qwen3-0.6B")
  if name == "claude":
    from . import claude
    return claude
  if name == "latest-gpt":
    from . import latest_gpt
    return latest_gpt
  raise ValueError(
    f"unknown harness {name!r}; valid: {sorted(_AVAILABLE)}"
  )


def get_active() -> Any:
  """Return the active harness, loading it lazily on first call."""
  global _active, _active_name
  if _active is not None:
    return _active
  name = os.environ.get("REVERIE_HARNESS", DEFAULT_HARNESS).strip()
  if not name:
    name = DEFAULT_HARNESS
  _active = _resolve(name)
  _active_name = name
  return _active


def get_active_name() -> str:
  """Return the string name of the active harness (after ``get_active`` has been called)."""
  if _active is None:
    return os.environ.get("REVERIE_HARNESS", DEFAULT_HARNESS).strip() or DEFAULT_HARNESS
  return _active_name


def reset() -> None:
  """Drop the cached active harness. Mostly useful in tests / the rebuild script."""
  global _active, _active_name
  _active = None
  _active_name = ""
