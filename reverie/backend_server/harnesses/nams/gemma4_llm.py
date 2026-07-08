"""
Gemma 4 E4B LLM harness for the NAMS-backed reverie harnesses.

This is a thin wrapper around the existing ``harnesses/gemma4.py``
``_Gemma4Harness`` -- we deliberately **reuse** that module's
thinking-strip + JSON-retry + completion-split logic instead of copy-pasting
it. The only thing this wrapper adds is the NAMS ``LLMProvider`` shim needed
for extraction mode C ("harness-llm").

The active harness is registered in ``harnesses/__init__.py`` as
``gemma4-e4b-nams``; ``reverie.py`` sets ``REVERIE_HARNESS=gemma4-e4b-nams``
and ``harnesses.get_active()`` then returns an instance of this class.
"""
from __future__ import annotations

import asyncio
from typing import Any

from harnesses import gemma4 as _gemma4_mod
from .llm_harness import LLMHarness, _SDKLLMProvider, make_completion


class Gemma4NamsLLM(LLMHarness, _gemma4_mod._Gemma4Harness):
  """Gemma 4 E4B LLM harness, NAMS-aware.

  Multiple-inherits from :class:`LLMHarness` (for the NAMS provider shim and
  type clarity) and :class:`harnesses.gemma4._Gemma4Harness` (for every LLM
  call method -- chat_request, safe_chat_response_json, get_embedding, etc.).
  No method bodies are duplicated; the gemma4 base class already implements
  the full ``harnesses/base.py`` contract.
  """

  def __init__(self, use_thinking: bool = False):
    super().__init__(model_id="google/gemma-4-E4B-it", use_thinking=use_thinking)
    # Override the embedder label is intentionally NOT done -- the gemma4
    # base class already sets embedder_name = "BAAI/bge-small-en-v1.5", which
    # is exactly what we want for this harness.

  def as_nams_llm_provider(self) -> Any:
    """Build a NAMS ``LLMProvider`` shim that routes ``complete()`` calls
    through this harness's ``_generate``. Used only when the user picks
    extraction mode C at startup; in mode A this method is never called.

    The NAMS SDK accepts any object that implements the ``LLMProvider``
    protocol (an async ``complete(messages, **kwargs) -> ChatCompletion``).
    We adapt the (sync) Gemma 4 generator into that contract with
    ``asyncio.to_thread`` so the SDK's event loop isn't blocked while the
    model decodes.
    """
    return _Gemma4NamsProvider(self)


class _Gemma4NamsProvider(_SDKLLMProvider):
  """Async adapter conforming to NAMS's ``LLMProvider`` Protocol, backed by a
  :class:`Gemma4NamsLLM`'s generator.

  Explicitly subclasses the SDK Protocol, exposes the required ``model``
  attribute, matches ``complete``'s keyword-only signature, and returns NAMS's
  real ``Completion`` (via :func:`make_completion`). The NAMS extraction
  pipeline calls ``provider.complete(messages, ...)``; we turn each
  ``ChatMessage`` into the dict shape ``_Gemma4Harness._generate`` expects and
  run the (CPU/GPU-bound) generation off the event loop thread.
  """

  def __init__(self, harness: Gemma4NamsLLM):
    self._harness = harness
    self.model = f"local/{getattr(harness, 'model_id', 'gemma-4-E4B-it')}"

  async def complete(self, messages, *, temperature: float = 0.0,
                     max_tokens: int | None = None,
                     stop=None, timeout: float | None = None):
    harness = self._harness
    msgs = [
      {"role": m.role if hasattr(m, "role") else m.get("role"),
       "content": m.content if hasattr(m, "content") else m.get("content")}
      for m in messages
    ]
    def _call() -> str:
      return harness._generate(
        msgs,
        max_new_tokens=max_tokens or 256,
        temperature=temperature if temperature and temperature > 0 else 0.0,
        top_p=_gemma4_mod._DEFAULT_TOP_P,
        top_k=_gemma4_mod._DEFAULT_TOP_K,
        kind="nams_extraction",
      )

    text = await asyncio.to_thread(_call)
    return make_completion(text, self.model)


def build(use_thinking: bool = False) -> Gemma4NamsLLM:
  """Factory called by ``harnesses/__init__.py`` to build the
  ``gemma4-e4b-nams`` harness."""
  return Gemma4NamsLLM(use_thinking=use_thinking)
