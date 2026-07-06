"""
Abstract LLM harness interface for the NAMS-backed reverie harnesses.

This mirrors the public surface documented in ``harnesses/base.py`` so that
the existing ``persona/prompt_template/gpt_structure.py`` facade can dispatch
to whichever concrete LLM harness the user picked at startup. The cognitive
modules in ``harnesses/nams/cognitive_modules/`` call these methods via the
facade, so they remain LLM-agnostic.

Two concrete implementations live alongside this file:

  * :mod:`harnesses.nams.gemma4_llm` -- reuses ``harnesses/gemma4.py``
  * :mod:`harnesses.nams.gpt_llm`    -- modern ``openai>=1.0`` SDK
"""
from __future__ import annotations

from typing import Any, Callable, Optional


class LLMHarness:
  """Vendor-neutral LLM + embedding interface.

  Concrete harnesses may be either module-level function collections (matching
  the historical ``harnesses/base.py`` duck-typed contract) or instances of
  this class. The NAMS cognitive modules access the active LLM harness via
  ``harnesses.get_active()``, which is what ``gpt_structure.py`` already
  defers to.
  """

  #: Short identifier for the embedding model (persisted into a sim's
  #: ``meta.json`` for cross-embedder fork detection).
  embedder_name: str = "unknown"

  #: Label substituted into ``gpt_param["engine"]`` for debug printouts.
  engine_label: str = "unknown"

  # ----- bare requests -----------------------------------------------------

  def llm_request(self, prompt: str, params: dict) -> str:
    """Legacy completion-style call. ``params`` mirrors the old
    ``gpt_parameter`` dict (engine, max_tokens, temperature, top_p,
    frequency_penalty, presence_penalty, stream, stop)."""
    raise NotImplementedError

  def chat_request(self, prompt: str) -> str:
    """One-shot chat call, default-tier model."""
    raise NotImplementedError

  def chat_request_strong(self, prompt: str) -> str:
    """One-shot chat call, stronger-tier model."""
    raise NotImplementedError

  def chat_single_request(self, prompt: str) -> str:
    """One-shot chat call without try/except wrapping."""
    raise NotImplementedError

  # ----- retry / validate / clean loops ------------------------------------

  def safe_generate_response(self, prompt, params, repeat=5,
                             fail_safe_response="error",
                             func_validate: Optional[Callable] = None,
                             func_clean_up: Optional[Callable] = None,
                             verbose=False) -> str:
    raise NotImplementedError

  def safe_chat_response(self, prompt, repeat=3,
                         fail_safe_response="error",
                         func_validate: Optional[Callable] = None,
                         func_clean_up: Optional[Callable] = None,
                         verbose=False) -> str:
    raise NotImplementedError

  def safe_chat_response_json(self, prompt, example_output,
                              special_instruction, repeat=3,
                              fail_safe_response="error",
                              func_validate: Optional[Callable] = None,
                              func_clean_up: Optional[Callable] = None,
                              verbose=False):
    raise NotImplementedError

  def safe_chat_response_json_strong(self, prompt, example_output,
                                     special_instruction, repeat=3,
                                     fail_safe_response="error",
                                     func_validate: Optional[Callable] = None,
                                     func_clean_up: Optional[Callable] = None,
                                     verbose=False):
    raise NotImplementedError

  # ----- embeddings --------------------------------------------------------

  def get_embedding(self, text: str, model: Optional[str] = None) -> list[float]:
    raise NotImplementedError

  # ----- NAMS LLM provider shim -------------------------------------------
  #
  # When the user picks extraction mode C ("harness-llm"), the NAMS SDK's
  # internal LLM extractor needs a Provider. Rather than re-implement the
  # provider interface twice, we expose a small async callable here that
  # concrete harnesses can build a Provider from. See nams_memory.py for how
  # this is wired into MemorySettings(llm=...).

  def as_nams_llm_provider(self) -> Any:
    """Return an object implementing the NAMS ``LLMProvider`` protocol, or
    raise ``NotImplementedError`` if the harness does not support being used
    as the NAMS extraction LLM. Only invoked in extraction mode C."""
    raise NotImplementedError(
      f"{type(self).__name__} does not provide a NAMS LLM provider; "
      "use extraction mode 'no-llm' with this harness."
    )


# --------------------------------------------------------------------------- #
# Generic NAMS LLM provider adapter
# --------------------------------------------------------------------------- #
#
# In the mixed-harness ("multi-harness") mode, the active LLM harness is a
# plain ``harnesses/gemma4.py`` ``_Gemma4Harness`` (not the NAMS-specific
# ``Gemma4NamsLLM`` subclass), because most personas in the sim run on the
# legacy JSON memory and only one (e.g. Klaus) runs on NAMS. The plain
# harness doesn't override ``as_nams_llm_provider``, so for extraction mode
# C we adapt it generically: any harness that exposes a
# ``_generate(messages, *, max_new_tokens, temperature, top_p, top_k, kind)``
# method (the gemma4 contract) can serve as the NAMS extraction LLM by
# deferring to that method off the SDK's event-loop thread.
#
# Routing through ``_generate`` (instead of calling the model directly) means
# every NAMS extraction call is logged via ``prompt_log.log_call`` with full
# request/response context, exactly like the cognitive-module LLM calls --
# satisfying the "all LLM calls logged with full context" requirement.


class NamsLLMProvider:
  """Minimal async adapter that satisfies the NAMS ``LLMProvider`` protocol
  by deferring to a harness's ``_generate``.

  Works for any harness whose ``_generate`` matches the gemma4 signature
  (``messages`` list of ``{"role", "content"}`` dicts, plus the keyword
  sampling params). The GPT harnesses use their own ``as_nams_llm_provider``
  and never hit this class.
  """

  def __init__(self, harness: Any,
               top_p: float = 0.95, top_k: int = 64):
    self._harness = harness
    self._top_p = top_p
    self._top_k = top_k

  async def complete(self, messages, *, temperature: float = 0.0,
                     max_tokens: int | None = None, **_: Any):
    harness = self._harness
    msgs = [
      {"role": (m.role if hasattr(m, "role") else m.get("role")),
       "content": (m.content if hasattr(m, "content") else m.get("content"))}
      for m in messages
    ]

    def _call() -> str:
      return harness._generate(
        msgs,
        max_new_tokens=max_tokens or 256,
        temperature=temperature if temperature and temperature > 0 else 0.0,
        top_p=self._top_p,
        top_k=self._top_k,
        kind="nams_extraction",
      )

    import asyncio
    text = await asyncio.to_thread(_call)

    class _Completion:
      def __init__(self, content: str):
        self.content = content

    return _Completion(text)


def nams_llm_provider_for(harness: Any) -> Any:
  """Return a NAMS ``LLMProvider`` for ``harness``.

  Uses the harness's own ``as_nams_llm_provider()`` when it has one (the
  NAMS-specific harnesses), else falls back to the generic
  :class:`NamsLLMProvider` adapter (the plain gemma4 harness in mixed mode).
  """
  if hasattr(harness, "as_nams_llm_provider"):
    try:
      return harness.as_nams_llm_provider()
    except NotImplementedError:
      pass
  return NamsLLMProvider(harness)
