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
