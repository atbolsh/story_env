"""
NAMS-backed reverie harnesses.

Two new harnesses that drive the cognitive modules with a Neo4j Agent Memory
System backing store running on a **local bolt Neo4j** (no NAMS hosted API
keys; see ``docker-compose.yml`` at the repo root):

  * ``gemma4-e4b-nams``  -- local Gemma 4 E4B + local BGE embedder + NAMS bolt
  * ``latest-gpt-nams``  -- modern OpenAI chat models + OpenAI embeddings +
                            NAMS bolt

Both harnesses share the same:

  * DB-only memory layer       -- :mod:`harnesses.nams.nams_memory`
  * Cognitive modules          -- :mod:`harnesses.nams.cognitive_modules`
  * Prompt templates           -- the existing ``persona/prompt_template/``
                                  package (reused unchanged via the
                                  ``gpt_structure`` facade, which dispatches
                                  to the active harness's LLM methods)
  * JSON -> NAMS one-way import -- :mod:`harnesses.nams.json_to_nams_import`
  * sync -> async bridge       -- :mod:`harnesses.nams.async_bridge`

They differ only in the LLM-call mechanics, expressed via the
:class:`harnesses.nams.llm_harness.LLMHarness` interface:

  * :mod:`harnesses.nams.gemma4_llm` -- reuses ``harnesses/gemma4.py``
    (``build("google/gemma-4-E4B-it")``) so the thinking-strip + JSON-retry
    + completion-split logic is not copy-pasted.
  * :mod:`harnesses.nams.gpt_llm`    -- modern ``openai>=1.0`` SDK with
    ``response_format={"type": "json_object"}`` for the JSON paths.

Selection is via ``REVERIE_HARNESS`` (set by ``reverie.py``'s startup prompt)
plus ``REVERIE_NAMS_EXTRACTION`` (``no-llm`` or ``harness-llm``; see
:mod:`harnesses.nams.nams_memory`).
"""
from __future__ import annotations

from .nams_memory import NamsMemory, NAMS_EXTRACTION_NO_LLM, NAMS_EXTRACTION_HARNESS_LLM, build_memory_settings
from .nams_persona import NamsPersona

__all__ = [
  "NamsMemory",
  "NamsPersona",
  "NAMS_EXTRACTION_NO_LLM",
  "NAMS_EXTRACTION_HARNESS_LLM",
  "build_memory_settings",
]
