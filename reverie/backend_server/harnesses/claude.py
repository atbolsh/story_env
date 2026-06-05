"""
Anthropic Claude harness (stub).

Scaffolded but not yet implemented. Every public entry point raises
``NotImplementedError`` with a clear message so misconfiguration fails
loudly rather than silently producing garbage.

When this lands, it should mirror the harness interface in ``base.py``:
- generation via Anthropic's messages API
- embeddings via ``BAAI/bge-small-en-v1.5`` (matching the Gemma 4 harness,
  so saves can be forked between them without re-embedding)
"""

from __future__ import annotations

embedder_name = "anthropic/UNIMPLEMENTED"

_MSG = (
  "claude harness is scaffolded but not implemented yet; "
  "select 'legacy-gpt' or 'gemma4-e2b' / 'gemma4-e4b' for now."
)


def _raise(*_args, **_kwargs):
  raise NotImplementedError(_MSG)


llm_request = _raise
chat_request = _raise
chat_request_strong = _raise
chat_single_request = _raise
safe_generate_response = _raise
safe_chat_response = _raise
safe_chat_response_json = _raise
safe_chat_response_json_strong = _raise
get_embedding = _raise
