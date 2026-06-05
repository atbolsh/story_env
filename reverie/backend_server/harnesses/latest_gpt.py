"""
Latest OpenAI chat-model harness (stub).

Scaffolded but not yet implemented. The legacy GPT harness pins
``openai<1.0`` so the migration to the modern OpenAI SDK (and its
gpt-4o / gpt-5.x-class models) is a separate piece of work. Every public
entry point here raises ``NotImplementedError`` with a clear message.
"""

from __future__ import annotations

embedder_name = "openai/UNIMPLEMENTED"

_MSG = (
  "latest-gpt harness is scaffolded but not implemented yet; "
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
