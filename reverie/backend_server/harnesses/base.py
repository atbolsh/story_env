"""
Harness interface (documentation only).

Each harness module / object must expose the following module-level callables.
Duck-typed -- there's no abstract base class, since the cognitive modules
historically pull everything via ``from persona.prompt_template.gpt_structure
import *`` and the facade simply re-exports these names.

Required attributes:

  embedder_name : str
    Short identifier for the embedding model the harness uses (e.g.
    ``"openai/text-embedding-ada-002"`` or ``"BAAI/bge-small-en-v1.5"``).
    Persisted into a simulation save's ``meta.json`` so we can detect
    cross-embedder fork attempts.

Required functions:

  llm_request(prompt: str, params: dict) -> str
    Legacy completion-style call. ``params`` mirrors the old
    ``gpt_parameter`` dict (engine, max_tokens, temperature, top_p,
    frequency_penalty, presence_penalty, stream, stop).

  chat_request(prompt: str) -> str
    One-shot chat call, default-tier model. Returns the assistant text.

  chat_request_strong(prompt: str) -> str
    One-shot chat call, stronger-tier model. For harnesses that only have
    one model (e.g. Gemma 4), this falls back to ``chat_request``.

  chat_single_request(prompt: str) -> str
    Like ``chat_request`` but without try/except wrapping (the historical
    contract from ``ChatGPT_single_request``).

  safe_generate_response(prompt, params, repeat=5, fail_safe_response="error",
                         func_validate=None, func_clean_up=None,
                         verbose=False) -> str
    Validate/clean/retry loop on top of ``llm_request``.

  safe_chat_response(prompt, repeat=3, fail_safe_response="error",
                     func_validate=None, func_clean_up=None,
                     verbose=False) -> str
    Validate/clean/retry loop on top of ``chat_request``, free-form output.

  safe_chat_response_json(prompt, example_output, special_instruction,
                          repeat=3, fail_safe_response="error",
                          func_validate=None, func_clean_up=None,
                          verbose=False) -> str | bool
    Validate/clean/retry loop, but wraps the prompt with JSON-output
    instructions and parses ``{"output": ...}`` from the model's reply.
    Returns ``False`` on total failure (matches historical contract).

  safe_chat_response_json_strong(...) -> str | bool
    Same as above, but routes through ``chat_request_strong``.

  get_embedding(text: str, model: str | None = None) -> list[float]
    Embedding for retrieval. ``model`` is a hint that the legacy harness
    forwards to OpenAI; other harnesses may ignore it.
"""
