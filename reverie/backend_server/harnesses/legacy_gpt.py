"""
Legacy GPT harness.

Behavior of the original Park et al. 2023 ``gpt_structure.py`` -- chat
completion calls plus the legacy completion-engine remap
(text-davinci-002/003 -> gpt-3.5-turbo). The only structural differences are
that (a) the function names are vendor-neutral (``llm_request`` etc.), with
the facade in ``persona/prompt_template/gpt_structure.py`` re-exporting the
old vendor-flavored names as aliases for backward compatibility with
defunct/test files, and (b) the SDK surface is the modern ``openai>=1.0``
client API (``client.chat.completions.create``) rather than the historical
0.28 module-level ``openai.ChatCompletion.create`` -- the call semantics,
models, retry loops, and prompt/response logging are unchanged.
"""

from __future__ import annotations

import json
import random  # noqa: F401  -- preserved for code that does `from harnesses... import *`
import time

from reverie_config import openai_api_key

from . import prompt_log

# OpenAI SDK bootstrap. Lazy so a user who only ever runs the gemma4
# harnesses does not need the openai package installed. Migrated from the
# 0.28 ``openai.ChatCompletion`` / ``openai.Embedding`` module-level API to
# the >=1.0 client API (KNOWN_WEAKNESSES §3.9) so this harness coexists with
# the NAMS ``[openai]`` extra, which requires openai>=1.0.
_client = None


def _ensure_client():
  global _client
  if _client is not None:
    return _client
  from openai import OpenAI  # type: ignore
  if not openai_api_key:
    raise RuntimeError(
      "OPENAI_API_KEY is not set; the legacy-gpt harness needs it. "
      "Put it in .env at the repo root (see README.md)."
    )
  _client = OpenAI(api_key=openai_api_key)
  return _client

embedder_name = "openai/text-embedding-ada-002"

# Label substituted into gpt_param["engine"] by run_gpt_prompt.py. Kept as the
# historical completion-engine name so llm_request's legacy remap (see
# _LEGACY_COMPLETION_ENGINE_TO_CHAT_MODEL) keeps resolving it to gpt-3.5-turbo.
engine_label = "text-davinci-003"


def _temp_sleep(seconds: float = 0.1) -> None:
  time.sleep(seconds)


def _chat_call(model: str, messages: list, kind: str, **params) -> str:
  """``client.chat.completions.create`` plus exact prompt/response logging
  (see ``prompt_log``). Raises on failure -- after logging the error record
  -- so each caller keeps its historical try/except semantics."""
  try:
    completion = _ensure_client().chat.completions.create(
      model=model, messages=messages, **params
    )
    out = completion.choices[0].message.content or ""
    prompt_log.log_call(
      harness="legacy-gpt", model=model, kind=kind,
      request={"messages": messages}, params=params or None,
      response={"raw": out, "returned": out},
    )
    return out
  except Exception as e:
    prompt_log.log_call(
      harness="legacy-gpt", model=model, kind=kind,
      request={"messages": messages}, params=params or None,
      error=f"{type(e).__name__}: {e}",
    )
    raise


# ---------------------------------------------------------------------------
# Bare requests
# ---------------------------------------------------------------------------

def chat_single_request(prompt: str) -> str:
  """One-shot chat call. No try/except, with a small sleep. (Historically
  ``ChatGPT_single_request``.)"""
  _temp_sleep()
  return _chat_call(
    "gpt-3.5-turbo", [{"role": "user", "content": prompt}], "chat_single"
  )


def chat_request_strong(prompt: str) -> str:
  """One-shot chat call, stronger-tier model. (Historically ``GPT4_request``.)"""
  _temp_sleep()
  try:
    return _chat_call(
      "gpt-4", [{"role": "user", "content": prompt}], "chat_strong"
    )
  except Exception:
    print("ChatGPT ERROR")
    return "ChatGPT ERROR"


def chat_request(prompt: str) -> str:
  """One-shot chat call, default-tier model. (Historically ``ChatGPT_request``.)"""
  try:
    return _chat_call(
      "gpt-3.5-turbo", [{"role": "user", "content": prompt}], "chat"
    )
  except Exception:
    print("ChatGPT ERROR")
    return "ChatGPT ERROR"


# ---------------------------------------------------------------------------
# JSON-wrapped chat retry loops
# ---------------------------------------------------------------------------

def safe_chat_response_json_strong(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  """JSON-wrapped retry loop, stronger-tier model.
  (Historically ``GPT4_safe_generate_response``.)"""
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print("CHAT GPT PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = chat_request_strong(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print("---- repeat count: \n", i, curr_gpt_response)
        print(curr_gpt_response)
        print("~~~~")

    except Exception:
      pass

  return False


def safe_chat_response_json(prompt,
                            example_output,
                            special_instruction,
                            repeat=3,
                            fail_safe_response="error",
                            func_validate=None,
                            func_clean_up=None,
                            verbose=False):
  """JSON-wrapped retry loop, default-tier model.
  (Historically ``ChatGPT_safe_generate_response``.)"""
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print("CHAT GPT PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = chat_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print("---- repeat count: \n", i, curr_gpt_response)
        print(curr_gpt_response)
        print("~~~~")

    except Exception:
      pass

  return False


def safe_chat_response(prompt,
                       repeat=3,
                       fail_safe_response="error",
                       func_validate=None,
                       func_clean_up=None,
                       verbose=False):
  """Free-form chat retry loop (no JSON wrapping).
  (Historically ``ChatGPT_safe_generate_response_OLD``.)"""
  if verbose:
    print("CHAT GPT PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = chat_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose:
        print(f"---- repeat count: {i}")
        print(curr_gpt_response)
        print("~~~~")

    except Exception:
      pass
  print("FAIL SAFE TRIGGERED")
  return fail_safe_response


# ---------------------------------------------------------------------------
# Legacy completion-style request + safe loop
# ---------------------------------------------------------------------------

# OpenAI deprecated the GPT-3 completion models (text-davinci-002/003) that the
# original paper used. Map the legacy engine names this codebase still passes in
# (via gpt_parameter["engine"]) onto a currently available chat model.
_LEGACY_COMPLETION_ENGINE_TO_CHAT_MODEL = {
  "text-davinci-003": "gpt-3.5-turbo",
  "text-davinci-002": "gpt-3.5-turbo",
}


def llm_request(prompt: str, gpt_parameter: dict) -> str:
  """Legacy completion-style request. (Historically ``GPT_request``.)"""
  _temp_sleep()
  engine = gpt_parameter["engine"]
  model = _LEGACY_COMPLETION_ENGINE_TO_CHAT_MODEL.get(engine, engine)
  try:
    return _chat_call(
      model,
      [{"role": "user", "content": prompt}],
      "completion",
      temperature=gpt_parameter["temperature"],
      max_tokens=gpt_parameter["max_tokens"],
      top_p=gpt_parameter["top_p"],
      frequency_penalty=gpt_parameter["frequency_penalty"],
      presence_penalty=gpt_parameter["presence_penalty"],
      stream=gpt_parameter["stream"],
      stop=gpt_parameter["stop"],
    )
  except Exception as e:
    print(f"GPT_request error ({type(e).__name__}): {e}")
    return "TOKEN LIMIT EXCEEDED"


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
  if verbose:
    print(prompt)

  for i in range(repeat):
    curr_gpt_response = llm_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt):
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose:
      print("---- repeat count: ", i, curr_gpt_response)
      print(curr_gpt_response)
      print("~~~~")
  return fail_safe_response


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embedding(text: str, model: str | None = "text-embedding-ada-002"):
  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"
  resp = _ensure_client().embeddings.create(
    input=[text], model=model or "text-embedding-ada-002"
  )
  return resp.data[0].embedding
