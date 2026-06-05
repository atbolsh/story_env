"""
Legacy GPT harness.

This is the verbatim behavior of the original Park et al. 2023 ``gpt_structure.py``
-- OpenAI 0.28 chat completion calls, plus the legacy completion-engine remap
(text-davinci-002/003 -> gpt-3.5-turbo). The only structural difference is that
the function names are vendor-neutral (``llm_request`` etc.); the facade in
``persona/prompt_template/gpt_structure.py`` re-exports the old vendor-flavored
names as aliases for backward compatibility with defunct/test files.
"""

from __future__ import annotations

import json
import random  # noqa: F401  -- preserved for code that does `from harnesses... import *`
import time

import openai

from reverie_config import openai_api_key

openai.api_key = openai_api_key

embedder_name = "openai/text-embedding-ada-002"


def _temp_sleep(seconds: float = 0.1) -> None:
  time.sleep(seconds)


# ---------------------------------------------------------------------------
# Bare requests
# ---------------------------------------------------------------------------

def chat_single_request(prompt: str) -> str:
  """One-shot chat call. No try/except, with a small sleep. (Historically
  ``ChatGPT_single_request``.)"""
  _temp_sleep()
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
  )
  return completion["choices"][0]["message"]["content"]


def chat_request_strong(prompt: str) -> str:
  """One-shot chat call, stronger-tier model. (Historically ``GPT4_request``.)"""
  _temp_sleep()
  try:
    completion = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[{"role": "user", "content": prompt}],
    )
    return completion["choices"][0]["message"]["content"]
  except Exception:
    print("ChatGPT ERROR")
    return "ChatGPT ERROR"


def chat_request(prompt: str) -> str:
  """One-shot chat call, default-tier model. (Historically ``ChatGPT_request``.)"""
  try:
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": prompt}],
    )
    return completion["choices"][0]["message"]["content"]
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
    response = openai.ChatCompletion.create(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      temperature=gpt_parameter["temperature"],
      max_tokens=gpt_parameter["max_tokens"],
      top_p=gpt_parameter["top_p"],
      frequency_penalty=gpt_parameter["frequency_penalty"],
      presence_penalty=gpt_parameter["presence_penalty"],
      stream=gpt_parameter["stream"],
      stop=gpt_parameter["stop"],
    )
    return response["choices"][0]["message"]["content"]
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
  return openai.Embedding.create(
    input=[text], model=model or "text-embedding-ada-002"
  )["data"][0]["embedding"]
