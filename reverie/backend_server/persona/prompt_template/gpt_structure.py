"""
Author: Joon Sung Park (joonspk@stanford.edu)
Maintained: reworked into a vendor-neutral facade over the ``harnesses``
package so reverie can drive its agents with different LLM backends
(legacy OpenAI, local Gemma 4, ...).

Public API (vendor-neutral; preferred):

  llm_request(prompt, params)
  chat_request(prompt)
  chat_request_strong(prompt)
  chat_single_request(prompt)
  safe_generate_response(...)
  safe_chat_response(...)
  safe_chat_response_json(...)
  safe_chat_response_json_strong(...)
  get_embedding(text, model=None)
  generate_prompt(curr_input, prompt_lib_file)

Backward-compatible aliases (deprecated -- defunct/test files still import
these via ``from persona.prompt_template.gpt_structure import *``):

  GPT_request                        -> llm_request
  ChatGPT_request                    -> chat_request
  GPT4_request                       -> chat_request_strong
  ChatGPT_single_request             -> chat_single_request
  ChatGPT_safe_generate_response_OLD -> safe_chat_response
  ChatGPT_safe_generate_response     -> safe_chat_response_json
  GPT4_safe_generate_response        -> safe_chat_response_json_strong
"""

import os
import random  # noqa: F401  -- historical: callers do `import *`
import string  # noqa: F401  -- historical: callers do `import *`
import time    # noqa: F401  -- historical: callers do `import *`

from harnesses import get_active

# Anchored so the prompt-template files resolve regardless of the process's
# current working directory (e.g. when reverie.py is launched from the repo
# root rather than from reverie/backend_server/). Callers throughout the
# codebase pass repo-relative paths like "persona/prompt_template/v2/...".
_BACKEND_SERVER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Vendor-neutral public API -- each function defers to the active harness.
# ---------------------------------------------------------------------------


def llm_request(prompt, params):
  """Legacy completion-style request. ``params`` mirrors the historical
  ``gpt_parameter`` dict."""
  return get_active().llm_request(prompt, params)


def chat_request(prompt):
  """One-shot chat call, default-tier model."""
  return get_active().chat_request(prompt)


def chat_request_strong(prompt):
  """One-shot chat call, stronger-tier model."""
  return get_active().chat_request_strong(prompt)


def chat_single_request(prompt):
  """One-shot chat call without try/except; sleeps briefly first."""
  return get_active().chat_single_request(prompt)


def safe_generate_response(prompt, params, repeat=5, fail_safe_response="error",
                           func_validate=None, func_clean_up=None, verbose=False):
  return get_active().safe_generate_response(
    prompt, params, repeat=repeat, fail_safe_response=fail_safe_response,
    func_validate=func_validate, func_clean_up=func_clean_up, verbose=verbose,
  )


def safe_chat_response(prompt, repeat=3, fail_safe_response="error",
                       func_validate=None, func_clean_up=None, verbose=False):
  return get_active().safe_chat_response(
    prompt, repeat=repeat, fail_safe_response=fail_safe_response,
    func_validate=func_validate, func_clean_up=func_clean_up, verbose=verbose,
  )


def safe_chat_response_json(prompt, example_output, special_instruction,
                            repeat=3, fail_safe_response="error",
                            func_validate=None, func_clean_up=None,
                            verbose=False):
  return get_active().safe_chat_response_json(
    prompt, example_output, special_instruction,
    repeat=repeat, fail_safe_response=fail_safe_response,
    func_validate=func_validate, func_clean_up=func_clean_up, verbose=verbose,
  )


def safe_chat_response_json_strong(prompt, example_output, special_instruction,
                                   repeat=3, fail_safe_response="error",
                                   func_validate=None, func_clean_up=None,
                                   verbose=False):
  return get_active().safe_chat_response_json_strong(
    prompt, example_output, special_instruction,
    repeat=repeat, fail_safe_response=fail_safe_response,
    func_validate=func_validate, func_clean_up=func_clean_up, verbose=verbose,
  )


def get_embedding(text, model=None):
  return get_active().get_embedding(text, model=model)


def active_engine():
  """Engine label of the active harness, for the ``engine`` field of the
  legacy ``gpt_param`` dicts (and the debug printouts that echo them). Only
  the legacy-gpt harness actually consumes the value; local harnesses ignore
  it, so this is primarily so logs reflect the model really being used."""
  return get_active().engine_label


# ---------------------------------------------------------------------------
# Pure-text helper (no model, kept here so it works before any harness is
# initialized).
# ---------------------------------------------------------------------------


def generate_prompt(curr_input, prompt_lib_file):
  """Takes in the current input (e.g. comment that you want to classify) and
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: ``!<INPUT>!`` -- this
  function replaces this substr with the actual ``curr_input`` to produce the
  final prompt that will be sent to the LLM.

  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the prompt file.
  RETURNS:
    a str prompt that will be sent to the LLM."""
  if type(curr_input) == type("string"):
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  # Resolve repo-relative prompt paths against the backend_server dir so the
  # simulation works no matter what CWD it was launched from.
  if not os.path.isabs(prompt_lib_file):
    prompt_lib_file = os.path.join(_BACKEND_SERVER_DIR, prompt_lib_file)

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


# ---------------------------------------------------------------------------
# Deprecated vendor-flavored aliases. Kept so existing `from ... import *`
# in defunct_run_gpt_prompt.py and test.py keeps working. New code should
# use the vendor-neutral names above.
# ---------------------------------------------------------------------------

GPT_request = llm_request
GPT4_request = chat_request_strong
ChatGPT_request = chat_request
ChatGPT_single_request = chat_single_request
ChatGPT_safe_generate_response = safe_chat_response_json
GPT4_safe_generate_response = safe_chat_response_json_strong
ChatGPT_safe_generate_response_OLD = safe_chat_response


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50,
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0,
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response, prompt=""):
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1:
      return False
    return True

  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.strip()

  output = safe_generate_response(prompt, gpt_parameter, 5, "rest",
                                  __func_validate, __func_clean_up, True)
  print(output)
