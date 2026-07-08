"""
Modern OpenAI chat-model LLM harness for the NAMS-backed reverie harnesses.

This is the long-overdue migration of the legacy GPT harness from the
``openai==0.28`` SDK to the modern ``openai>=1.0`` SDK (KNOWN_WEAKNESSES §3.9).
It uses ``client.chat.completions.create`` with the ``response_format =
{"type": "json_object"}`` mode for the JSON-wrapped paths (closing the
brittle ``rfind('}')`` parsing in the legacy harness, §3.7), and OpenAI's
``text-embedding-3-small`` for retrieval embeddings.

Registered in ``harnesses/__init__.py`` as ``latest-gpt-nams``; selected via
``REVERIE_HARNESS=latest-gpt-nams``.

The model tier is configurable via the ``OPENAI_CHAT_MODEL`` and
``OPENAI_CHAT_STRONG_MODEL`` env vars (defaulting to ``gpt-4o-mini`` and
``gpt-4o``). The embedder model is configurable via ``OPENAI_EMBED_MODEL``
(default ``text-embedding-3-small``).
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Callable, Optional

from reverie_config import openai_api_key

from harnesses import prompt_log
from .llm_harness import LLMHarness, _SDKLLMProvider, make_completion

# --- model selection -------------------------------------------------------

_DEFAULT_CHAT_MODEL = "gpt-4o-mini"
_DEFAULT_STRONG_MODEL = "gpt-4o"
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def _chat_model() -> str:
  return os.environ.get("OPENAI_CHAT_MODEL", "").strip() or _DEFAULT_CHAT_MODEL


def _strong_model() -> str:
  return os.environ.get("OPENAI_CHAT_STRONG_MODEL", "").strip() or _DEFAULT_STRONG_MODEL


def _embed_model() -> str:
  return os.environ.get("OPENAI_EMBED_MODEL", "").strip() or _DEFAULT_EMBED_MODEL


# --- OpenAI SDK bootstrap --------------------------------------------------

_client = None


def _ensure_client():
  """Lazily construct the modern OpenAI client.

  Importing ``openai`` is deferred so a user who only ever runs the gemma4
  harnesses does not need the openai package installed.
  """
  global _client
  if _client is not None:
    return _client
  from openai import OpenAI  # type: ignore
  if not openai_api_key:
    raise RuntimeError(
      "OPENAI_API_KEY is not set; the latest-gpt-nams harness needs it. "
      "Put it in .env at the repo root (see README.md)."
    )
  _client = OpenAI(api_key=openai_api_key)
  return _client


def _embedder_name_for_settings() -> str:
  return f"openai/{_embed_model()}"


def _temp_sleep(seconds: float = 0.1) -> None:
  time.sleep(seconds)


def _extract_first_json_object(s: str) -> Optional[str]:
  """Find the first balanced ``{...}`` substring in ``s``, ignoring braces
  inside string literals. Returns ``None`` if not found.

  ``response_format=json_object`` makes the model emit JSON reliably, but we
  keep this defensive parser for the rare case where the wrapper prompt
  smuggles in a preamble. Mirrors ``harnesses/gemma4.py``'s helper.
  """
  start = s.find("{")
  if start == -1:
    return None
  depth = 0
  in_str = False
  esc = False
  for i in range(start, len(s)):
    c = s[i]
    if in_str:
      if esc:
        esc = False
      elif c == "\\":
        esc = True
      elif c == '"':
        in_str = False
      continue
    if c == '"':
      in_str = True
    elif c == "{":
      depth += 1
    elif c == "}":
      depth -= 1
      if depth == 0:
        return s[start:i + 1]
  return None


class LatestGPTNamsLLM(LLMHarness):
  """Modern OpenAI chat-model harness implementing the LLMHarness contract."""

  def __init__(self):
    self.engine_label = _chat_model()
    self.embedder_name = _embedder_name_for_settings()

  # ------------------------------------------------------------------ bare

  def _chat_call(self, model: str, messages: list, kind: str, **params) -> str:
    client = _ensure_client()
    try:
      completion = client.chat.completions.create(
        model=model, messages=messages, **params
      )
      out = completion.choices[0].message.content or ""
      prompt_log.log_call(
        harness="latest-gpt-nams", model=model, kind=kind,
        request={"messages": messages}, params=params or None,
        response={"raw": out, "returned": out},
      )
      return out
    except Exception as e:
      prompt_log.log_call(
        harness="latest-gpt-nams", model=model, kind=kind,
        request={"messages": messages}, params=params or None,
        error=f"{type(e).__name__}: {e}",
      )
      raise

  def chat_single_request(self, prompt: str) -> str:
    _temp_sleep()
    return self._chat_call(
      _chat_model(), [{"role": "user", "content": prompt}], "chat_single"
    )

  def chat_request(self, prompt: str) -> str:
    try:
      _temp_sleep()
      return self._chat_call(
        _chat_model(), [{"role": "user", "content": prompt}], "chat"
      )
    except Exception:
      traceback.print_exc()
      print("ChatGPT ERROR")
      return "ChatGPT ERROR"

  def chat_request_strong(self, prompt: str) -> str:
    try:
      _temp_sleep()
      return self._chat_call(
        _strong_model(), [{"role": "user", "content": prompt}], "chat_strong"
      )
    except Exception:
      traceback.print_exc()
      print("ChatGPT ERROR")
      return "ChatGPT ERROR"

  # ------------------------------------------------------------ safe loops

  def _safe_json(self, *, strong: bool, prompt: str, example_output: Any,
                 special_instruction: str, repeat: int = 3,
                 fail_safe_response="error",
                 func_validate: Optional[Callable] = None,
                 func_clean_up: Optional[Callable] = None,
                 verbose: bool = False):
    model = _strong_model() if strong else _chat_model()
    client = _ensure_client()
    system_prompt = (
      "You are a strict JSON-emitting assistant. Output exactly one JSON "
      'object and nothing else. The object must have a single key "output". '
      'Example: {"output": "..."}.'
    )
    user_prompt = (
      '"""\n' + prompt + '\n"""\n'
      + f"Output the response to the prompt above in json. {special_instruction}\n"
      + "Example output json:\n"
      + '{"output": "' + str(example_output) + '"}'
    )
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
    ]
    if verbose:
      print("LATEST-GPT-NAMS PROMPT")
      print(user_prompt)
    for i in range(repeat):
      try:
        completion = client.chat.completions.create(
          model=model, messages=messages,
          response_format={"type": "json_object"},
          temperature=0.2,
        )
        raw = (completion.choices[0].message.content or "").strip()
        prompt_log.log_call(
          harness="latest-gpt-nams", model=model, kind="chat_json",
          request={"messages": messages},
          params={"response_format": {"type": "json_object"}, "temperature": 0.2},
          response={"raw": raw, "returned": raw},
        )
        obj_str = _extract_first_json_object(raw)
        if obj_str is None:
          end_index = raw.rfind('}') + 1
          obj_str = raw[:end_index] if end_index > 0 else raw
        parsed = json.loads(obj_str)
        curr = parsed["output"]
        if not isinstance(curr, str):
          curr = str(curr)
        if func_validate(curr, prompt=user_prompt):
          return func_clean_up(curr, prompt=user_prompt)
        if verbose:
          print(f"---- repeat count: {i}")
          print(curr)
          print("~~~~")
      except Exception:
        if verbose:
          traceback.print_exc()
    return False

  def safe_chat_response_json(self, prompt, example_output,
                              special_instruction, repeat=3,
                              fail_safe_response="error",
                              func_validate=None, func_clean_up=None,
                              verbose=False):
    return self._safe_json(
      strong=False, prompt=prompt, example_output=example_output,
      special_instruction=special_instruction, repeat=repeat,
      fail_safe_response=fail_safe_response,
      func_validate=func_validate, func_clean_up=func_clean_up, verbose=verbose,
    )

  def safe_chat_response_json_strong(self, prompt, example_output,
                                     special_instruction, repeat=3,
                                     fail_safe_response="error",
                                     func_validate=None, func_clean_up=None,
                                     verbose=False):
    return self._safe_json(
      strong=True, prompt=prompt, example_output=example_output,
      special_instruction=special_instruction, repeat=repeat,
      fail_safe_response=fail_safe_response,
      func_validate=func_validate, func_clean_up=func_clean_up, verbose=verbose,
    )

  def safe_chat_response(self, prompt, repeat=3, fail_safe_response="error",
                         func_validate=None, func_clean_up=None, verbose=False):
    if verbose:
      print("LATEST-GPT-NAMS PROMPT")
      print(prompt)
    for i in range(repeat):
      try:
        curr = self.chat_request(prompt).strip()
        if func_validate(curr, prompt=prompt):
          return func_clean_up(curr, prompt=prompt)
        if verbose:
          print(f"---- repeat count: {i}")
          print(curr)
          print("~~~~")
      except Exception:
        if verbose:
          traceback.print_exc()
    print("FAIL SAFE TRIGGERED")
    return fail_safe_response

  # ----------------------------------------------------- legacy completion

  def llm_request(self, prompt: str, gpt_parameter: dict) -> str:
    """Legacy completion-style request. The davinci engines are gone; we
    route to the chat model. The original prompts end mid-sentence, so we
    keep the historical contract by handing the model the prompt as a user
    turn and letting it continue."""
    _temp_sleep()
    engine = gpt_parameter.get("engine", _chat_model())
    model = engine if engine.startswith("gpt-") else _chat_model()
    try:
      return self._chat_call(
        model,
        [{"role": "user", "content": prompt}],
        "completion",
        temperature=gpt_parameter.get("temperature", 0.7),
        max_tokens=gpt_parameter.get("max_tokens", 256),
        top_p=gpt_parameter.get("top_p", 1.0),
        frequency_penalty=gpt_parameter.get("frequency_penalty", 0.0),
        presence_penalty=gpt_parameter.get("presence_penalty", 0.0),
        stop=gpt_parameter.get("stop", None),
      )
    except Exception as e:
      print(f"GPT_request error ({type(e).__name__}): {e}")
      return "TOKEN LIMIT EXCEEDED"

  def safe_generate_response(self, prompt, gpt_parameter, repeat=5,
                             fail_safe_response="error",
                             func_validate=None, func_clean_up=None,
                             verbose=False):
    if verbose:
      print(prompt)
    for i in range(repeat):
      curr = self.llm_request(prompt, gpt_parameter)
      if func_validate(curr, prompt=prompt):
        return func_clean_up(curr, prompt=prompt)
      if verbose:
        print("---- repeat count: ", i, curr)
        print(curr)
        print("~~~~")
    return fail_safe_response

  # ----------------------------------------------------------- embeddings

  def get_embedding(self, text: str, model: Optional[str] = None) -> list[float]:
    client = _ensure_client()
    text = (text or "").replace("\n", " ")
    if not text:
      text = "this is blank"
    emb_model = model or _embed_model()
    resp = client.embeddings.create(input=[text], model=emb_model)
    return resp.data[0].embedding

  # -------------------------------------------------- NAMS LLM provider shim

  def as_nams_llm_provider(self) -> Any:
    """Build a NAMS ``LLMProvider`` shim that routes ``complete()`` calls
    through this harness's chat completions. Used only in extraction mode C."""
    return _OpenAINamsProvider(self)


class _OpenAINamsProvider(_SDKLLMProvider):
  """Async adapter conforming to NAMS's ``LLMProvider`` Protocol by deferring
  to the (sync) OpenAI chat client via ``asyncio.to_thread``.

  Explicitly subclasses the SDK Protocol, exposes the required ``model``
  attribute, matches ``complete``'s keyword-only signature, and returns NAMS's
  real ``Completion`` (via :func:`make_completion`)."""

  def __init__(self, harness: LatestGPTNamsLLM):
    self._harness = harness
    self.model = f"openai/{_chat_model()}"

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
      return harness._chat_call(
        _chat_model(), msgs, "nams_extraction",
        temperature=temperature if temperature > 0 else 0.0,
        max_tokens=max_tokens or 256,
      )

    import asyncio
    text = await asyncio.to_thread(_call)
    return make_completion(text, self.model)


def build() -> LatestGPTNamsLLM:
  """Factory called by ``harnesses/__init__.py`` to build the
  ``latest-gpt-nams`` harness."""
  return LatestGPTNamsLLM()
