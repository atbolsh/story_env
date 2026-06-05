"""
Local Gemma 4 harness.

Drives reverie's cognitive modules with a locally-hosted Gemma 4 model
(``E2B`` or ``E4B``, instruction-tuned variants) via the HuggingFace
Transformers library. Pairs generation with a small local embedding model
(``BAAI/bge-small-en-v1.5``) so retrieval works without any network.

Public surface matches ``harnesses/base.py``. The module exposes a
``build(model_id)`` factory which returns an object with bound methods
mirroring the harness interface; ``harnesses/__init__.py`` calls
``build("google/gemma-4-E2B-it")`` or ``build("google/gemma-4-E4B-it")``
depending on the selected harness name.

Model and embedder are *lazy-loaded* on first use, so just importing this
module (or calling ``build``) does not download or load any weights.
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from typing import Any, Callable, Optional

# Recommended sampling for Gemma 4 (per the model card):
#   temperature=1.0, top_p=0.95, top_k=64.
# For JSON-output paths we drop temperature/top_p to be much more conservative.
_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_TOP_P = 0.95
_DEFAULT_TOP_K = 64
_JSON_TEMPERATURE = 0.4
_JSON_TOP_P = 0.9
_JSON_TOP_K = 40

_DEFAULT_MAX_NEW_TOKENS = 256


def _maybe_hf_login() -> None:
  """If ``HF_TOKEN`` is in the environment, log in to speed up model pulls."""
  token = os.environ.get("HF_TOKEN", "").strip()
  if not token:
    return
  try:
    from huggingface_hub import login as hf_login
    hf_login(token=token, add_to_git_credential=False)
  except Exception as e:
    print(f"[gemma4] huggingface_hub.login failed ({type(e).__name__}): {e}")


def _apply_stop_sequences(text: str, stop) -> str:
  """OpenAI-style stop-sequence trimming. ``stop`` may be a str or list."""
  if not stop:
    return text
  if isinstance(stop, str):
    stop = [stop]
  earliest = -1
  for s in stop:
    if not s:
      continue
    idx = text.find(s)
    if idx != -1 and (earliest == -1 or idx < earliest):
      earliest = idx
  return text if earliest == -1 else text[:earliest]


def _extract_first_json_object(s: str) -> Optional[str]:
  """Find the first balanced ``{...}`` substring in ``s``, ignoring braces
  inside string literals. Returns ``None`` if not found.

  More forgiving than ``s[:s.rfind('}')+1]`` when the model emits prose
  before the JSON or wraps it in markdown fences.
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


class _Gemma4Harness:
  """Singleton-per-model-id harness.

  Heavy state (model, processor, embedder) is lazy-loaded so the registry can
  build the object cheaply.
  """

  def __init__(self, model_id: str):
    self.model_id = model_id
    self.embedder_name = "BAAI/bge-small-en-v1.5"
    self._model = None
    self._processor = None
    self._embedder = None
    self._device = None

  # ------------------------------------------------------------------ load
  def _ensure_model(self) -> None:
    if self._model is not None:
      return
    _maybe_hf_login()
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"[gemma4] loading {self.model_id} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(self.model_id)
    model = AutoModelForCausalLM.from_pretrained(
      self.model_id,
      dtype="auto",
      device_map="auto",
    )
    model.eval()
    self._processor = processor
    self._model = model
    self._device = model.device
    print(f"[gemma4] loaded in {time.time() - t0:.1f}s on {self._device}")

  def _ensure_embedder(self) -> None:
    if self._embedder is not None:
      return
    _maybe_hf_login()
    from sentence_transformers import SentenceTransformer
    try:
      import torch
      device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
      device = "cpu"
    print(f"[gemma4] loading embedder {self.embedder_name} on {device} ...")
    t0 = time.time()
    self._embedder = SentenceTransformer(self.embedder_name, device=device)
    print(f"[gemma4] embedder ready in {time.time() - t0:.1f}s")

  # -------------------------------------------------------------- generate
  def _generate(self,
                messages: list,
                max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
                temperature: float = _DEFAULT_TEMPERATURE,
                top_p: float = _DEFAULT_TOP_P,
                top_k: int = _DEFAULT_TOP_K,
                stop=None) -> str:
    """Core chat-template generation. Returns the assistant text."""
    self._ensure_model()
    import torch

    text = self._processor.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False,
    )
    inputs = self._processor(text=text, return_tensors="pt").to(self._device)
    input_len = inputs["input_ids"].shape[-1]

    do_sample = temperature is not None and temperature > 0
    gen_kwargs = dict(
      max_new_tokens=max(1, int(max_new_tokens)),
      do_sample=do_sample,
    )
    if do_sample:
      gen_kwargs["temperature"] = float(temperature)
      gen_kwargs["top_p"] = float(top_p)
      gen_kwargs["top_k"] = int(top_k)

    with torch.inference_mode():
      outputs = self._model.generate(**inputs, **gen_kwargs)

    response = self._processor.decode(
      outputs[0][input_len:], skip_special_tokens=False
    )
    try:
      parsed = self._processor.parse_response(response)
    except Exception:
      parsed = response

    if isinstance(parsed, dict):
      text_out = parsed.get("content") or parsed.get("text") or ""
    else:
      text_out = str(parsed)

    return _apply_stop_sequences(text_out, stop)

  # ============================================================ public API
  # Bare requests
  def chat_single_request(self, prompt: str) -> str:
    return self._generate(
      [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
      ],
      max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
    )

  def chat_request(self, prompt: str) -> str:
    try:
      return self._generate(
        [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt},
        ],
        max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
      )
    except Exception:
      traceback.print_exc()
      print("Gemma4 ERROR")
      return "Gemma4 ERROR"

  def chat_request_strong(self, prompt: str) -> str:
    # Gemma 4 harness has only one model loaded; "strong" = same model, but
    # we allow a higher max_new_tokens to mirror the legacy GPT-4 contract.
    try:
      return self._generate(
        [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt},
        ],
        max_new_tokens=512,
      )
    except Exception:
      traceback.print_exc()
      print("Gemma4 ERROR")
      return "Gemma4 ERROR"

  # JSON-wrapped retry loops
  def _safe_json(self,
                 strong: bool,
                 prompt: str,
                 example_output: Any,
                 special_instruction: str,
                 repeat: int = 3,
                 fail_safe_response="error",
                 func_validate: Optional[Callable] = None,
                 func_clean_up: Optional[Callable] = None,
                 verbose: bool = False):
    system_prompt = (
      "You are a strict JSON-emitting assistant. Output exactly one JSON "
      "object and nothing else. No prose, no commentary, no markdown fences. "
      'The object must have a single key "output". Example: {"output": "..."}.'
    )
    user_prompt = (
      '"""\n' + prompt + '\n"""\n'
      + f"Output the response to the prompt above in json. {special_instruction}\n"
      + "Example output json:\n"
      + '{"output": "' + str(example_output) + '"}'
    )
    if verbose:
      print("GEMMA4 PROMPT")
      print(user_prompt)

    max_new_tokens = 512 if strong else 384
    for i in range(repeat):
      try:
        raw = self._generate(
          [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
          ],
          max_new_tokens=max_new_tokens,
          temperature=_JSON_TEMPERATURE,
          top_p=_JSON_TOP_P,
          top_k=_JSON_TOP_K,
        ).strip()

        obj_str = _extract_first_json_object(raw)
        if obj_str is None:
          # Fall back to the historical "trim to last brace" heuristic.
          end_index = raw.rfind('}') + 1
          obj_str = raw[:end_index] if end_index > 0 else raw

        parsed_obj = json.loads(obj_str)
        curr_response = parsed_obj["output"]

        if func_validate(curr_response, prompt=user_prompt):
          return func_clean_up(curr_response, prompt=user_prompt)

        if verbose:
          print("---- repeat count: \n", i, curr_response)
          print(curr_response)
          print("~~~~")

      except Exception:
        if verbose:
          traceback.print_exc()

    return False

  def safe_chat_response_json(self, prompt, example_output, special_instruction,
                              repeat=3, fail_safe_response="error",
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
      print("GEMMA4 PROMPT")
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

  # Legacy completion-style request + safe loop
  def llm_request(self, prompt: str, gpt_parameter: dict) -> str:
    """Mimics the legacy completion-style ``GPT_request`` contract."""
    try:
      # The legacy code treats temperature=0 as deterministic. With Gemma 4
      # we approximate that by disabling sampling.
      temp = float(gpt_parameter.get("temperature", _DEFAULT_TEMPERATURE))
      top_p = float(gpt_parameter.get("top_p", _DEFAULT_TOP_P))
      max_new_tokens = int(gpt_parameter.get("max_tokens", _DEFAULT_MAX_NEW_TOKENS))
      stop = gpt_parameter.get("stop", None)
      # The legacy prompts were authored for text-davinci style continuation,
      # not chat. A neutral system prompt keeps things grounded.
      messages = [
        {"role": "system",
         "content": "Continue the user's text directly. Be concise and follow the prompt's format exactly."},
        {"role": "user", "content": prompt},
      ]
      return self._generate(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temp if temp > 0 else 0.0,
        top_p=top_p,
        top_k=_DEFAULT_TOP_K,
        stop=stop,
      )
    except Exception as e:
      print(f"[gemma4] llm_request error ({type(e).__name__}): {e}")
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

  # Embeddings
  def get_embedding(self, text: str, model: Optional[str] = None):
    self._ensure_embedder()
    text = (text or "").replace("\n", " ").strip()
    if not text:
      text = "this is blank"
    vec = self._embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()


def build(model_id: str) -> _Gemma4Harness:
  """Return a fresh harness object for ``model_id``."""
  return _Gemma4Harness(model_id)
