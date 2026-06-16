"""
Local Qwen3-0.6B harness.

Drives reverie's cognitive modules with a locally-hosted Qwen3-0.6B model via
the HuggingFace Transformers library, pairing generation with the same small
local embedder used by the Gemma harness (``BAAI/bge-small-en-v1.5``) so
retrieval works fully offline and sims stay fork-compatible across the local
harnesses.

Public surface matches ``harnesses/base.py``. ``harnesses/__init__.py`` calls
``build("Qwen/Qwen3-0.6B")`` (or ``build("Qwen/Qwen3-0.6B", use_thinking=True)``
for the ``*-thinking`` harness variant). Model and embedder are *lazy-loaded* on
first use, so importing this module (or calling ``build``) downloads/loads
nothing.

Qwen3 specifics / best practices (per the Qwen3 model card):

  * Qwen3 is a hybrid "thinking" model. By default we run it in **non-thinking
    mode** (``enable_thinking=False`` in the chat template) for the agent loop:
    the cognitive prompts want terse, format-following completions, and
    ``<think>`` blocks both bloat the output and break the downstream cleaners
    (the same lesson that motivated ``enable_thinking=False`` for Gemma). Any
    stray ``<think>...</think>`` is stripped defensively.
  * When ``use_thinking`` is set the chat template enables the reasoning block
    (``enable_thinking=True``), each call's token budget is scaled by
    ``_THINKING_MAX_NEW_TOKENS_MULT``, and the ``<think>...</think>`` content is
    logged but stripped from every returned value -- so memories, conversations,
    and JSON behave as if no reasoning block existed.
  * Recommended non-thinking sampling: ``temperature=0.7, top_p=0.8,
    top_k=20, min_p=0``. Recommended thinking sampling: ``temperature=0.6,
    top_p=0.95, top_k=20, min_p=0``. JSON-output paths drop the temperature for
    stability.
  * The model card warns against greedy decoding for *thinking* mode; in
    non-thinking mode greedy is acceptable, and we honor the legacy
    ``temperature=0`` contract (deterministic prompts like decide_to_talk) by
    disabling sampling. In thinking mode we never go greedy: a deterministic
    request is bumped to the recommended thinking temperature instead.

The legacy completion-prompt handling, stop-sequence trimming, and JSON
extraction are shared with the Gemma harness (same upstream prompt templates).
"""

from __future__ import annotations

import json
import time
import traceback
from typing import Any, Callable, Optional

from . import prompt_log
# Shared, model-agnostic prompt helpers (legacy completion-prompt splitting,
# OpenAI-style stop trimming, tolerant first-JSON-object extraction).
from .gemma4 import (
  _apply_stop_sequences,
  _extract_first_json_object,
  _maybe_hf_login,
  _split_completion_prompt,
)

# Recommended sampling for Qwen3 in non-thinking mode (per the model card):
#   temperature=0.7, top_p=0.8, top_k=20, min_p=0.
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_P = 0.8
_DEFAULT_TOP_K = 20
_DEFAULT_MIN_P = 0.0
# JSON-output paths: keep top_p/top_k but be much more conservative on temp.
_JSON_TEMPERATURE = 0.3
_JSON_TOP_P = 0.8
_JSON_TOP_K = 20

_DEFAULT_MAX_NEW_TOKENS = 256

# Recommended sampling for Qwen3 in thinking mode (per the model card):
#   temperature=0.6, top_p=0.95, top_k=20, min_p=0. Greedy decoding is
#   explicitly discouraged in thinking mode (it can cause endless repetition).
_THINKING_TEMPERATURE = 0.6
_THINKING_TOP_P = 0.95
_THINKING_TOP_K = 20
# Thinking spends part of the budget on the <think> block before the answer, so
# scale every call's token budget up (mirrors the Gemma thinking harness).
_THINKING_MAX_NEW_TOKENS_MULT = 2


def _split_qwen_thinking(text: str) -> tuple:
  """Return ``(answer, thinking_or_None)`` splitting off a ``<think>...</think>``
  block. Qwen3 emits one only with thinking enabled; we always split defensively
  so a stray (or truncated) block never reaches the cleaners or memories.
  ``answer`` is the text with the block removed; ``thinking`` is the reasoning
  (or ``None`` if absent)."""
  if "<think>" not in text:
    return text, None
  open_idx = text.find("<think>")
  close = text.rfind("</think>")
  if close != -1:
    thinking = text[open_idx + len("<think>"):close].strip()
    answer = (text[:open_idx] + text[close + len("</think>"):]).lstrip()
    return answer, (thinking or None)
  # Unclosed think block (e.g. truncated): drop everything from the tag on.
  thinking = text[open_idx + len("<think>"):].strip()
  answer = text[:open_idx].rstrip()
  return answer, (thinking or None)


class _QwenHarness:
  """Singleton-per-model-id harness. Heavy state is lazy-loaded so the registry
  can build the object cheaply."""

  def __init__(self, model_id: str, use_thinking: bool = False):
    self.model_id = model_id
    # When True, the chat template enables Qwen3's <think> reasoning block. The
    # reasoning is logged (see prompt_log) but stripped from every value handed
    # back to the cognitive modules, so memories / conversations / JSON behave
    # exactly as if it were never generated.
    self.use_thinking = bool(use_thinking)
    # Label substituted into gpt_param["engine"] by run_gpt_prompt.py (purely
    # informational for this harness -- llm_request never reads "engine").
    self.engine_label = model_id
    self.embedder_name = "BAAI/bge-small-en-v1.5"
    self._model = None
    self._tokenizer = None
    self._embedder = None
    self._device = None

  # ------------------------------------------------------------------ load
  def _ensure_model(self) -> None:
    if self._model is not None:
      return
    _maybe_hf_login()
    import torch  # noqa: F401  -- ensures torch is importable before generate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[qwen] loading {self.model_id} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    model = AutoModelForCausalLM.from_pretrained(
      self.model_id,
      dtype="auto",
      device_map="auto",
    )
    model.eval()
    self._tokenizer = tokenizer
    self._model = model
    self._device = model.device
    print(f"[qwen] loaded in {time.time() - t0:.1f}s on {self._device}")

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
    print(f"[qwen] loading embedder {self.embedder_name} on {device} ...")
    t0 = time.time()
    self._embedder = SentenceTransformer(self.embedder_name, device=device)
    print(f"[qwen] embedder ready in {time.time() - t0:.1f}s")

  # -------------------------------------------------------------- generate
  def _generate(self,
                messages: list,
                max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
                temperature: float = _DEFAULT_TEMPERATURE,
                top_p: float = _DEFAULT_TOP_P,
                top_k: int = _DEFAULT_TOP_K,
                stop=None,
                kind: str = "chat") -> str:
    """Core chat-template generation. Returns the assistant text.

    If the final message has role=``assistant`` we treat it as a pre-fill and
    continue it (``continue_final_message=True``) rather than open a new turn,
    returning only the model's continuation -- mirroring the legacy completion
    contract. ``kind`` tags the call type in the prompt-pair log.
    """
    if self.use_thinking:
      # Scale the token budget so the <think> block doesn't crowd out the
      # answer, and never go greedy (the card warns it loops in thinking mode).
      max_new_tokens = max_new_tokens * _THINKING_MAX_NEW_TOKENS_MULT
      if temperature is None or temperature <= 0:
        temperature = _THINKING_TEMPERATURE
        top_p = _THINKING_TOP_P
        top_k = _THINKING_TOP_K
    log_params = {
      "max_new_tokens": max_new_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "top_k": top_k,
      "stop": stop,
      "thinking": self.use_thinking,
    }
    text = None
    try:
      self._ensure_model()
      import torch

      last_is_assistant = bool(messages) and messages[-1].get("role") == "assistant"
      text = self._tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not last_is_assistant,
        continue_final_message=last_is_assistant,
        enable_thinking=self.use_thinking,
      )
      inputs = self._tokenizer([text], return_tensors="pt").to(self._device)
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
        gen_kwargs["min_p"] = _DEFAULT_MIN_P

      with torch.inference_mode():
        outputs = self._model.generate(**inputs, **gen_kwargs)

      text_out = self._tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
      )
      # Strip the reasoning block from everything we return; only the log keeps
      # it (via the dedicated ``thinking`` field).
      text_out, thinking_out = _split_qwen_thinking(text_out)

      returned = _apply_stop_sequences(text_out, stop)
      response_record = {"raw": text_out, "returned": returned}
      if thinking_out:
        response_record["thinking"] = thinking_out
      prompt_log.log_call(
        harness="qwen", model=self.model_id, kind=kind,
        request={"messages": messages, "rendered_prompt": text},
        params=log_params,
        response=response_record,
      )
      return returned
    except Exception as e:
      prompt_log.log_call(
        harness="qwen", model=self.model_id, kind=kind,
        request={"messages": messages, "rendered_prompt": text},
        params=log_params,
        error=f"{type(e).__name__}: {e}",
      )
      raise

  # ============================================================ public API
  # Bare requests
  def chat_single_request(self, prompt: str) -> str:
    return self._generate(
      [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
      ],
      max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
      kind="chat_single",
    )

  def chat_request(self, prompt: str) -> str:
    try:
      return self._generate(
        [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt},
        ],
        max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
        kind="chat",
      )
    except Exception:
      traceback.print_exc()
      print("Qwen ERROR")
      return "Qwen ERROR"

  def chat_request_strong(self, prompt: str) -> str:
    # One model loaded; "strong" = same model with a higher token budget, to
    # mirror the legacy GPT-4 contract.
    try:
      return self._generate(
        [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt},
        ],
        max_new_tokens=512,
        kind="chat_strong",
      )
    except Exception:
      traceback.print_exc()
      print("Qwen ERROR")
      return "Qwen ERROR"

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
      print("QWEN PROMPT")
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
          kind="chat_json",
        ).strip()

        obj_str = _extract_first_json_object(raw)
        if obj_str is None:
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
      print("QWEN PROMPT")
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
    """Mimics the legacy completion-style ``GPT_request`` contract. Legacy
    prompts (authored for ``text-davinci-003``) often end mid-sentence; we
    split off the trailing partial line as an assistant pre-fill so the chat
    template continues it instead of opening a fresh interpretive turn."""
    try:
      temp = float(gpt_parameter.get("temperature", _DEFAULT_TEMPERATURE))
      top_p = float(gpt_parameter.get("top_p", _DEFAULT_TOP_P))
      max_new_tokens = int(gpt_parameter.get("max_tokens", _DEFAULT_MAX_NEW_TOKENS))
      stop = gpt_parameter.get("stop", None)

      user_content, pre_fill = _split_completion_prompt(prompt)
      messages = [
        {"role": "system",
         "content": "Continue the user's text directly. Be concise and follow the prompt's format exactly."},
        {"role": "user", "content": user_content},
      ]
      if pre_fill:
        messages.append({"role": "assistant", "content": pre_fill})

      return self._generate(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temp if temp > 0 else 0.0,
        top_p=top_p,
        top_k=_DEFAULT_TOP_K,
        stop=stop,
        kind="completion",
      )
    except Exception as e:
      print(f"[qwen] llm_request error ({type(e).__name__}): {e}")
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


def build(model_id: str, use_thinking: bool = False) -> _QwenHarness:
  """Return a fresh harness object for ``model_id``.

  ``use_thinking`` enables Qwen3's ``<think>`` reasoning block; the reasoning is
  logged but stripped from every returned value.
  """
  return _QwenHarness(model_id, use_thinking=use_thinking)
