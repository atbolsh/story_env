"""
Local Gemma 4 harness.

Drives reverie's cognitive modules with a locally-hosted Gemma 4 model
(``E2B`` or ``E4B``, instruction-tuned variants) via the HuggingFace
Transformers library. Pairs generation with a small local embedding model
(``BAAI/bge-small-en-v1.5``) so retrieval works without any network.

Public surface matches ``harnesses/base.py``. The module exposes a
``build(model_id, use_thinking=False)`` factory which returns an object with
bound methods mirroring the harness interface; ``harnesses/__init__.py`` calls
``build("google/gemma-4-E2B-it")`` or ``build("google/gemma-4-E4B-it")``
depending on the selected harness name, passing ``use_thinking=True`` for the
``*-thinking`` harness variants.

Thinking mode: when ``use_thinking`` is set, the chat template turns on Gemma
4's reasoning channel (``enable_thinking=True``) and every call's token budget
is scaled by ``_THINKING_MAX_NEW_TOKENS_MULT``. The generated reasoning is
recorded in the prompt-pair log (as a dedicated ``thinking`` field) but is
*stripped from every value returned to the cognitive modules* -- so memories,
conversations, and JSON outputs behave exactly as if no reasoning channel
existed.

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

from . import prompt_log

# Recommended sampling for Gemma 4 (per the model card):
#   temperature=1.0, top_p=0.95, top_k=64.
# The same sampling is recommended in both thinking and non-thinking mode, so
# the thinking variants reuse these values (only the token budget changes).
# For JSON-output paths we drop temperature/top_p to be much more conservative.
_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_TOP_P = 0.95
_DEFAULT_TOP_K = 64
_JSON_TEMPERATURE = 0.4
_JSON_TOP_P = 0.9
_JSON_TOP_K = 40

_DEFAULT_MAX_NEW_TOKENS = 256

# With thinking enabled the model spends a chunk of its budget on the reasoning
# channel before the answer, so we scale every call's token budget up. Gemma 4's
# small variants (E2B/E4B in particular) "reason extensively by default" and will
# happily burn an entire modest budget on hidden reasoning, leaving *nothing* for
# the answer -- we observed exactly this: a thinking block truncated mid-sentence
# and an empty answer that then failed JSON parsing. The recommended fix when you
# can't pass an explicit max_thinking_tokens knob is a generous budget so the
# answer survives, so we use 4x (the model card's thinking examples use ~1024
# tokens vs the ~256-512 we use otherwise).
_THINKING_MAX_NEW_TOKENS_MULT = 4

# Gemma 4's thinking mode wraps its reasoning as:
#   <|channel>thought\n ...reasoning... <channel|> ...final answer...
# ``processor.parse_response`` normally splits this for us, but we also strip
# defensively (see ``_split_gemma_thinking``).
_GEMMA_THOUGHT_OPEN = "<|channel>thought"
_GEMMA_THOUGHT_CLOSE = "<channel|>"


def _split_gemma_thinking(text: str) -> tuple:
  """Return ``(answer, thinking_or_None)``.

  Gemma 4's thinking mode emits a ``<|channel>thought ... <channel|>`` span that
  precedes the final answer. ``processor.parse_response`` usually separates the
  two, but we strip defensively here too so a stray thought channel never leaks
  into memories, conversations, or JSON outputs. ``answer`` is the text with the
  thought span removed; ``thinking`` is the reasoning (or ``None`` if absent).
  """
  if not text:
    return text, None
  open_idx = text.find(_GEMMA_THOUGHT_OPEN)
  if open_idx == -1:
    return text, None
  close_idx = text.find(_GEMMA_THOUGHT_CLOSE, open_idx)
  if close_idx == -1:
    # Truncated thought block with no final answer: drop everything from the
    # tag on, keeping only any text that preceded it (normally empty).
    thinking = text[open_idx + len(_GEMMA_THOUGHT_OPEN):].strip()
    answer = text[:open_idx].strip()
    return answer, (thinking or None)
  thinking = text[open_idx + len(_GEMMA_THOUGHT_OPEN):close_idx].strip()
  answer = (text[:open_idx] + text[close_idx + len(_GEMMA_THOUGHT_CLOSE):]).strip()
  return answer, (thinking or None)


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


def _split_completion_prompt(prompt: str) -> tuple:
  """Detect legacy text-completion-style prompts and split off the trailing
  partial line for assistant pre-fill.

  Background: prompts in ``persona/prompt_template/v*/`` were authored for the
  ``text-davinci-003`` *completion* API and often end mid-sentence so the model
  literally continues the text, e.g.::

      ...(total duration in minutes 60):
      1) Isabella is

  Routing that through Gemma 4's chat template wraps it in a fresh user turn,
  and Gemma 4 then produces an interpretive *response* instead of strictly
  continuing the line -- a response that may drop the few-shot format. We saw
  this in practice on ``task_decomp_v3``: Gemma 4 emitted ``(06:00-06:05)``
  time ranges where the few-shot example clearly showed
  ``(duration in minutes: 5, minutes left: 55)``, which then crashed the
  downstream cleaner.

  The fix is to keep the bulk of the prompt as the user message but pre-fill
  the trailing partial line as the assistant's reply, then ask the chat
  template to *continue* that final message (transformers'
  ``continue_final_message=True``). Gemma 4 then sees the partial line as its
  own prior output and is far more likely to keep the legacy format.

  Heuristic: a prompt is "completion-style" iff its last non-empty line, after
  stripping whitespace, does *not* end in ``.``, ``!``, or ``?``. That catches
  the partial-list-item endings (``"1) Isabella is"``, ``"Action sequence: ["``,
  ``"... 1) wake up..., 2)"``) without grabbing fully-formed chat prompts
  (which all of the safe_chat_response_json paths are, via _safe_json's
  wrapper).

  Returns ``(user_content, assistant_pre_fill)``. If the prompt doesn't look
  completion-style, returns ``(prompt, "")``.
  """
  rstripped = prompt.rstrip()
  if not rstripped:
    return prompt, ""
  last_nl = rstripped.rfind("\n")
  if last_nl == -1:
    return prompt, ""
  last_line = rstripped[last_nl + 1:].strip()
  if not last_line:
    return prompt, ""
  if last_line[-1] in ".!?":
    return prompt, ""
  return rstripped[:last_nl], last_line


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

  def __init__(self, model_id: str, use_thinking: bool = False):
    self.model_id = model_id
    # When True, the chat template enables Gemma 4's reasoning channel. The
    # reasoning is logged (see prompt_log) but stripped from every value handed
    # back to the cognitive modules, so memories / conversations / JSON behave
    # exactly as if it were never generated.
    self.use_thinking = bool(use_thinking)
    # Label substituted into gpt_param["engine"] by run_gpt_prompt.py (purely
    # informational for this harness -- llm_request never reads "engine").
    self.engine_label = model_id
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
                stop=None,
                kind: str = "chat",
                enable_thinking: Optional[bool] = None) -> str:
    """Core chat-template generation. Returns the assistant text.

    If the final message in ``messages`` has role=``assistant``, we treat that
    as a pre-fill and ask the chat template to continue it rather than open a
    new model turn (``continue_final_message=True`` /
    ``add_generation_prompt=False``). The returned text is just the model's
    *continuation*, not the pre-fill (that mirrors the legacy completion
    contract, where the caller already has the prompt text and only wants the
    new tokens).

    ``kind`` tags the call type in the prompt-pair log (see ``prompt_log``);
    every call through here -- including each retry of the safe_* loops --
    produces one log record when ``REVERIE_PROMPT_LOG`` is set.
    """
    # ``enable_thinking`` lets a caller override the harness-level setting for a
    # single call (e.g. _safe_json disables thinking on its final retry); when
    # left as None we honor the harness default.
    use_thinking = self.use_thinking if enable_thinking is None else bool(enable_thinking)
    # With thinking on, scale the token budget so the reasoning channel doesn't
    # eat into the answer (the answer is what the caller actually consumes).
    if use_thinking:
      max_new_tokens = max_new_tokens * _THINKING_MAX_NEW_TOKENS_MULT
    log_params = {
      "max_new_tokens": max_new_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "top_k": top_k,
      "stop": stop,
      "thinking": use_thinking,
    }
    text = None
    try:
      self._ensure_model()
      import torch

      last_is_assistant = bool(messages) and messages[-1].get("role") == "assistant"
      text = self._processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not last_is_assistant,
        continue_final_message=last_is_assistant,
        enable_thinking=use_thinking,
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

      thinking_out = None
      if isinstance(parsed, dict):
        text_out = parsed.get("content") or parsed.get("text") or ""
        thinking_out = parsed.get("thinking")
      else:
        text_out = str(parsed)

      # Defensive split: guarantee no thought channel survives into the text we
      # return, even if parse_response didn't separate it. The reasoning is only
      # ever surfaced via the log's ``thinking`` field, never to the caller.
      text_out, leaked_thinking = _split_gemma_thinking(text_out)
      if leaked_thinking and not thinking_out:
        thinking_out = leaked_thinking

      returned = _apply_stop_sequences(text_out, stop)
      response_record = {"raw": text_out, "returned": returned}
      if thinking_out:
        response_record["thinking"] = thinking_out
      prompt_log.log_call(
        harness="gemma4", model=self.model_id, kind=kind,
        request={"messages": messages, "rendered_prompt": text},
        params=log_params,
        response=response_record,
      )
      return returned
    except Exception as e:
      prompt_log.log_call(
        harness="gemma4", model=self.model_id, kind=kind,
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
        kind="chat_strong",
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
    if self.use_thinking:
      # Gemma 4's small variants over-reason and can spend the whole budget on
      # the hidden thought channel without ever emitting the answer. Per Google's
      # thinking-mode guidance, when you can't pass an explicit max_thinking_tokens
      # the recommended lever is an in-prompt brevity instruction; we also tell it
      # not to second-guess, since the failure we saw was a self-correction loop
      # ("Wait... Actually... let me reconsider...") that never reached an answer.
      system_prompt += (
        " Think briefly: keep your reasoning to a few sentences at most, do not "
        "restate the prompt or second-guess yourself, then commit and output the "
        "JSON object."
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
        # Thinking-off-on-the-final-retry (intentional, slightly hacky): with
        # thinking enabled these models sometimes burn the entire token budget
        # reasoning and return an empty/garbled answer that won't parse. The same
        # models answer structured JSON cleanly with thinking *off* (the
        # non-thinking harnesses completed full days without this failure), so as
        # a last-ditch effort -- after spending the earlier attempts with thinking
        # on -- we disable it for the final attempt to maximize the chance of a
        # usable answer before giving up. ``None`` keeps the harness default.
        attempt_thinking = False if (self.use_thinking and i == repeat - 1) else None
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
          enable_thinking=attempt_thinking,
        ).strip()

        obj_str = _extract_first_json_object(raw)
        if obj_str is None:
          # Fall back to the historical "trim to last brace" heuristic.
          end_index = raw.rfind('}') + 1
          obj_str = raw[:end_index] if end_index > 0 else raw

        parsed_obj = json.loads(obj_str)
        curr_response = parsed_obj["output"]
        # The legacy validators/cleaners (written for OpenAI, which always
        # returned text) call str methods like ``.strip()``. A model that emits
        # e.g. {"output": 1} would hand them an int and raise inside validation,
        # silently failing every retry. Coerce non-strings back to the string
        # contract so {"output": 1} behaves like {"output": "1"}.
        if not isinstance(curr_response, str):
          curr_response = str(curr_response)

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
    """Mimics the legacy completion-style ``GPT_request`` contract.

    Legacy prompts were authored for ``text-davinci-003`` (pure completion),
    so many of them end mid-sentence -- e.g. ``"1) Isabella is"`` -- expecting
    the model to literally continue the text. When fed straight into Gemma 4's
    chat template, the model sees a fresh user turn instead and produces an
    interpretive reply that often drops the few-shot format. We split such
    prompts via ``_split_completion_prompt`` so the trailing partial line
    becomes an assistant pre-fill; ``_generate`` then asks the chat template
    to continue that message.
    """
    try:
      # The legacy code treats temperature=0 as deterministic. With Gemma 4
      # we approximate that by disabling sampling.
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


def build(model_id: str, use_thinking: bool = False) -> _Gemma4Harness:
  """Return a fresh harness object for ``model_id``.

  ``use_thinking`` enables Gemma 4's reasoning channel; the reasoning is logged
  but stripped from every returned value.
  """
  return _Gemma4Harness(model_id, use_thinking=use_thinking)
