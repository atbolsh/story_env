"""
Exact prompt/response pair logging (JSONL + human-readable transcript).

When the ``REVERIE_PROMPT_LOG`` environment variable is set to a file path,
every model call made by a harness appends one JSON object per line to that
file, capturing the *exact* request (post-wrapping, post-chat-template) and
the raw response. This includes every retry inside the ``safe_*`` loops --
which the ``print_run_prompts`` console blocks do not show -- so it is the
authoritative record for post-session debugging.

Because one-record-per-line JSONL is unreadable in an editor (a single
record can be several hundred KB of escaped text on one line), each record
is *also* appended in a pretty, multi-line form to a sibling transcript
file: same path with the ``.jsonl`` suffix replaced by ``.txt``. The JSONL
stays the machine-readable source of truth; the ``.txt`` is for humans
(vim/less-friendly, one banner-delimited block per call, in the same order
as the JSONL records).

Record schema (fields are null/absent where not applicable):

  ts        ISO-8601 timestamp with offset
  harness   harness identifier (e.g. ``"gemma4"``, ``"legacy-gpt"``)
  model     concrete model id (e.g. ``"google/gemma-4-E2B-it"``,
            ``"gpt-3.5-turbo"``)
  kind      call type: ``"chat"`` | ``"chat_strong"`` | ``"chat_single"`` |
            ``"chat_json"`` | ``"completion"``
  request   exact request payload. Always has ``"messages"``; the gemma4
            harness also includes ``"rendered_prompt"``, the full
            chat-templated string fed to the tokenizer (byte-exact model
            input, special tokens included).
  params    sampling parameters used for the call
  response  ``{"raw": ..., "returned": ...}`` -- raw model text before any
            stop-sequence trimming, and the text actually returned to the
            caller (both *before* the caller's validate/clean-up step).
            Thinking-mode harnesses add a ``"thinking"`` field with the model's
            reasoning; that reasoning is stripped from ``raw``/``returned`` and
            is never fed back into memories, conversations, or prompts.
  error     ``"ExcType: message"`` if the call raised; the record is still
            written so failed calls are visible in the log

Logging must never break a simulation: any I/O or serialization failure
degrades to a one-time console warning and the call proceeds normally.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
from typing import Any, Optional

_lock = threading.Lock()
_warned = False
_seq = 0


def _log_path() -> str:
  return os.environ.get("REVERIE_PROMPT_LOG", "").strip()


def _text_log_path(jsonl_path: str) -> str:
  """Sibling human-readable transcript path for a given JSONL path."""
  if jsonl_path.endswith(".jsonl"):
    return jsonl_path[:-len(".jsonl")] + ".txt"
  return jsonl_path + ".txt"


def enabled() -> bool:
  """True iff prompt-pair logging is currently configured."""
  return bool(_log_path())


_BANNER = "=" * 78
_RULE_WIDTH = 78


def _rule(label: str) -> str:
  """A section divider like ``--- user ----------------``."""
  head = f"--- {label} "
  return head + "-" * max(0, _RULE_WIDTH - len(head))


def _format_record_text(record: dict, seq: int) -> str:
  """Render one log record as a vim/less-friendly multi-line block."""
  lines = [
    _BANNER,
    f"call #{seq}  {record['ts']}  {record['harness']}  "
    f"{record['model']}  [{record['kind']}]",
  ]
  if record.get("params") is not None:
    lines.append("params: " + json.dumps(record["params"], ensure_ascii=False,
                                         default=str))
  lines.append("")

  request = record.get("request")
  messages = None
  if isinstance(request, dict):
    messages = request.get("messages")
  if isinstance(messages, list):
    for i, msg in enumerate(messages):
      if not isinstance(msg, dict):
        lines += [_rule("message"), str(msg)]
        continue
      role = str(msg.get("role", "?"))
      # A trailing assistant message is a pre-fill the model continues, not
      # a reply; label it so transcripts read unambiguously.
      if role == "assistant" and i == len(messages) - 1:
        role = "assistant (pre-fill)"
      lines += [_rule(role), str(msg.get("content", ""))]
  elif request is not None:
    lines += [_rule("request"),
              json.dumps(request, ensure_ascii=False, indent=2, default=str)]

  response = record.get("response")
  if isinstance(response, dict):
    returned = response.get("returned")
    raw = response.get("raw")
    thinking = response.get("thinking")
    # Reasoning (thinking-mode harnesses only). Shown for the record but it is
    # stripped from ``returned``/``raw`` and never reaches memories or prompts.
    if thinking:
      lines += [_rule("thinking (stripped from output)"), str(thinking)]
    lines += [_rule("response"), str(returned)]
    if raw != returned:
      lines += [_rule("raw response (before stop-trim)"), str(raw)]
  elif response is not None:
    lines += [_rule("response"),
              json.dumps(response, ensure_ascii=False, indent=2, default=str)]

  if record.get("error") is not None:
    lines += [_rule("ERROR"), str(record["error"])]

  lines.append("")
  return "\n".join(lines) + "\n"


def log_call(*,
             harness: str,
             model: str,
             kind: str,
             request: Any,
             params: Optional[dict] = None,
             response: Any = None,
             error: Optional[str] = None) -> None:
  """Append one prompt/response record to the JSONL log (and its pretty
  ``.txt`` sibling), if enabled."""
  global _warned, _seq
  path = _log_path()
  if not path:
    return
  try:
    record = {
      "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
      "harness": harness,
      "model": model,
      "kind": kind,
      "request": request,
      "params": params,
      "response": response,
    }
    if error is not None:
      record["error"] = error
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock:
      _seq += 1
      seq = _seq
      parent = os.path.dirname(path)
      if parent:
        os.makedirs(parent, exist_ok=True)
      with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
      with open(_text_log_path(path), "a", encoding="utf-8") as f:
        f.write(_format_record_text(record, seq))
  except Exception as e:
    if not _warned:
      _warned = True
      print(f"[prompt_log] disabled after write failure "
            f"({type(e).__name__}): {e}")
