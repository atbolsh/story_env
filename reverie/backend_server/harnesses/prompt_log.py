"""
Exact prompt/response pair logging (JSONL).

When the ``REVERIE_PROMPT_LOG`` environment variable is set to a file path,
every model call made by a harness appends one JSON object per line to that
file, capturing the *exact* request (post-wrapping, post-chat-template) and
the raw response. This includes every retry inside the ``safe_*`` loops --
which the ``print_run_prompts`` console blocks do not show -- so it is the
authoritative record for post-session debugging.

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
            caller (both *before* the caller's validate/clean-up step)
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


def _log_path() -> str:
  return os.environ.get("REVERIE_PROMPT_LOG", "").strip()


def enabled() -> bool:
  """True iff prompt-pair logging is currently configured."""
  return bool(_log_path())


def log_call(*,
             harness: str,
             model: str,
             kind: str,
             request: Any,
             params: Optional[dict] = None,
             response: Any = None,
             error: Optional[str] = None) -> None:
  """Append one prompt/response record to the JSONL log, if enabled."""
  global _warned
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
      parent = os.path.dirname(path)
      if parent:
        os.makedirs(parent, exist_ok=True)
      with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
  except Exception as e:
    if not _warned:
      _warned = True
      print(f"[prompt_log] disabled after write failure "
            f"({type(e).__name__}): {e}")
