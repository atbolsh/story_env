"""
Memory-retrieval logging (JSONL + human-readable transcript).

This records every *cosine-similarity* memory retrieval -- i.e. every call to
``persona.cognitive_modules.retrieve.new_retrieve`` -- so that, for each
retrieval, we capture:

  * the **seed statement** (the "focal point" the memory stream was scored
    against -- the answer to "cosine similarity *to what?*"), and
  * the **cosine similarity that resulted** for each returned node (the raw
    ``cos_sim`` between the seed's embedding and the node's embedding), along
    with the recency / importance components and the final combined score that
    actually decided the ranking.

This is deliberately a *separate* log from the prompt-pair log
(``prompt_log``): the prompt log shows the statements *after* they have been
flattened into a prompt, but it does not show *why* those particular
statements were chosen (their similarity scores) or what they were compared
against. This log answers that.

Scope note: only the cosine path (``new_retrieve``) is logged. The other
retrieval path -- ``AssociativeMemory.retrieve_relevant_events`` /
``retrieve_relevant_thoughts``, used by the reaction prompts
(``decide_to_talk`` / ``decide_to_react``) -- is a keyword set-intersection,
not a cosine ranking, so it has no similarity score to record and is out of
scope here.

Path resolution (first match wins):

  1. ``REVERIE_MEMORY_RETRIEVAL_LOG``  -- explicit path, if set.
  2. derived from ``REVERIE_PROMPT_LOG`` -- same directory, filename
     ``memory_retrieval_log_<harness>_<stamp>.jsonl`` so the keyword, the LLM
     under test, and the date are all in the filename (matching the spirit of
     the other logs, which encode the same in their run directory name).
  3. otherwise disabled.

As with ``prompt_log``, each JSONL record is also appended in pretty,
banner-delimited form to a sibling ``.txt`` file, and any I/O failure
degrades to a one-time console warning rather than breaking the simulation.
"""

from __future__ import annotations

import datetime
import inspect
import json
import os
import threading
from typing import Any, Optional

_lock = threading.Lock()
_warned = False
_seq = 0

# Resolved once (lazily, on first write) so a run writes to a single file even
# though the harness/stamp are only knowable at runtime.
_resolved_path: Optional[str] = None
_run_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _harness_name() -> str:
  """Active harness name (e.g. ``"gemma4-e4b"``), best-effort."""
  try:
    from . import get_active_name
    return get_active_name() or "unknown"
  except Exception:
    return os.environ.get("REVERIE_HARNESS", "unknown").strip() or "unknown"


def _model_id() -> str:
  """Concrete model id of the active harness, best-effort (no weight load)."""
  try:
    from . import _active  # type: ignore
    if _active is not None:
      return str(getattr(_active, "engine_label", "")) or _harness_name()
  except Exception:
    pass
  return _harness_name()


def _derive_from_prompt_log(prompt_log_path: str) -> str:
  """Build the memory-retrieval log path next to the prompt-pair log, with the
  keyword + harness + stamp in the filename."""
  parent = os.path.dirname(prompt_log_path)
  fname = f"memory_retrieval_log_{_harness_name()}_{_run_stamp}.jsonl"
  return os.path.join(parent, fname) if parent else fname


def _log_path() -> str:
  global _resolved_path
  if _resolved_path is not None:
    return _resolved_path
  explicit = os.environ.get("REVERIE_MEMORY_RETRIEVAL_LOG", "").strip()
  if explicit:
    _resolved_path = explicit
    return _resolved_path
  prompt_log_path = os.environ.get("REVERIE_PROMPT_LOG", "").strip()
  if prompt_log_path:
    _resolved_path = _derive_from_prompt_log(prompt_log_path)
    return _resolved_path
  _resolved_path = ""
  return _resolved_path


def _text_log_path(jsonl_path: str) -> str:
  if jsonl_path.endswith(".jsonl"):
    return jsonl_path[:-len(".jsonl")] + ".txt"
  return jsonl_path + ".txt"


def enabled() -> bool:
  """True iff memory-retrieval logging is currently configured."""
  return bool(_log_path())


def _caller() -> str:
  """Identify the cognitive-module call site that triggered the retrieval,
  skipping frames internal to this module and ``new_retrieve`` itself."""
  try:
    for fr in inspect.stack()[1:]:
      fname = os.path.basename(fr.filename)
      if fname == "memory_retrieval_log.py":
        continue
      if fr.function == "new_retrieve":
        continue
      return f"{fr.function} ({fname}:{fr.lineno})"
  except Exception:
    pass
  return "?"


_BANNER = "=" * 78
_RULE_WIDTH = 78


def _rule(label: str) -> str:
  head = f"--- {label} "
  return head + "-" * max(0, _RULE_WIDTH - len(head))


def _format_record_text(record: dict, seq: int) -> str:
  """Render one retrieval record as a vim/less-friendly multi-line block."""
  w = record.get("weights") or {}
  formula = ""
  if w:
    gw = w.get("gw") or []
    formula = (f"score = recency_w({w.get('recency_w')})*recency_norm*{gw[0] if len(gw)>0 else '?'}"
               f" + relevance_w({w.get('relevance_w')})*relevance_norm*{gw[1] if len(gw)>1 else '?'}"
               f" + importance_w({w.get('importance_w')})*importance_norm*{gw[2] if len(gw)>2 else '?'}")
  lines = [
    _BANNER,
    f"retrieval #{seq}  {record['ts']}  {record['harness']}  "
    f"{record['model']}",
    f"caller: {record.get('caller', '?')}",
    f"seed (focal point): {json.dumps(record.get('focal_point', ''), ensure_ascii=False)}",
    f"requested: {record.get('n_requested')}   returned: {record.get('n_returned')}",
  ]
  if formula:
    lines.append(formula)
  lines.append(_rule("cos_raw = true cos-sim to seed; w_* are the weighted "
                     "contributions that sum to score"))
  results = record.get("results") or []
  if results:
    lines.append(f"{'rank':>4}  {'cos_raw':>7}  {'w_rec':>7}  {'w_rel':>7}  "
                 f"{'w_imp':>7}  {'score':>7}  statement")
    for r in results:
      def _f(x):
        try:
          return f"{float(x):.3f}"
        except Exception:
          return str(x)
      lines.append(
        f"{r.get('rank',''):>4}  {_f(r.get('cosine_raw')):>7}  "
        f"{_f(r.get('w_recency')):>7}  {_f(r.get('w_relevance')):>7}  "
        f"{_f(r.get('w_importance')):>7}  {_f(r.get('score')):>7}  "
        f"{str(r.get('statement',''))}"
      )
  else:
    lines.append("(no nodes returned)")
  lines.append("")
  return "\n".join(lines) + "\n"


def log_retrieval(*,
                  focal_point: str,
                  n_requested: int,
                  results: list,
                  weights: Optional[dict] = None,
                  caller: Optional[str] = None) -> None:
  """Append one memory-retrieval record (JSONL + pretty ``.txt`` sibling), if
  enabled.

  ``results`` is a list of dicts, in the final ranked order, each with keys:
  ``rank``, ``node_id``, ``type``, ``statement``, ``cosine_raw`` (true cosine
  similarity to the seed), ``relevance_norm`` / ``recency_norm`` /
  ``importance_norm`` (the min-max-normalized component values the ranking
  uses), ``recency_raw`` / ``importance_raw``, the weighted contributions
  ``w_recency`` / ``w_relevance`` / ``w_importance`` (which sum to ``score``),
  and ``score`` (the final ranking value).

  ``weights`` carries the ``recency_w`` / ``relevance_w`` / ``importance_w``
  persona weights and the ``gw`` global multipliers, so the score formula is
  self-documenting.
  """
  global _warned, _seq
  path = _log_path()
  if not path:
    return
  try:
    record = {
      "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
      "harness": _harness_name(),
      "model": _model_id(),
      "caller": caller if caller is not None else _caller(),
      "focal_point": focal_point,
      "n_requested": n_requested,
      "n_returned": len(results),
      "weights": weights,
      "results": results,
    }
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
      print(f"[memory_retrieval_log] disabled after write failure "
            f"({type(e).__name__}): {e}")
