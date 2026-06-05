"""
Rebuild every embedding inside a saved reverie simulation under a target
LLM harness's embedder.

Each persona's associative memory is persisted as a ``{description_string:
vector}`` dict in ``personas/<name>/bootstrap_memory/associative_memory/
embeddings.json`` (see ``persona/memory_structures/associative_memory.py``
lines 65, 149-150). This script walks every persona, re-encodes each
description under the target harness's ``get_embedding``, writes the new
mapping back, and updates ``reverie/meta.json``'s ``embedder`` field.

Use case: you built up a simulation under ``legacy-gpt`` (which uses
OpenAI's ``text-embedding-ada-002``, dim 1536) and now want to keep
running it under ``gemma4-e4b`` (which uses ``BAAI/bge-small-en-v1.5``,
dim 384). Without rebuilding, cosine similarity between the new query
embedding and the old stored embeddings is meaningless and retrieval
will misbehave.

Usage::

    cd reverie/backend_server
    python rebuild_embeddings.py --sim <sim_name> --to <harness_name> \\
        [--inplace | --out <new_sim_name>]

Default is non-destructive: copies the sim to ``<sim_name>__rebuilt`` (or
``--out <new_sim_name>``) and rebuilds there. ``--inplace`` rewrites the
original; you'll get a big warning first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make ``harnesses`` and ``reverie_config`` importable when this script is
# launched from ``reverie/backend_server``.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
  sys.path.insert(0, str(_THIS_DIR))

import harnesses  # noqa: E402
from global_methods import copyanything  # noqa: E402
from reverie_config import fs_storage  # noqa: E402


def _read_json(path: Path):
  with open(path) as f:
    return json.load(f)


def _write_json(path: Path, obj) -> None:
  with open(path, "w") as f:
    json.dump(obj, f)


def _rebuild_persona(persona_dir: Path, embed_fn) -> tuple[int, int, int]:
  """Re-embed one persona's ``embeddings.json``. Returns (count, old_dim, new_dim)."""
  emb_path = persona_dir / "bootstrap_memory" / "associative_memory" / "embeddings.json"
  if not emb_path.exists():
    return 0, 0, 0
  embeddings = _read_json(emb_path)
  if not embeddings:
    _write_json(emb_path, embeddings)
    return 0, 0, 0

  old_dim = 0
  for v in embeddings.values():
    if isinstance(v, list):
      old_dim = len(v)
      break

  new_embeddings: dict = {}
  new_dim = 0
  for description in embeddings.keys():
    vec = embed_fn(description)
    new_embeddings[description] = vec
    if not new_dim and isinstance(vec, list):
      new_dim = len(vec)

  _write_json(emb_path, new_embeddings)
  return len(new_embeddings), old_dim, new_dim


def rebuild(sim_name: str, target_harness: str,
            inplace: bool = False, out_sim: str | None = None) -> str:
  """Drive the rebuild. Returns the name of the sim that was modified."""
  src = Path(fs_storage) / sim_name
  if not src.exists():
    raise SystemExit(f"sim {sim_name!r} not found at {src}")

  meta_path = src / "reverie" / "meta.json"
  if not meta_path.exists():
    raise SystemExit(f"meta.json not found at {meta_path}")
  meta = _read_json(meta_path)
  current_embedder = meta.get("embedder", "unknown")

  print(f"[rebuild] sim:             {sim_name}")
  print(f"[rebuild] current embedder: {current_embedder}")
  print(f"[rebuild] target harness:   {target_harness}")

  # Resolve and PROBE the target harness *before* copying anything, so we fail
  # fast on stub harnesses (claude / latest-gpt) instead of producing a
  # half-rebuilt copy when the sim happens to have no embeddings.
  os.environ["REVERIE_HARNESS"] = target_harness
  harnesses.reset()
  try:
    active = harnesses.get_active()
  except NotImplementedError as e:
    raise SystemExit(f"target harness {target_harness!r} is not implemented: {e}")
  except Exception as e:
    raise SystemExit(f"failed to activate target harness {target_harness!r}: {e}")

  new_embedder_name = active.embedder_name
  print(f"[rebuild] new embedder:    {new_embedder_name}")

  try:
    probe = active.get_embedding("rebuild_embeddings probe")
    if not isinstance(probe, list) or not probe:
      raise RuntimeError(
        f"get_embedding returned unexpected value: {type(probe).__name__}"
      )
  except NotImplementedError as e:
    raise SystemExit(
      f"target harness {target_harness!r} does not implement get_embedding: {e}"
    )
  except Exception as e:
    raise SystemExit(
      f"target harness {target_harness!r} probe failed "
      f"({type(e).__name__}): {e}"
    )
  print(f"[rebuild] embedder probe OK (dim {len(probe)})")

  if inplace:
    print("[rebuild] mode: IN-PLACE (the original sim will be overwritten)")
    target_sim = sim_name
    target_dir = src
  else:
    target_sim = out_sim or f"{sim_name}__rebuilt"
    target_dir = Path(fs_storage) / target_sim
    if target_dir.exists():
      raise SystemExit(
        f"refusing to overwrite existing sim {target_sim!r} at {target_dir}; "
        f"delete it first or pick a different --out value"
      )
    print(f"[rebuild] mode: COPY -> {target_sim}")
    print(f"[rebuild] copying {src} -> {target_dir} ...")
    copyanything(str(src), str(target_dir))

  target_meta_path = target_dir / "reverie" / "meta.json"
  target_meta = _read_json(target_meta_path)
  persona_names = target_meta.get("persona_names", [])
  if not persona_names:
    print("[rebuild] WARNING: no persona_names in meta.json; nothing to do.")
    return target_sim

  embed_fn = active.get_embedding

  t0 = time.time()
  total_entries = 0
  for name in persona_names:
    persona_dir = target_dir / "personas" / name
    print(f"[rebuild] persona: {name}")
    count, old_dim, new_dim = _rebuild_persona(persona_dir, embed_fn)
    if count == 0:
      print(f"  no embeddings found (skipped)")
    else:
      print(f"  {count} entries re-embedded; dim {old_dim} -> {new_dim}")
    total_entries += count

  target_meta["embedder"] = new_embedder_name
  _write_json(target_meta_path, target_meta)

  elapsed = time.time() - t0
  print(f"[rebuild] DONE. {total_entries} total entries re-embedded "
        f"in {elapsed:.1f}s.")
  print(f"[rebuild] meta['embedder'] -> {new_embedder_name}")
  print(f"[rebuild] final sim:        {target_sim}")
  return target_sim


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(
    description=(
      "Rebuild a saved simulation's per-persona embeddings under a target "
      "LLM harness's embedder."
    )
  )
  ap.add_argument("--sim", required=True,
                  help="name of the simulation folder under fs_storage")
  ap.add_argument("--to", required=True, dest="to_harness",
                  help="target harness name (e.g. legacy-gpt, gemma4-e2b, "
                       "gemma4-e4b)")
  mx = ap.add_mutually_exclusive_group()
  mx.add_argument("--inplace", action="store_true",
                  help="overwrite the source sim in place (destructive!)")
  mx.add_argument("--out", default=None,
                  help="name for the rebuilt copy (default: <sim>__rebuilt)")
  return ap.parse_args()


def main() -> None:
  args = _parse_args()

  if args.inplace:
    print("=" * 72)
    print("WARNING: --inplace will overwrite the source sim's embeddings.json")
    print("files and meta.json. Make sure you have a backup if you care about")
    print("the original embeddings.")
    print("=" * 72)
    confirm = input("Type 'yes' to proceed: ").strip().lower()
    if confirm != "yes":
      raise SystemExit("aborted.")

  rebuild(
    sim_name=args.sim,
    target_harness=args.to_harness,
    inplace=args.inplace,
    out_sim=args.out,
  )


if __name__ == "__main__":
  main()
