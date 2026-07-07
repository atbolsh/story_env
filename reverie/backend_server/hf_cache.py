"""Auto-place the HuggingFace cache on a volume with enough free space.

On Vast.ai instances the image's default ``HF_HOME`` is typically
``/workspace/.hf_home`` -- but ``/workspace`` is a *small* persistent volume
(often 10G) that cannot hold a ~16G model like ``google/gemma-4-E4B-it``. The
download then fails mid-stream with ``Not enough free disk space`` and a
``Can't load the model`` OSError, even though other mounts (the container
overlay ``/`` or ``/dev/shm``) have plenty of room.

This module is imported very early -- before any ``huggingface_hub`` /
``transformers`` import -- so it can repoint ``HF_HOME`` / ``HF_HUB_CACHE``
in time for the env-var read that happens at HF import time.

Behaviour:

  * Gated on ``/workspace`` existing, so it only fires on Vast.ai-style boxes.
    On a normal dev machine (no ``/workspace``) it is a complete no-op.
  * If the currently-configured cache volume already has >= ``min_free_gb``
    free, it does nothing (the default is fine).
  * Otherwise it scans writable mounts reported by ``/proc/mounts``, prefers
    persistent (non-tmpfs) volumes over tmpfs, and picks the one with the
    most free space that meets ``min_free_gb``. It then sets ``HF_HOME`` and
    ``HF_HUB_CACHE`` to a ``hf_cache/`` subdir on that volume and prints a
    clear one-line banner.
  * If no writable volume has enough space it prints a loud warning and
    leaves the env untouched (the download will then fail naturally with the
    informative space error).

Note on persistence: the container overlay ``/`` and ``/dev/shm`` are NOT
persistent across Vast.ai instance restarts, so caching there means
re-downloading weights on each boot. ``/workspace`` IS persistent but on
small-workspace boxes it simply cannot fit the model -- there is no good
answer here other than resizing the workspace volume in the Vast.ai instance
config. This helper picks "works now" over "persists", which is the right
tradeoff for smoke testing; for long-running work, resize ``/workspace``.
"""
from __future__ import annotations

import os
import re
import shutil

# Headroom over the largest weights file we expect to cache (Gemma 4 E4B
# safetensors is ~16G). 20G gives comfortable slack for the blobs + temp
# download partials.
_MIN_FREE_GB_DEFAULT = 20

_done = False


def _free_gb(path: str) -> float:
  try:
    return shutil.disk_usage(path).free / (1024 ** 3)
  except Exception:
    return 0.0


def _current_cache_dir() -> str:
  if os.environ.get("HF_HUB_CACHE"):
    return os.environ["HF_HUB_CACHE"]
  if os.environ.get("HF_HOME"):
    return os.path.join(os.environ["HF_HOME"], "hub")
  return os.path.join(os.path.expanduser("~"), ".cache",
                      "huggingface", "hub")


def _decode_mount_point(mp: str) -> str:
  """``/proc/mounts`` octal-escapes spaces/tabs as ``\\040`` etc."""
  try:
    return re.sub(r"\\([0-7]{3})",
                  lambda m: chr(int(m.group(1), 8)), mp)
  except Exception:
    return mp


def _candidate_mounts() -> list:
  """Return ``[(mount_point, is_tmpfs, free_gb), ...]`` for writable mounts
  of real filesystems (skips proc/sysfs/devtmpfs/cgroup/etc.)."""
  pseudo = {"sysfs", "proc", "devtmpfs", "devpts", "cgroup", "cgroup2",
            "mqueue", "hugetlbfs", "fusectl", "binfmt_misc", "securityfs",
            "pstore", "debugfs", "tracefs", "configfs", "rpc_pipefs",
            "autofs", "bpf", "tracefs"}
  out = []
  try:
    with open("/proc/mounts") as f:
      for line in f:
        parts = line.split()
        if len(parts) < 3:
          continue
        _dev, mp_raw, fstype = parts[0], parts[1], parts[2]
        mp = _decode_mount_point(mp_raw)
        if not mp.startswith("/"):
          continue
        is_tmp = fstype in ("tmpfs", "ramfs", "shm")
        if fstype in pseudo and not is_tmp:
          continue
        if not os.path.isdir(mp):
          continue
        # Writable check: skip mounts we can't write to (e.g. ro bind mounts).
        if not os.access(mp, os.W_OK):
          continue
        out.append((mp, is_tmp, _free_gb(mp)))
  except Exception:
    pass
  return out


def configure_hf_cache_if_vast(min_free_gb: float = _MIN_FREE_GB_DEFAULT) -> None:
  """Repoint ``HF_HOME`` / ``HF_HUB_CACHE`` to a volume with enough space,
  but only on Vast.ai-style boxes (``/workspace`` exists) and only when the
  currently-configured cache volume is too small. Idempotent."""
  global _done
  if _done:
    return
  _done = True

  # Gate: only on Vast.ai-style instances.
  if not os.path.isdir("/workspace"):
    return

  cur = _current_cache_dir()
  cur_free = _free_gb(cur)
  if cur_free >= min_free_gb:
    # Default location has enough headroom; don't touch anything.
    return

  # Default volume too small -- find a better one.
  cands = _candidate_mounts()
  persistent = [(mp, f) for (mp, is_tmp, f) in cands
                if not is_tmp and f >= min_free_gb]
  volatile = [(mp, f) for (mp, is_tmp, f) in cands
              if is_tmp and f >= min_free_gb]
  persistent.sort(key=lambda x: x[1], reverse=True)
  volatile.sort(key=lambda x: x[1], reverse=True)
  ranked = persistent + volatile
  if not ranked:
    print(f"[hf_cache] WARNING: /workspace is present (Vast.ai) and the "
          f"current HF cache volume {cur!r} has only {cur_free:.1f}G free "
          f"(< {min_free_gb}G needed), but no other writable mount has "
          f"enough space either. The model download will likely fail. "
          f"Free up space or set HF_HOME manually.", flush=True)
    return

  choice, free = ranked[0]
  kind = "persistent" if not any(c[0] == choice and c[1] for c in cands) else "tmpfs (lost on restart)"
  new_home = os.path.join(choice, "hf_cache")
  os.environ["HF_HOME"] = new_home
  os.environ["HF_HUB_CACHE"] = os.path.join(new_home, "hub")
  print(f"[hf_cache] Vast.ai: default HF cache volume {cur!r} has only "
        f"{cur_free:.1f}G free (< {min_free_gb}G needed for the model "
        f"download). Repointing HF_HOME -> {new_home} on {choice} "
        f"({free:.1f}G free, {kind}).", flush=True)
