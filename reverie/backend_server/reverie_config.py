"""
Reverie backend configuration.

Replaces the old hand-rolled `utils.py` (which had the OpenAI key hardcoded
and lived under `.gitignore`). This module is committable: secrets come from
the repo-root `.env` file via python-dotenv, and only path/debug constants
are hardcoded.

On import, `.env` at the repo root is loaded into `os.environ` if it exists.
This is a no-op if the env vars are already set (e.g. by `run_remote_servers.sh`
which sources `.env` into each tmux window before launching the backend).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

# Secrets / per-user values: from .env (or the surrounding shell env).
openai_api_key = os.environ.get("OPENAI_API_KEY", "")

# Asset / storage paths -- relative to the backend_server CWD, since the
# README's documented invocation is `cd reverie/backend_server && python reverie.py`.
maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose
debug = True
