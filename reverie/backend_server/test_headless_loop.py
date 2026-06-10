"""
Headless smoke test for the self-stepping backend loop.

Forks the base 3-persona sim into a throwaway sim, stubs out Persona.move
(no LLM calls), runs start_server for 5 steps WITHOUT any frontend running,
and asserts that the backend produced every per-step artifact on its own.
Cleans up the throwaway sim folder afterwards.

Run from reverie/backend_server (the documented backend CWD).
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, ".")

import reverie
from reverie_config import fs_storage, fs_temp_storage
from persona.persona import Persona

SIM = "_headless_smoke_test"
FORK = "base_the_ville_isabella_maria_klaus"

# Belt and braces: remove leftovers from a previous run.
if os.path.exists(f"{fs_storage}/{SIM}"):
    shutil.rmtree(f"{fs_storage}/{SIM}")

# ReverieServer.__init__ overwrites the frontend signaling files; snapshot
# them so this test leaves temp_storage exactly as it found it.
_signal_backup = {}
for fname in ("curr_sim_code.json", "curr_step.json"):
    fpath = f"{fs_temp_storage}/{fname}"
    _signal_backup[fpath] = open(fpath).read() if os.path.exists(fpath) else None


def stub_move(self, maze, personas, curr_tile, curr_time):
    # Shuffle one tile right and back, no cognition.
    x, y = curr_tile
    nxt = (x + 1, y) if x % 2 == 0 else (x - 1, y)
    return nxt, "\U0001f9ea", f"stub testing @ the_ville:test:arena:object"


Persona.move = stub_move

rs = reverie.ReverieServer(FORK, SIM)
reverie.rs = rs  # open_server normally sets this global

start_step = rs.step
assert start_step == 0, f"expected fork at step 0, got {start_step}"

t0 = time.time()
rs.start_server(5)
elapsed = time.time() - t0

sim_folder = f"{fs_storage}/{SIM}"
errors = []

if rs.step != 5:
    errors.append(f"rs.step == {rs.step}, expected 5")
if elapsed > 10:
    errors.append(f"5 steps took {elapsed:.1f}s -- loop is waiting on something")

for n in range(5):
    if not os.path.exists(f"{sim_folder}/movement/{n}.json"):
        errors.append(f"missing movement/{n}.json")
for n in range(6):
    if not os.path.exists(f"{sim_folder}/environment/{n}.json"):
        errors.append(f"missing environment/{n}.json")

# Movement targets of step N must equal positions in environment N+1.
for n in range(5):
    with open(f"{sim_folder}/movement/{n}.json") as f:
        mv = json.load(f)
    with open(f"{sim_folder}/environment/{n+1}.json") as f:
        env = json.load(f)
    for name, det in mv["persona"].items():
        got = (env[name]["x"], env[name]["y"])
        want = tuple(det["movement"])
        if got != want:
            errors.append(f"step {n}: {name} env {got} != movement {want}")
        if env[name]["maze"] != "the_ville":
            errors.append(f"step {n}: {name} maze field {env[name]['maze']!r}")

with open(f"{fs_temp_storage}/curr_step.json") as f:
    published = json.load(f)["step"]
if published != 5:
    errors.append(f"curr_step.json says {published}, expected 5")

shutil.rmtree(sim_folder)
for fpath, contents in _signal_backup.items():
    if contents is None:
        if os.path.exists(fpath):
            os.remove(fpath)
    else:
        with open(fpath, "w") as f:
            f.write(contents)

if errors:
    print("FAIL")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print(f"PASS: 5 headless steps in {elapsed:.2f}s, all files consistent")
