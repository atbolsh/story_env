

# Generative Agents: Interactive Simulacra of Human Behavior 

<p align="center" width="100%">
<img src="cover.png" alt="Smallville" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

This repository accompanies our research paper titled "[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)." It contains our core simulation module for  generative agents—computational agents that simulate believable human behaviors—and their game environment. Below, we document the steps for setting up the simulation environment on your local machine and for replaying the simulation as a demo animation.

## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Isabella_Rodriguez.png" alt="Generative Isabella">   Setting Up the Environment 
You need to (1) install Python dependencies and (2) provide an OpenAI API key (and any other secrets) via a `.env` file at the repo root.

### Step 1. Install requirements.txt
Install everything listed in `requirements.txt` (I strongly recommend a virtualenv first). The codebase targets Python 3.12 with Django 5.2 LTS and a trimmed dependency set:

    pip install -r requirements.txt

### Step 2. Create a `.env` file at the repo root
The backend reads its OpenAI API key (and any future model/API secrets) from a `.env` file located next to this README. `.env` is gitignored, so it stays local to your checkout. Minimum contents:

```
OPENAI_API_KEY=sk-...your-key-here...
```

`reverie/backend_server/reverie_config.py` loads this file via `python-dotenv` at import time, exposing `openai_api_key`, the asset paths, and the `debug` flag to the rest of the backend. You no longer need to create a hand-rolled `utils.py`.

## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Klaus_Mueller.png" alt="Generative Klaus">   Running a Simulation 
To run a new simulation, you will need to concurrently start two servers: the environment server and the agent simulation server.

### Step 1. Starting the Environment Server
Again, the environment is implemented as a Django project, and as such, you will need to start the Django server. To do this, first navigate to `environment/frontend_server` (this is where `manage.py` is located) in your command line. Then run the following command:

    python manage.py runserver

Then, on your favorite browser, go to [http://localhost:8000/](http://localhost:8000/). If you see a message that says, "Your environment server is up and running," your server is running properly. Ensure that the environment server continues to run while you are running the simulation, so keep this command-line tab open! (Note: I recommend using either Chrome or Safari. Firefox might produce some frontend glitches, although it should not interfere with the actual simulation.)

### Step 2. Starting the Simulation Server
Open up another command line (the one you used in Step 1 should still be running the environment server, so leave that as it is). Navigate to `reverie/backend_server` and run `reverie.py`.

    python reverie.py
This will start the simulation server. It first prints the list of available **model harnesses** and asks "Select model harness [legacy-gpt]: ". Type the name of the backend you want to drive the agents with (press Enter to accept the default). The choices are:

| Harness | Backend |
| --- | --- |
| `legacy-gpt` (default) | OpenAI GPT-3.5 / GPT-4 via the `openai==0.28` SDK |
| `gemma4-e2b` / `gemma4-e4b` | Local Gemma 4 E2B / E4B via transformers |
| `gemma4-e2b-thinking` / `gemma4-e4b-thinking` | Same Gemma 4 models with **thinking mode** on |
| `qwen3-0.6b` | Local Qwen3-0.6B (non-thinking) via transformers |
| `qwen3-0.6b-thinking` | Qwen3-0.6B with **thinking mode** on |
| `gemma4-e4b-nams` / `gemma4-e4b-nams-thinking` | Local Gemma 4 E4B + **NAMS** graph memory on a local Neo4j (no API keys) |
| `latest-gpt-nams` | Modern OpenAI chat models (`openai>=1.0`) + **NAMS** graph memory on a local Neo4j |

The `*-thinking` variants are just convenience aliases for the same model with its reasoning channel enabled: the model "thinks" before answering, and that reasoning is written to the prompt-pair logs (the `thinking` field) but is **stripped from everything else** — it never enters an agent's memory, a conversation, or any JSON the cognitive modules parse. Thinking runs also raise each call's token budget (2x) to leave room for the reasoning. Pick one the same way you'd pick any other harness; there's no separate flag to set.

#### NAMS-backed harnesses (`gemma4-e4b-nams`, `latest-gpt-nams`)

These harnesses replace the JSON memory stream with a **Neo4j Agent Memory System (NAMS)** graph running on a **local Neo4j database** — no Neo4j Aura / NAMS hosted API keys involved. The per-character long-term memory (identity, events, thoughts, plans, schedule, relationships) lives as POLE+O entities + temporal Facts in the graph; only `scratch.json` (transient state) and `spatial_memory.json` (world layout) stay on disk. See `reverie/backend_server/harnesses/nams/` for the implementation.

**One-time local DB setup:**

There are two supported layouts depending on your host:

- **Docker host** (a box with a running Docker daemon): use the included compose file (bolt on `7687`, HTTP browser on `7474`):

       docker compose up -d neo4j

  The compose file sets the password from the `NEO4J_PASSWORD` env var (default `password`). Data persists in a named volume, so a sim's memory graph survives `docker compose down` + `up`. Use `docker compose down -v` to wipe the database (e.g. before re-importing a forked JSON bootstrap into a clean graph).

- **Bare-metal / no-Docker host** (e.g. a rented GPU container on Vast.ai where the Docker daemon isn't available but Neo4j is installed — or can be apt-installed — directly): use the idempotent launch script:

       bash scripts/vast_neo4j_baremetal_launch.sh

  This installs Neo4j Community + OpenJDK 17 via apt if missing, writes the NAMS config block into `/etc/neo4j/neo4j.conf`, writes `/etc/neo4j/apoc.conf` (v5 requires APOC settings in their own file), downloads the matching APOC core jar, sets the initial password to `password`, starts Neo4j, and verifies bolt + APOC. Safe to re-run on every fresh box. The harness then connects with the same defaults (`bolt://localhost:7687`, user `neo4j`, password `password`) — no `.env` change needed.

2. Tell the harness how to reach the database (defaults shown):

       export NEO4J_URI="bolt://localhost:7687"
       export NEO4J_USER="neo4j"
       export NEO4J_PASSWORD="password"

   You can put these in your `.env` next to this README (alongside `OPENAI_API_KEY`); `reverie_config.py` already loads `.env` into the environment at import time.

3. Install the NAMS extras (already in `requirements.txt`). The `neo4j-agent-memory` SDK requires **Python 3.10+** (the rest of this repo targets 3.12), so use a 3.10+ interpreter for the `*-nams` harnesses:

       pip install -r requirements.txt
       python -m spacy download en_core_web_sm

   The package ships as `neo4j-agent-memory` on PyPI (https://pypi.org/project/neo4j-agent-memory/). The `[sentence-transformers]` extra wires the local BGE embedder used by `gemma4-e4b-nams`; the `[openai]` extra wires OpenAI embeddings used by `latest-gpt-nams`; `[spacy]` + `[gliner]` drive the NERS entity/relation extraction pipeline that ages short-term messages into long-term POLE+O facts. `glirel` (relation extraction) is pulled in separately.

**Per-character isolation.** Each character's semantic model must be his own — never mutually visible to another character. Neo4j **Community Edition** (the free GPLv3 image in `docker-compose.yml`) supports only one database per instance, so isolation is achieved at the *instance* level:

- **When at most ONE persona in a sim is on NAMS** (e.g. `midnight_test.sh` with only Klaus Mueller on NAMS): the single `docker compose up -d neo4j` instance is enough. That one persona's graph lives in it and isolation is trivially satisfied. No extra setup. This is the default path.

- **When TWO OR MORE personas in the same sim are on NAMS:** launch one dedicated Community container per persona with `scripts/nams_db.sh`. Each gets its own container, its own data volume, and its own bolt port; their memory graphs are physically separate. The script writes a registry (`nams_databases.json`) that the harness reads at startup to route each persona to its own instance. This keeps the whole stack on the free GPLv3 Community Edition — no Neo4j Enterprise license flag, no account.

       # Launch dedicated containers for two NAMS personas
       scripts/nams_db.sh up "Klaus Mueller" "Isabella Rodriguez"

       # See what's running + dump files on disk
       scripts/nams_db.sh list

       # When the run is over: save each persona, then stop its container,
       # then spin the server down yourself.
       scripts/nams_db.sh teardown "Klaus Mueller"
       scripts/nams_db.sh teardown "Isabella Rodriguez"

       # Later, on any machine with the .dump files + this repo:
       scripts/nams_db.sh up "Klaus Mueller" "Isabella Rodriguez"
       scripts/nams_db.sh load "Klaus Mueller" nams_dumps/klaus_mueller__<timestamp>.dump
       scripts/nams_db.sh load "Isabella Rodriguez" nams_dumps/isabella_rodriguez__<timestamp>.dump

**Save-file format.** `scripts/nams_db.sh save`/`teardown` use Neo4j's native `neo4j-admin database dump` output — a binary archive of the store files, one `.dump` file per persona (`nams_dumps/<sanitized>__<UTC-timestamp>.dump`). It is not human-readable; it is only meaningful to `neo4j-admin database load` on the same (or newer) Neo4j major version. Restored with `scripts/nams_db.sh load <persona> <dump_file>`. `save` is non-destructive (the persona's container is restarted after the dump); `teardown` saves + stops the container (and with `--purge` also drops the volume + registry entry), leaving the server clean for you to spin down.

`scripts/nams_db.sh --help` prints the full subcommand reference. The single-instance compose path and the per-persona-container path coexist: the harness falls back to `bolt://localhost:7687` for any persona not in the registry, so a sim mixing one NAMS persona (in the compose instance) with JSON-backed personas just works without touching `nams_db.sh`.

**Bare-metal save / wipe / load (no Docker).** On hosts running Neo4j directly (the `vast_neo4j_baremetal_launch.sh` path above), the equivalent tool is `scripts/nams_baremetal_db.sh`. When only one persona is on NAMS, the entire `neo4j` database IS that persona's memory graph, so "save Klaus's memories" = "dump the neo4j database":

       # Download Klaus's memories for offline analysis (e.g. tomorrow morning)
       bash scripts/nams_baremetal_db.sh save logs/klaus_run1.dump

       # Wipe the DB clean between runs (in addition to --force-import)
       bash scripts/nams_baremetal_db.sh wipe

       # Reinstate a previously saved memory graph
       bash scripts/nams_baremetal_db.sh load logs/klaus_run1.dump

       # Show running state + bolt + node counts by label
       bash scripts/nams_baremetal_db.sh status

Same `.dump` format as the Docker path (Neo4j native `neo4j-admin database dump` binary archive). `midnight_test.sh` calls `save` after each run and `wipe` between runs automatically on bare-metal hosts, dropping each run's `klaus_memories_<run>.dump` into that run's `logs/midnight_<run>_<stamp>/` dir.

**First run against a JSON-forked sim.** When you fork a sim that was built up under a legacy JSON harness (e.g. `base_the_ville_isabella_maria_klaus`), the first `*-nams` run automatically imports each persona's `bootstrap_memory/` (spatial, identity, schedule, `associative_memory/nodes.json`) into the graph as POLE+O entities + Facts. The import is gated by a `graph_exists` check, so re-runs against an already-imported graph are no-ops. To force a fresh re-import, wipe the database (`docker compose down -v && docker compose up -d neo4j` for the single instance, `scripts/nams_db.sh down "Klaus Mueller" --purge` for a per-persona container, or `scripts/nams_baremetal_db.sh wipe` on a bare-metal host).

**Extraction mode prompt.** After selecting a `*-nams` harness, reverie asks:

    NAMS extraction mode:
      [A] no-llm      -- spaCy + GLiNER + GLiREL only (deterministic)
      [C] harness-llm -- + harness chat LLM for raw-message extraction
    Select extraction mode [A]:

Mode **A** runs the NAMS extractor pipeline without any LLM stage (air-gapped, deterministic) — the harness LLM is still used by the cognitive modules directly (reflection insights, conversation summaries, poignancy scoring) via explicit `add_fact` calls. Mode **C** additionally wires the harness chat LLM as the NAMS extractor's LLM stage so raw short-term text is also LLM-summarized into facts. Both modes keep the same cognitive-module behavior; the difference is only whether NAMS's *internal* extractor gets an LLM stage.

Browse any character's graph at `http://localhost:7474` (user `neo4j`, password from `NEO4J_PASSWORD`). Useful Cypher for a quick sanity check:

    MATCH (f:Fact) WHERE f.metadata CONTAINS '"kind": "plan"' RETURN f LIMIT 5;
    MATCH (a:Entity {type:'PERSON'})-[r:TALKED_WITH]->(b:Entity) RETURN a,r,b LIMIT 5;
    MATCH (t:ReasoningTrace) RETURN t.task, t.outcome LIMIT 5;

After the harness, a prompt will appear asking the following: "Enter the name of the forked simulation: ". To start a 3-agent simulation with Isabella Rodriguez, Maria Lopez, and Klaus Mueller, type the following:
    
    base_the_ville_isabella_maria_klaus
The prompt will then ask, "Enter the name of the new simulation: ". Type any name to denote your current simulation (e.g., just "test-simulation" will do for now).

    test-simulation
Keep the simulator server running. At this stage, it will display the following prompt: "Enter option: "

### Step 3. Running and Saving the Simulation
To run the simulation, type the following command in your simulation server in response to the prompt, "Enter option":

    run <step-count>
Note that you will want to replace `<step-count>` above with an integer indicating the number of game steps you want to simulate. For instance, if you want to simulate 100 game steps, you should input `run 100`. One game step represents 10 seconds in the game.

To watch the simulation, navigate to [http://localhost:8000/simulator_home](http://localhost:8000/simulator_home) in your browser. You should see the map of Smallville, along with a list of active agents on the map. You can move around the map using your keyboard arrows. The browser is a *pure observer*: the simulation advances on its own whether or not the page is open, and you can close, refresh, or background the tab at any time -- it simply re-attaches at the live step. (This differs from the original release, where the backend would not advance unless a browser tab was open and focused.)

Once the simulation finishes running, the "Enter option" prompt will re-appear. At this point, you can simulate more steps by re-entering the run command with your desired game steps, exit the simulation without saving by typing `exit`, or save and exit by typing `fin`.

The saved simulation can be accessed the next time you run the simulation server by providing the name of your simulation as the forked simulation. This will allow you to restart your simulation from the point where you left off.

### Step 4. Replaying a Simulation
You can replay a simulation that you have already run simply by having your environment server running and navigating to the following address in your browser: `http://localhost:8000/replay/<simulation-name>/<starting-time-step>`. Please make sure to replace `<simulation-name>` with the name of the simulation you want to replay, and `<starting-time-step>` with the integer time-step from which you wish to start the replay.

For instance, by visiting the following link, you will initiate a pre-simulated example, starting at time-step 1:  
[http://localhost:8000/replay/July1_the_ville_isabella_maria_klaus-step-3-20/1/](http://localhost:8000/replay/July1_the_ville_isabella_maria_klaus-step-3-20/1/)

**Simulation names containing a dot (`.`).** Some harness names contain a dot
(e.g. `qwen3-0.6b`), so sims saved by `midnight_test.sh` get names like
`midnight_qwen3-0.6b_2026-06-17_18-07-25`. The `replay` and `demo` routes accept
dots in the simulation name, so you can paste the name verbatim — just be sure to
keep the **trailing slash**, which the routes require:  
`http://localhost:8000/replay/midnight_qwen3-0.6b_2026-06-17_18-07-25/1/`

### Step 5. Demoing a Simulation
You may have noticed that all character sprites in the replay look identical. We would like to clarify that the replay function is primarily intended for debugging purposes and does not prioritize optimizing the size of the simulation folder or the visuals. To properly demonstrate a simulation with appropriate character sprites, you will need to compress the simulation first. To do this, open the `compress_sim_storage.py` file located in the `reverie` directory using a text editor. Then, execute the `compress` function with the name of the target simulation as its input. By doing so, the simulation file will be compressed, making it ready for demonstration.

To start the demo, go to the following address on your browser: `http://localhost:8000/demo/<simulation-name>/<starting-time-step>/<simulation-speed>`. Note that `<simulation-name>` and `<starting-time-step>` denote the same things as mentioned above. `<simulation-speed>` can be set to control the demo speed, where 1 is the slowest, and 5 is the fastest. For instance, visiting the following link will start a pre-simulated example, beginning at time-step 1, with a medium demo speed:  
[http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/](http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/)

### Tips
We've noticed that OpenAI's API can hang when it reaches the hourly rate limit. When this happens, you may need to restart your simulation. For now, we recommend saving your simulation often as you progress to ensure that you lose as little of the simulation as possible when you do need to stop and rerun it. Running these simulations, at least as of early 2023, could be somewhat costly, especially when there are many agents in the environment.

### Running on a Remote GPU Host (e.g. vast.ai)
The reverie backend and the Django frontend communicate **only via the shared filesystem** under `environment/frontend_server/{storage,temp_storage}`. That means both servers have to live on the same host — but only the browser-facing port (8000) needs to be reachable from your laptop. SSH port forwarding handles the rest.

In all of the examples below, replace `PORT`, `USER`, and `HOST` with the values from the ssh command your GPU provider gave you (for vast.ai, that's the `Direct ssh connect` button).

1. On the remote box, clone this repo, install `requirements.txt`, and drop your `.env` (API keys) next to `run_remote_servers.sh`. To push the env file from your laptop, you can use the included helper:

       ./ssh2scp.sh "ssh -p PORT USER@HOST -L 8080:localhost:8080" \
                    .env /root/generative_agents/.env
       # then run the printed scp command

2. Start both servers on the remote inside a tmux session:

       ./run_remote_servers.sh
       tmux attach -t gen_agents
       # Ctrl-b n / Ctrl-b p to switch between the frontend and backend windows
       # Ctrl-b d to detach (servers keep running)

3. From your laptop, add the frontend port to your existing ssh command using the helper:

       ./ssh2tunnel.sh "ssh -p PORT USER@HOST -L 8080:localhost:8080"
       # -> ssh -p PORT USER@HOST -L 8080:localhost:8080 -L 8000:localhost:8000

   Run that command (or `eval "$(./ssh2tunnel.sh '...')"`). While the tunnel is up, the remote Django server is reachable from your local browser at [http://localhost:8000/simulator_home](http://localhost:8000/simulator_home), exactly as in the local setup above. Drive the simulation by typing `run N`, `save`, `fin`, etc. into the `backend` window of the remote tmux session.

## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Maria_Lopez.png" alt="Generative Maria">   Simulation Storage Location
All simulations that you save will be located in `environment/frontend_server/storage`, and all compressed demos will be located in `environment/frontend_server/compressed_storage`. 

## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Sam_Moore.png" alt="Generative Sam">   Customization

There are two ways to optionally customize your simulations. 

### Author and Load Agent History
First is to initialize agents with unique history at the start of the simulation. To do this, you would want to 1) start your simulation using one of the base simulations, and 2) author and load agent history. More specifically, here are the steps:

#### Step 1. Starting Up a Base Simulation 
There are two base simulations included in the repository: `base_the_ville_n25` with 25 agents, and `base_the_ville_isabella_maria_klaus` with 3 agents. Load one of the base simulations by following the steps until step 2 above. 

#### Step 2. Loading a History File 
Then, when prompted with "Enter option: ", you should load the agent history by responding with the following command:

    call -- load history the_ville/<history_file_name>.csv
Note that you will need to replace `<history_file_name>` with the name of an existing history file. There are two history files included in the repo as examples: `agent_history_init_n25.csv` for `base_the_ville_n25` and `agent_history_init_n3.csv` for `base_the_ville_isabella_maria_klaus`. These files include semicolon-separated lists of memory records for each of the agents—loading them will insert the memory records into the agents' memory stream.

#### Step 3. Further Customization 
To customize the initialization by authoring your own history file, place your file in the following folder: `environment/frontend_server/static_dirs/assets/the_ville`. The column format for your custom history file will have to match the example history files included. Therefore, we recommend starting the process by copying and pasting the ones that are already in the repository.

### Create New Base Simulations
For a more involved customization, you will need to author your own base simulation files. The most straightforward approach would be to copy and paste an existing base simulation folder, renaming and editing it according to your requirements. This process will be simpler if you decide to keep the agent names unchanged. However, if you wish to change their names or increase the number of agents that the Smallville map can accommodate, you might need to directly edit the map using the [Tiled](https://www.mapeditor.org/) map editor.


## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Eddy_Lin.png" alt="Generative Eddy">   Authors and Citation 

**Authors:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein

Please cite our paper if you use the code or data in this repository. 
```
@inproceedings{Park2023GenerativeAgents,  
author = {Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},  
title = {Generative Agents: Interactive Simulacra of Human Behavior},  
year = {2023},  
publisher = {Association for Computing Machinery},  
address = {New York, NY, USA},  
booktitle = {In the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23)},  
keywords = {Human-AI interaction, agents, generative AI, large language models},  
location = {San Francisco, CA, USA},  
series = {UIST '23}
}
```

## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Wolfgang_Schulz.png" alt="Generative Wolfgang">   Acknowledgements

We encourage you to support the following three amazing artists who have designed the game assets for this project, especially if you are planning to use the assets included here for your own project: 
* Background art: [PixyMoon (@_PixyMoon\_)](https://twitter.com/_PixyMoon_)
* Furniture/interior design: [LimeZu (@lime_px)](https://twitter.com/lime_px)
* Character design: [ぴぽ (@pipohi)](https://twitter.com/pipohi)

In addition, we thank Lindsay Popowski, Philip Guo, Michael Terry, and the Center for Advanced Study in the Behavioral Sciences (CASBS) community for their insights, discussions, and support. Lastly, all locations featured in Smallville are inspired by real-world locations that Joon has frequented as an undergraduate and graduate student---he thanks everyone there for feeding and supporting him all these years.


