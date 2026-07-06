"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: reverie.py
Description: This is the main program for running generative agent simulations
that defines the ReverieServer class. This class maintains and records all  
states related to the simulation. The primary mode of interaction for those  
running the simulation should be through the open_server function, which  
enables the simulator to input command-line prompts for running and saving  
the simulation, among other tasks.

Release note (June 14, 2023) -- Reverie implements the core simulation 
mechanism described in my paper entitled "Generative Agents: Interactive 
Simulacra of Human Behavior." If you are reading through these lines after 
having read the paper, you might notice that I use older terms to describe 
generative agents and their cognitive modules here. Most notably, I use the 
term "personas" to refer to generative agents, "associative memory" to refer 
to the memory stream, and "reverie" to refer to the overarching simulation 
framework.
"""
# --- single-copy global_methods bootstrap -----------------------------------
# Put the repo's shared/ dir (the one canonical global_methods.py) on sys.path
# so ``import global_methods`` resolves there no matter which working directory
# this process was launched from. Walks up from this file to find shared/.
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isdir(_os.path.join(_d, "shared")):
  _d = _os.path.dirname(_d)
_shared = _os.path.join(_d, "shared")
if _os.path.isdir(_shared) and _shared not in _sys.path:
  _sys.path.insert(0, _shared)
del _os, _sys, _d, _shared
# ----------------------------------------------------------------------------
import json
import numpy
import datetime
import pickle
import time
import math
import os
import shutil
import traceback

from global_methods import *
from reverie_config import *
from maze import *
from persona.persona import *
import harnesses

##############################################################################
#                                  REVERIE                                   #
##############################################################################

class ReverieServer: 
  def __init__(self, 
               fork_sim_code,
               sim_code):
    # FORKING FROM A PRIOR SIMULATION:
    # <fork_sim_code> indicates the simulation we are forking from. 
    # Interestingly, all simulations must be forked from some initial 
    # simulation, where the first simulation is "hand-crafted".
    self.fork_sim_code = fork_sim_code
    fork_folder = f"{fs_storage}/{self.fork_sim_code}"

    # <sim_code> indicates our current simulation. The first step here is to 
    # copy everything that's in <fork_sim_code>, but edit its 
    # reverie/meta/json's fork variable. 
    self.sim_code = sim_code
    sim_folder = f"{fs_storage}/{self.sim_code}"
    copyanything(fork_folder, sim_folder)

    # The backend writes per-step movement files to <sim_folder>/movement/N.json
    # (see start_server). Base simulations don't ship with this directory, so
    # create it on fork to avoid a FileNotFoundError on the first step.
    os.makedirs(f"{sim_folder}/movement", exist_ok=True)

    with open(f"{sim_folder}/reverie/meta.json") as json_file:  
      reverie_meta = json.load(json_file)

    # Embedder-compatibility guard. Cosine similarities across two different
    # embedding spaces are meaningless, so warn loudly if the user is forking
    # a sim that was built up under a different embedder than the active
    # harness uses. We don't hard-fail (the user might intentionally rebuild
    # via reverie/backend_server/rebuild_embeddings.py).
    forked_embedder = reverie_meta.get("embedder")
    try:
      active_embedder = harnesses.get_active().embedder_name
    except Exception:
      active_embedder = None
    if forked_embedder and active_embedder and forked_embedder != active_embedder:
      print("=" * 72)
      print(f"WARNING: forked sim {self.fork_sim_code!r} was built with "
            f"embedder {forked_embedder!r}, but the active harness "
            f"({harnesses.get_active_name()!r}) uses {active_embedder!r}.")
      print("Cosine similarity across embedder families is not meaningful;")
      print("retrieval will misbehave until you rebuild this sim's embeddings:")
      print(f"    python rebuild_embeddings.py --sim {self.sim_code} "
            f"--to {harnesses.get_active_name()}")
      print("=" * 72)
    elif forked_embedder is None and active_embedder:
      print(f"[reverie] forked sim has no recorded embedder; assuming "
            f"compatibility with active embedder {active_embedder!r}.")

    with open(f"{sim_folder}/reverie/meta.json", "w") as outfile: 
      reverie_meta["fork_sim_code"] = fork_sim_code
      if active_embedder:
        reverie_meta["embedder"] = active_embedder
      outfile.write(json.dumps(reverie_meta, indent=2))

    # LOADING REVERIE'S GLOBAL VARIABLES
    # The start datetime of the Reverie: 
    # <start_datetime> is the datetime instance for the start datetime of 
    # the Reverie instance. Once it is set, this is not really meant to 
    # change. It takes a string date in the following example form: 
    # "June 25, 2022"
    # e.g., ...strptime(June 25, 2022, "%B %d, %Y")
    self.start_time = datetime.datetime.strptime(
                        f"{reverie_meta['start_date']}, 00:00:00",  
                        "%B %d, %Y, %H:%M:%S")
    # <curr_time> is the datetime instance that indicates the game's current
    # time. This gets incremented by <sec_per_step> amount everytime the world
    # progresses (that is, everytime curr_env_file is recieved). 
    self.curr_time = datetime.datetime.strptime(reverie_meta['curr_time'], 
                                                "%B %d, %Y, %H:%M:%S")
    # <sec_per_step> denotes the number of seconds in game time that each 
    # step moves foward. 
    self.sec_per_step = reverie_meta['sec_per_step']
    
    # <maze> is the main Maze instance. Note that we pass in the maze_name
    # (e.g., "double_studio") to instantiate Maze. 
    # e.g., Maze("double_studio")
    self.maze = Maze(reverie_meta['maze_name'])
    
    # <step> denotes the number of steps that our game has taken. A step here
    # literally translates to the number of moves our personas made in terms
    # of the number of tiles. 
    self.step = reverie_meta['step']

    # SETTING UP PERSONAS IN REVERIE
    # <personas> is a dictionary that takes the persona's full name as its 
    # keys, and the actual persona instance as its values.
    # This dictionary is meant to keep track of all personas who are part of
    # the Reverie instance. 
    # e.g., ["Isabella Rodriguez"] = Persona("Isabella Rodriguezs")
    self.personas = dict()
    # <personas_tile> is a dictionary that contains the tile location of
    # the personas (!-> NOT px tile, but the actual tile coordinate).
    # The tile take the form of a set, (row, col). 
    # e.g., ["Isabella Rodriguez"] = (58, 39)
    self.personas_tile = dict()
    
    # # <persona_convo_match> is a dictionary that describes which of the two
    # # personas are talking to each other. It takes a key of a persona's full
    # # name, and value of another persona's full name who is talking to the 
    # # original persona. 
    # # e.g., dict["Isabella Rodriguez"] = ["Maria Lopez"]
    # self.persona_convo_match = dict()
    # # <persona_convo> contains the actual content of the conversations. It
    # # takes as keys, a pair of persona names, and val of a string convo. 
    # # Note that the key pairs are *ordered alphabetically*. 
    # # e.g., dict[("Adam Abraham", "Zane Xu")] = "Adam: baba \n Zane:..."
    # self.persona_convo = dict()

    # Loading in all personas.
    init_env_file = f"{sim_folder}/environment/{str(self.step)}.json"
    init_env = json.load(open(init_env_file))

    # NAMS-backed personas.
    #
    # Two ways a persona ends up on NAMS (Neo4j Agent Memory System on local
    # bolt Neo4j) instead of the legacy JSON AssociativeMemory:
    #
    #   1. Dedicated *-nams harness (gemma4-e4b-nams / latest-gpt-nams): every
    #      persona in the sim is a NamsPersona. Selected interactively at
    #      startup; back-compat with the single-harness flow.
    #   2. Mixed ("multi-harness") mode: the active harness is a plain LLM
    #      harness (e.g. gemma4-e4b) for the whole sim, but a *subset* of
    #      personas -- named in the REVERIE_NAMS_PERSONAS env var (a comma-
    #      separated list of full persona names) -- run on NAMS while the
    #      rest keep the JSON memory. This is what midnight_test.sh uses to
    #      put only the gentrification scholar (Klaus) on NAMS and leave
    #      Isabella and Maria on the legacy JSON memory.
    #
    # In both cases the first boot of a NAMS persona against a JSON-forked sim
    # runs the one-way JSON -> NAMS importer (gated by graph_exists). See
    # harnesses/nams/json_to_nams_import.py.
    active_name = harnesses.get_active_name() or ""
    dedicated_nams = "-nams" in active_name
    nams_persona_csv = os.environ.get("REVERIE_NAMS_PERSONAS", "").strip()
    nams_persona_names = {n.strip() for n in nams_persona_csv.split(",") if n.strip()}
    if dedicated_nams and not nams_persona_names:
      # Back-compat: a dedicated *-nams harness with no per-persona override
      # puts every persona on NAMS.
      nams_persona_names = set(reverie_meta['persona_names'])

    if nams_persona_names:
      from harnesses.nams import (
        NamsMemory, NamsPersona, NAMS_EXTRACTION_NO_LLM,
      )
      from harnesses.nams.json_to_nams_import import import_persona_bootstrap
      nams_extraction_mode = os.environ.get(
        "REVERIE_NAMS_EXTRACTION", NAMS_EXTRACTION_NO_LLM)
      nams_llm_harness = harnesses.get_active()
      nams_embedder_name = getattr(
        nams_llm_harness, "embedder_name", "BAAI/bge-small-en-v1.5")

      def _build_nams(persona_name):
        return NamsMemory(
          session_id=persona_name,
          embedder_name=nams_embedder_name,
          extraction_mode=nams_extraction_mode,
          llm_harness=nams_llm_harness,
        )
    else:
      nams_persona_names = set()

    n_persona_total = len(reverie_meta['persona_names'])
    n_persona_i = 0
    for persona_name in reverie_meta['persona_names']:
      n_persona_i += 1
      persona_folder = f"{sim_folder}/personas/{persona_name}"
      p_x = init_env[persona_name]["x"]
      p_y = init_env[persona_name]["y"]
      if persona_name in nams_persona_names:
        print(f"[reverie] loading persona {n_persona_i}/{n_persona_total}: "
              f"{persona_name!r} (NAMS -- connecting to Neo4j + initializing "
              f"the no-llm extraction pipeline; first run downloads spaCy + "
              f"GLiNER + GLiREL models, which is silent and can take minutes)...",
              flush=True)
        nams = _build_nams(persona_name)
        try:
          if not nams.graph_exists():
            print(f"[reverie] importing JSON bootstrap -> NAMS for "
                  f"{persona_name!r}...", flush=True)
            import_persona_bootstrap(
              nams=nams, bootstrap_dir=f"{persona_folder}/bootstrap_memory",
            )
        except Exception as e:
          print(f"[reverie] NAMS import for {persona_name!r} failed "
                f"({type(e).__name__}: {e}); continuing with empty graph.",
                flush=True)
        curr_persona = NamsPersona(
          persona_name, persona_folder,
          nams_memory=nams, llm_harness=nams_llm_harness,
        )
        print(f"[reverie] loaded persona {persona_name!r} (NAMS).", flush=True)
      else:
        print(f"[reverie] loading persona {n_persona_i}/{n_persona_total}: "
              f"{persona_name!r} (JSON memory)...", flush=True)
        curr_persona = Persona(persona_name, persona_folder)
        print(f"[reverie] loaded persona {persona_name!r} (JSON).", flush=True)

      self.personas[persona_name] = curr_persona
      self.personas_tile[persona_name] = (p_x, p_y)
      self.maze.tiles[p_y][p_x]["events"].add(curr_persona.scratch
                                              .get_curr_event_and_desc())

    # REVERIE SETTINGS PARAMETERS:  
    # <server_sleep> denotes the amount of time that our while loop rests each
    # cycle; this is to not kill our machine. 
    self.server_sleep = 0.1

    # SIGNALING THE FRONTEND SERVER: 
    # curr_sim_code.json contains the current simulation code, and
    # curr_step.json contains the current step of the simulation. These are 
    # used to communicate the code and step information to the frontend. 
    # curr_step.json is refreshed by start_server after every step, so a
    # browser that opens (or refreshes) at any point attaches at the live
    # step. The frontend only ever reads these files.
    curr_sim_code = dict()
    curr_sim_code["sim_code"] = self.sim_code
    with open(f"{fs_temp_storage}/curr_sim_code.json", "w") as outfile: 
      outfile.write(json.dumps(curr_sim_code, indent=2))
    
    self.signal_curr_step()


  def signal_curr_step(self): 
    """
    Publish the current step number to temp_storage/curr_step.json so the
    frontend observer can find out where the live simulation is. Purely
    informational for the frontend; the backend never reads it back.
    """
    with open(f"{fs_temp_storage}/curr_step.json", "w") as outfile: 
      outfile.write(json.dumps({"step": self.step}, indent=2))


  def save(self): 
    """
    Save all Reverie progress -- this includes Reverie's global state as well
    as all the personas.  

    INPUT
      None
    OUTPUT 
      None
      * Saves all relevant data to the designated memory directory
    """
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    # Save Reverie meta information.
    reverie_meta = dict() 
    reverie_meta["fork_sim_code"] = self.fork_sim_code
    reverie_meta["start_date"] = self.start_time.strftime("%B %d, %Y")
    reverie_meta["curr_time"] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
    reverie_meta["sec_per_step"] = self.sec_per_step
    reverie_meta["maze_name"] = self.maze.maze_name
    reverie_meta["persona_names"] = list(self.personas.keys())
    reverie_meta["step"] = self.step
    try:
      reverie_meta["embedder"] = harnesses.get_active().embedder_name
    except Exception:
      pass
    reverie_meta_f = f"{sim_folder}/reverie/meta.json"
    with open(reverie_meta_f, "w") as outfile: 
      outfile.write(json.dumps(reverie_meta, indent=2))

    # Save the personas.
    for persona_name, persona in self.personas.items(): 
      save_folder = f"{sim_folder}/personas/{persona_name}/bootstrap_memory"
      persona.save(save_folder)


  def start_path_tester_server(self): 
    """
    Starts the path tester server. This is for generating the spatial memory
    that we need for bootstrapping a persona's state. 

    To use this, you need to open server and enter the path tester mode, and
    open the front-end side of the browser. 

    INPUT 
      None
    OUTPUT 
      None
      * Saves the spatial memory of the test agent to the path_tester_env.json
        of the temp storage. 
    """
    def print_tree(tree): 
      def _print_tree(tree, depth):
        dash = " >" * depth

        if type(tree) == type(list()): 
          if tree:
            print (dash, tree)
          return 

        for key, val in tree.items(): 
          if key: 
            print (dash, key)
          _print_tree(val, depth+1)
      
      _print_tree(tree, 0)

    # <curr_vision> is the vision radius of the test agent. Recommend 8 as 
    # our default. 
    curr_vision = 8
    # <s_mem> is our test spatial memory. 
    s_mem = dict()

    # The main while loop for the test agent. 
    while (True): 
      try: 
        curr_dict = {}
        tester_file = fs_temp_storage + "/path_tester_env.json"
        if check_if_file_exists(tester_file): 
          with open(tester_file) as json_file: 
            curr_dict = json.load(json_file)
            os.remove(tester_file)
          
          # Current camera location
          curr_sts = self.maze.sq_tile_size
          curr_camera = (int(math.ceil(curr_dict["x"]/curr_sts)), 
                         int(math.ceil(curr_dict["y"]/curr_sts))+1)
          curr_tile_det = self.maze.access_tile(curr_camera)

          # Initiating the s_mem
          world = curr_tile_det["world"]
          if curr_tile_det["world"] not in s_mem: 
            s_mem[world] = dict()

          # Iterating throughn the nearby tiles.
          nearby_tiles = self.maze.get_nearby_tiles(curr_camera, curr_vision)
          for i in nearby_tiles: 
            i_det = self.maze.access_tile(i)
            if (curr_tile_det["sector"] == i_det["sector"] 
                and curr_tile_det["arena"] == i_det["arena"]): 
              if i_det["sector"] != "": 
                if i_det["sector"] not in s_mem[world]: 
                  s_mem[world][i_det["sector"]] = dict()
              if i_det["arena"] != "": 
                if i_det["arena"] not in s_mem[world][i_det["sector"]]: 
                  s_mem[world][i_det["sector"]][i_det["arena"]] = list()
              if i_det["game_object"] != "": 
                if (i_det["game_object"] 
                    not in s_mem[world][i_det["sector"]][i_det["arena"]]):
                  s_mem[world][i_det["sector"]][i_det["arena"]] += [
                                                         i_det["game_object"]]

        # Incrementally outputting the s_mem and saving the json file. 
        print ("= " * 15)
        out_file = fs_temp_storage + "/path_tester_out.json"
        with open(out_file, "w") as outfile: 
          outfile.write(json.dumps(s_mem, indent=2))
        print_tree(s_mem)

      except:
        pass

      time.sleep(self.server_sleep * 10)


  def start_server(self, int_counter): 
    """
    The main backend server of Reverie. 
    This function steps the simulation forward on its own: it reads the
    environment file for the current step, calls on each persona to make
    decisions based on the world state, writes their moves to the movement
    file, and then writes the next step's environment file itself.

    Historical note: the environment file used to be written by the frontend
    browser (the Phaser game loop POSTed persona positions back after
    animating each step), which meant the simulation only advanced while a
    browser tab was open and focused. But the positions the frontend echoed
    back were exactly the movement targets the backend had computed one step
    earlier, so the round trip carried no information. The backend now
    closes that loop itself, and the browser is a pure observer: it can be
    opened, closed, or refreshed at any time without the agents ever
    noticing.
    INPUT
      int_counter: Integer value for the number of steps left for us to take
                   in this iteration. 
    OUTPUT 
      None
    """
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    # When a persona arrives at a game object, we give a unique event
    # to that object. 
    # e.g., ('double studio[...]:bed', 'is', 'unmade', 'unmade')
    # Later on, before this cycle ends, we need to return that to its 
    # initial state, like this: 
    # e.g., ('double studio[...]:bed', None, None, None)
    # So we need to keep track of which event we added. 
    # <game_obj_cleanup> is used for that. 
    game_obj_cleanup = dict()

    # The main while loop of Reverie. 
    while (True): 
      # Done with this iteration if <int_counter> reaches 0. 
      if int_counter == 0: 
        break

      # <curr_env_file> records where every persona stands at the current
      # step. The backend wrote it at the end of the previous step (or, for
      # step 0 / a fork, it ships with the simulation folder), so it is
      # always already present -- there is no waiting on the frontend.
      curr_env_file = f"{sim_folder}/environment/{self.step}.json"
      env_retrieved = False
      if check_if_file_exists(curr_env_file):
        # If we have an environment file, it means we have a new perception
        # input to our personas. So we first retrieve it.
        try: 
          # Try and save block for robustness of the while loop.
          with open(curr_env_file) as json_file:
            new_env = json.load(json_file)
            env_retrieved = True
        except: 
          pass
      
        if env_retrieved: 
          # This is where we go through <game_obj_cleanup> to clean up all 
          # object actions that were used in this cylce. 
          for key, val in game_obj_cleanup.items(): 
            # We turn all object actions to their blank form (with None). 
            self.maze.turn_event_from_tile_idle(key, val)
          # Then we initialize game_obj_cleanup for this cycle. 
          game_obj_cleanup = dict()

          # We first move our personas in the backend environment to match 
          # the frontend environment. 
          for persona_name, persona in self.personas.items(): 
            # <curr_tile> is the tile that the persona was at previously. 
            curr_tile = self.personas_tile[persona_name]
            # <new_tile> is the tile that the persona will move to right now,
            # during this cycle. 
            new_tile = (new_env[persona_name]["x"], 
                        new_env[persona_name]["y"])

            # We actually move the persona on the backend tile map here. 
            self.personas_tile[persona_name] = new_tile
            self.maze.remove_subject_events_from_tile(persona.name, curr_tile)
            self.maze.add_event_from_tile(persona.scratch
                                         .get_curr_event_and_desc(), new_tile)

            # Now, the persona will travel to get to their destination. *Once*
            # the persona gets there, we activate the object action.
            if not persona.scratch.planned_path: 
              # We add that new object action event to the backend tile map. 
              # At its creation, it is stored in the persona's backend. 
              game_obj_cleanup[persona.scratch
                               .get_curr_obj_event_and_desc()] = new_tile
              self.maze.add_event_from_tile(persona.scratch
                                     .get_curr_obj_event_and_desc(), new_tile)
              # We also need to remove the temporary blank action for the 
              # object that is currently taking the action. 
              blank = (persona.scratch.get_curr_obj_event_and_desc()[0], 
                       None, None, None)
              self.maze.remove_event_from_tile(blank, new_tile)

          # Then we need to actually have each of the personas perceive and
          # move. The movement for each of the personas comes in the form of
          # x y coordinates where the persona will move towards. e.g., (50, 34)
          # This is where the core brains of the personas are invoked. 
          movements = {"persona": dict(), 
                       "meta": dict()}
          for persona_name, persona in self.personas.items(): 
            # <next_tile> is a x,y coordinate. e.g., (58, 9)
            # <pronunciatio> is an emoji. e.g., "\ud83d\udca4"
            # <description> is a string description of the movement. e.g., 
            #   writing her next novel (editing her novel) 
            #   @ double studio:double studio:common room:sofa
            next_tile, pronunciatio, description = persona.move(
              self.maze, self.personas, self.personas_tile[persona_name], 
              self.curr_time)
            movements["persona"][persona_name] = {}
            movements["persona"][persona_name]["movement"] = next_tile
            movements["persona"][persona_name]["pronunciatio"] = pronunciatio
            movements["persona"][persona_name]["description"] = description
            movements["persona"][persona_name]["chat"] = (persona
                                                          .scratch.chat)

          # Include the meta information about the current stage in the 
          # movements dictionary. 
          movements["meta"]["curr_time"] = (self.curr_time 
                                             .strftime("%B %d, %Y, %H:%M:%S"))

          # We then write the personas' movements to a file that will be sent 
          # to the frontend server. 
          # Example json output: 
          # {"persona": {"Maria Lopez": {"movement": [58, 9]}},
          #  "persona": {"Klaus Mueller": {"movement": [38, 12]}}, 
          #  "meta": {curr_time: <datetime>}}
          curr_move_file = f"{sim_folder}/movement/{self.step}.json"
          with open(curr_move_file, "w") as outfile: 
            outfile.write(json.dumps(movements, indent=2))

          # The world advances by acknowledging the moves ourselves: each
          # persona lands exactly on the movement target we just computed,
          # so we write the next step's environment file directly. (The
          # frontend used to do this by animating the moves and POSTing the
          # resulting positions back; see the docstring.)
          next_env = dict()
          for persona_name in self.personas: 
            mv = movements["persona"][persona_name]["movement"]
            next_env[persona_name] = {"maze": self.maze.maze_name, 
                                      "x": mv[0], 
                                      "y": mv[1]}
          next_env_file = f"{sim_folder}/environment/{self.step + 1}.json"
          with open(next_env_file, "w") as outfile: 
            outfile.write(json.dumps(next_env, indent=2))

          # After this cycle, the world takes one step forward, and the
          # current time moves by <sec_per_step> amount.
          self.step += 1
          self.curr_time += datetime.timedelta(seconds=self.sec_per_step)
          # Let any attached observer know where the live simulation is.
          self.signal_curr_step()
          # Per-step progress so the operator can see the sim advancing (and
          # so a silent hang is distinguishable from a slow-but-healthy run).
          print(f"[reverie] step {self.step}/{int_counter + self.step} "
                f"({self.curr_time.strftime('%B %d, %Y, %H:%M:%S')})",
                flush=True)

          int_counter -= 1
          continue
          
      # We only reach here if the environment file for the current step is
      # missing or unreadable (should not happen in normal operation).
      # Sleep so we don't burn our machines. 
      time.sleep(self.server_sleep)


  def open_server(self): 
    """
    Open up an interactive terminal prompt that lets you run the simulation 
    step by step and probe agent state. 

    INPUT 
      None
    OUTPUT
      None
    """
    print ("Note: The agents in this simulation package are computational")
    print ("constructs powered by generative agents architecture and LLM. We")
    print ("clarify that these agents lack human-like agency, consciousness,")
    print ("and independent decision-making.\n---")

    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    while True: 
      sim_command = input("Enter option: ")
      sim_command = sim_command.strip()
      ret_str = ""

      try: 
        if sim_command.lower() in ["f", "fin", "finish", "save and finish"]: 
          # Finishes the simulation environment and saves the progress. 
          # Example: fin
          self.save()
          break

        elif sim_command.lower() == "start path tester mode": 
          # Starts the path tester and removes the currently forked sim files.
          # Note that once you start this mode, you need to exit out of the
          # session and restart in case you want to run something else. 
          shutil.rmtree(sim_folder) 
          self.start_path_tester_server()

        elif sim_command.lower() == "exit": 
          # Finishes the simulation environment but does not save the progress
          # and erases all saved data from current simulation. 
          # Example: exit 
          shutil.rmtree(sim_folder) 
          break 

        elif sim_command.lower() == "save": 
          # Saves the current simulation progress. 
          # Example: save
          self.save()

        elif sim_command[:3].lower() == "run": 
          # Runs the number of steps specified in the prompt.
          # Example: run 1000
          int_count = int(sim_command.split()[-1])
          rs.start_server(int_count)

        elif ("print persona schedule" 
              in sim_command[:22].lower()): 
          # Print the decomposed schedule of the persona specified in the 
          # prompt.
          # Example: print persona schedule Isabella Rodriguez
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                      .scratch.get_str_daily_schedule_summary())

        elif ("print all persona schedule" 
              in sim_command[:26].lower()): 
          # Print the decomposed schedule of all personas in the world. 
          # Example: print all persona schedule
          for persona_name, persona in self.personas.items(): 
            ret_str += f"{persona_name}\n"
            ret_str += f"{persona.scratch.get_str_daily_schedule_summary()}\n"
            ret_str += f"---\n"

        elif ("print hourly org persona schedule" 
              in sim_command.lower()): 
          # Print the hourly schedule of the persona specified in the prompt.
          # This one shows the original, non-decomposed version of the 
          # schedule.
          # Ex: print persona schedule Isabella Rodriguez
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                      .scratch.get_str_daily_schedule_hourly_org_summary())

        elif ("print persona current tile" 
              in sim_command[:26].lower()): 
          # Print the x y tile coordinate of the persona specified in the 
          # prompt. 
          # Ex: print persona current tile Isabella Rodriguez
          ret_str += str(self.personas[" ".join(sim_command.split()[-2:])]
                      .scratch.curr_tile)

        elif ("print persona chatting with buffer" 
              in sim_command.lower()): 
          # Print the chatting with buffer of the persona specified in the 
          # prompt.
          # Ex: print persona chatting with buffer Isabella Rodriguez
          curr_persona = self.personas[" ".join(sim_command.split()[-2:])]
          for p_n, count in curr_persona.scratch.chatting_with_buffer.items(): 
            ret_str += f"{p_n}: {count}"

        elif ("print persona associative memory (event)" 
              in sim_command.lower()):
          # Print the associative memory (event) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (event) Isabella Rodriguez
          ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_events())

        elif ("print persona associative memory (thought)" 
              in sim_command.lower()): 
          # Print the associative memory (thought) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (thought) Isabella Rodriguez
          ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_thoughts())

        elif ("print persona associative memory (chat)" 
              in sim_command.lower()): 
          # Print the associative memory (chat) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (chat) Isabella Rodriguez
          ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_chats())

        elif ("print persona spatial memory" 
              in sim_command.lower()): 
          # Print the spatial memory of the persona specified in the prompt
          # Ex: print persona spatial memory Isabella Rodriguez
          self.personas[" ".join(sim_command.split()[-2:])].s_mem.print_tree()

        elif ("print current time" 
              in sim_command[:18].lower()): 
          # Print the current time of the world. 
          # Ex: print current time
          ret_str += f'{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}\n'
          ret_str += f'steps: {self.step}'

        elif ("print tile event" 
              in sim_command[:16].lower()): 
          # Print the tile events in the tile specified in the prompt 
          # Ex: print tile event 50, 30
          cooordinate = [int(i.strip()) for i in sim_command[16:].split(",")]
          for i in self.maze.access_tile(cooordinate)["events"]: 
            ret_str += f"{i}\n"

        elif ("print tile details" 
              in sim_command.lower()): 
          # Print the tile details of the tile specified in the prompt 
          # Ex: print tile event 50, 30
          cooordinate = [int(i.strip()) for i in sim_command[18:].split(",")]
          for key, val in self.maze.access_tile(cooordinate).items(): 
            ret_str += f"{key}: {val}\n"

        elif ("call -- analysis" 
              in sim_command.lower()): 
          # Starts a stateless chat session with the agent. It does not save 
          # anything to the agent's memory. 
          # Ex: call -- analysis Isabella Rodriguez
          persona_name = sim_command[len("call -- analysis"):].strip() 
          self.personas[persona_name].open_convo_session("analysis")

        elif ("call -- load history" 
              in sim_command.lower()): 
          curr_file = maze_assets_loc + "/" + sim_command[len("call -- load history"):].strip() 
          # call -- load history the_ville/agent_history_init_n3.csv

          rows = read_file_to_list(curr_file, header=True, strip_trail=True)[1]
          clean_whispers = []
          for row in rows: 
            agent_name = row[0].strip() 
            whispers = row[1].split(";")
            whispers = [whisper.strip() for whisper in whispers]
            for whisper in whispers: 
              clean_whispers += [[agent_name, whisper]]

          load_history_via_whisper(self.personas, clean_whispers)

        print (ret_str)

      except:
        traceback.print_exc()
        print ("Error.")
        pass


def _run_interactive():
  """Interactive prompt: the original reverie.py flow. Used when no CLI args
  are passed. Picks a harness, optionally a NAMS extraction mode, a fork sim
  and a new sim name, then drops into the open_server() REPL."""
  available = harnesses.available_names()
  default_name = harnesses.DEFAULT_HARNESS
  print("Available model harnesses:")
  for name, desc in available.items():
    marker = " (default)" if name == default_name else ""
    print(f"  {name}{marker}: {desc}")
  harness_choice = input(
    f"Select model harness [{default_name}]: "
  ).strip().lower() or default_name
  if harness_choice not in available:
    raise SystemExit(
      f"unknown harness {harness_choice!r}; pick one of {sorted(available)}"
    )
  os.environ["REVERIE_HARNESS"] = harness_choice
  print(f"[reverie] using harness: {harness_choice}")

  # NAMS-backed harnesses need a second startup answer: the extraction mode.
  #   A / no-llm     -- spaCy + GLiNER + GLiREL only (air-gapped, deterministic)
  #   C / harness-llm -- spaCy + GLiNER for raw messages + the harness chat
  #                     LLM as the NAMS extraction LLM stage (richer facts)
  # Both modes still call add_fact directly from the reflect / converse
  # cognitive modules; mode C additionally lets NAMS's internal LLM
  # extractor run on raw short-term text. See
  # harnesses/nams/nams_memory.py for how this wires into MemorySettings.
  is_nams = "-nams" in harness_choice
  if is_nams:
    print("NAMS extraction mode:")
    print("  [A] no-llm      -- spaCy + GLiNER + GLiREL only (deterministic)")
    print("  [C] harness-llm -- + harness chat LLM for raw-message extraction")
    ext_choice = (input("Select extraction mode [A]: ").strip().lower()
                  or "a")
    if ext_choice.startswith("c"):
      os.environ["REVERIE_NAMS_EXTRACTION"] = "harness-llm"
    else:
      os.environ["REVERIE_NAMS_EXTRACTION"] = "no-llm"
    print(f"[reverie] NAMS extraction mode: {os.environ['REVERIE_NAMS_EXTRACTION']}")

  origin = input("Enter the name of the forked simulation: ").strip()
  target = input("Enter the name of the new simulation: ").strip()

  rs = ReverieServer(origin, target)
  rs.open_server()


def _run_cli():
  """CLI / headless mode. Selected by passing any command-line argument.

  This is the entry point for unattended runs (midnight_test.sh, CI, the
  smoke tests). It supports two shapes:

    1. Full headless sim:
         reverie.py --harness gemma4-e4b \
                     --fork <sim> --target <sim> \
                     --steps 8640 \
                     [--nams-personas "Klaus Mueller"] \
                     [--nams-extraction no-llm|harness-llm]
       Forks <fork> into <target>, runs <steps> steps, saves, exits. No
       interactive REPL. In mixed mode, --nams-personas lists the persona
       names that run on NAMS (everyone else stays on the legacy JSON
       memory); the active --harness is the LLM for *all* personas.

    2. Import-only (translate JSON bootstrap -> Neo4j, no sim run):
         reverie.py --import-nams-only \
                     --fork <sim> \
                     --nams-personas "Klaus Mueller" \
                     [--embedder BAAI/bge-small-en-v1.5] \
                     [--force-import]
       For each named persona, loads its bootstrap_memory/ from the fork sim
       and writes it into the local Neo4j as POLE+O entities + Facts, then
       exits. No sim is forked, no steps run, no LLM is loaded (the importer
       bypasses the NERS pipeline and writes facts directly, so extraction
       mode is irrelevant here). --force-import wipes the persona's existing
       session first so re-import is clean.

  Every LLM call (cognitive-module calls AND NAMS extraction-LLM calls in
  mode C) is logged to $REVERIE_PROMPT_LOG with full request/response
  context, exactly as in interactive mode.
  """
  import argparse

  parser = argparse.ArgumentParser(
    prog="reverie.py",
    description=("Headless / CLI mode for reverie. Pass no args for the "
                 "original interactive prompt."),
  )
  parser.add_argument("--harness", help="LLM harness name (see harnesses/__init__.py).")
  parser.add_argument("--fork", help="Forked simulation code (under frontend storage/).")
  parser.add_argument("--target", help="New simulation code to create.")
  parser.add_argument("--steps", type=int,
                      help="Number of steps to run (headless). Saves + exits after.")
  parser.add_argument("--nams-personas", default="",
                      help=("Comma-separated persona names that run on NAMS "
                            "(mixed mode). Everyone else stays on JSON memory. "
                            "With a dedicated *-nams harness, leave empty to "
                            "put all personas on NAMS."))
  parser.add_argument("--nams-extraction",
                      choices=["no-llm", "harness-llm"], default="no-llm",
                      help=("NAMS extraction mode for the NAMS personas. "
                            "no-llm = spaCy+GLiNER+GLiREL only; "
                            "harness-llm = + the active harness chat LLM as "
                            "the NAMS extraction LLM stage."))
  parser.add_argument("--embedder",
                      default="BAAI/bge-small-en-v1.5",
                      help=("Embedder id for --import-nams-only (no harness "
                            "is loaded, so we pass the string directly to "
                            "the NAMS SDK). Ignored for the full run, which "
                            "takes the embedder from the active harness."))
  parser.add_argument("--import-nams-only", action="store_true",
                      help=("Just translate the named personas' JSON "
                            "bootstrap_memory into the local Neo4j and exit; "
                            "do not fork or run any steps."))
  parser.add_argument("--force-import", action="store_true",
                      help=("With --import-nams-only, wipe each persona's "
                            "existing NAMS session before importing so the "
                            "import is clean and idempotent."))
  args = parser.parse_args()

  # --- import-only path ---------------------------------------------------
  if args.import_nams_only:
    if not args.fork:
      parser.error("--import-nams-only requires --fork")
    if not args.nams_personas:
      parser.error("--import-nams-only requires --nams-personas")
    _run_import_nams_only(
      fork_sim=args.fork,
      persona_names=[n.strip() for n in args.nams_personas.split(",")
                     if n.strip()],
      embedder_name=args.embedder,
      force=args.force_import,
    )
    return

  # --- full headless run --------------------------------------------------
  if not args.harness:
    parser.error("--harness is required for a headless run")
  if not args.fork or not args.target:
    parser.error("--fork and --target are required for a headless run")
  if args.steps is None:
    parser.error("--steps is required for a headless run")

  available = harnesses.available_names()
  if args.harness not in available:
    raise SystemExit(
      f"unknown harness {args.harness!r}; pick one of {sorted(available)}"
    )
  os.environ["REVERIE_HARNESS"] = args.harness
  if args.nams_personas:
    os.environ["REVERIE_NAMS_PERSONAS"] = args.nams_personas
  os.environ["REVERIE_NAMS_EXTRACTION"] = args.nams_extraction
  print(f"[reverie] CLI: harness={args.harness} fork={args.fork} "
        f"target={args.target} steps={args.steps}", flush=True)
  if args.nams_personas:
    print(f"[reverie] NAMS personas: {args.nams_personas} "
          f"(extraction={args.nams_extraction})", flush=True)
  print("[reverie] booting: building harness + forking sim + loading personas "
        "(first run downloads the Gemma model + NAMS spaCy/GLiNER/GLiREL "
        "models; this can take several minutes -- progress will print as each "
        "phase starts)", flush=True)

  rs = ReverieServer(args.fork, args.target)
  # Headless mode always saves, no matter how the run ends:
  #   * clean completion -> save + print "[reverie] COMPLETED:"
  #   * crash (exception propagates out of start_server) -> save partial +
  #     print "[reverie] CRASHED:"
  #   * Ctrl-C (midnight_test.sh budget/stall timeout) -> save partial +
  #     print "[reverie] INTERRUPTED:" + re-raise
  # The orchestrator (midnight_test.sh) greps the backend log for the
  # "COMPLETED" marker to distinguish a clean finish from a crash where the
  # process also exited (both leave backend_running() false).
  print(f"[reverie] all personas loaded; starting step loop: {args.steps} "
        f"steps. The first step loads the Gemma model onto the GPU (one-time, "
        f"several GB) -- silent until the model is ready, then each step prints "
        f"progress.", flush=True)
  try:
    rs.start_server(args.steps)
  except KeyboardInterrupt:
    print("[reverie] INTERRUPTED (Ctrl-C); saving partial progress...")
    try:
      rs.save()
    except Exception:
      traceback.print_exc()
    raise
  except Exception:
    traceback.print_exc()
    print("[reverie] CRASHED: saving partial progress before exit...")
    try:
      rs.save()
    except Exception:
      traceback.print_exc()
    return

  rs.save()
  print(f"[reverie] COMPLETED: {args.steps} steps, sim {args.target} saved.")


def _run_import_nams_only(*, fork_sim: str, persona_names: list,
                          embedder_name: str, force: bool):
  """Translate each named persona's JSON bootstrap_memory (under
  ``<fs_storage>/<fork_sim>/personas/<name>/bootstrap_memory``) into the
  local Neo4j as POLE+O entities + Facts, then exit. No sim fork, no LLM
  load, no steps. The importer bypasses the NERS pipeline (it writes facts
  directly via add_fact/add_entity/cypher_write), so extraction mode is
  irrelevant and we hard-code no-llm to avoid needing a harness object.

  With ``force=True``, wipe each persona's existing session + facts +
  entities + reasoning traces first so the import is idempotent.
  """
  from harnesses.nams import NamsMemory, NAMS_EXTRACTION_NO_LLM
  from harnesses.nams.json_to_nams_import import import_persona_bootstrap

  fork_folder = f"{fs_storage}/{fork_sim}"
  if not os.path.isdir(fork_folder):
    raise SystemExit(
      f"fork sim {fork_sim!r} not found at {fork_folder}")

  for name in persona_names:
    bootstrap_dir = (f"{fork_folder}/personas/{name}/bootstrap_memory")
    if not os.path.isdir(bootstrap_dir):
      raise SystemExit(
        f"persona {name!r} bootstrap not found at {bootstrap_dir}")

    print(f"[import-nams-only] {name!r}: opening NAMS session...")
    # llm_harness=None is fine: importer bypasses NERS, and we hard-code
    # no-llm so build_memory_settings never touches the harness.
    nams = NamsMemory(
      session_id=name,
      embedder_name=embedder_name,
      extraction_mode=NAMS_EXTRACTION_NO_LLM,
      llm_harness=None,
    )
    try:
      _ = nams.client  # force connect
      if force:
        print(f"[import-nams-only] {name!r}: --force-import; wiping existing "
              f"session...")
        _wipe_nams_session(nams, name)
      if (not force) and nams.graph_exists():
        print(f"[import-nams-only] {name!r}: graph already exists; skipping. "
              f"(Use --force-import to re-import.)")
        continue
      print(f"[import-nams-only] {name!r}: importing JSON bootstrap -> NAMS...")
      report = import_persona_bootstrap(
        nams=nams, bootstrap_dir=os.path.abspath(bootstrap_dir),
      )
      print(f"[import-nams-only] {name!r}: {report}")
    finally:
      nams.close()


def _wipe_nams_session(nams, name: str):
  """Delete every NAMS node tied to this persona name (Conversation + its
  Messages, Facts naming the persona, the persona's PERSON Entity, and the
  persona's ReasoningTraces). Used by --import-nams-only --force-import so
  re-import is clean and idempotent."""
  try:
    nams.clear_session()
  except Exception as e:
    print(f"[import-nams-only] clear_session for {name!r} failed "
          f"({type(e).__name__}: {e}); continuing with the node-level wipe.")
  for q, params in (
    ("MATCH (f:Fact) WHERE f.subject = $name OR f.object = $name "
     "DETACH DELETE f", {"name": name}),
    ("MATCH (e:Entity {name: $name}) DETACH DELETE e", {"name": name}),
    ("MATCH (t:ReasoningTrace {session_id: $name}) DETACH DELETE t",
     {"name": name}),
  ):
    try:
      nams.cypher_write(q, params)
    except Exception as e:
      print(f"[import-nams-only] wipe query failed for {name!r} "
            f"({type(e).__name__}: {e}); continuing.")


if __name__ == '__main__':
  import sys as _sys
  if len(_sys.argv) > 1:
    _run_cli()
  else:
    _run_interactive()




















































