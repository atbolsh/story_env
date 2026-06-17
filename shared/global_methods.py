"""
File: global_methods.py
Description: Small, dependency-light helpers shared across the whole project --
the reverie backend, the Django frontend, and the standalone scripts.

This is the *single* canonical copy. It lives in ``<repo>/shared`` and each
process puts that directory on ``sys.path`` (see the short bootstrap near the
top of every entry point), so ``from global_methods import *`` resolves here no
matter which server or script imported it.

Originally authored by Joon Sung Park (joonspk@stanford.edu).
"""

import csv
import errno
import os
import shutil
from os import listdir

# Kept imported (even where unused below) because several callers rely on
# ``from global_methods import *`` re-exporting these names.
import datetime as dt
import math
import pathlib
import random
import string
import sys
import time

import numpy


def create_folder_if_not_there(curr_path):
  """
  Checks if a folder in the curr_path exists. If it does not exist, creates
  the folder.
  Note that if the curr_path designates a file location, it will operate on
  the folder that contains the file. But the function also works even if the
  path designates to just a folder.
  ARGS:
    curr_path: path to check / create the containing folder for.
  RETURNS:
    True: if a new folder is created
    False: if a new folder is not created
  """
  outfolder_name = curr_path.split("/")
  if len(outfolder_name) != 1:
    # This checks if the curr path is a file or a folder.
    if "." in outfolder_name[-1]:
      outfolder_name = outfolder_name[:-1]

    outfolder_name = "/".join(outfolder_name)
    if not os.path.exists(outfolder_name):
      os.makedirs(outfolder_name)
      return True

  return False


def write_list_of_list_to_csv(curr_list_of_list, outfile):
  """
  Writes a list of list to csv.
  Unlike write_list_to_csv_line, it writes the entire csv in one shot.
  ARGS:
    curr_list_of_list: list to write. The list comes in the following form:
               [['key1', 'val1-1', 'val1-2'...],
                ['key2', 'val2-1', 'val2-2'...],]
    outfile: name of the csv file to write
  RETURNS:
    None
  """
  create_folder_if_not_there(outfile)
  with open(outfile, "w") as f:
    writer = csv.writer(f)
    writer.writerows(curr_list_of_list)


def write_list_to_csv_line(line_list, outfile):
  """
  Writes one line to a csv file.
  Unlike write_list_of_list_to_csv, this opens an existing outfile and then
  appends a line to that file.
  This also works if the file does not exist already.
  ARGS:
    line_list: list to write. The list comes in the following form:
               ['key1', 'val1-1', 'val1-2'...]
               Importantly, this is NOT a list of list.
    outfile: name of the csv file to write
  RETURNS:
    None
  """
  create_folder_if_not_there(outfile)

  # Opening the file first so we can write incrementally as we progress.
  with open(outfile, "a") as curr_file:
    csv.writer(curr_file).writerow(line_list)


def read_file_to_list(curr_file, header=False, strip_trail=True):
  """
  Reads in a csv file to a list of list. If header is True, it returns a
  tuple with (header row, all rows).
  ARGS:
    curr_file: path to the current csv file.
  RETURNS:
    List of list where the component lists are the rows of the file.
  """
  analysis_list = []
  with open(curr_file) as f_analysis_file:
    data_reader = csv.reader(f_analysis_file, delimiter=",")
    for row in data_reader:
      if strip_trail:
        row = [i.strip() for i in row]
      analysis_list += [row]
  if not header:
    return analysis_list
  return analysis_list[0], analysis_list[1:]


def read_file_to_set(curr_file, col=0):
  """
  Reads in a "single column" of a csv file to a set.
  ARGS:
    curr_file: path to the current csv file.
  RETURNS:
    Set with all items in a single column of a csv file.
  """
  analysis_set = set()
  with open(curr_file) as f_analysis_file:
    data_reader = csv.reader(f_analysis_file, delimiter=",")
    for row in data_reader:
      analysis_set.add(row[col])
  return analysis_set


def get_row_len(curr_file):
  """
  Get the number of rows in a csv file.
  ARGS:
    curr_file: path to the current csv file.
  RETURNS:
    The number of rows
    False if the file does not exist
  """
  try:
    analysis_set = set()
    with open(curr_file) as f_analysis_file:
      data_reader = csv.reader(f_analysis_file, delimiter=",")
      for row in data_reader:
        analysis_set.add(row[0])
    return len(analysis_set)
  except Exception:
    return False


def check_if_file_exists(curr_file):
  """
  Checks if a file exists.
  ARGS:
    curr_file: path to the current csv file.
  RETURNS:
    True if the file exists
    False if the file does not exist
  """
  try:
    with open(curr_file):
      pass
    return True
  except Exception:
    return False


def find_filenames(path_to_dir, suffix=".csv"):
  """
  Given a directory, find all files that end with the provided suffix and
  return their paths.
  ARGS:
    path_to_dir: Path to the current directory
    suffix: The target suffix.
  RETURNS:
    A list of paths to all files in the directory.
  """
  filenames = listdir(path_to_dir)
  return [path_to_dir + "/" + filename
          for filename in filenames if filename.endswith(suffix)]


def average(list_of_val):
  """
  Finds the average of the numbers in a list.
  ARGS:
    list_of_val: a list of numeric values
  RETURNS:
    The average of the values
  """
  return sum(list_of_val) / float(len(list_of_val))


def std(list_of_val):
  """
  Finds the std of the numbers in a list.
  ARGS:
    list_of_val: a list of numeric values
  RETURNS:
    The std of the values
  """
  return numpy.std(list_of_val)


def copyanything(src, dst):
  """
  Copy over everything in the src folder to dst folder.
  ARGS:
    src: address of the source folder
    dst: address of the destination folder
  RETURNS:
    None
  """
  try:
    shutil.copytree(src, dst)
  except OSError as exc:  # python >2.5
    if exc.errno in (errno.ENOTDIR, errno.EINVAL):
      shutil.copy(src, dst)
    else:
      raise


def _split_top_level_parens(s):
  """
  Split a string into its text outside any parentheses and the contents of
  its top-level "(...)" groups.
  ARGS:
    s: the string to split.
  RETURNS:
    (base, groups): <base> is the text outside all parentheses (stripped);
    <groups> is a list of the top-level parenthetical contents, in order.
    An unbalanced trailing "(" group is included without its closing paren.
  """
  base_chars = []
  groups = []
  depth = 0
  start = None
  for i, c in enumerate(s):
    if c == "(":
      if depth == 0:
        start = i
      depth += 1
    elif c == ")":
      if depth > 0:
        depth -= 1
        if depth == 0:
          groups += [s[start + 1:i]]
          start = None
    elif depth == 0:
      base_chars += [c]
  if depth > 0 and start is not None:
    groups += [s[start + 1:]]
  return "".join(base_chars).strip(), groups


def flatten_parentheticals(s):
  """
  Collapse degenerate self-nested parentheticals in an action description.

  Local models (see the 2026-06-10 midnight run analysis) sometimes echo a
  task back into its own subtask, and repeated decomposition then compounds
  it into exponential garbage like:
      "task (task) (task (task)) (task (task) (task (task))) (real subtask)"
  This recursively drops every parenthetical group that is just an echo of
  the base text and keeps the last informative one, restoring the canonical
  "task (subtask)" shape. Legitimate nesting whose content differs from the
  base text -- e.g. "getting ready (checking mic, camera, etc.)" -- is left
  intact.
  ARGS:
    s: the action/task description.
  RETURNS:
    The flattened description.
  """
  base, groups = _split_top_level_parens(s)
  kept = None
  for g in groups:
    g = flatten_parentheticals(g).strip()
    # Drop a leading echo of the base text inside the group.
    if base and g.lower().startswith(base.lower()):
      g = g[len(base):].strip(" ()-,")
    if g and (not base or g.lower() != base.lower()):
      kept = g
  if not base:
    return kept or ""
  return f"{base} ({kept})" if kept else base


def sanitize_action_description(desp, max_len=350):
  """
  Sanitize a generated action/task description before it is saved into sim
  state (schedules, memory nodes) and fed back into later prompts.

  Fixes the two systematic degenerations observed in the 2026-06-10 midnight
  benchmark runs:
    1. the "conversing about conversing about ..." prefix echo (gemma4-e2b),
    2. exponential parenthetical self-nesting of task descriptions
       (gemma4-e4b; one prompt grew to 542 KB and OOMed the GPU).
  Also normalizes whitespace and caps the length so no single description can
  blow up prompt sizes, regardless of how it degenerated.
  ARGS:
    desp: the description string (non-strings are returned unchanged).
    max_len: hard cap on the returned length; generous on purpose -- its job
      is to stop runaway growth, not to trim normal content.
  RETURNS:
    The sanitized description.
  """
  if not isinstance(desp, str):
    return desp
  desp = " ".join(desp.split())
  while "conversing about conversing about" in desp:
    desp = desp.replace("conversing about conversing about",
                        "conversing about")
  desp = flatten_parentheticals(desp)
  if len(desp) > max_len:
    cut = desp[:max_len].rsplit(" ", 1)[0]
    desp = cut + "..."
  return desp
