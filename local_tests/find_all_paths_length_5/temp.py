"""
temp_paths_average_deg15.py

Computes the missing entries for color_average_deg_15:
  - Appends G_compute_time + TOTAL_TIME to equal_degs_times_paths.log
  - Writes per-S PASS/FAIL to paths_color_average_deg_15.log

Run from the new-graph-measures/ directory:
  python temp_paths_average_deg15.py
"""

import json
import logging
import os
import sys
import time

import networkx as nx

from local_tests.find_all_paths_length_5.PathMotifCalculator import PathMotifCalculator

# ── Config ────────────────────────────────────────────────────────────────────

RUN_NAME   = "color_average_deg_15"
NUM_S      = 1000

BASE_DIR        = os.path.join(os.getcwd(), "local_tests")
INPUT_DIR       = os.path.join(BASE_DIR, f"input_{RUN_NAME}")
COMPARE_LOG_DIR = os.path.join(BASE_DIR, "find_all_paths_length_5", "logs", "compare_results")

LOG_FILE  = os.path.join(COMPARE_LOG_DIR, f"paths_{RUN_NAME}.log")
TIMES_LOG = os.path.join(COMPARE_LOG_DIR, "equal_degs_times_paths.log")

# ── Sanity checks ─────────────────────────────────────────────────────────────

with open(TIMES_LOG) as f:
    times_content = f.read()
if f"{RUN_NAME} | TOTAL_TIME=" in times_content:
    print(f"TOTAL_TIME for {RUN_NAME} already present in {TIMES_LOG} — nothing to do.")
    sys.exit(0)

if os.path.exists(LOG_FILE):
    print(f"Per-S log already exists: {LOG_FILE} — delete it first to regenerate.")
    sys.exit(0)

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_graph_file(path):
    graph = nx.Graph()
    with open(path) as f:
        data = json.load(f)
    for node in data["nodes"]:
        graph.add_node(node["id"], color=node["color"])
    for edge in data["links"]:
        graph.add_edge(edge["source"], edge["target"])
    return graph

def compute_paths(path):
    G = read_graph_file(path)
    calc = PathMotifCalculator(G, directed=False)
    return calc.build()

# ── Loggers ───────────────────────────────────────────────────────────────────

times_logger = logging.getLogger("equal_degs_times_paths")
times_logger.setLevel(logging.INFO)
th = logging.FileHandler(TIMES_LOG)  # append mode
th.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
times_logger.addHandler(th)

logger = logging.getLogger(f"paths_{RUN_NAME}")
logger.setLevel(logging.INFO)
lh = logging.FileHandler(LOG_FILE)
lh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(lh)

# ── Compute G ─────────────────────────────────────────────────────────────────

total_start = time.perf_counter()

print(f"Computing G paths for {RUN_NAME} ...")
g_start = time.perf_counter()
g_paths = compute_paths(os.path.join(INPUT_DIR, "G_non_induced.json"))
g_time  = time.perf_counter() - g_start

times_logger.info(f"{RUN_NAME} | G_compute_time={g_time:.4f}s")
print(f"G computed in {g_time:.2f}s")

# ── Process S_1 … S_1000 ─────────────────────────────────────────────────────

false_pos = 0

for i in range(1, NUM_S + 1):
    s_paths  = compute_paths(os.path.join(INPUT_DIR, f"S_{i}.json"))
    feasible = all(g_paths.get(m, 0) >= cnt for m, cnt in s_paths.items())

    logger.info(f"PATH {'PASS' if feasible else 'FAIL'} S_{i}")

    if i > 10 and feasible:
        false_pos += 1

    if i % 100 == 0:
        print(f"  Done S_{i}  (false positives so far: {false_pos})")

total_time = time.perf_counter() - total_start
times_logger.info(f"{RUN_NAME} | TOTAL_TIME={total_time:.4f}s")

print(f"\nDone.")
print(f"G compute time: {g_time:.2f}s")
print(f"Total time:     {total_time:.2f}s")
print(f"False positives (S_11..S_1000): {false_pos}")
print(f"Per-S log:   {LOG_FILE}")
print(f"Times log:   {TIMES_LOG}")