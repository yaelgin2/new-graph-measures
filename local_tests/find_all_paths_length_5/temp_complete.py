"""
temp_complete_missing_paths.py

Completes two missing entries in the paths embedded experiments:

  1. g_den_15_embedded_den_3_average_5
     Log exists but has only S_1..S_97 — appends S_98..S_100.

  2. g_den_15_embedded_den_5_rare_6
     Log entirely missing — computes all S_1..S_100 from scratch.

After both logs are complete, rewrites the summary logs so they contain
clean, deduplicated, correctly-ordered entries.

Run from the new-graph-measures/ directory:
  python temp_complete_missing_paths.py
"""

import json
import logging
import os
import re
import sys

import networkx as nx

from local_tests.find_all_paths_length_5.PathMotifCalculator import PathMotifCalculator

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR     = os.path.join(os.getcwd(), "local_tests")
PATHS_LOG    = os.path.join(BASE_DIR, "find_all_paths_length_5", "logs")
SUMMARY_3    = os.path.join(PATHS_LOG, "summary_paths_different_distributions_3.log")
SUMMARY_5    = os.path.join(PATHS_LOG, "summary_paths_different_distributions_5.log")

TASKS = [
    {
        "name":        "g_den_15_embedded_den_3_average_5",
        "input_dir":   os.path.join(BASE_DIR, "input_color_average_deg_3"),
        "graph_file":  "g_den_15_embedded_den_3_average_5.json",
        "log_file":    os.path.join(PATHS_LOG, "g_den_15_embedded_den_3_average_5.log"),
        "summary":     SUMMARY_3,
        "start_from":  98,   # S_1..S_97 already logged; resume from S_98
        "num_s":       100,
    },
    {
        "name":        "g_den_15_embedded_den_5_rare_6",
        "input_dir":   os.path.join(BASE_DIR, "input_color_rare_deg_5"),
        "graph_file":  "g_den_15_embedded_den_5_rare_6.json",
        "log_file":    os.path.join(PATHS_LOG, "g_den_15_embedded_den_5_rare_6.log"),
        "summary":     SUMMARY_5,
        "start_from":  1,    # entirely missing
        "num_s":       100,
    },
]

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

def make_logger(name, log_file, mode="a"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.FileHandler(log_file, mode=mode)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(h)
    return logger

def count_fp_from_log(log_file, num_s):
    """Count false positives (PASS for S_11..S_num_s) from a completed log."""
    fp = 0
    pattern = re.compile(r'PATH (PASS|FAIL) S_(\d+)')
    with open(log_file) as f:
        for line in f:
            m = pattern.search(line)
            if m and m.group(1) == "PASS":
                idx = int(m.group(2))
                if idx > 10:
                    fp += 1
    return fp

def already_logged(log_file, s_idx):
    """Check if S_s_idx is already in the log."""
    if not os.path.exists(log_file):
        return False
    pattern = re.compile(rf'\bS_{s_idx}\b')
    with open(log_file) as f:
        return any(pattern.search(line) for line in f)

# ── Process each task ─────────────────────────────────────────────────────────

for task in TASKS:
    name       = task["name"]
    log_file   = task["log_file"]
    start_from = task["start_from"]
    num_s      = task["num_s"]
    input_dir  = task["input_dir"]

    print(f"\n{'='*60}")
    print(f"Task: {name}  (S_{start_from}..S_{num_s})")

    # Load G
    g_path = os.path.join(input_dir, f"{name}.json")
    # G file may be in the graphs_by_density dir instead
    if not os.path.exists(g_path):
        g_path = os.path.join(BASE_DIR, "graphs_by_density_5"
                              if "den_5" in name else "graphs_by_density_3",
                              f"{name}.json")
    print(f"Loading G from {g_path} ...")
    g_paths = compute_paths(g_path)

    logger = make_logger(name, log_file, mode="a")
    fp = 0

    for i in range(start_from, num_s + 1):
        if already_logged(log_file, i):
            print(f"  S_{i} already logged, skipping")
            continue

        s_path  = os.path.join(input_dir, f"S_{i}.json")
        s_paths = compute_paths(s_path)
        feasible = all(g_paths.get(m, 0) >= cnt for m, cnt in s_paths.items())
        logger.info(f"PATH {'PASS' if feasible else 'FAIL'} S_{i}")

        if i > 10 and feasible:
            fp += 1

        if i % 10 == 0:
            print(f"  Done S_{i}")

    # Count total fp from full log (covers both pre-existing and new entries)
    total_fp = count_fp_from_log(log_file, num_s)
    print(f"  Complete. False positives (S_11..S_{num_s}): {total_fp}")

print(f"\n{'='*60}")
print("All tasks complete.")

# ── Clean up summary logs ─────────────────────────────────────────────────────
# For each summary file: read all entries, deduplicate (keep last value for
# each key), recompute AVERAGE lines, write back in clean order.

def clean_summary(summary_path, embedded_den):
    """
    Rebuild a clean summary log from scratch by:
    1. Parsing all per-graph log files directly (ground truth)
    2. Writing deduplicated, correctly-ordered entries
    """
    if not os.path.exists(summary_path):
        print(f"[SKIP] Summary not found: {summary_path}")
        return

    from graphMeasures.feature_calculators import MotifsNodeCalculator

    densities = [5, 8, 10, 13, 15] if embedded_den == 3 else [8, 10, 13, 15]
    colors    = ["uniform", "average", "rare"]

    # Parse all existing per-graph logs to get ground truth fp values
    fp_values = {}   # name -> fp count
    for g_den in densities:
        for color in colors:
            for j in range(10):
                name     = f"g_den_{g_den}_embedded_den_{embedded_den}_{color}_{j}"
                log_file = os.path.join(PATHS_LOG, f"{name}.log")
                if os.path.exists(log_file):
                    fp_values[name] = count_fp_from_log(log_file, 100)

    # Read existing summary to get timestamps (preserve original timestamps)
    ts_map = {}   # name -> timestamp string
    avg_ts_map = {}
    ts_pattern  = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (.+?) \| path_only')
    with open(summary_path) as f:
        for line in f:
            m = ts_pattern.match(line.strip())
            if m:
                ts, key = m.group(1), m.group(2)
                if "AVERAGE" in line:
                    avg_ts_map[key] = ts
                else:
                    ts_map[key] = ts

    # Write clean summary
    backup = summary_path + ".bak"
    os.rename(summary_path, backup)
    print(f"Backed up {summary_path} → {backup}")

    with open(summary_path, "w") as out:
        for g_den in densities:
            for color in colors:
                color_fps = []
                for j in range(10):
                    name = f"g_den_{g_den}_embedded_den_{embedded_den}_{color}_{j}"
                    if name not in fp_values:
                        continue
                    fp  = fp_values[name]
                    ts  = ts_map.get(name, "1970-01-01 00:00:00,000")
                    out.write(f"{ts} - {name} | path_only={fp}\n")
                    color_fps.append(fp)

                if color_fps:
                    avg_key = f"g_den_{g_den}_embedded_den_{embedded_den}_{color}"
                    avg_val = sum(color_fps) / len(color_fps)
                    avg_ts  = avg_ts_map.get(avg_key, ts_map.get(
                        f"g_den_{g_den}_embedded_den_{embedded_den}_{color}_9",
                        "1970-01-01 00:00:00,000"))
                    out.write(f"{avg_ts} - {avg_key} | AVERAGE path_only={avg_val:.1f}\n")

    print(f"Rewrote clean summary: {summary_path}")


print("\nCleaning summary logs ...")
clean_summary(SUMMARY_3, embedded_den=3)
clean_summary(SUMMARY_5, embedded_den=5)
print("Done.")