"""
FANMOD+ Benchmark Script
Runs LocalFANMOD on graphs and converts output to the same colored motif IDs
used by the Kavosh implementation, so results are directly comparable.

Motif number encoding (matching IsomorphismGenerator):
  - nodes labeled 0..3
  - bits = edge presence over combinations(range(4), 2) = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
  - motif_number = BitArray(bits).uint

Colored motif number:
  color_int = sum(colors[i] << (8*(3-i)) for i in range(4))
  colored_id = (canonical_motif << 32) + color_int

  Then canonicalize: for each permutation in motif_to_minimal_motif[raw_motif][1],
  apply permutation to colors, compute color_int, keep minimum.
  Final id = (canonical_motif << 32) + min_color_int
"""

import os
import re
import json
import time
import pickle
import subprocess
import tempfile
from collections import defaultdict
from itertools import combinations
import networkx as nx

# ── Paths ──────────────────────────────────────────────────────────────────────
FANMOD_PLUS_BIN = os.path.expanduser("~/new-graph-measures/FANMODPlus/build/LocalFANMOD")
VARIATIONS_PKL  = ("graphMeasures/feature_calculators/node_features_calculators/"
                   "calculators/motif_variations/4_undirected_colored.pkl")

GRAPH_DIR   = "local_tests/graphs_by_density_3"
OUT_DIR     = "local_tests/other_algorithms_tests/fanmod_plus"
LOG_DIR     = "local_tests/other_algorithms_tests/logs"
TMP_INPUT   = "local_tests/other_algorithms_tests/fanmod_tmp_input.txt"
TMP_OUTPUT  = "local_tests/other_algorithms_tests/fanmod_tmp_output.csv"

MOTIF_SIZE       = 4
NUM_RANDOM_GRAPHS = 0
COLOR_KEY        = "color"

DENSITIES     = [5, 8, 10, 13, 15]
DISTRIBUTIONS = ["uniform", "average", "rare"]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── Load variations pickle ─────────────────────────────────────────────────────
with open(VARIATIONS_PKL, "rb") as f:
    motif_to_minimal_motif = pickle.load(f)
# motif_to_minimal_motif[raw_motif] = (canonical_motif, [permutations])

# ── Helper: edges → raw motif number ──────────────────────────────────────────
_PAIRS = list(combinations(range(4), 2))  # (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
_PAIR_INDEX = {p: i for i, p in enumerate(_PAIRS)}

def edges_to_motif_number(edges):
    """edges: iterable of (u,v) with u,v in 0..3 (unordered). Returns BitArray uint."""
    bits = [False] * 6
    for u, v in edges:
        key = (min(u,v), max(u,v))
        if key in _PAIR_INDEX:
            bits[_PAIR_INDEX[key]] = True
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val

# ── Helper: colored motif ID (matching Kavosh) ─────────────────────────────────
def colors_to_color_int(colors):
    """colors: sequence of 4 ints. Matches colors_tuple_and_motif_number_to_colored_motif_number."""
    n = len(colors)
    ci = 0
    for i, c in enumerate(colors):
        ci += c << (8 * (n - 1 - i))
    return ci

def canonical_colored_motif_id(raw_motif, node_colors):
    """
    Given raw motif number and list of 4 node colors (in LEDA node order 0..3),
    return the canonical colored motif ID matching Kavosh's encoding.
    """
    if raw_motif not in motif_to_minimal_motif:
        return None
    canonical_motif, permutations = motif_to_minimal_motif[raw_motif]
    min_color_int = min(
        colors_to_color_int(tuple(node_colors[p] for p in perm))
        for perm in permutations
    )
    return (canonical_motif << (8 * 4)) + min_color_int

# ── FANMOD+ output parser ──────────────────────────────────────────────────────
def parse_fanmod_output(filepath):
    """
    Parse FANMOD+ output CSV (which interleaves CSV lines with LEDA graph blocks).
    Returns dict: {colored_motif_id: count}
    """
    with open(filepath, "r") as f:
        content = f.read()

    results = defaultdict(int)

    # Split into per-motif blocks. Each block starts with: ID,Frequency,Count
    # We split on lines matching that pattern.
    blocks = re.split(r'(?=^\d+,[\d.e+\-]+,\d+$)', content, flags=re.MULTILINE)

    for block in blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue
        m = re.match(r'^(\d+),([\d.e+\-]+),(\d+)$', lines[0])
        if not m:
            continue
        count = int(m.group(3))

        # Extract node colors from LEDA #nodes section: lines like |{N}|
        node_colors = [int(x) for x in re.findall(r'\|\{(\d+)\}\|', block)]
        if len(node_colors) != 4:
            continue

        # Extract edges from LEDA #edges section: lines like "u v 1 |{}|"
        # LEDA nodes are 1-indexed
        edge_matches = re.findall(r'^(\d+) (\d+) 1\b', block, re.MULTILINE)
        edges = [(int(a)-1, int(b)-1) for a, b in edge_matches]
        if not edges:
            continue

        raw_motif = edges_to_motif_number(edges)
        colored_id = canonical_colored_motif_id(raw_motif, node_colors)
        if colored_id is None:
            # disconnected or unknown topology — skip
            continue

        results[colored_id] += count

    return dict(results)

# ── Write FANMOD+ input file ───────────────────────────────────────────────────
def write_fanmod_input(G, filepath):
    """
    FANMOD+ undirected input: one line per edge (not duplicated)
      src_id dst_id src_color dst_color
    Node IDs must be 0-based integers.
    """
    # Remap node IDs to contiguous 0-based ints
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    colors = {node_to_idx[n]: G.nodes[n].get(COLOR_KEY, 0) for n in nodes}

    with open(filepath, "w") as f:
        for u, v in G.edges():
            ui, vi = node_to_idx[u], node_to_idx[v]
            f.write(f"{ui} {vi} {colors[ui]} {colors[vi]}\n")

# ── Main benchmark loop ────────────────────────────────────────────────────────
log_path = os.path.join(LOG_DIR, "fanmod_plus.log")

with open(log_path, "w") as log_file:
    log_file.write("graph,time_seconds,num_colored_motifs,status\n")

    for den in DENSITIES:
        for dist in DISTRIBUTIONS:
            graph_path = os.path.join(
                GRAPH_DIR, f"g_den_{den}_embedded_den_3_{dist}_0.json"
            )
            if not os.path.isfile(graph_path):
                print(f"SKIP (not found): {graph_path}")
                continue

            # Load graph
            with open(graph_path) as gf:
                data = json.load(gf)

            G = nx.node_link_graph(data)

            print(f"\nRunning on {graph_path} ...")

            # Write input
            write_fanmod_input(G, TMP_INPUT)

            # Run FANMOD+
            cmd = [
                FANMOD_PLUS_BIN,
                "-i", TMP_INPUT,
                "-o", TMP_OUTPUT,
                "-s", str(MOTIF_SIZE),
                "-r", str(NUM_RANDOM_GRAPHS),
                "-V",
            ]

            t0 = time.time()
            try:
                result = subprocess.run(
                    cmd, text=True,
                    stdout=subprocess.PIPE, stderr=None,  # stderr streams live
                    timeout=3600
                )
                elapsed = time.time() - t0

                if result.returncode != 0:
                    raise RuntimeError(f"FANMOD+ exited with code {result.returncode}")

                # Parse output
                motif_counts = parse_fanmod_output(TMP_OUTPUT)

                # Save counts
                out_path = os.path.join(OUT_DIR, f"g_den_{den}_embedded_den_3_{dist}_0.json")
                with open(out_path, "w") as out_f:
                    # Convert int keys to str for JSON
                    json.dump({str(k): v for k, v in motif_counts.items()}, out_f, indent=2)

                status = "ok"
                print(f"  Done in {elapsed:.2f}s — {len(motif_counts)} distinct colored motifs")

            except subprocess.TimeoutExpired:
                elapsed = time.time() - t0
                status = "timeout"
                print(f"  TIMEOUT after {elapsed:.0f}s")

            except Exception as e:
                elapsed = time.time() - t0
                status = f"error: {e}"
                print(f"  ERROR: {e}")

            log_file.write(f"{graph_path},{elapsed:.4f},{len(motif_counts) if status=='ok' else 0},{status}\n")
            log_file.flush()

print("\nDone. Log:", log_path)