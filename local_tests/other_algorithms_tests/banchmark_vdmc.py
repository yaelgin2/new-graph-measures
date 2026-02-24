"""
Run VDMC (MotifsNodeCalculator) on the benchmark graphs and save results as JSON.
Output format matches FANMOD+ results: {str(colored_motif_id): count}
Output location: local_tests/other_algorithms_tests/vdmc/
"""

import json
import os
import time
import pickle

import networkx as nx

from graphMeasures.feature_calculators import MotifsNodeCalculator
from graphMeasures.loggers import PrintLogger

# ── Config ─────────────────────────────────────────────────────────────────────
GRAPH_DIR = "local_tests/graphs_by_density_3"
OUT_DIR   = "local_tests/other_algorithms_tests/vdmc"
LOG_DIR   = "local_tests/other_algorithms_tests/logs"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CONFIGURATION = {
    "colored_directed_variations_3":   "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_directed_colored.pkl",
    "colored_undirected_variations_3": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_undirected_colored.pkl",
    "colored_directed_variations_4":   "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_directed_colored.pkl",
    "colored_undirected_variations_4": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_undirected_colored.pkl",
}

MOTIF_SIZE    = 4
DENSITIES     = [5, 8, 10, 13, 15]
DISTRIBUTIONS = ["uniform", "average", "rare"]


def read_graph(path):
    G = nx.Graph()
    with open(path) as f:
        data = json.load(f)
    for node in data["nodes"]:
        G.add_node(node["id"], color=node["color"])
    for edge in data["links"]:
        G.add_edge(edge["source"], edge["target"])
    return G


log_path = os.path.join(LOG_DIR, "vdmc.log")

with open(log_path, "w") as log_file:
    log_file.write("graph,time_seconds,num_colored_motifs,status\n")

    for den in DENSITIES:
        for dist in DISTRIBUTIONS:
            name       = f"g_den_{den}_embedded_den_3_{dist}_0"
            graph_path = os.path.join(GRAPH_DIR, f"{name}.json")
            out_path   = os.path.join(OUT_DIR, f"{name}.json")

            if not os.path.isfile(graph_path):
                print(f"SKIP (not found): {graph_path}")
                continue

            print(f"\nRunning VDMC on {name} ...")
            G = read_graph(graph_path)

            t0 = time.time()
            try:
                calc = MotifsNodeCalculator(
                    graph=G,
                    colores_loaded=True,
                    configuration=CONFIGURATION,
                    level=MOTIF_SIZE,
                    calc_nodes=False,
                    calc_edges=False,
                    count_motifs=True,
                    logger=PrintLogger(),
                )
                result = calc.build()
                elapsed = time.time() - t0

                motif_sum = result.get(MotifsNodeCalculator.MOTIF_SUM_KEY, {})

                # Save as JSON with string keys (matching FANMOD+ format)
                with open(out_path, "w") as f:
                    json.dump({str(k): v for k, v in motif_sum.items()}, f, indent=2)

                status = "ok"
                print(f"  Done in {elapsed:.2f}s — {len(motif_sum)} distinct colored motifs")

            except Exception as e:
                elapsed = time.time() - t0
                status = f"error: {e}"
                motif_sum = {}
                print(f"  ERROR: {e}")

            log_file.write(f"{name},{elapsed:.4f},{len(motif_sum)},{status}\n")
            log_file.flush()

print(f"\nDone. Results in {OUT_DIR}, log: {log_path}")