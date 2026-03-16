"""
Shared S-motif cache.

All 4 test scripts import get_cached_S_motifs() from here.
Cache lives in  local_tests/temp/s_motif_cache/
and is keyed by the path of the S json file relative to local_tests/,
so S files from different input dirs never collide.
"""

import os
import pickle

from graphMeasures.feature_calculators import MotifsNodeCalculator
import networkx as nx

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.getcwd(), "local_tests")
CACHE_DIR  = os.path.join(BASE_DIR, "temp", "s_motif_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── VDMC configuration (same for all scripts) ────────────────────────────────
CONFIGURATION = {
    "colored_directed_variations_3":   "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_directed_colored.pkl",
    "colored_undirected_variations_3": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_undirected_colored.pkl",
    "colored_directed_variations_4":   "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_directed_colored.pkl",
    "colored_undirected_variations_4": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_undirected_colored.pkl",
}

MOTIF_SIZE = 4


def _read_graph(path: str) -> nx.Graph:
    import json
    G = nx.Graph()
    with open(path) as f:
        data = json.load(f)
    for node in data["nodes"]:
        G.add_node(node["id"], color=node["color"])
    for edge in data["links"]:
        G.add_edge(edge["source"], edge["target"])
    return G


def _cache_key(s_path: str) -> str:
    """
    Stable filename derived from the path relative to local_tests/.
    e.g. input_color_uniform_deg_3/S_1.json  ->  input_color_uniform_deg_3__S_1.pkl
    Works even if s_path is absolute.
    """
    try:
        rel = os.path.relpath(s_path, BASE_DIR)
    except ValueError:          # different drive on Windows
        rel = s_path
    # replace path separators with __ and swap extension
    key = rel.replace(os.sep, "__").replace("/", "__").replace(".json", ".pkl")
    return key


def get_cached_S_motifs(s_path: str) -> dict:
    """
    Return the MOTIF_SUM_KEY dict for the S graph at s_path.
    Computes and caches on first call; returns cached result on subsequent calls
    regardless of which script calls it.
    """
    cache_file = os.path.join(CACHE_DIR, _cache_key(s_path))

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    S = _read_graph(s_path)
    calc = MotifsNodeCalculator(
        graph=S,
        colores_loaded=True,
        configuration=CONFIGURATION,
        level=MOTIF_SIZE,
        calc_nodes=False,
        calc_edges=False,
        count_motifs=True,
    )
    s_motifs = calc.build()[MotifsNodeCalculator.MOTIF_SUM_KEY]

    with open(cache_file, "wb") as f:
        pickle.dump(s_motifs, f)

    return s_motifs