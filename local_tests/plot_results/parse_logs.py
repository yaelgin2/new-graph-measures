"""
parse_logs.py — Shared parsing library for all table/plot generation scripts.

Parses log files from:
  - local_tests/induced/logs/
  - local_tests/non_induced/logs/
  - local_tests/find_all_paths_length_5/logs/
  - ../PROJECT_RUN_PATTERN/pattern_finder/results/

Returns structured dicts/dataframes used by table and plot scripts.
"""

import os
import re
import sys

# ── Path roots ────────────────────────────────────────────────────────────────

def get_base_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

BASE_DIR          = get_base_dir()
INDUCED_LOG_DIR   = os.path.join(BASE_DIR, "induced",  "logs")
NON_IND_LOG_DIR   = os.path.join(BASE_DIR, "non_induced", "logs")
PATHS_LOG_DIR     = os.path.join(BASE_DIR, "find_all_paths_length_5", "logs")
PATTERN_RESULT_DIR = os.path.join(BASE_DIR, "..", "PROJECT_RUN_PATTERN",
                                  "pattern_finder", "results")

INDUCED_CMP   = os.path.join(INDUCED_LOG_DIR,  "compare_results")
NON_IND_CMP   = os.path.join(NON_IND_LOG_DIR,  "compare_results")
PATHS_CMP     = os.path.join(PATHS_LOG_DIR,    "compare_results")

ALGORITHMS = ["induced", "non_induced", "paths", "pattern_finder"]

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDED_DENSITIES   = [3, 5]
GRAPH_DENSITIES_DEN3 = [5, 8, 10, 13, 15]
GRAPH_DENSITIES_DEN5 = [8, 10, 13, 15]
EQUAL_DEG_DENSITIES  = [3, 5, 8, 15]
COLOR_DISTRIBUTIONS  = ["uniform", "average", "rare"]
REAL_GRAPHS          = ["Mutagenicity", "DHFR-MD"]
NUM_VERSIONS         = 10   # j = 0..9
NUM_S_EMBEDDED       = 100  # experiments 1/2: 100 S per graph
NUM_S_TIMED          = 1000 # experiments 3/4/5/6: 1000 S per graph
EMBEDDED_SKIP        = 10   # S_1..S_10 are always embedded → exclude from FP


def missing(path):
    """Return True and print warning if path does not exist."""
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return True
    return False


# ── Per-graph PASS/FAIL log parser ────────────────────────────────────────────

def _keyword_for_algo(algo):
    if algo == "induced":
        return "SUM"
    if algo == "non_induced":
        return "SUM"
    if algo == "paths":
        return "PATH"
    raise ValueError(f"Unknown algo for keyword: {algo}")


def parse_per_graph_log(path, algo, num_s, warn_embedded=True):
    """
    Parse a per-graph PASS/FAIL log.

    Returns:
        dict: {s_index (int): bool passed}  — only indices 1..num_s

    Crashes with loud error if any S in 1..10 is marked FAIL
    (i.e. algorithm incorrectly says embedded S is not in G).
    """
    if missing(path):
        return None

    keyword = _keyword_for_algo(algo)
    results = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            # e.g. "2026-03-04 16:10:29,226 - SUM PASS S_1"
            #      "2026-03-04 16:10:29,226 - PATH FAIL S_42"
            m = re.search(rf'{keyword} (PASS|FAIL) S_(\d+)', line)
            if m:
                verdict = m.group(1) == "PASS"
                idx     = int(m.group(2))
                results[idx] = verdict

    if warn_embedded:
        for idx in range(1, EMBEDDED_SKIP + 1):
            if idx in results and not results[idx]:
                print(f"CRITICAL ERROR: {path} — S_{idx} (embedded) marked as FAIL by {algo}. "
                      f"This should never happen.", file=sys.stderr)
                sys.exit(1)

    return results


def count_false_positives(results, total_s, skip=EMBEDDED_SKIP):
    """
    False positives = S passes check but is NOT in G.
    S_1..S_skip are embedded (always in G) → excluded.
    Remaining S_(skip+1)..S_total_s: a PASS is a false positive.
    """
    if results is None:
        return None
    return sum(1 for i in range(skip + 1, total_s + 1)
               if results.get(i, False))


# ── Summary log parsers ───────────────────────────────────────────────────────

def parse_summary_induced_non_induced(path):
    """
    Parse summary_induced_motifs.log / summary_non_induced_motifs.log
    or summary_3.log / summary_5.log in compare_results/.

    Returns dict: {graph_file_name: fp_count}
    Also handles AVERAGE lines → {key_avg: float}
    """
    if missing(path):
        return {}

    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            # "2026-03-02 15:30:21,532 - g_den_5_embedded_den_3_uniform_0 | sum_only=0"
            m = re.search(r'- (.+?) \| sum_only=(\S+)', line)
            if m:
                results[m.group(1)] = _parse_num(m.group(2))
                continue
            # AVERAGE line
            m = re.search(r'- (.+?) \| AVERAGE sum_only=(\S+)', line)
            if m:
                results[m.group(1) + "__AVG"] = float(m.group(2))
    return results


def parse_summary_paths(path):
    """
    Parse path summary logs — handles three formats:
      path_only=N
      path_only_false_pos=N
    and AVERAGE lines.
    """
    if missing(path):
        return {}

    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.search(r'- (.+?) \| path_only(?:_false_pos)?=(\S+)', line)
            if m:
                key = m.group(1)
                val = _parse_num(m.group(2))
                if "AVERAGE" in line:
                    results[key + "__AVG"] = float(m.group(2))
                else:
                    results[key] = val
    return results


def _parse_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


# ── Timing log parsers ────────────────────────────────────────────────────────

def parse_time_log(path):
    """
    Parse timing log files.
    Format: "timestamp - <run_name> | G_compute_time=Xs"
            "timestamp - <run_name> | TOTAL_TIME=Xs"

    Returns dict: {run_name: {"G_time": float, "total_time": float}}
    """
    if missing(path):
        return {}

    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.search(r'- (.+?) \| G_compute_time=([0-9.]+)s', line)
            if m:
                name = m.group(1)
                results.setdefault(name, {})["G_time"] = float(m.group(2))
                continue
            m = re.search(r'- (.+?) \| TOTAL_TIME=([0-9.]+)s', line)
            if m:
                name = m.group(1)
                results.setdefault(name, {})["total_time"] = float(m.group(2))
                continue
            m = re.search(r'- (.+?) \| S_total_time=([0-9.]+)s', line)
            if m:
                name = m.group(1)
                results.setdefault(name, {})["S_time"] = float(m.group(2))
    return results


def parse_s_time_log(path):
    """
    Parse s_times_*.log files.
    Returns dict: {run_name: {"G_time": float, "total_time": float}}
    Same format as parse_time_log.
    """
    return parse_time_log(path)


# ── Pattern finder parsers ────────────────────────────────────────────────────

def parse_pattern_summary(folder_path):
    """
    Parse PATTERN_SUMMARY.log in a pattern finder output folder.

    Returns:
        {
          "total_pattern_time": float,   # total seconds for pattern creation
          "avg_pattern_time":   float,
          "num_patterns":       int,
        }
    """
    path = os.path.join(folder_path, "PATTERN_SUMMARY.log")
    if missing(path):
        return None

    total_time = None
    avg_time   = None
    num_runs   = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'Run (\d+): (SUCCESS|FAILURE)', line)
            if m:
                num_runs = max(num_runs, int(m.group(1)))
                continue
            m = re.search(r'Total pattern finding time: ([0-9.]+) seconds', line)
            if m:
                total_time = float(m.group(1))
                continue
            m = re.search(r'Average time per pattern: ([0-9.]+) seconds', line)
            if m:
                avg_time = float(m.group(1))

    if total_time is None:
        raise ValueError(f"PATTERN_SUMMARY.log missing 'Total pattern finding time' in {folder_path}")

    return {
        "total_pattern_time": total_time,
        "avg_pattern_time":   avg_time,
        "num_patterns":       num_runs,
    }


def parse_pattern_results_file(path, folder_name):
    """
    Parse a RESULTS_*.log file from pattern finder.

    Returns:
        {
          "not_in_g":        set of S indices found not in G,
          "search_time":     float (seconds),
          "false_positives": int,
        }

    Crashes loudly if any S in 1..10 appears in not_in_g.
    """
    if missing(path):
        return None

    not_in_g    = set()
    search_time = None

    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        m = re.search(r'Total subgraph testing time: ([0-9.]+) seconds', line)
        if m:
            search_time = float(m.group(1))
            continue
        if re.match(r'^\d+$', line):
            not_in_g.add(int(line))

    if search_time is None:
        raise ValueError(
            f"RESULTS file missing 'Total subgraph testing time': {path}"
        )

    # Check embedded S's were not incorrectly flagged
    for idx in range(1, EMBEDDED_SKIP + 1):
        if idx in not_in_g:
            print(
                f"CRITICAL ERROR IN PATTERN FINDER: {path} — "
                f"S_{idx} (embedded, should be in G) was flagged as NOT in G. "
                f"THIS SHOULD NEVER HAPPEN.",
                file=sys.stderr
            )
            sys.exit(1)

    false_positives = (NUM_S_TIMED - EMBEDDED_SKIP) - len(not_in_g)

    return {
        "not_in_g":        not_in_g,
        "search_time":     search_time,
        "false_positives": false_positives,
    }


def parse_pattern_folder(folder_path):
    """
    Parse an entire pattern finder output folder.

    Returns:
        {
          "pattern_summary":  {...},  # from PATTERN_SUMMARY.log
          "results": {
              graph_key: {
                  "not_in_g": set,
                  "search_time": float,
                  "false_positives": int,
              }
          }
        }
    """
    if not os.path.isdir(folder_path):
        print(f"[MISSING FOLDER] {folder_path}")
        return None

    summary = parse_pattern_summary(folder_path)

    results = {}
    for fname in os.listdir(folder_path):
        if not fname.startswith("RESULTS_") or not fname.endswith(".log"):
            continue
        graph_key = fname[len("RESULTS_"):-len(".log")]
        fpath     = os.path.join(folder_path, fname)
        results[graph_key] = parse_pattern_results_file(fpath, folder_path)

    return {"pattern_summary": summary, "results": results}


def get_pattern_folder_name(color_dist, density):
    """Map (color_dist, density) → OUTPUT folder name."""
    return f"OUTPUT_input_color_{color_dist}_deg_{density}"


# ── High-level data loaders ───────────────────────────────────────────────────

def load_equal_deg_fp(algo):
    """
    Load false positive counts for equal-density experiment (exp 3).
    Returns dict: {(color_dist, density): fp_count or None}
    """
    out = {}
    for density in EQUAL_DEG_DENSITIES:
        for color in COLOR_DISTRIBUTIONS:
            key = (color, density)
            run_name = f"color_{color}_deg_{density}"

            if algo in ("induced", "non_induced"):
                algo_prefix = "induced" if algo == "induced" else "non_induced"
                log = os.path.join(
                    INDUCED_CMP if algo == "induced" else NON_IND_CMP,
                    f"{algo_prefix}_color_{color}_deg_{density}.log"
                )
                res = parse_per_graph_log(log, algo, NUM_S_TIMED)
                out[key] = count_false_positives(res, NUM_S_TIMED) if res else None

            elif algo == "paths":
                log = os.path.join(PATHS_CMP, f"paths_color_{color}_deg_{density}.log")
                res = parse_per_graph_log(log, "paths", NUM_S_TIMED)
                out[key] = count_false_positives(res, NUM_S_TIMED) if res else None

            elif algo == "pattern_finder":
                folder = os.path.join(
                    PATTERN_RESULT_DIR,
                    get_pattern_folder_name(color, density)
                )
                data = parse_pattern_folder(folder)
                if data is None:
                    out[key] = None
                else:
                    # G_induced and G_non_induced — use G_induced as the equal-deg G
                    r = data["results"].get("G_induced")
                    out[key] = r["false_positives"] if r else None

    return out


def load_embedded_fp(algo, embedded_den):
    """
    Load false positive counts for embedded experiments (exp 1/2).
    10 graphs × 100 S, averaged.
    Returns dict: {(graph_density, color_dist): avg_fp or None}
    """
    graph_dens = GRAPH_DENSITIES_DEN3 if embedded_den == 3 else GRAPH_DENSITIES_DEN5
    out = {}
    for g_den in graph_dens:
        for color in COLOR_DISTRIBUTIONS:
            fps = []
            for j in range(NUM_VERSIONS):
                name = f"g_den_{g_den}_embedded_den_{embedded_den}_{color}_{j}"

                if algo == "induced":
                    log = os.path.join(INDUCED_LOG_DIR, f"{name}.log")
                elif algo == "non_induced":
                    log = os.path.join(NON_IND_LOG_DIR, f"{name}.log")
                elif algo == "paths":
                    log = os.path.join(PATHS_LOG_DIR, f"{name}.log")
                else:
                    log = None

                if algo == "pattern_finder":
                    folder = os.path.join(
                        PATTERN_RESULT_DIR,
                        get_pattern_folder_name(color, embedded_den)
                    )
                    data = parse_pattern_folder(folder)
                    if data is None:
                        fps.append(None)
                    else:
                        r = data["results"].get(name)
                        fps.append(r["false_positives"] if r else None)
                else:
                    res = parse_per_graph_log(log, algo, NUM_S_TIMED)
                    fp  = count_false_positives(res, NUM_S_TIMED) if res else None
                    fps.append(fp)

            valid = [x for x in fps if x is not None]
            out[(g_den, color)] = sum(valid) / len(valid) if valid else None

    return out


def load_timed_embedded_fp(algo, embedded_den):
    """
    Load false positive counts for timed embedded experiments (exp 4/5).
    1 graph × 1000 S (graph index _0).
    Returns dict: {(graph_density, color_dist): fp or None}
    """
    graph_dens = GRAPH_DENSITIES_DEN3 if embedded_den == 3 else GRAPH_DENSITIES_DEN5
    out = {}
    for g_den in graph_dens:
        for color in COLOR_DISTRIBUTIONS:
            name = f"g_den_{g_den}_embedded_den_{embedded_den}_{color}_0"

            if algo == "induced":
                log = os.path.join(INDUCED_CMP, f"{name}.log")
            elif algo == "non_induced":
                log = os.path.join(NON_IND_CMP, f"{name}.log")
            elif algo == "paths":
                log = os.path.join(PATHS_CMP, f"{name}.log")
            elif algo == "pattern_finder":
                folder = os.path.join(
                    PATTERN_RESULT_DIR,
                    get_pattern_folder_name(color, embedded_den)
                )
                data = parse_pattern_folder(folder)
                if data is None:
                    out[(g_den, color)] = None
                    continue
                r = data["results"].get(name)
                out[(g_den, color)] = r["false_positives"] if r else None
                continue
            else:
                out[(g_den, color)] = None
                continue

            res = parse_per_graph_log(log, algo, NUM_S_TIMED)
            out[(g_den, color)] = count_false_positives(res, NUM_S_TIMED) if res else None

    return out


def load_equal_deg_timing(algo):
    """
    Load G_time and total_time for equal-density experiment.
    Returns dict: {(color_dist, density): {"G_time": float, "total_time": float} or None}
    """
    if algo == "induced":
        times = parse_time_log(os.path.join(INDUCED_CMP, "equal_degs_times_induced_motifs.log"))
        prefix = "color"
    elif algo == "non_induced":
        times = parse_time_log(os.path.join(NON_IND_CMP, "equal_degs_times_non_induced_motifs.log"))
        prefix = "color"
    elif algo == "paths":
        times = parse_time_log(os.path.join(PATHS_CMP, "equal_degs_times_paths.log"))
        prefix = "color"
    elif algo == "pattern_finder":
        out = {}
        for density in EQUAL_DEG_DENSITIES:
            for color in COLOR_DISTRIBUTIONS:
                folder = os.path.join(
                    PATTERN_RESULT_DIR,
                    get_pattern_folder_name(color, density)
                )
                data = parse_pattern_folder(folder)
                if data is None:
                    out[(color, density)] = None
                    continue
                ps = data["pattern_summary"]
                r  = data["results"].get("G_induced")
                if r is None:
                    out[(color, density)] = None
                    continue
                preprocess = ps["total_pattern_time"]
                search     = r["search_time"]
                out[(color, density)] = {
                    "G_time":     search,
                    "total_time": preprocess + search,
                    "preprocess": preprocess,
                }
        return out
    else:
        return {}

    out = {}
    for density in EQUAL_DEG_DENSITIES:
        for color in COLOR_DISTRIBUTIONS:
            run = f"color_{color}_deg_{density}"
            out[(color, density)] = times.get(run)
    return out


def load_embedded_timing(algo, embedded_den):
    """
    Load G_time and total_time for timed embedded experiments (exp 4/5).
    Returns dict: {(graph_density, color_dist): {"G_time": float, "total_time": float} or None}
    """
    graph_dens = GRAPH_DENSITIES_DEN3 if embedded_den == 3 else GRAPH_DENSITIES_DEN5

    if algo == "induced":
        fname = f"times_{embedded_den}.log"
        times = parse_time_log(os.path.join(INDUCED_CMP, fname))
    elif algo == "non_induced":
        fname = f"times_{embedded_den}.log"
        times = parse_time_log(os.path.join(NON_IND_CMP, fname))
    elif algo == "paths":
        fname = f"times_paths_{embedded_den}.log"
        times = parse_time_log(os.path.join(PATHS_CMP, fname))
    elif algo == "pattern_finder":
        out = {}
        for g_den in graph_dens:
            for color in COLOR_DISTRIBUTIONS:
                folder = os.path.join(
                    PATTERN_RESULT_DIR,
                    get_pattern_folder_name(color, embedded_den)
                )
                data = parse_pattern_folder(folder)
                if data is None:
                    out[(g_den, color)] = None
                    continue
                name = f"g_den_{g_den}_embedded_den_{embedded_den}_{color}_0"
                r  = data["results"].get(name)
                ps = data["pattern_summary"]
                if r is None:
                    out[(g_den, color)] = None
                    continue
                out[(g_den, color)] = {
                    "G_time":     r["search_time"],
                    "total_time": ps["total_pattern_time"] + r["search_time"],
                    "preprocess": ps["total_pattern_time"],
                }
        return out
    else:
        return {}

    out = {}
    for g_den in graph_dens:
        for color in COLOR_DISTRIBUTIONS:
            name = f"g_den_{g_den}_embedded_den_{embedded_den}_{color}_0"
            out[(g_den, color)] = times.get(name)
    return out


def load_real_graph_fp(algo):
    """
    Returns dict: {graph_name: fp_count or None}
    """
    out = {}
    for gname in REAL_GRAPHS:
        if algo == "induced":
            log = os.path.join(INDUCED_CMP, f"{gname}.log")
            res = parse_per_graph_log(log, "induced", NUM_S_TIMED,
                                      warn_embedded=False)
            # real graphs: no embedded S's, all PASS = false positive
            out[gname] = (sum(1 for v in res.values() if v)
                          if res else None)
        elif algo == "non_induced":
            log = os.path.join(NON_IND_CMP, f"{gname}.log")
            res = parse_per_graph_log(log, "non_induced", NUM_S_TIMED,
                                      warn_embedded=False)
            out[gname] = (sum(1 for v in res.values() if v)
                          if res else None)
        elif algo == "paths":
            log = os.path.join(PATHS_CMP, f"paths_{gname}.log")
            res = parse_per_graph_log(log, "paths", NUM_S_TIMED,
                                      warn_embedded=False)
            out[gname] = (sum(1 for v in res.values() if v)
                          if res else None)
        elif algo == "pattern_finder":
            folder = os.path.join(PATTERN_RESULT_DIR, "OUTPUT_NCI109")
            data   = parse_pattern_folder(folder)
            if data is None:
                out[gname] = None
            else:
                r = data["results"].get(gname)
                if r is None:
                    print(f"[MISSING] Pattern finder result for {gname} in OUTPUT_NCI109")
                    out[gname] = None
                else:
                    # real graphs: no embedded S's guaranteed → fp = 1000 - not_in_g
                    out[gname] = NUM_S_TIMED - len(r["not_in_g"])
    return out


def load_real_graph_timing(algo):
    """
    Returns dict: {graph_name: {"G_time": float, "total_time": float} or None}
    """
    out = {}
    if algo == "induced":
        times = parse_time_log(os.path.join(INDUCED_CMP, "real_graphs_times.log"))
        for gname in REAL_GRAPHS:
            out[gname] = times.get(gname)
    elif algo == "non_induced":
        times = parse_time_log(os.path.join(NON_IND_CMP, "real_graphs_times.log"))
        for gname in REAL_GRAPHS:
            out[gname] = times.get(gname)
    elif algo == "paths":
        times = parse_time_log(os.path.join(PATHS_CMP, "real_graphs_paths_times.log"))
        for gname in REAL_GRAPHS:
            out[gname] = times.get(gname)
    elif algo == "pattern_finder":
        folder = os.path.join(PATTERN_RESULT_DIR, "OUTPUT_NCI109")
        data   = parse_pattern_folder(folder)
        for gname in REAL_GRAPHS:
            if data is None:
                out[gname] = None
            else:
                r  = data["results"].get(gname)
                ps = data["pattern_summary"]
                if r is None:
                    out[gname] = None
                else:
                    out[gname] = {
                        "G_time":     r["search_time"],
                        "total_time": ps["total_pattern_time"] + r["search_time"],
                        "preprocess": ps["total_pattern_time"],
                    }
    return out


def load_s_timing(algo, density):
    """
    Load S computation timing (exp 7).
    Returns dict: {run_name: {"G_time": float, "total_time": float} or None}
    Only average over first 100 S's worth of timing for consistency.
    """
    if algo == "induced":
        fname = f"s_times_induced_{density}.log" if density not in (3, 5) else \
                ("s_times_induced.log" if density == 3 else "s_times_induced_5.log")
        path  = os.path.join(INDUCED_CMP, fname)
    elif algo == "non_induced":
        fname = f"s_times_non_induced_{density}.log" if density not in (3, 5) else \
                ("s_times_non_induced.log" if density == 3 else "s_times_non_induced_5.log")
        path  = os.path.join(NON_IND_CMP, fname)
    elif algo == "paths":
        fname = f"s_times_paths_deg_{density}.log"
        path  = os.path.join(PATHS_CMP, fname)
    else:
        return {}

    return parse_s_time_log(path)


def load_per_s_results_for_layering(embedded_den, g_den, color, j=0):
    """
    Load per-S PASS/FAIL for all algorithms for a single graph.
    Used for layered/overlap analysis.
    Returns dict: {algo: {s_idx: bool}}
    """
    name   = f"g_den_{g_den}_embedded_den_{embedded_den}_{color}_{j}"
    result = {}

    # induced
    log = os.path.join(INDUCED_CMP, f"{name}.log")
    result["induced"] = parse_per_graph_log(log, "induced", NUM_S_TIMED)

    # non_induced
    log = os.path.join(NON_IND_CMP, f"{name}.log")
    result["non_induced"] = parse_per_graph_log(log, "non_induced", NUM_S_TIMED)

    # paths
    log = os.path.join(PATHS_CMP, f"{name}.log")
    result["paths"] = parse_per_graph_log(log, "paths", NUM_S_TIMED)

    # pattern_finder
    folder = os.path.join(
        PATTERN_RESULT_DIR,
        get_pattern_folder_name(color, embedded_den)
    )
    data = parse_pattern_folder(folder)
    if data and data["results"].get(name):
        not_in_g = data["results"][name]["not_in_g"]
        # Convert to same format: True = PASS (FP), False = FAIL (correctly rejected)
        pf_results = {}
        for i in range(1, NUM_S_TIMED + 1):
            pf_results[i] = (i not in not_in_g)
        result["pattern_finder"] = pf_results
    else:
        result["pattern_finder"] = None

    return result
