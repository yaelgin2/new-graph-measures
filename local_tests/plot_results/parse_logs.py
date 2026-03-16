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
    # parse_logs.py lives in local_tests/plot_results/
    # BASE_DIR resolves to local_tests/
    return os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

BASE_DIR          = get_base_dir()
INDUCED_LOG_DIR   = os.path.join(BASE_DIR, "induced",  "logs")
NON_IND_LOG_DIR   = os.path.join(BASE_DIR, "non_induced", "logs")
PATHS_LOG_DIR     = os.path.join(BASE_DIR, "find_all_paths_length_5", "logs")
# PROJECT_RUN_PATTERN is a sibling of the repo root (new-graph-measures)
# BASE_DIR = .../new-graph-measures/local_tests/
# BASE_DIR/.. = .../new-graph-measures/
# BASE_DIR/../.. = parent of the repo = where PROJECT_RUN_PATTERN lives
PATTERN_RESULT_DIR = os.path.realpath(os.path.join(BASE_DIR, "..", "..",
                                                    "PROJECT_RUN_PATTERN",
                                                    "pattern_finder", "results"))

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


# Global registry of missing files — populated as parsing runs
_missing_files = []

def missing(path):
    """Return True, print and record warning if path does not exist."""
    if not os.path.exists(path):
        msg = f"[MISSING] {path}"
        print(msg)
        _missing_files.append(path)
        return True
    return False

def get_missing_files():
    """Return a copy of all missing files encountered so far."""
    return list(_missing_files)

def clear_missing_files():
    """Reset the missing files registry (call before each script run)."""
    _missing_files.clear()


# ── Per-graph PASS/FAIL log parser ────────────────────────────────────────────

def _keyword_for_algo(algo):
    if algo == "induced":
        return "SUM"
    if algo == "non_induced":
        return "SUM"
    if algo == "paths":
        return "PATH"
    raise ValueError(f"Unknown algo for keyword: {algo}")


def parse_per_graph_log(path, algo, num_s, warn_embedded=True, s_start=1):
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

    # Two supported log formats:
    #   (1) "timestamp - SUM/PATH PASS/FAIL S_N"  — original per-graph logs (100 S)
    #   (2) "timestamp - S_N PASS/FAIL"            — compare_results logs (1000 S)
    result_line_pattern    = re.compile(rf'{keyword} (PASS|FAIL) S_(\d+)')
    keywordless_pattern    = re.compile(r'S_(\d+) (PASS|FAIL)')
    has_any_result = False

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if any(skip in line for skip in ["Starting run", "====", "Progress:"]):
                continue
            is_timestamp_line = re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+', line)
            m = result_line_pattern.search(line)
            if m:
                has_any_result = True
                results[int(m.group(2))] = (m.group(1) == "PASS")
            else:
                m2 = keywordless_pattern.search(line)
                if m2:
                    has_any_result = True
                    results[int(m2.group(1))] = (m2.group(2) == "PASS")
                elif is_timestamp_line:
                    pass  # summary/timing line — skip
                elif line:
                    raise ValueError(
                        f"FORMAT ERROR in {path}:\n"
                        f"  Unexpected line (not a timestamp log line and not empty):\n"
                        f"  >>> {line!r}\n"
                        f"  Expected: 'timestamp - {keyword} PASS/FAIL S_N' or 'timestamp - S_N PASS/FAIL'"
                    )

    if warn_embedded:
        for idx in range(1, EMBEDDED_SKIP + 1):
            if idx in results and not results[idx]:
                print(f"CRITICAL ERROR: {path} — S_{idx} (embedded) marked as FAIL by {algo}. "
                      f"This should never happen.", file=sys.stderr)
                sys.exit(1)

    # Warn about any expected S indices that are absent from the log
    missing_s = [i for i in range(s_start, s_start + num_s) if i not in results]
    if missing_s:
        ranges = []
        start = missing_s[0]
        end   = missing_s[0]
        for idx in missing_s[1:]:
            if idx == end + 1:
                end = idx
            else:
                ranges.append(f"S_{start}" if start == end else f"S_{start}..S_{end}")
                start = end = idx
        ranges.append(f"S_{start}" if start == end else f"S_{start}..S_{end}")
        msg = f"{path}: {len(missing_s)} S entries absent from log: {', '.join(ranges)}"
        print(f"[MISSING] {msg}")
        _missing_files.append(msg)

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

    Entries where either G_compute_time or TOTAL_TIME is absent (e.g. the run
    is still in progress) are excluded and a [MISSING] warning is printed so
    callers can treat them as None without KeyError.
    """
    if missing(path):
        return {}

    raw = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.search(r'- (.+?) \| G_compute_time=([0-9.]+)s', line)
            if m:
                name = m.group(1)
                raw.setdefault(name, {})["G_time"] = float(m.group(2))
                continue
            m = re.search(r'- (.+?) \| TOTAL(?:_BATCH)?_TIME=([0-9.]+)s', line)
            if m:
                name = m.group(1)
                raw.setdefault(name, {})["total_time"] = float(m.group(2))
                continue
            m = re.search(r'- (.+?) \| S_total_time=([0-9.]+)s', line)
            if m:
                name = m.group(1)
                raw.setdefault(name, {})["S_time"] = float(m.group(2))

    results = {}
    for name, times in raw.items():
        if "G_time" not in times or "total_time" not in times:
            missing_keys = [k for k in ("G_time", "total_time") if k not in times]
            msg = (f"{path}: run '{name}' is missing timing fields {missing_keys} "
                   f"— log may be incomplete (run still in progress?). Treating as N/A.")
            print(f"[MISSING] {msg}")
            _missing_files.append(f"{path} [{name}]")
        else:
            results[name] = times
    return results


def parse_s_time_log(path):
    """
    Parse s_times_*.log files.
    Format: "timestamp - /path/to/input_color_{color}_deg_{density} | S_N time=Xs"

    Returns dict: {
        "color_{color}_deg_{density}": {
            "avg_s_time": float,   # mean seconds per S (over all S entries found)
            "total_time": float,   # sum of all S times (alias used by callers)
        }
    }
    """
    if missing(path):
        return {}

    # Accumulate times per (color, density) key extracted from the input dir path
    from collections import defaultdict
    buckets = defaultdict(list)

    dir_key_pattern = re.compile(r'input_color_(\w+)_deg_(\d+)')
    time_pattern    = re.compile(r'S_\d+ time=([0-9.]+)s')

    with open(path) as f:
        for line in f:
            line = line.strip()
            dm = dir_key_pattern.search(line)
            tm = time_pattern.search(line)
            if dm and tm:
                key = f"color_{dm.group(1)}_deg_{dm.group(2)}"
                buckets[key].append(float(tm.group(1)))

    results = {}
    for key, times in buckets.items():
        avg = sum(times) / len(times)
        results[key] = {
            "avg_s_time": avg,
            "total_time": avg,   # callers use total_time; for S logs this is avg per S
        }
    return results


# ── Pattern finder parsers ────────────────────────────────────────────────────

def _sum_pattern_times_from_logs(folder_path):
    """
    Fallback: sum 'Time taken: X seconds' from every logs/run_N_output.log
    when PATTERN_SUMMARY.log is missing the total.

    Returns (total_time, num_patterns) or raises ValueError if logs/ is empty/absent.
    """
    logs_dir = os.path.join(folder_path, "logs")
    if not os.path.isdir(logs_dir):
        raise ValueError(
            f"PATTERN_SUMMARY.log has no 'Total pattern finding time' AND "
            f"logs/ folder is absent — cannot compute preprocessing time for {folder_path}"
        )

    run_files = sorted(
        f for f in os.listdir(logs_dir)
        if re.match(r'run_\d+_output\.log', f)
    )
    if not run_files:
        raise ValueError(
            f"PATTERN_SUMMARY.log has no 'Total pattern finding time' AND "
            f"logs/ folder contains no run_N_output.log files in {folder_path}"
        )

    total = 0.0
    for fname in run_files:
        fpath   = os.path.join(logs_dir, fname)
        content = open(fpath).read()
        m = re.search(r'Time taken: ([0-9.]+) seconds', content)
        if m is None:
            raise ValueError(
                f"FORMAT ERROR in {fpath}:\n"
                f"  Could not find 'Time taken: X seconds' line.\n"
                f"  Expected it near the end of each run_N_output.log."
            )
        total += float(m.group(1))

    print(f"  [INFO] Computed preprocessing time from {len(run_files)} run logs "
          f"in {logs_dir}: {total:.3f}s")
    return total, len(run_files)


def parse_pattern_summary(folder_path):
    """
    Parse PATTERN_SUMMARY.log in a pattern finder output folder.

    If the file exists but is missing the 'Total pattern finding time' line
    (e.g. the run was interrupted before the summary was written), falls back
    to summing 'Time taken: X seconds' from every logs/run_N_output.log.

    Returns:
        {
          "total_pattern_time": float,   # total seconds for pattern creation
          "avg_pattern_time":   float or None,
          "num_patterns":       int,
          "time_source":        "summary" | "run_logs",  # where the time came from
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

    time_source = "summary"
    if total_time is None:
        # PATTERN_SUMMARY.log exists but total line is absent — fall back to run logs
        print(f"  [WARN] PATTERN_SUMMARY.log in {folder_path} has no "
              f"'Total pattern finding time' — summing from run logs instead.")
        total_time, run_log_count = _sum_pattern_times_from_logs(folder_path)
        time_source = "run_logs"
        if num_runs == 0:
            num_runs = run_log_count
        if avg_time is None and num_runs > 0:
            avg_time = total_time / num_runs

    return {
        "total_pattern_time": total_time,
        "avg_pattern_time":   avg_time,
        "num_patterns":       num_runs,
        "time_source":        time_source,
    }


def parse_pattern_results_file(path, folder_name, warn_embedded=True):
    """
    Parse a RESULTS_*.log file from pattern finder.

    Args:
        warn_embedded: if True (default), halts if any S in 1..10 is in not_in_g
                       (they are embedded so must be in G).
                       Set False for real graphs where S1-S10 are NOT embedded
                       and may legitimately be absent from G.

    Returns:
        {
          "not_in_g":        set of S indices found not in G,
          "search_time":     float (seconds),
          "false_positives": int,
        }
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

    if warn_embedded:
        for idx in range(1, EMBEDDED_SKIP + 1):
            if idx in not_in_g:
                print(
                    f"CRITICAL ERROR IN PATTERN FINDER: {path} — "
                    f"S_{idx} (embedded, should be in G) was flagged as NOT in G. "
                    f"THIS SHOULD NEVER HAPPEN.",
                    file=sys.stderr
                )
                sys.exit(1)
        # FP = S's that passed (were not rejected) among non-embedded S's only
        false_positives = (NUM_S_TIMED - EMBEDDED_SKIP) - len(not_in_g)
    else:
        # Real graphs: S1-S10 are NOT embedded, all 1000 S's are candidates.
        # FP = S's that passed (not in not_in_g) = total S's minus those correctly rejected.
        false_positives = NUM_S_TIMED - len(not_in_g)

    return {
        "not_in_g":        not_in_g,
        "search_time":     search_time,
        "false_positives": false_positives,
    }


def parse_pattern_folder(folder_path, warn_embedded=True):
    """
    Parse an entire pattern finder output folder.

    Args:
        warn_embedded: passed through to parse_pattern_results_file.
                       Set False for real-graph folders (OUTPUT_NCI109) where
                       S1-S10 are not embedded and may legitimately be absent from G.

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
        msg = f"[MISSING] {folder_path}"
        print(msg)
        _missing_files.append(folder_path)
        return None

    summary = parse_pattern_summary(folder_path)

    results = {}
    for fname in os.listdir(folder_path):
        if not fname.startswith("RESULTS_") or not fname.endswith(".log"):
            continue
        graph_key = fname[len("RESULTS_"):-len(".log")]
        fpath     = os.path.join(folder_path, fname)
        results[graph_key] = parse_pattern_results_file(fpath, folder_path,
                                                        warn_embedded=warn_embedded)

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
                    # These logs contain 100 S's (not 1000) — pass correct num_s
                    res = parse_per_graph_log(log, algo, 100)
                    fp  = count_false_positives(res, 100) if res else None
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
                                      warn_embedded=False, s_start=0)
            # real graphs: S's are 0-indexed; all PASS = false positive
            out[gname] = (sum(1 for v in res.values() if v)
                          if res else None)
        elif algo == "non_induced":
            log = os.path.join(NON_IND_CMP, f"{gname}.log")
            res = parse_per_graph_log(log, "non_induced", NUM_S_TIMED,
                                      warn_embedded=False, s_start=0)
            out[gname] = (sum(1 for v in res.values() if v)
                          if res else None)
        elif algo == "paths":
            log = os.path.join(PATHS_CMP, f"paths_{gname}.log")
            res = parse_per_graph_log(log, "paths", NUM_S_TIMED,
                                      warn_embedded=False, s_start=0)
            out[gname] = (sum(1 for v in res.values() if v)
                          if res else None)
        elif algo == "pattern_finder":
            folder = os.path.join(PATTERN_RESULT_DIR, "OUTPUT_NCI109")
            data   = parse_pattern_folder(folder, warn_embedded=False)
            if data is None:
                out[gname] = None
            else:
                r = data["results"].get(gname)
                if r is None:
                    msg = f"[MISSING] Pattern finder result key '{gname}' in {os.path.join(PATTERN_RESULT_DIR, 'OUTPUT_NCI109')}"
                    print(msg)
                    _missing_files.append(msg)
                    out[gname] = None
                else:
                    # real graphs: fp comes from parse_pattern_results_file (warn_embedded=False)
                    out[gname] = r["false_positives"]
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
        data   = parse_pattern_folder(folder, warn_embedded=False)
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
        fname = f"s_times_induced_{density}.log"
        path  = os.path.join(INDUCED_CMP, fname)
    elif algo == "non_induced":
        fname = f"s_times_non_induced_{density}.log"
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

# ── Script runner — used by every table/plot script ──────────────────────────

EXIT_FORMAT_ERROR = 2   # exit code for log format problems
EXIT_CRASH        = 1   # exit code for any other exception

def run_script_main(main_fn):
    """
    Wrapper called by every table/plot script instead of plain main().

    Behaviour:
      - Clears the missing-files registry before running.
      - On ValueError containing 'FORMAT ERROR': prints full error,
        prints [FORMAT_ERROR] marker, exits with code 2 so run_all.py
        halts the entire run immediately.
      - On any other exception: prints traceback, exits with code 1
        so run_all.py also halts.
      - On success: prints [MISSING_SUMMARY] line listing every missing
        file that was encountered (empty list if none), then exits 0.
    """
    import traceback
    clear_missing_files()
    try:
        main_fn()
    except ValueError as e:
        if "FORMAT ERROR" in str(e):
            # Extract the bad filename from the error message (first line after "FORMAT ERROR in ")
            import re as _re
            _fname_match = _re.search(r'FORMAT ERROR in ([^:]+):', str(e))
            _bad_file = _fname_match.group(1).strip() if _fname_match else "(unknown file)"
            print(f"\n{'!'*60}", file=sys.stderr)
            print(f"FORMAT ERROR — unexpected content in log file:", file=sys.stderr)
            print(f"  FILE: {_bad_file}", file=sys.stderr)
            print(f"{'─'*60}", file=sys.stderr)
            print(str(e), file=sys.stderr)
            print(f"{'!'*60}", file=sys.stderr)
            print(f"[FORMAT_ERROR] file={_bad_file} | {e}", file=sys.stderr)
            # Still emit the missing summary before dying
            _print_missing_summary()
            sys.exit(EXIT_FORMAT_ERROR)
        else:
            traceback.print_exc()
            _print_missing_summary()
            sys.exit(EXIT_CRASH)
    except Exception:
        traceback.print_exc()
        _print_missing_summary()
        sys.exit(EXIT_CRASH)

    _print_missing_summary()


def _print_missing_summary():
    files = get_missing_files()
    # Always emit this line — run_all.py parses it
    print(f"[MISSING_SUMMARY] count={len(files)}")
    for f in files:
        print(f"[MISSING_FILE] {f}")