"""
run_all.py
Runs all table and plot generation scripts in local_tests/plot_results/.
Saves a status log: local_tests/plot_results/run_status.log

Exit codes from scripts:
  0 = success (may have missing files)
  1 = unexpected crash
  2 = FORMAT ERROR in a log file — entire run halts immediately

For every script the exact list of missing log files is printed and logged.
"""
import os
import sys
import subprocess
import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_LOG  = os.path.join(SCRIPTS_DIR, "run_status.log")

SCRIPTS = [
    "plot_01_fp_bar_embedded3.py",
    "plot_02_fp_bar_embedded5.py",
    "plot_03_fp_heatmap.py",
    "plot_04_layered.py",
    "plot_05_equal_deg.py",
    "plot_06_real_graphs.py",
    "plot_07_timing.py",
    "plot_08_timing_scaling.py",
    "plot_09_s_timing.py",
]

EXIT_FORMAT_ERROR = 2


def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr


def parse_script_output(stdout, stderr):
    """
    Extract structured info from a script's output.

    Returns:
        missing_files: list of paths (from [MISSING_FILE] lines)
        has_missing_summary: bool (was [MISSING_SUMMARY] line present?)
        is_format_error: bool ([FORMAT_ERROR] marker found)
        other_stdout: list of stdout lines that are not marker lines
        other_stderr: list of stderr lines that are not marker lines
    """
    missing_files       = []
    has_missing_summary = False
    is_format_error     = False
    other_stdout        = []
    other_stderr        = []

    for line in stdout.splitlines():
        if line.startswith("[MISSING_FILE] "):
            missing_files.append(line[len("[MISSING_FILE] "):].strip())
        elif line.startswith("[MISSING_SUMMARY]"):
            has_missing_summary = True
        else:
            other_stdout.append(line)

    format_error_file = None
    for line in stderr.splitlines():
        if line.startswith("[FORMAT_ERROR]"):
            is_format_error = True
            # Extract "file=<path>" from the marker
            import re as _re
            _m = _re.search(r'file=([^ |]+)', line)
            if _m:
                format_error_file = _m.group(1).strip()
        else:
            other_stderr.append(line)

    return missing_files, has_missing_summary, is_format_error, format_error_file, other_stdout, other_stderr


def _divider():
    return "=" * 70


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines = [f"Run started: {timestamp}", _divider(), ""]

    completed         = []
    missing_data      = []
    failed_script     = None
    all_missing_files = []   # across all scripts, in order (deduped at summary)

    for script in SCRIPTS:
        sep = "-" * 55
        print(f"\n{sep}\nRunning: {script}")

        code, stdout, stderr = run_script(script)
        missing_files, has_summary, is_fmt_err, fmt_err_file, out_lines, err_lines = \
            parse_script_output(stdout, stderr)

        # ── Determine status ──────────────────────────────────────────────
        if code == EXIT_FORMAT_ERROR or (code != 0 and is_fmt_err):
            status = "FORMAT ERROR"
        elif code != 0:
            status = "FAILED"
        elif missing_files:
            status = "MISSING DATA"
        else:
            status = "COMPLETED"

        # ── Console output ────────────────────────────────────────────────
        print(f"  Status: {status}")

        if missing_files:
            all_missing_files.extend(missing_files)
            print(f"  Missing files ({len(missing_files)}):")
            for f in missing_files:
                print(f"    {f}")

        if not has_summary and code == 0:
            print("  WARNING: script exited 0 but [MISSING_SUMMARY] line was absent "
                  "(parse_logs may not have been imported correctly)")

        if code != 0:
            print("  STDERR:")
            for l in err_lines:
                print(f"    {l}")

        # ── Log lines ─────────────────────────────────────────────────────
        log_lines.append(f"[{status:15s}] {script}")
        if status == "FORMAT ERROR" and fmt_err_file:
            log_lines.append(f"  Could not parse log file: {fmt_err_file}")
        if missing_files:
            log_lines.append(f"  Missing files ({len(missing_files)}):")
            for f in missing_files:
                log_lines.append(f"    {f}")
        for l in out_lines:
            if l.strip():
                log_lines.append(f"  {l}")
        if code != 0:
            log_lines.append(f"  Exit code: {code}")
            log_lines.append(f"  STDERR:")
            for l in err_lines:
                log_lines.append(f"    {l}")
        log_lines.append("")

        # ── Track results ─────────────────────────────────────────────────
        if status in ("FORMAT ERROR", "FAILED"):
            failed_script = script
            _write_log(log_lines, completed, missing_data,
                       failed_script, all_missing_files, timestamp,
                       halted=True)
            if status == "FORMAT ERROR":
                print(f"\n{'!'*55}")
                print(f"FORMAT ERROR — halting entire run.")
                if fmt_err_file:
                    print(f"  Could not parse log file: {fmt_err_file}")
                print(f"  Triggered by script:    {script}")
                print(f"Fix the log file format before re-running.")
                print(f"Full details in: {STATUS_LOG}")
                print(f"{'!'*55}")
            else:
                print(f"\nScript {script} FAILED (exit {code}) — halting entire run.")
                print(f"Full details in: {STATUS_LOG}")
            _print_all_missing(all_missing_files)
            sys.exit(code)

        elif status == "MISSING DATA":
            missing_data.append(script)
        else:
            completed.append(script)

    # ── All scripts done ──────────────────────────────────────────────────
    _write_log(log_lines, completed, missing_data,
               failed_script=None, all_missing_files=all_missing_files,
               timestamp=timestamp, halted=False)

    print(f"\n{_divider()}")
    print(f"All scripts finished.")
    print(f"  Completed:    {len(completed)}")
    print(f"  Missing data: {len(missing_data)}")
    print(f"  Failed:       0")
    _print_all_missing(all_missing_files)
    print(f"\nStatus log: {STATUS_LOG}")


def _print_all_missing(all_missing_files):
    unique = sorted(set(all_missing_files))
    print(f"\n{_divider()}")
    print(f"ALL MISSING LOG FILES ({len(unique)} unique):")
    if unique:
        for f in unique:
            print(f"  {f}")
    else:
        print("  (none)")


def _write_log(log_lines, completed, missing_data, failed_script,
               all_missing_files, timestamp, halted):
    unique_missing = sorted(set(all_missing_files))

    summary = [
        _divider(),
        f"Run ended: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Halted early: {'YES — ' + failed_script if halted else 'no'}",
        "",
        f"COMPLETED     ({len(completed)}  scripts): " +
            (", ".join(completed) if completed else "none"),
        "",
        f"MISSING DATA  ({len(missing_data)} scripts): " +
            (", ".join(missing_data) if missing_data else "none"),
        "",
        f"FAILED:  " + (failed_script or "none"),
        "",
        _divider(),
        f"ALL MISSING LOG FILES ({len(unique_missing)} unique):",
    ]
    if unique_missing:
        for f in unique_missing:
            summary.append(f"  {f}")
    else:
        summary.append("  (none)")

    summary += [
        "",
        "NOTES:",
        "  • 'MISSING DATA' scripts still produced output — missing inputs assumed 0.",
        "  • Re-run after all log files exist to get fully accurate results.",
        "  • On FORMAT ERROR or FAILED the run was halted at that script.",
    ]

    with open(STATUS_LOG, "w") as fh:
        fh.write("\n".join(log_lines + summary) + "\n")


if __name__ == "__main__":
    main()