"""
run_all.py
Runs all table and plot generation scripts in local_tests/plot_results/.
Saves a status log: local_tests/plot_results/run_status.log

For each script, logs whether it:
  - COMPLETED (output file written successfully)
  - MISSING DATA (completed but some inputs were missing — output still generated)
  - FAILED (exception raised)
"""
import os
import sys
import subprocess
import datetime
import importlib.util

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_LOG  = os.path.join(SCRIPTS_DIR, "run_status.log")

SCRIPTS = [
    "table_01_fp_embedded3.py",
    "table_02_fp_embedded5.py",
    "table_03_fp_equal_deg.py",
    "table_04_layered.py",
    "table_05_pairwise.py",
    "table_06_real_graphs.py",
    "table_07_timing_equal_deg.py",
    "table_08_timing_embedded.py",
    "table_09_s_timing.py",
    "plot_01_fp_bar_embedded3.py",
    "plot_02_fp_bar_embedded5.py",
    "plot_03_fp_heatmap.py",
    "plot_04_layered.py",
    "plot_05_fp_vs_density.py",
    "plot_06_real_graphs.py",
    "plot_07_timing.py",
    "plot_08_timing_scaling.py",
    "plot_09_s_timing.py",
    "plot_10_efficiency_frontier.py",
    "plot_11_color_effect.py",
]

def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"Run at: {timestamp}", "=" * 60, ""]

    completed    = []
    missing_data = []
    failed       = []

    for script in SCRIPTS:
        print(f"\n{'='*50}\nRunning: {script}")
        code, stdout, stderr = run_script(script)

        has_missing = "[MISSING]" in stdout or "[MISSING]" in stderr
        if code != 0:
            status = "FAILED"
            failed.append(script)
            print(f"  FAILED:\n{stderr}")
        elif has_missing:
            status = "MISSING DATA"
            missing_data.append(script)
            print(f"  MISSING DATA (output generated with 0s for missing files)")
        else:
            status = "COMPLETED"
            completed.append(script)
            print(f"  OK")

        lines.append(f"[{status:15s}] {script}")
        if stdout.strip():
            for l in stdout.strip().splitlines():
                lines.append(f"              {l}")
        if code != 0 and stderr.strip():
            for l in stderr.strip().splitlines()[:10]:
                lines.append(f"  ERROR: {l}")
        lines.append("")

    lines += [
        "=" * 60,
        f"COMPLETED    ({len(completed)}): " + ", ".join(completed),
        "",
        f"MISSING DATA ({len(missing_data)}): " + ", ".join(missing_data),
        "",
        f"FAILED       ({len(failed)}): "    + (", ".join(failed) if failed else "none"),
        "",
        "NOTE: 'MISSING DATA' outputs are final plots/tables but use 0 for",
        "      missing log files. Re-run after all log files are available",
        "      to get fully accurate results.",
    ]

    with open(STATUS_LOG, "w") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*60}")
    print(f"Status log written to: {STATUS_LOG}")
    print(f"Completed:    {len(completed)}")
    print(f"Missing data: {len(missing_data)}")
    print(f"Failed:       {len(failed)}")

if __name__ == "__main__":
    main()
