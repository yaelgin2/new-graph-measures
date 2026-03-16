import subprocess
import os

# ---- paths ----

SCRIPT = "/home/cohent59/new-graph-measures/local_tests/find_all_paths_length_5/s_to_leaves.py"

INPUT_DIR = "/home/cohent59/new-graph-measures/local_tests/input_color_rare_deg_3"

OUTPUT_ROOT = "/home/cohent59/new-graph-measures/local_tests/find_all_paths_length_5/input_color_rare_deg_3_paths"


os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---- iterate S_1 ... S_1000 ----

for i in range(1, 1001):

    input_file = os.path.join(INPUT_DIR, f"S_{i}.json")

    if not os.path.exists(input_file):
        print(f"Skipping missing {input_file}")
        continue

    output_folder = os.path.join(OUTPUT_ROOT, f"S_{i}")
    os.makedirs(output_folder, exist_ok=True)

    cmd = [
        "python",
        SCRIPT,
        "--sub_file",
        input_file,
        "--folder",
        output_folder,
    ]

    print("Running:", " ".join(cmd))

    subprocess.run(cmd, check=True)

print("Done.")
