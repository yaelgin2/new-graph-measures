"""
Compare FANMOD+ results to VDMC (induced colored motif counts).

VDMC pickle format: {node_id: {colored_motif_id: count}}
  - counts are per-node participation counts
  - each 4-node motif is counted once per member node → divide sum by 4

FANMOD+ json format: {str(colored_motif_id): count}
  - counts are global (each motif counted once)
"""

import os
import json
import pickle

FANMOD_DIR = "local_tests/other_algorithms_tests/fanmod_plus"
VDMC_DIR   = "/home/cohent59/new-graph-measures/local_tests/induced/cache"

DENSITIES     = [5, 8, 10, 13, 15]
DISTRIBUTIONS = ["uniform", "average", "rare"]

MOTIF_SIZE = 4  # divide VDMC node-sums by this

def load_fanmod(den, dist):
    path = os.path.join(FANMOD_DIR, f"g_den_{den}_embedded_den_3_{dist}_0.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def load_vdmc(den, dist):
    path = os.path.join(VDMC_DIR, f"g_den_{den}_embedded_den_3_{dist}_0.pkl")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        per_node = pickle.load(f)
    # Aggregate: sum counts across all nodes, divide by MOTIF_SIZE
    totals = {}
    for node_counts in per_node.values():
        for motif_id, cnt in node_counts.items():
            totals[motif_id] = totals.get(motif_id, 0) + cnt
    # Each motif counted once per node (4 nodes per motif)
    return {mid: total // MOTIF_SIZE for mid, total in totals.items()}

def compare(fanmod, vdmc, label):
    all_motifs = set(fanmod.keys()) | set(vdmc.keys())
    mismatches = []
    for mid in sorted(all_motifs):
        fc = fanmod.get(mid, 0)
        vc = vdmc.get(mid, 0)
        if fc != vc:
            mismatches.append((mid, fc, vc))
    return mismatches

for den in DENSITIES:
    for dist in DISTRIBUTIONS:
        label = f"g_den_{den}_embedded_den_3_{dist}_0"
        print(f"\n{'='*60}")
        print(f"Comparing: {label}")

        fanmod = load_fanmod(den, dist)
        vdmc   = load_vdmc(den, dist)

        if fanmod is None:
            print("  SKIP: FANMOD+ output not found")
            continue
        if vdmc is None:
            print("  SKIP: VDMC cache not found")
            continue

        print(f"  FANMOD+ motifs: {len(fanmod)}, total count: {sum(fanmod.values())}")
        print(f"  VDMC    motifs: {len(vdmc)},  total count: {sum(vdmc.values())}")

        mismatches = compare(fanmod, vdmc, label)

        if not mismatches:
            print("  ✓ MATCH: all motif counts agree")
        else:
            print(f"  ✗ MISMATCH: {len(mismatches)} motifs differ")
            print(f"  {'Motif ID':<20} {'FANMOD+':>10} {'VDMC':>10} {'Diff':>10}")
            print(f"  {'-'*50}")
            for mid, fc, vc in mismatches[:20]:  # show first 20
                print(f"  {mid:<20} {fc:>10} {vc:>10} {fc-vc:>+10}")
            if len(mismatches) > 20:
                print(f"  ... and {len(mismatches)-20} more")
            print("\nStopping after first mismatched graph.")
            break
    else:
        continue
    break

print("\nDone.")
