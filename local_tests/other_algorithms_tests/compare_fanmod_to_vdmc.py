"""
Compare FANMOD+ results to VDMC (induced colored motif counts).
"""

import os
import json

FANMOD_DIR    = "local_tests/other_algorithms_tests/fanmod_plus"
VDMC_DIR      = "local_tests/other_algorithms_tests/vdmc"
DENSITIES     = [5, 8, 10, 13, 15]
DISTRIBUTIONS = ["uniform", "average", "rare"]


def load_fanmod(den, dist):
    path = os.path.join(FANMOD_DIR, f"g_den_{den}_embedded_den_3_{dist}_0.json")
    if not os.path.isfile(path):
        return None, None, False
    with open(path) as f:
        raw = json.load(f)
    if "motifs" in raw:
        motifs        = {int(k): v for k, v in raw["motifs"].items()}
        header_total  = raw.get("header_total_subgraphs")
        has_imprecise = raw.get("has_imprecise_counts", False)
    else:
        motifs        = {int(k): v for k, v in raw.items()}
        header_total  = None
        has_imprecise = True
    return motifs, header_total, has_imprecise


def load_vdmc(den, dist):
    path = os.path.join(VDMC_DIR, f"g_den_{den}_embedded_den_3_{dist}_0.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def counts_match(fc, vc):
    if fc == vc:
        return True
    # Only consider rounding if large enough that FANMOD's 5 sig figs lose precision
    if vc >= 100000:
        return fc == round(float(f"{vc:.4e}"))
    return False


def compare(fanmod, vdmc):
    mismatches = []
    for mid in sorted(set(fanmod.keys()) | set(vdmc.keys())):
        fc = fanmod.get(mid, 0)
        vc = vdmc.get(mid, 0)
        if not counts_match(fc, vc):
            mismatches.append((mid, fc, vc))
    return mismatches


done = False
for den in DENSITIES:
    for dist in DISTRIBUTIONS:
        label = f"g_den_{den}_embedded_den_3_{dist}_0"
        print(f"\n{'='*60}")
        print(f"Comparing: {label}")

        fanmod, header_total, has_imprecise = load_fanmod(den, dist)
        vdmc = load_vdmc(den, dist)

        if fanmod is None:
            print("  SKIP: FANMOD+ output not found")
            continue
        if vdmc is None:
            print("  SKIP: VDMC output not found")
            continue

        fanmod_total = header_total if header_total is not None else sum(fanmod.values())
        vdmc_total   = sum(vdmc.values())

        print(f"  FANMOD+ motifs: {len(fanmod)}, total count: {fanmod_total}"
              + (" (from header)" if header_total is not None else ""))
        print(f"  VDMC    motifs: {len(vdmc)},  total count: {vdmc_total}")

        if has_imprecise:
            parsed_sum = sum(fanmod.values())
            if header_total and header_total != parsed_sum:
                print(f"  WARNING: {header_total - parsed_sum} counts lost to scientific notation rounding")

        mismatches = compare(fanmod, vdmc)

        if not mismatches:
            print("  ✓ MATCH: all motif counts agree")
        else:
            print(f"  ✗ MISMATCH: {len(mismatches)} motifs differ")
            print(f"  {'Motif ID':<20} {'FANMOD+':>10} {'VDMC':>10} {'Diff':>10}")
            print(f"  {'-'*50}")
            for mid, fc, vc in mismatches[:20]:
                print(f"  {mid:<20} {fc:>10} {vc:>10} {fc-vc:>+10}")
            if len(mismatches) > 20:
                print(f"  ... and {len(mismatches)-20} more")
            done = True
            break
    if done:
        break

print("\nDone.")