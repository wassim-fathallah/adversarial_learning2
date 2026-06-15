"""
Aggregate AADA multi-seed runs into FFB-style mean ± std.

Reads long_term_memory.json, groups the stored runs for a dataset, and prints
mean ± std of accuracy and per-attribute P-rule across seeds.

    python aggregate_seeds.py --dataset adult
    python aggregate_seeds.py --dataset adult --last 10   # only the last 10 runs
"""

import argparse
import json
import os
from collections import defaultdict
from statistics import mean, stdev

HERE = os.path.dirname(os.path.abspath(__file__))
# long_term_memory.json lives in the parent package dir (adversarial_fairness/).
MEM = os.path.join(HERE, "..", "long_term_memory.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--last", type=int, default=None, help="use only the last N runs")
    args = ap.parse_args()

    data = json.load(open(MEM, encoding="utf-8-sig"))
    runs = []
    for key, entries in data.items():
        if key.split("|")[0].lower() == args.dataset.lower():
            runs.extend(entries)
    if not runs:
        raise SystemExit(f"No runs for dataset '{args.dataset}' in {MEM}")
    if args.last:
        runs = runs[-args.last:]

    accs = [r["accuracy_final"] * 100 for r in runs]   # stored as fraction
    seeds = [r.get("seed") for r in runs]
    prules = defaultdict(list)
    for r in runs:
        for attr, v in (r.get("p_rules_final") or {}).items():
            prules[attr].append(v)

    def ms(xs):
        return (mean(xs), stdev(xs) if len(xs) > 1 else 0.0)

    print(f"\nDataset: {args.dataset}   |   runs: {len(runs)}   |   seeds: {seeds}\n")
    a_m, a_s = ms(accs)
    print(f"  Accuracy   : {a_m:6.2f} ± {a_s:.2f}")
    for attr, vals in prules.items():
        p_m, p_s = ms(vals)
        print(f"  P-rule {attr:12s}: {p_m:6.2f} ± {p_s:.2f}   (min {min(vals):.1f}, max {max(vals):.1f})")
    print()


if __name__ == "__main__":
    main()
