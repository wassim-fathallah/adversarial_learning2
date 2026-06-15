#!/usr/bin/env python3
"""
Run the adversarial fairness pipeline with multiple seeds (FFB-style sweep),
for one, several, or ALL datasets.

Usage:
    # all 8 datasets, FFB's 10 seeds (default)
    python run_multiple_seeds.py --seeds 14159 26535 89793 23846 26433 83279 50288 41971 69399 37510

    # one dataset, 10 seeds
    python run_multiple_seeds.py --dataset adult --seeds 14159 26535 89793 23846 26433 83279 50288 41971 69399 37510

    # a few datasets, default 5 seeds
    python run_multiple_seeds.py --dataset adult german compas

    # seed range
    python run_multiple_seeds.py --dataset adult --seeds-range 42 51
"""

import argparse
import os
import subprocess
import sys

# main.py lives in the parent package dir; this script sits in adversarial_fairness/seeds/.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# all datasets AADA supports (matches DATASET_PRESETS in main.py, incl. HIMS-Tunisia)
ALL_DATASETS = ["adult", "bank", "compas", "german", "kdd", "acs", "utkface", "HIMS-Tunisia"]


def run_one(dataset_arg: str, seed: int, other_args: list) -> int:
    cmd = [sys.executable, os.path.join(_PARENT, "main.py"),
           "--dataset", dataset_arg, "--seed", str(seed)] + other_args
    print(f"\n{'='*60}\n  DATASET={dataset_arg}  SEED={seed}\n{'='*60}")
    return subprocess.run(cmd, cwd=_PARENT).returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial fairness pipeline with multiple seeds, over one or many datasets",
        epilog="Other args (--epochs, --iterations, etc.) are passed through to main.py",
    )
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help=f"One or more dataset names. Omit for ALL: {ALL_DATASETS}")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 999],
                        help="List of seeds (default: 42 123 456 789 999)")
    parser.add_argument("--seeds-range", type=int, nargs=2, metavar=("START", "END"),
                        help="Inclusive range: --seeds-range 42 51 runs 42..51")
    args, unknown = parser.parse_known_args()

    datasets = args.dataset if args.dataset else ALL_DATASETS
    seeds = args.seeds
    if args.seeds_range:
        start, end = args.seeds_range
        seeds = list(range(start, end + 1))

    total = len(datasets) * len(seeds)
    print(f"\nDatasets ({len(datasets)}): {datasets}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Total runs: {total}")

    failed = []
    done = 0
    for ds in datasets:                       # dataset-outer: fills memory 10 runs/dataset
        for seed in seeds:
            done += 1
            print(f"\n>>> [{done}/{total}] {ds} seed={seed}")
            if run_one(ds, seed, unknown) != 0:
                failed.append(f"{ds}|seed={seed}")

    print(f"\n{'='*60}\n  SUMMARY: {total - len(failed)}/{total} runs OK")
    if failed:
        print(f"  FAILED: {failed}")
        sys.exit(1)
    print(f"  ALL RUNS COMPLETED\n{'='*60}\n")


if __name__ == "__main__":
    main()
