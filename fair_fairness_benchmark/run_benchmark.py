"""
Run the FFB benchmark on the migration dataset exactly as in the FFB paper.
- Sweeps all lambda values from Table 6 of the paper
- Runs all 3 sensitive attributes: sex, coastal_origin, educ_level
- Runs 3 seeds for reliability
- Logs every run to WandB (project: ffb_migration)

Usage:
    python run_benchmark.py                  # full sweep (~hours)
    python run_benchmark.py --quick          # 3 lam values, 1 seed (~minutes)
    python run_benchmark.py --method erm     # single method only
    python run_benchmark.py --sens sex       # single sensitive attribute only
    python run_benchmark.py --no_wandb       # disable wandb (offline mode)
"""

import subprocess
import sys
import os
import argparse
import time

SRC = os.path.join(os.path.dirname(__file__), "src")

# FFB Table 6 — lambda ranges per method
METHODS = {
    "erm": {
        "script": "ffb_tabular_erm.py",
        "lam_values": [0.0],           # baseline, no lambda
        "extra_args": [],
    },
    "diffdp": {
        "script": "ffb_tabular_diffdp.py",
        "lam_values": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0],
        "extra_args": [],
    },
    "diffeopp": {
        "script": "ffb_tabular_diffeopp.py",
        "lam_values": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0],
        "extra_args": [],
    },
    "diffeodd": {
        "script": "ffb_tabular_diffeodd.py",
        "lam_values": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0],
        "extra_args": [],
    },
    "pr": {
        "script": "ffb_tabular_pr.py",
        "lam_values": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.50, 0.6, 0.7, 0.8, 0.9, 1.0],
        "extra_args": [],
    },
    "hsic": {
        "script": "ffb_tabular_hsic.py",
        "lam_values": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000],
        "extra_args": [],
    },
    "adv": {
        "script": "ffb_tabular_adv.py",
        "lam_values": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0],
        "extra_args": ["--num_epochs", "300"],
        "skip_common": ["--num_training_steps", "--wandb_project"],
    },
    "adv_gr": {
        "script": "ffb_tabular_adv_gr.py",
        "lam_values": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0],
        "extra_args": [],
    },
    "laftr": {
        "script": "ffb_tabular_laftr.py",
        "lam_values": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        "extra_args": [],
        "lam_arg": "--A_z",
    },
}

SENSITIVE_ATTRS = ["sex", "coastal_origin", "educ_level"]
SEEDS = [1314, 42, 123]

# Quick mode: reduced sweep for testing
QUICK_LAM = {
    "erm":      [0.0],
    "diffdp":   [0.5, 1.0, 2.0],
    "diffeopp": [0.5, 1.0, 2.0],
    "diffeodd": [0.5, 1.0, 2.0],
    "pr":       [0.1, 0.3, 0.5],
    "hsic":     [100, 300, 500],
    "adv":      [0.5, 1.0, 2.0],
    "adv_gr":   [0.5, 1.0, 2.0],
    "laftr":    [0.5, 1.0, 2.0],
}


def run_one(script, lam, sensitive_attr, seed, extra_args, skip_common, lam_arg, wandb_project, no_wandb, dry_run):
    method_name = script.replace("ffb_tabular_", "").replace(".py", "")

    common = {
        "--num_training_steps": "150",
        "--batch_size":         "128",
        "--wandb_project":      wandb_project,
    }
    # Remove args this script doesn't support
    for k in skip_common:
        common.pop(k, None)

    cmd = [sys.executable, os.path.join(SRC, script),
           "--dataset",        "migration",
           "--sensitive_attr", sensitive_attr,
           "--seed",           str(seed),
           "--exp_name",       f"migration_{method_name}_{sensitive_attr}_lam{lam}_seed{seed}",
    ]
    for k, v in common.items():
        cmd += [k, v]
    cmd += extra_args

    # ERM has no lambda argument (lam=0.0 signals baseline)
    if lam != 0.0:
        cmd += [lam_arg, str(lam)]

    if no_wandb:
        env = {**os.environ, "WANDB_MODE": "offline"}
    else:
        env = os.environ.copy()

    label = f"{script.replace('ffb_tabular_','').replace('.py','')} | sens={sensitive_attr} | lam={lam} | seed={seed}"

    if dry_run:
        print(f"[DRY RUN] {label}")
        print("  " + " ".join(cmd))
        return True

    print(f"\n{'='*70}")
    print(f"  Running: {label}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, cwd=SRC, env=env)
    success = result.returncode == 0
    status = "OK" if success else f"FAILED (exit {result.returncode})"
    print(f"  >> {status}: {label}")
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",    action="store_true", help="Quick mode: 3 lam values, 1 seed")
    parser.add_argument("--method",   type=str, default=None, help="Run a single method (e.g. erm, hsic, adv)")
    parser.add_argument("--sens",     type=str, default=None, help="Single sensitive attr (sex, coastal_origin, educ_level)")
    parser.add_argument("--seed",     type=int, default=None, help="Single seed")
    parser.add_argument("--wandb_project", type=str, default="ffb_migration", help="WandB project name")
    parser.add_argument("--no_wandb", action="store_true", help="Run in offline mode (no WandB upload)")
    parser.add_argument("--dry_run",  action="store_true", help="Print commands without running")
    args = parser.parse_args()

    methods     = {args.method: METHODS[args.method]} if args.method else METHODS
    sens_attrs  = [args.sens] if args.sens else SENSITIVE_ATTRS
    seeds       = [args.seed] if args.seed else ([SEEDS[0]] if args.quick else SEEDS)

    total = sum(
        len(QUICK_LAM[m] if args.quick else METHODS[m]["lam_values"]) * len(sens_attrs) * len(seeds)
        for m in methods
    )

    print(f"\nFFB Migration Benchmark Sweep")
    print(f"  Methods         : {list(methods.keys())}")
    print(f"  Sensitive attrs : {sens_attrs}")
    print(f"  Seeds           : {seeds}")
    print(f"  Total runs      : {total}")
    print(f"  WandB project   : {args.wandb_project}")
    print(f"  Quick mode      : {args.quick}")
    print()

    completed, failed = 0, []
    t0 = time.time()

    for method_name, cfg in methods.items():
        lam_values = QUICK_LAM[method_name] if args.quick else cfg["lam_values"]
        for sens in sens_attrs:
            for seed in seeds:
                for lam in lam_values:
                    ok = run_one(
                        script=cfg["script"],
                        lam=lam,
                        sensitive_attr=sens,
                        seed=seed,
                        extra_args=cfg["extra_args"],
                        skip_common=cfg.get("skip_common", []),
                        lam_arg=cfg.get("lam_arg", "--lam"),
                        wandb_project=args.wandb_project,
                        no_wandb=args.no_wandb,
                        dry_run=args.dry_run,
                    )
                    if ok:
                        completed += 1
                    else:
                        failed.append(f"{method_name}|{sens}|lam={lam}|seed={seed}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Done: {completed}/{total} runs in {elapsed/60:.1f} min")
    if failed:
        print(f"  Failed runs ({len(failed)}):")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
