# -*- coding: utf-8 -*-
"""
Run the FFB tabular methods on an UPLOADED ("generic") dataset.

Reads datasets/generic/config.json (written by the Streamlit upload button):
    {"csv_name": "data.csv", "target_attr": "...", "sensitive_attrs": [...], "drop_cols": [...]}

For each method × sensitive attribute × lambda × seed it shells out to the
corresponding src/ffb_tabular_*.py script (with WANDB disabled). Results land in
results/generic_{method}_{attr}_lam{lam}_seed{seed}.json and are picked up
automatically by the dashboard.

Usage:
    python run_generic_sweep.py                     # full sweep, all methods, 10 seeds
    python run_generic_sweep.py --seeds 42          # quick: 1 seed
    python run_generic_sweep.py --methods erm,adv   # subset of methods
"""

import argparse
import json
import os
import subprocess
import sys

HERE    = os.path.dirname(os.path.abspath(__file__))
SRC     = os.path.join(HERE, "src")
GENERIC = os.path.join(HERE, "datasets", "generic")

# method -> (script, lambda grid). None lambda = script takes no --lam (single point).
METHODS = {
    "erm":   ("ffb_tabular_erm.py",   [None]),
    "adv":   ("ffb_tabular_adv.py",   [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]),
    "laftr": ("ffb_tabular_laftr.py", [None]),
    "hsic":  ("ffb_tabular_hsic.py",  [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]),
    "pr":    ("ffb_tabular_pr.py",    [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
}
SEEDS_FULL = [14159, 26535, 89793, 23846, 26433, 83279, 50288, 41971, 69399, 37510]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="erm,adv,laftr,hsic,pr",
                    help="comma list from: erm,adv,laftr,hsic,pr")
    ap.add_argument("--sensitive_attrs", default="",
                    help="comma list; defaults to config.json sensitive_attrs")
    ap.add_argument("--seeds", default="",
                    help="comma list of ints; defaults to the full 10-seed set")
    ap.add_argument("--python", default=sys.executable,
                    help="python interpreter to run the FFB scripts with")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(GENERIC, "config.json")):
        raise SystemExit(f"No config.json in {GENERIC} — upload a dataset first.")
    cfg = json.load(open(os.path.join(GENERIC, "config.json")))

    sens    = [a.strip() for a in args.sensitive_attrs.split(",") if a.strip()] or cfg["sensitive_attrs"]
    seeds   = [int(s) for s in args.seeds.split(",") if s.strip()] or SEEDS_FULL
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in METHODS]

    jobs = []
    for m in methods:
        script, lams = METHODS[m]
        for attr in sens:
            for lam in lams:
                for seed in seeds:
                    jobs.append((m, script, attr, lam, seed))

    total = len(jobs)
    print(f"[sweep] dataset=generic | methods={methods} | attrs={sens} | "
          f"seeds={len(seeds)} | total runs={total}", flush=True)

    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"

    done = failed = 0
    for (m, script, attr, lam, seed) in jobs:
        done += 1
        cmd = [args.python, script, "--dataset", "generic",
               "--sensitive_attr", attr, "--seed", str(seed)]
        if lam is not None:
            cmd += ["--lam", str(lam)]
        lam_s = "-" if lam is None else lam
        print(f"[{done}/{total}] {m} attr={attr} lam={lam_s} seed={seed}", flush=True)
        r = subprocess.run(cmd, cwd=SRC, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            failed += 1
            last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "unknown error"
            print(f"      FAILED ({m}/{attr}/lam{lam_s}/seed{seed}): {last}", flush=True)

    print(f"[sweep] DONE — {total - failed}/{total} succeeded, {failed} failed. "
          f"Results in results/generic_*.json", flush=True)


if __name__ == "__main__":
    main()
