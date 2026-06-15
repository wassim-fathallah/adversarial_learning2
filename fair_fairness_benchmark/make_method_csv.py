"""
Aggregate FFB per-run result JSONs into a per-lambda summary CSV.

For a given method it groups all runs by (lambda, dataset, sensitive_attr),
averages test/acc and test/prule across seeds, and writes one row per group:

    LAMBDA,Dataset,Sensitive_Attribute,Accuracy,Prule

This matches the hand-made HSIC_all.csv format.

Usage:
    python make_method_csv.py --method laftr               # -> LAFTR_all.csv
    python make_method_csv.py --method hsic --out HSIC.csv
    python make_method_csv.py --method laftr --min_lam 0.01 # drop lam<=0 (unlabeled)
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def last_step(history):
    return history[-1] if history else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="e.g. laftr, hsic, pr, adv, diffdp")
    ap.add_argument("--out", default=None, help="output CSV path (default: <METHOD>_all.csv)")
    ap.add_argument("--min_lam", type=float, default=0.0,
                    help="drop runs with lambda <= this (laftr downloads default to 0.0 when "
                         "A_z was not captured; use 0.0 after re-downloading with the A_z fix)")
    args = ap.parse_args()

    out = args.out or os.path.join(HERE, f"{args.method.upper()}_all.csv")

    pattern = os.path.join(RESULTS, f"*_{args.method}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matching {pattern}")

    # (lam, dataset, attr) -> list of (acc, prule)
    groups = defaultdict(list)
    skipped = 0
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8-sig"))
        except Exception as e:
            print(f"  skip (unreadable): {os.path.basename(f)} -> {e}")
            skipped += 1
            continue
        m = d.get("metadata", {})
        h = last_step(d.get("history", []))
        lam  = m.get("lam")
        ds   = m.get("dataset")
        attr = m.get("sensitive_attr")
        acc  = h.get("test/acc")
        pr   = h.get("test/prule")
        if lam is None or acc is None or pr is None:
            skipped += 1
            continue
        if float(lam) <= args.min_lam:
            skipped += 1
            continue
        groups[(float(lam), ds, attr)].append((acc, pr))

    rows = []
    for (lam, ds, attr), vals in groups.items():
        n = len(vals)
        acc_mean = sum(v[0] for v in vals) / n
        pr_mean  = sum(v[1] for v in vals) / n
        rows.append((lam, ds, attr, acc_mean, pr_mean, n))

    # sort like HSIC_all.csv: by lambda, then dataset, then attribute
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["LAMBDA", "Dataset", "Sensitive_Attribute", "Accuracy", "Prule"])
        for lam, ds, attr, acc, pr, n in rows:
            # integer lambda printed without trailing .0 (matches HSIC file: 50, 100, ...)
            lam_s = str(int(lam)) if float(lam).is_integer() else str(lam)
            w.writerow([lam_s, ds, attr, f"{acc:.5f}", f"{pr:.5f}"])

    n_seeds = sorted({len(v) for v in groups.values()})
    print(f"[make_method_csv] method={args.method}")
    print(f"  files read     : {len(files)}  (skipped {skipped})")
    print(f"  groups (rows)  : {len(rows)}")
    print(f"  seeds/group    : {n_seeds}")
    print(f"  datasets       : {sorted({r[1] for r in rows})}")
    print(f"  lambdas        : {sorted({r[0] for r in rows})}")
    print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()
