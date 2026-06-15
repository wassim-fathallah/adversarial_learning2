"""
Aggregate ALL FFB results for a SINGLE dataset into one summary CSV.

Companion to make_method_csv.py: that one fixes a method and spans datasets;
this one fixes a dataset (e.g. HIMS-Tunisia) and spans every method. It groups all
runs by (method, lambda, sensitive_attr), averages test/acc and test/prule across
seeds, and writes one row per group:

    Method,LAMBDA,Sensitive_Attribute,Accuracy,Prule

Usage:
    python make_dataset_csv.py --dataset HIMS-Tunisia            # -> HIMS-Tunisia_all.csv
    python make_dataset_csv.py --dataset adult --out ADULT.csv
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# stable display/sort order for methods
METHOD_ORDER = ["erm", "diffdp", "diffeopp", "diffeodd", "pr", "hsic", "adv", "adv_gr", "laftr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. HIMS-Tunisia, adult, kdd, acs")
    ap.add_argument("--out", default=None, help="output CSV path (default: <DATASET>_all.csv)")
    args = ap.parse_args()

    out = args.out or os.path.join(HERE, f"{args.dataset}_all.csv")

    pattern = os.path.join(RESULTS, f"{args.dataset}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matching {pattern}")

    # (method, lam, attr) -> list of (acc, prule)
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
        h = (d.get("history") or [{}])[-1]
        method = m.get("method")
        lam    = m.get("lam")
        attr   = m.get("sensitive_attr")
        acc    = h.get("test/acc")
        pr     = h.get("test/prule")
        if None in (method, lam, attr, acc, pr):
            skipped += 1
            continue
        groups[(method, float(lam), attr)].append((acc, pr))

    rows = []
    for (method, lam, attr), vals in groups.items():
        n = len(vals)
        acc_mean = sum(v[0] for v in vals) / n
        pr_mean  = sum(v[1] for v in vals) / n
        rows.append((method, lam, attr, acc_mean, pr_mean, n))

    def method_key(m):
        return METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER)

    # sort: method (canonical order), then lambda, then attribute
    rows.sort(key=lambda r: (method_key(r[0]), r[1], r[2]))

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Method", "LAMBDA", "Sensitive_Attribute", "Accuracy", "Prule"])
        for method, lam, attr, acc, pr, n in rows:
            lam_s = str(int(lam)) if float(lam).is_integer() else str(lam)
            w.writerow([method, lam_s, attr, f"{acc:.5f}", f"{pr:.5f}"])

    print(f"[make_dataset_csv] dataset={args.dataset}")
    print(f"  files read     : {len(files)}  (skipped {skipped})")
    print(f"  groups (rows)  : {len(rows)}")
    print(f"  methods        : {sorted({r[0] for r in rows}, key=method_key)}")
    print(f"  attrs          : {sorted({r[2] for r in rows})}")
    print(f"  seeds/group    : {sorted({len(v) for v in groups.values()})}")
    print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()
