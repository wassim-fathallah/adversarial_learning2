"""
Insert the HIMS-Tunisia ("HIMS") dataset rows into per-method comparison CSVs.

The target CSVs are lambda-major (header: LAMBDA,Dataset,Sensitive_Attribute,
Accuracy,Prule; each lambda block lists every dataset). For each target we read
the HIMS-Tunisia rows for that method from HIMS-Tunisia_all.csv and append the 3
HIMS-Tunisia rows (sex, coastal_origin, educ_level) at the END of each matching
lambda block, labelled with --label (default HIMS). Lambda values are matched by
float and re-emitted using the target file's own lambda string.

A .bak copy of every target is written before it is modified.
"""

import argparse
import csv
import os
import shutil
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MIG  = os.path.join(HERE, "HIMS-Tunisia_all.csv")
ATTR_ORDER = ["sex", "coastal_origin", "educ_level"]


def load_hims_tunisia(method):
    """method -> {float(lambda): {attr: (acc_str, prule_str)}} from HIMS-Tunisia_all.csv"""
    out = defaultdict(dict)
    for r in csv.DictReader(open(MIG, encoding="utf-8-sig")):
        if r["Method"] != method:
            continue
        out[float(r["LAMBDA"])][r["Sensitive_Attribute"]] = (r["Accuracy"], r["Prule"])
    return out


def insert(target, method, label):
    rows = list(csv.reader(open(target, encoding="utf-8-sig")))
    header, body = rows[0], rows[1:]
    mig = load_hims_tunisia(method)

    # group body rows by lambda, preserving first-seen order and the original lambda string
    blocks = OrderedDict()           # lam_str -> list of rows
    for row in body:
        blocks.setdefault(row[0], []).append(row)

    out = [header]
    used = set()
    for lam_str, blk in blocks.items():
        out.extend(blk)
        lam_f = float(lam_str)
        if lam_f in mig:
            for attr in ATTR_ORDER:
                if attr in mig[lam_f]:
                    acc, pr = mig[lam_f][attr]
                    out.append([lam_str, label, attr, acc, pr])
            used.add(lam_f)

    missing = sorted(set(mig) - used)
    shutil.copyfile(target, target + ".bak")
    with open(target, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(out)

    print(f"  inserted {len(used)} lambda blocks x ~3 rows into {os.path.basename(target)} "
          f"(label={label})")
    if missing:
        print(f"    NOTE: HIMS-Tunisia lambdas not present in target (skipped): {missing}")


def relabel(target, old, new):
    rows = list(csv.reader(open(target, encoding="utf-8-sig")))
    shutil.copyfile(target, target + ".bak")
    n = 0
    for row in rows[1:]:
        if len(row) > 1 and row[1] == old:
            row[1] = new
            n += 1
    with open(target, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"  relabelled {n} rows '{old}' -> '{new}' in {os.path.basename(target)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--method", help="HIMS-Tunisia method to pull rows for (insert mode)")
    ap.add_argument("--label", default="HIMS")
    ap.add_argument("--relabel_from", help="relabel mode: existing dataset name to rename")
    args = ap.parse_args()

    if args.relabel_from:
        relabel(args.target, args.relabel_from, args.label)
    else:
        insert(args.target, args.method, args.label)
