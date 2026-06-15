# -*- coding: utf-8 -*-
"""
FFB comparison table generator.
================================

For every FFB method × dataset × sensitive attribute we run a sweep over many
(lambda, seed) configurations. From that sweep we extract THREE operating points
(the selection criteria), and at each we also report DP / EOdd / EOpp:

  1. Max-Acc   : lambda with max mean accuracy
                 -> report its accuracy AND P-rule, dDP, dEOdd, dEOpp.
  2. Max-Prule : lambda with max mean P-rule (disparate impact)
                 -> report its P-rule AND accuracy, dDP, dEOdd, dEOpp.
  3. Trade-off : lambda with max mean min(acc, P-rule). If some lambda has BOTH
                 acc>=80 and P-rule>=80 this gives the most balanced such point;
                 otherwise it gives the point closest to the 80/80 corner.

Every attribute always gets ALL THREE rows (no merging). When two criteria pick
the same lambda their rows show identical values; single-lambda methods (ERM,
LAFTR on tabular) show three identical rows.

Our own method ("Ours") does NOT sweep — it adapts lambda online and stops at a
single operating point. So we show one row per (dataset, attr): the final result
if the stop condition was reached, otherwise the best accuracy/best-P-rule point
that was tracked. (Stored in long_term_memory.json.)

Aggregation across seeds: MEAN. (10 seeds for FFB tabular datasets, 3 for HIMS-Tunisia.)

Metric scale: FFB logs everything in percent (acc, prule, dp, eodd, eopp ∈ [0,100]).
We keep that. "Ours" accuracy is stored as a fraction → multiplied by 100; P-rule is
already a percentage.

Output:
  - readable tables printed to console
  - LaTeX (booktabs) tables written to comparison_tables.tex
"""

import os
import re
import json
import glob
from collections import defaultdict
from statistics import mean

#
# Paths & configuration
#

HERE        = os.path.dirname(os.path.abspath(__file__))
FFB_RESULTS = os.path.join(HERE, "..", "fair_fairness_benchmark", "results")
OURS_MEMORY = os.path.join(HERE, "long_term_memory.json")
LATEX_OUT   = os.path.join(HERE, "comparison_tables.tex")

# Set True to append our adaptive method's row to each dataset block.
# False -> tables show ONLY the FFB methods' results.
INCLUDE_OURS = True

# Methods to include, in display order.  'diffdp' is excluded (only 2 stray files).
METHODS = ["erm", "adv", "laftr", "hsic", "pr"]
METHOD_NAMES = {
    "erm":   "ERM (baseline)",
    "adv":   "AdvDebias",
    "laftr": "LAFTR",
    "hsic":  "HSIC",
    "pr":    "PrejudiceRemover",
}

# Dataset display order + pretty names
DATASET_ORDER = ["adult", "bank_marketing", "compas", "german", "kdd", "acs",
                 "utkface", "HIMS-Tunisia"]
DATASET_NAMES = {
    "adult": "Adult", "bank_marketing": "Bank", "compas": "COMPAS",
    "german": "German", "kdd": "KDD Census", "acs": "ACS Income",
    "utkface": "UTKFace", "HIMS-Tunisia": "HIMS-Tunisia",
}

# How to read "Ours" out of long_term_memory.json, per FFB dataset name.
#   key      : the memory key to use for this dataset
#   attr_map : {ours_attr_name_in_memory : ffb_attr_name}  (for row alignment)
# Keys below match the runs reported in the thesis (tab:ours_summary), plus KDD/ACS.
OURS_CONFIG = {
    "adult":          {"key": "adult|income|race,sex",
                       "attr_map": {"race": "race", "sex": "sex"}},
    "compas":         {"key": "compas|two_year_recid|race,sex",
                       "attr_map": {"race": "race", "sex": "sex"}},
    "german":         {"key": "german|Class|Age,Sex",
                       "attr_map": {"Age": "age", "Sex": "sex"}},
    "bank_marketing": {"key": "bank|y|age,job",
                       "attr_map": {"age": "age"}},
    "kdd":            {"key": "kdd|income|race,sex",
                       "attr_map": {"race": "race", "sex": "sex"}},
    "acs":            {"key": "acs|PINCP|RAC1P,SEX",
                       "attr_map": {"RAC1P": "race", "SEX": "sex"}},
    "utkface":        {"key": "utkface|age|ethnicity,gender",
                       "attr_map": {"gender": "Gender", "ethnicity": "Race"}},
    "HIMS-Tunisia":   {"key": "HIMS-Tunisia|legal_entry|Gender,educ_level,region_origin",
                       "attr_map": {"Gender": "sex",
                                    "region_origin": "coastal_origin",
                                    "educ_level": "educ_level"}},
}

# Pretty attribute labels for tables
ATTR_LABELS = {
    "sex": "sex", "race": "race", "age": "age",
    "Gender": "gender", "Race": "race",
    "coastal_origin": "coastal", "educ_level": "educ",
}


#
# Parse FFB result files
#

_FNAME_RE = re.compile(r"^(?P<prefix>.+)_lam(?P<lam>[0-9.]+)_seed(?P<seed>[0-9]+)\.json$")
_KNOWN_METHODS = {"adv", "erm", "hsic", "laftr", "pr", "diffdp"}


def _split_prefix(prefix: str):
    """'bank_marketing_adv_age' -> ('bank_marketing', 'adv', 'age')."""
    parts = prefix.split("_")
    for i, p in enumerate(parts):
        if p in _KNOWN_METHODS:
            dataset = "_".join(parts[:i])
            method  = p
            attr    = "_".join(parts[i + 1:])
            return dataset, method, attr
    return None, None, None


def load_ffb():
    """Returns nested dict: data[dataset][method][attr][lam] = {metric: [values across seeds]}"""
    files = glob.glob(os.path.join(FFB_RESULTS, "*.json"))
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(list)))))

    for path in files:
        m = _FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        dataset, method, attr = _split_prefix(m.group("prefix"))
        if method not in METHODS:
            continue
        lam = float(m.group("lam"))
        try:
            hist = json.load(open(path))["history"][-1]
        except (json.JSONDecodeError, IndexError, KeyError):
            continue
        bucket = data[dataset][method][attr][lam]
        bucket["acc"].append(hist.get("test/acc"))
        bucket["prule"].append(hist.get("test/prule"))
        bucket["dp"].append(hist.get("test/dp"))
        bucket["eodd"].append(hist.get("test/eodd"))
        bucket["eopp"].append(hist.get("test/eopp"))
    return data


def _mean_over_seeds(lam_dict):
    """{lam: {metric: [vals]}} -> {lam: {metric: mean}} (skips None)."""
    out = {}
    for lam, metrics in lam_dict.items():
        row = {}
        for k, vals in metrics.items():
            clean = [v for v in vals if v is not None]
            row[k] = mean(clean) if clean else None
        row["_n_seeds"] = max(len(v) for v in metrics.values())
        out[lam] = row
    return out


# Selection criteria. DP / EOdd / EOpp are reported columns, NOT selection
# objectives. Each objective is (label, scorer, required_metrics) where the
# chosen lambda maximizes scorer(row):
#   Max-Acc   : highest accuracy
#   Max-Prule : highest P-rule
#   Trade-off : highest min(acc, P-rule). If a lambda has BOTH acc>=80 and
#               P-rule>=80 this picks the most balanced such point; if none
#               satisfies both, it picks the point closest to the 80/80 corner
#               (the one whose weaker metric is as high as possible).
def _score_acc(r):      return r["acc"]
def _score_prule(r):    return r["prule"]
def _score_tradeoff(r): return min(r["acc"], r["prule"])

OBJECTIVES = [
    ("Max-Acc",   _score_acc,      ("acc",)),
    ("Max-Prule", _score_prule,    ("prule",)),
    ("Trade-off", _score_tradeoff, ("acc", "prule")),
]


def _is_degenerate(r):
    """
    True for a collapsed operating point: the model predicts (nearly) one class,
    so P-rule is trivially ~100 and ALL fairness gaps are ~0. These are not
    meaningful results and must be excluded from selection — otherwise Max-Prule
    would report e.g. 76.48/100/0/0/0 (the majority-class predictor).
    """
    prule = r.get("prule")
    dp, eodd, eopp = r.get("dp"), r.get("eodd"), r.get("eopp")
    if None in (prule, dp, eodd, eopp):
        return False
    return prule >= 99.0 and dp < 0.5 and eodd < 0.5 and eopp < 0.5


def select_operating_points(lam_means):
    """
    From {lam: {metric: mean}} return EXACTLY one row per objective in OBJECTIVES,
    always in the order: Max-Acc, Max-Prule, Trade-off.

    Degenerate collapse points (see _is_degenerate) are dropped first so the
    selection lands on real operating points with real DP/EOdd/EOpp values.

    No merging and no collapsing: every attribute gets all three rows, even when
    two criteria pick the same lambda (their rows will then show identical values)
    or when the method has a single lambda (all three rows identical). A row is
    only skipped if its required metric is entirely missing.

    Returns a list of (label, row) where row is that lambda's metric dict (+'lam').
    """
    # Drop degenerate collapse points; keep all only if nothing else remains.
    usable = {l: r for l, r in lam_means.items() if not _is_degenerate(r)}
    if not usable:
        usable = lam_means

    out = []
    for label, scorer, need in OBJECTIVES:
        valid = [l for l, r in usable.items()
                 if all(r.get(k) is not None for k in need)]
        if not valid:
            continue
        best_lam = max(valid, key=lambda l: scorer(usable[l]))
        row = dict(usable[best_lam]); row["lam"] = best_lam
        out.append((label, row))
    return out


#
# Load "Ours" from long_term_memory.json
#

def load_ours():
    """
    Returns {ffb_dataset: {ffb_attr: {acc, prule, reached}}}
    acc in percent. dDP/dEOdd/dEOpp are NOT persisted -> None.

    long_term_memory.json has been trimmed to exactly the report run per key,
    so we use the most recent run. If a dataset is later re-run (appending new
    runs), the most recent run is shown — re-trim to keep reproducing the report.
    """
    if not os.path.exists(OURS_MEMORY):
        return {}
    mem = json.load(open(OURS_MEMORY))
    out = {}
    for ds, cfg in OURS_CONFIG.items():
        runs = mem.get(cfg["key"])
        if not runs:
            continue
        run = runs[-1]
        acc = run.get("accuracy_final", 0) * 100.0
        prules = run.get("p_rules_final", {})
        fair = run.get("fairness_final", {})  # {attr: {dp, eodd, eopp}} — newer runs only
        reached = run.get("success", False)
        per_attr = {}
        for ours_attr, ffb_attr in cfg["attr_map"].items():
            if ours_attr in prules:
                fa = fair.get(ours_attr, {})
                per_attr[ffb_attr] = {
                    "acc": acc,
                    "prule": prules[ours_attr],
                    "reached": reached,
                    "dp":   fa.get("dp"),
                    "eodd": fa.get("eodd"),
                    "eopp": fa.get("eopp"),
                }
        if per_attr:
            out[ds] = per_attr
    return out


#
# Formatting helpers
#

def f(x):
    return "--" if x is None else f"{x:.2f}"


def attr_label(a):
    return ATTR_LABELS.get(a, a)


#
# Console output
#

def print_console(ffb, ours):
    for method in METHODS:
        print("\n" + "=" * 92)
        print(f"  METHOD: {METHOD_NAMES[method]}")
        print("=" * 92)
        hdr = f"{'Dataset':<10}{'Attr':<9}{'Select':<22}{'Acc':>8}{'P-rule':>9}{'dDP':>8}{'dEOdd':>8}{'dEOpp':>8}"
        print(hdr)
        print("-" * 104)
        for ds in DATASET_ORDER:
            if ds not in ffb or method not in ffb[ds]:
                continue
            attrs = sorted(ffb[ds][method].keys())
            for attr in attrs:
                lam_means = _mean_over_seeds(ffb[ds][method][attr])
                rows = select_operating_points(lam_means)
                if not rows:
                    continue
                for label, r in rows:
                    print(f"{DATASET_NAMES.get(ds, ds):<10}{attr_label(attr):<9}{label:<22}"
                          f"{f(r['acc']):>8}{f(r['prule']):>9}{f(r['dp']):>8}"
                          f"{f(r['eodd']):>8}{f(r['eopp']):>8}")
            # Ours rows for this dataset
            if INCLUDE_OURS and ds in ours:
                for ffb_attr, o in ours[ds].items():
                    tag = "final" if o["reached"] else "best"
                    print(f"{'> OURS':<10}{attr_label(ffb_attr):<9}{tag:<22}"
                          f"{f(o['acc']):>8}{f(o['prule']):>9}{f(o['dp']):>8}"
                          f"{f(o['eodd']):>8}{f(o['eopp']):>8}")
            print("-" * 104)


#
# LaTeX output
#

def latex_tables(ffb, ours):
    lines = []
    lines.append("% Auto-generated by compare_ffb.py")
    lines.append("% Per FFB sweep we extract operating points (mean over seeds):")
    lines.append("%   Max-Acc   = lambda with highest accuracy")
    lines.append("%   Max-Prule = lambda with highest P-rule")
    lines.append("%   Trade-off = lambda with highest min(acc, P-rule); best balanced")
    lines.append("%               point with both >=80 if attainable, else closest to it.")
    lines.append("% DP/EOdd/EOpp are reported (not selection criteria). Every attribute")
    lines.append("% always gets all 3 rows (Max-Acc, Max-Prule, Trade-off), no merging.")
    lines.append("")
    for method in METHODS:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        cap = (f"{METHOD_NAMES[method]} on the FFB benchmark. "
               r"For each sensitive attribute we report the accuracy-optimal (Max-Acc), "
               r"the P-rule-optimal (Max-Prule), and the best accuracy/P-rule "
               r"trade-off (Trade-off: highest $\min(\text{Acc},\text{P-rule})$) "
               r"operating points (mean over seeds). "
               r"$\Delta$DP, $\Delta$EOdd, $\Delta$EOpp are the fairness gaps at that "
               r"point (lower is fairer). All values in \%.")
        lines.append(r"\caption{" + cap + "}")
        lines.append(r"\begin{tabular}{lllrrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Dataset & Attr & Select & Acc & P-rule & $\Delta$DP & $\Delta$EOdd & $\Delta$EOpp \\")
        lines.append(r"\midrule")
        for ds in DATASET_ORDER:
            if ds not in ffb or method not in ffb[ds]:
                continue
            attrs = sorted(ffb[ds][method].keys())
            first = True
            for attr in attrs:
                lam_means = _mean_over_seeds(ffb[ds][method][attr])
                rows = select_operating_points(lam_means)
                if not rows:
                    continue
                for label, r in rows:
                    dscell = DATASET_NAMES.get(ds, ds) if first else ""
                    first = False
                    lines.append(
                        f"{dscell} & {attr_label(attr)} & {label} & "
                        f"{f(r['acc'])} & {f(r['prule'])} & {f(r['dp'])} & "
                        f"{f(r['eodd'])} & {f(r['eopp'])} \\\\")
            # Ours rows (optional)
            if INCLUDE_OURS and ds in ours:
                for ffb_attr, o in ours[ds].items():
                    tag = "Final" if o["reached"] else "Best"
                    lines.append(
                        f"\\textbf{{Ours}} & {attr_label(ffb_attr)} & {tag} & "
                        f"\\textbf{{{f(o['acc'])}}} & \\textbf{{{f(o['prule'])}}} & "
                        f"{f(o['dp'])} & {f(o['eodd'])} & {f(o['eopp'])} \\\\")
            lines.append(r"\midrule")
        if lines[-1] == r"\midrule":
            lines[-1] = r"\bottomrule"
        else:
            lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CSV dump (for handing exact numbers to a report-writing agent)
# ─────────────────────────────────────────────────────────────────────────────

CSV_OUT = os.path.join(HERE, "comparison_data.csv")


def write_csv(ffb, ours):
    rows = ["method,dataset,attribute,selection,acc,prule,dDP,dEOdd,dEOpp"]
    for method in METHODS:
        for ds in DATASET_ORDER:
            if ds not in ffb or method not in ffb[ds]:
                continue
            for attr in sorted(ffb[ds][method].keys()):
                for label, r in select_operating_points(_mean_over_seeds(ffb[ds][method][attr])):
                    rows.append(",".join([
                        METHOD_NAMES[method], DATASET_NAMES.get(ds, ds),
                        attr_label(attr), label,
                        f(r["acc"]), f(r["prule"]), f(r["dp"]), f(r["eodd"]), f(r["eopp"]),
                    ]))
            if INCLUDE_OURS and ds in ours:
                for ffb_attr, o in ours[ds].items():
                    rows.append(",".join([
                        "Ours", DATASET_NAMES.get(ds, ds), attr_label(ffb_attr),
                        "Final" if o["reached"] else "Best",
                        f(o["acc"]), f(o["prule"]), f(o["dp"]), f(o["eodd"]), f(o["eopp"]),
                    ]))
    with open(CSV_OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(FFB_RESULTS):
        raise SystemExit(f"FFB results not found: {FFB_RESULTS}")
    ffb  = load_ffb()
    ours = load_ours()

    print_console(ffb, ours)

    tex = latex_tables(ffb, ours)
    with open(LATEX_OUT, "w", encoding="utf-8") as fh:
        fh.write(tex)
    write_csv(ffb, ours)
    print("\n" + "=" * 92)
    print(f"LaTeX tables written to: {LATEX_OUT}")
    print(f"CSV data written to:     {CSV_OUT}")
    if not ours:
        print("WARNING: no 'Ours' rows — check OURS_CONFIG keys vs long_term_memory.json")
    else:
        missing_diffs = any(o["dp"] is None for ds in ours.values() for o in ds.values())
        if missing_diffs:
            print("NOTE: Ours dDP/dEOdd/dEOpp show '--' (not persisted in memory yet).")
    print("=" * 92)


if __name__ == "__main__":
    main()
