"""
AADA vs FFB operating-point figure.

One panel per (dataset, sensitive attribute). FFB methods (AdvDebias, PR, HSIC,
LAFTR) are plotted as their full lambda-sweep operating points (Accuracy vs
P-rule); AADA (our method) is a single big star at its (P-rule, Accuracy)
operating point. The P-rule >= 80 region is shaded.

Run:  python make_aada_vs_ffb.py
Out:  fig_aada_vs_ffb.png
"""

import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# method -> (csv path, color, marker). All rollup CSVs live in the repo so the
# figure regenerates on any machine.
METHODS = {
    "ERM":       (os.path.join(HERE, "ERM_all.csv"),     "#444444", "X"),
    "AdvDebias": (os.path.join(HERE, "ADV_all.csv"),     "#3b6fa0", "o"),
    "PR":        (os.path.join(HERE, "PRALL.csv"),       "#d9a441", "s"),
    "HSIC":      (os.path.join(HERE, "HSIC_all.csv"),    "#5a9e5a", "^"),
    "LAFTR":     (os.path.join(HERE, "LAFTR_all.csv"),   "#c0504d", "D"),
}
AADA_COLOR = "#7b4ea3"

# AADA operating points: (dataset, attr) -> (p_rule, accuracy)
AADA = {
    ("adult", "sex"):           (81.2, 83.96),
    ("adult", "race"):          (82.9, 83.96),
    ("german", "sex"):          (87.3, 68.20),
    ("german", "age"):          (80.6, 68.20),
    ("compas", "sex"):          (78.3, 65.08),
    ("compas", "race"):         (75.9, 65.08),
    ("bank_marketing", "age"):  (94.5, 91.3),
    ("kdd", "sex"):             (95.3, 95.1),
    ("kdd", "race"):            (97.8, 95.1),
    ("acs", "sex"):             (82.6, 73.5),
    ("acs", "race"):            (83.2, 73.5),
    ("utkface", "Gender"):      (84.0, 77.75),
    ("utkface", "Race"):        (81.0, 77.75),
    ("HIMS", "sex"):            (84.1, 82.2),
    ("HIMS", "coastal_origin"): (90.4, 82.2),
    ("HIMS", "educ_level"):     (88.3, 82.2),
}

# panel order (4x4 grid) — each dataset's attributes kept on the same row
PANEL_ORDER = [
    ("adult", "sex"),  ("adult", "race"),  ("compas", "sex"),     ("compas", "race"),
    ("german", "sex"), ("german", "age"),  ("kdd", "sex"),        ("kdd", "race"),
    ("acs", "sex"),    ("acs", "race"),    ("utkface", "Gender"), ("utkface", "Race"),
    ("bank_marketing", "age"), ("HIMS", "sex"), ("HIMS", "coastal_origin"), ("HIMS", "educ_level"),
]

# nice panel titles
NICE_DS = {"bank_marketing": "Bank", "kdd": "KDD", "acs": "ACS", "utkface": "UTKFace",
           "adult": "Adult", "german": "German", "compas": "COMPAS", "HIMS": "HIMS"}
NICE_ATTR = {"coastal_origin": "region", "educ_level": "educ", "Gender": "gender", "Race": "race"}


def load_points():
    """method -> {(dataset, attr): [(prule, acc), ...]}"""
    data = {}
    for m, (path, _, _) in METHODS.items():
        pts = defaultdict(list)
        if not os.path.exists(path):
            print(f"  WARN: missing {path}")
            data[m] = pts
            continue
        bad = 0
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            try:
                acc = float(r["Accuracy"]); pr = float(r["Prule"])
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            ds = "HIMS" if r["Dataset"] in ("migration", "HIMS-Tunisia") else r["Dataset"]
            pts[(ds, r["Sensitive_Attribute"])].append((pr, acc))
        if bad:
            print(f"  {m}: skipped {bad} unparseable row(s)")
        data[m] = pts
    return data


def load_utkface_adv():
    """UTKface adversarial-debiasing (run locally) -> {(utkface, attr): [(prule, acc), ...]},
    one point per lambda, averaged over whatever seeds are present."""
    by_attr_lam = defaultdict(list)   # (attr, lam) -> [(prule, acc), ...]
    for f in glob.glob(os.path.join(RESULTS, "utkface_adv_*.json")):
        try:
            d = json.load(open(f, encoding="utf-8-sig"))
        except Exception:
            continue
        m = d.get("metadata", {})
        h = (d.get("history") or [{}])[-1]
        attr = m.get("sensitive_attr"); lam = m.get("lam")
        acc = h.get("test/acc"); pr = h.get("test/prule")
        if None in (attr, lam, acc, pr):
            continue
        by_attr_lam[(attr, float(lam))].append((pr, acc))
    out = defaultdict(list)
    for (attr, lam), vals in by_attr_lam.items():
        n = len(vals)
        out[("utkface", attr)].append((sum(v[0] for v in vals) / n,
                                       sum(v[1] for v in vals) / n))
    return out


def main():
    data = load_points()

    # merge locally-run UTKface adversarial debiasing into the AdvDebias series
    utk_adv = load_utkface_adv()
    for key, pts in utk_adv.items():
        data["AdvDebias"][key] = pts
    if utk_adv:
        print(f"  AdvDebias: added UTKface points for {sorted(utk_adv)} "
              f"({sum(len(v) for v in utk_adv.values())} pts)")

    ncols, nrows = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 16))
    axes = axes.ravel()

    for ax, (ds, attr) in zip(axes, PANEL_ORDER):
        # FFB method clouds
        legend_handles = []
        for m, (_, color, marker) in METHODS.items():
            pts = data[m].get((ds, attr), [])
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.scatter(xs, ys, c=color, marker=marker, s=55, alpha=0.8,
                       edgecolors="white", linewidths=0.5, zorder=3)
            legend_handles.append(Line2D([0], [0], marker=marker, color="none",
                                         markerfacecolor=color, markeredgecolor="white",
                                         markersize=9, label=f"{m} ({len(pts)})"))
        # AADA star
        if (ds, attr) in AADA:
            pr, acc = AADA[(ds, attr)]
            ax.scatter([pr], [acc], c=AADA_COLOR, marker="*", s=600,
                       edgecolors="black", linewidths=1.2, zorder=5)
            legend_handles.append(Line2D([0], [0], marker="*", color="none",
                                         markerfacecolor=AADA_COLOR, markeredgecolor="black",
                                         markersize=18, label="AADA"))

        # 80% P-rule target region
        ax.axvspan(80, 105, color="#bfe3bf", alpha=0.45, zorder=0)
        ax.axvline(80, color="gray", ls="--", lw=1, zorder=1)

        ax.set_xlim(0, 105)
        ax.set_title(f"{NICE_DS.get(ds, ds)} / {NICE_ATTR.get(attr, attr)}", fontsize=13)
        ax.set_xlabel("P-rule (%)", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.grid(True, ls=":", alpha=0.4)
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=8, loc="best", framealpha=0.9)

    fig.suptitle("FFB fairness-control operating points vs AADA, by dataset and sensitive attribute",
                 fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(HERE, "fig_aada_vs_ffb.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"-> wrote {out}")


if __name__ == "__main__":
    main()
