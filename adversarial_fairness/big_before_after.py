# -*- coding: utf-8 -*-
"""Render the before/after debiasing KDE LARGE — one image per sensitive attribute,
so each subgroup curve is easy to read. Reads the saved report plotdata (no training).

    python big_before_after.py HIMS-10k
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except Exception:
    sns = None

name = sys.argv[1] if len(sys.argv) > 1 else "HIMS-10k"
HERE = os.path.dirname(os.path.abspath(__file__))
rep  = os.path.join(HERE, "reports")
d    = np.load(os.path.join(rep, f"report_{name}_plotdata.npz"))
meta = json.load(open(os.path.join(rep, f"report_{name}_plotmeta.json"), encoding="utf-8"))

probs, base = d["probs"], d["base_probs"]
codes = d["codes"]
attrs, target = meta["attrs"], meta["target_col"]
cmap = plt.get_cmap("tab10")

for i, a in enumerate(attrs):
    sj = meta["stats"][a]
    labels = {int(c): l for c, l in sj["labels"].items()}
    groups = {int(c): g for c, g in sj["groups"].items()}
    code_i = codes[i].astype(int)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for j, c in enumerate(sorted(groups)):
        col = cmap(j % 10)
        pf, pb = probs[code_i == c], base[code_i == c]
        lbl = labels.get(c, f"group {c}")
        if sns is not None and pf.size >= 5 and np.std(pf) > 1e-4:
            sns.kdeplot(x=pb, ax=ax, color=col, ls="--", lw=2.0, clip=(0, 1),
                        label=f"{lbl} — before")
            sns.kdeplot(x=pf, ax=ax, color=col, ls="-", lw=2.8, clip=(0, 1),
                        label=f"{lbl} — after")
        else:
            ax.axvline(np.mean(pb), color=col, ls="--", lw=2.0)
            ax.axvline(np.mean(pf), color=col, ls="-", lw=2.8)
    prb, pra = sj.get("p_rule_before"), sj["p_rule"]
    box = (f"P-rule\nbefore = {prb:.0f}%\nafter  = {pra:.0f}%"
           if prb is not None else f"P-rule after = {pra:.0f}%")
    ax.text(0.5, 0.98, box, transform=ax.transAxes, fontsize=12, va="top", ha="center",
            bbox=dict(boxstyle="round", facecolor="#eef3ff", edgecolor="#1f77b4", alpha=0.9))
    ax.set_title(f"HIMS-Tunisia — {a}: before (dashed) -> after (solid) debiasing", fontsize=14)
    ax.set_xlabel(f"P({target} = 1)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=11, loc="upper left")
    fig.tight_layout()
    out = os.path.join(rep, f"report_{name}_BA_{a}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[plot] {out}")
