# -*- coding: utf-8 -*-
"""Equalized-odds before/after bar chart from a saved run (memory file), titled
HIMS-Tunisia, no beta annotation.

    python plot_eodd.py real_mom_b09.json
"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "real_mom_b09.json"
d = json.load(open(path, encoding="utf-8-sig"))
e = d[next(iter(d))][-1]
attrs = ["Gender", "region_origin", "educ_level"]
fb = e.get("fairness_baseline", {}) or {}
ff = e.get("fairness_final", {}) or {}
before = [fb.get(a, {}).get("eodd", 0.0) for a in attrs]
after  = [ff.get(a, {}).get("eodd", 0.0) for a in attrs]

x = np.arange(len(attrs)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.6))
b1 = ax.bar(x - w / 2, before, w, label="before", color="#7f7f7f")
b2 = ax.bar(x + w / 2, after,  w, label="after",  color="#1f77b4")
ax.bar_label(b1, fmt="%.1f", fontsize=9); ax.bar_label(b2, fmt="%.1f", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(attrs)
ax.set_ylabel("Equalized-odds gap (%) - lower is fairer")
ax.set_title("HIMS-Tunisia - equalized odds before vs after")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig("fig_eodd_HIMS-Tunisia.png", dpi=150)
print("saved fig_eodd_HIMS-Tunisia.png  before", [round(v,2) for v in before], "after", [round(v,2) for v in after])
