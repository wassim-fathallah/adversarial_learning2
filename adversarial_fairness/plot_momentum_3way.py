# -*- coding: utf-8 -*-
"""
Per-attribute momentum comparison for HIMS-10k, EARLY-STOP aware.

Unlike plot_momentum_per_attr.py (which truncates both curves at one shared stop),
this plots EACH run to its OWN early-stop iteration and marks it — so you can see
that a faster-converging beta stops sooner. One panel per sensitive attribute.

Usage (pairs of  file  beta , 2 or 3 of them):
    python plot_momentum_3way.py synth_mom_b09.json 0.9 synth_mom_b07.json 0.7 synth_mom_b00.json 0
    python plot_momentum_3way.py synth_mom_b09.json 0.9 synth_mom_b07.json 0.7
"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESH = 80.0
OUT = "fig_momentum_3way_hims10k.png"
# colour/marker per curve, in argv order
STYLE = [("#1f77b4", "-o"), ("#2ca02c", "-^"), ("#d62728", "-s"), ("#9467bd", "-d")]


def load(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    key = next(iter(data))                       # isolated file → one key
    im = data[key][-1]["iteration_metrics"]
    iters = [m["iteration"] for m in im]
    attrs = list(im[-1]["p_rules"].keys())
    per_attr = {a: [m["p_rules"][a] for m in im] for a in attrs}
    return iters, attrs, per_attr


# Parse argv pairs: file beta file beta ...
args = sys.argv[1:]
runs = []
for i in range(0, len(args) - 1, 2):
    path, beta = args[i], args[i + 1]
    iters, attrs, per_attr = load(path)
    runs.append({"beta": beta, "iters": iters, "attrs": attrs, "pa": per_attr})

ref_attrs = runs[0]["attrs"]
fig, axes = plt.subplots(1, len(ref_attrs), figsize=(18, 6.2), sharey=True)
if len(ref_attrs) == 1:
    axes = [axes]

for ax, a in zip(axes, ref_attrs):
    for r, (col, mk) in zip(runs, STYLE):
        it, pa = r["iters"], r["pa"][a]
        lbl = (f"without momentum (β=0)" if float(r["beta"]) == 0
               else f"with momentum (β={r['beta']})")
        ax.plot(it, pa, mk, color=col, lw=2, ms=4, label=lbl)
        # mark this run's early-stop (its final iteration)
        ax.axvline(it[-1], ls=":", color=col, lw=1.1, alpha=0.6)
    ax.axhline(THRESH, ls="--", color="gray", lw=1)
    ax.set_title(a)
    ax.set_xlabel("Iteration")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("P-rule (%)")
axes[0].legend(loc="lower right", fontsize=9)
fig.suptitle("HIMS-10k — momentum comparison per sensitive attribute "
             "(grouped P-rule; dotted = each run's early-stop)")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"[plot] saved {OUT}")
for r in runs:
    stop = r["iters"][-1]
    mins = min(r["pa"][a][-1] for a in ref_attrs)
    print(f"  beta={r['beta']:>3}: stopped at iter {stop:2d}  | final P-rules "
          + ", ".join(f"{a}={r['pa'][a][-1]:.0f}" for a in ref_attrs))
