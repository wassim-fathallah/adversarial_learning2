# -*- coding: utf-8 -*-
"""Running-max ("best P-rule so far") view of a momentum run — monotonic by
construction. Shows the raw per-iteration P-rule faint behind the monotone curve,
so it is clearly a 'best achieved up to iter N' summary, not a falsified raw curve.

    python plot_momentum_monotone.py synth_mom_b09.json 0.9
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "synth_mom_b09.json"
beta = sys.argv[2] if len(sys.argv) > 2 else "0.9"
OUT  = f"fig_momentum_monotone_b{beta}.png"

data = json.load(open(path, encoding="utf-8-sig"))
im = data[next(iter(data))][-1]["iteration_metrics"]
iters = [m["iteration"] for m in im]
attrs = list(im[-1]["p_rules"].keys())

fig, axes = plt.subplots(1, len(attrs), figsize=(16, 4.8), sharey=True)
for ax, a in zip(axes, attrs):
    raw = np.array([m["p_rules"][a] for m in im])
    best = np.maximum.accumulate(raw)            # running max -> monotone non-decreasing
    ax.plot(iters, raw,  "-o", color="0.7", lw=1.2, ms=3, label="raw P-rule (noisy)")
    ax.plot(iters, best, "-o", color="#1f77b4", lw=2.4, ms=4, label="best so far (monotone)")
    ax.axhline(80, ls="--", color="gray", lw=1)
    ax.set_title(a); ax.set_xlabel("Iteration"); ax.grid(alpha=0.3)
axes[0].set_ylabel("P-rule (%)")
axes[0].legend(loc="lower right", fontsize=9)
fig.suptitle(f"HIMS-10k - beta={beta}: best-so-far P-rule (monotone) vs raw")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"[plot] saved {OUT}")
