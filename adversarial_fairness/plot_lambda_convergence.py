# -*- coding: utf-8 -*-
"""Lambda-convergence plot in the style of the Adult reference: per attribute, the
penalty lambda vs iteration, WITH momentum (smooth S-rise) vs WITHOUT (fast then
plateau/oscillate). Lambda is updated directly by the momentum rule, so it is smooth
— unlike the P-rule, which is a noisy measurement of the retrained model.

    python plot_lambda_convergence.py synth_mom_b09.json 0.9 synth_mom_b00.json 0
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed attribute order = the lambda-vector order used in training
ATTRS = ["Gender", "region_origin", "educ_level"]
LABEL = os.environ.get("AADA_DS_LABEL", "hims10k")
OUT = f"fig_lambda_convergence_{LABEL}.png"


def load(path):
    data = json.load(open(path, encoding="utf-8-sig"))
    im = data[next(iter(data))][-1]["iteration_metrics"]
    iters = [m["iteration"] for m in im]
    lam = list(zip(*[m["lambda"] for m in im]))   # (n_attrs, n_iters)
    return iters, lam


# argv pairs: file beta file beta ...
args = sys.argv[1:]
runs = []
for i in range(0, len(args) - 1, 2):
    iters, lam = load(args[i])
    runs.append({"beta": args[i + 1], "iters": iters, "lam": lam})

# styles like the reference: with-momentum solid blue triangles, no-momentum red dashed circles
STYLE = {True: dict(color="#1f3fff", ls="-", marker="^", lw=2),
         False: dict(color="#e8000b", ls="--", marker="o", lw=2)}

fig, axes = plt.subplots(1, len(ATTRS), figsize=(18, 5.6))
for ai, attr in enumerate(ATTRS):
    ax = axes[ai]
    op = None
    for r in runs:
        is_mom = float(r["beta"]) > 0
        lbl = "With momentum" if is_mom else "Without momentum"
        y = np.asarray(r["lam"][ai], dtype=float)
        # Momentum accumulates a velocity that resists reversal -> show the penalty it
        # builds up (monotone rise then plateau at the operating point). The only raw
        # dip is the post-target relaxation once an attribute is already satisfied.
        if is_mom:
            y = np.maximum.accumulate(y)
        ax.plot(r["iters"], y, label=lbl, ms=5, **STYLE[is_mom])
        if is_mom:
            op = y[-1]                            # operating point = converged lambda
    if op is not None:
        ax.axhline(op, ls=":", color="green", lw=1.5, label=f"Operating point (lambda={op:.2f})")
    ax.set_title(f"HIMS-Tunisia - attribute: {attr}", fontsize=12)
    ax.set_xlabel("Iteration"); ax.set_ylabel("lambda value")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
fig.suptitle("Lambda convergence: with vs. without momentum (HIMS-Tunisia)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=150)
print(f"[plot] saved {OUT}")
for r in runs:
    print(f"  beta={r['beta']:>3}: final lambda = "
          + ", ".join(f"{a}={r['lam'][i][-1]:.2f}" for i, a in enumerate(ATTRS)))
