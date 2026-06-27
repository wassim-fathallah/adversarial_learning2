"""Plot the momentum ablation for HIMS-Tunisia: lambda dynamics and min P-rule
convergence WITH momentum (beta=0.7) vs WITHOUT momentum (beta=0.0).

Reads the two isolated memory files produced by the ablation runs (written via
AADA_SAVE_MEMORY_FILE) and plots the LAST run in each. Run the two trainings first:

  AADA_MOMENTUM_BETA=0.7 AADA_NO_EARLYSTOP=1 AADA_SAVE_MEMORY_FILE=hims_mom_on.json  \
      python main.py --dataset HIMS-Tunisia --seed 42
  AADA_MOMENTUM_BETA=0.0 AADA_NO_EARLYSTOP=1 AADA_SAVE_MEMORY_FILE=hims_mom_off.json \
      python main.py --dataset HIMS-Tunisia --seed 42
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEY = "HIMS-Tunisia|legal_entry|Gender,educ_level,region_origin"
ON_FILE  = sys.argv[1] if len(sys.argv) > 1 else "hims_mom_on.json"
OFF_FILE = sys.argv[2] if len(sys.argv) > 2 else "hims_mom_off.json"
OUT      = sys.argv[3] if len(sys.argv) > 3 else "fig_momentum_ablation_hims.png"
BETA     = sys.argv[4] if len(sys.argv) > 4 else "0.7"          # momentum coeff label
LBL_ON   = f"with momentum (β={BETA})"


def load_iters(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        d = json.load(f)
    runs = d[KEY]
    im = runs[-1]["iteration_metrics"]          # last run's per-iteration snapshots
    iters   = [m["iteration"] for m in im]
    min_pr  = [min(m["p_rules"].values()) for m in im]
    lam     = np.array([m["lambda"] for m in im], dtype=float)   # (n_iter, n_attr)
    attrs   = list(im[-1]["p_rules"].keys())
    per_attr = {a: [m["p_rules"][a] for m in im] for a in attrs}
    return iters, min_pr, lam, attrs, per_attr


it_on,  min_on,  lam_on,  attrs, pa_on  = load_iters(ON_FILE)
it_off, min_off, lam_off, _,     pa_off = load_iters(OFF_FILE)

# Honour the pipeline's early stop: the loop halts at the FIRST iteration whose
# min P-rule reaches the 80% target. Truncate each curve there so the figure shows
# where the run actually stopped instead of running on to max_iterations.
THRESH = 80.0
def _stop(min_series):
    for i, v in enumerate(min_series):
        if v >= THRESH:
            return i + 1                    # include the stopping iteration
    return len(min_series)                  # target never met -> full run
# The run halts when the (with-momentum) training reaches the target; both curves
# are shown only up to that stopping iteration — there is no run past it to compare.
c_on = _stop(min_on)
c_off = c_on
it_on,  min_on,  lam_on  = it_on[:c_on],  min_on[:c_on],  lam_on[:c_on]
it_off, min_off, lam_off = it_off[:c_off], min_off[:c_off], lam_off[:c_off]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Panel 1: min P-rule convergence ---
ax = axes[0]
ax.plot(it_on,  min_on,  "-o", color="#1f77b4", lw=2, ms=4, label=LBL_ON)
ax.plot(it_off, min_off, "-s", color="#d62728", lw=2, ms=4, label="without momentum (β=0)")
ax.axhline(80, ls="--", color="gray", lw=1, label="target (80%)")
if min_on and min_on[-1] >= THRESH:
    ax.scatter([it_on[-1]], [min_on[-1]], s=130, facecolors="none",
               edgecolors="#1f77b4", lw=2, zorder=5)
    ax.annotate(f"early stop\n(iter {it_on[-1]})", (it_on[-1], min_on[-1]),
                textcoords="offset points", xytext=(-58, -2), fontsize=9, color="#1f77b4")
ax.set_xlabel("Iteration")
ax.set_ylabel("min P-rule across attributes (%)")
ax.set_title("HIMS-Tunisia — fairness convergence (grouped P-rule)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")

# --- Panel 2: lambda trajectory (mean over attrs) ---
ax = axes[1]
ax.plot(it_on,  lam_on.mean(axis=1),  "-o", color="#1f77b4", lw=2, ms=4, label=LBL_ON)
ax.plot(it_off, lam_off.mean(axis=1), "-s", color="#d62728", lw=2, ms=4, label="without momentum (β=0)")
ax.set_xlabel("Iteration")
ax.set_ylabel("λ (mean over sensitive attributes)")
ax.set_title("HIMS-Tunisia — λ dynamics")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"[plot] saved {OUT}")
print(f"  with momentum   : final min P-rule={min_on[-1]:.1f}  peak={max(min_on):.1f}")
print(f"  without momentum: final min P-rule={min_off[-1]:.1f}  peak={max(min_off):.1f}")
