"""Per-attribute momentum ablation for HIMS-Tunisia. Reuses the two isolated memory
files from the ablation runs (no re-training). One panel per sensitive attribute,
each comparing WITH momentum (beta=0.7) vs WITHOUT momentum (beta=0)."""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ON_FILE  = sys.argv[1] if len(sys.argv) > 1 else "hims_mom_on.json"
OFF_FILE = sys.argv[2] if len(sys.argv) > 2 else "hims_mom_off.json"
OUT      = sys.argv[3] if len(sys.argv) > 3 else "fig_momentum_per_attr_hims.png"
BETA     = sys.argv[4] if len(sys.argv) > 4 else "0.7"
LBL_ON   = f"with momentum (β={BETA})"


def load(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
        # Auto-detect the memory key (each isolated ablation file holds exactly one),
        # so this works for any dataset name (HIMS-Tunisia, HIMS-Synth10k, ...).
        key = next(iter(data))
        im = data[key][-1]["iteration_metrics"]
    iters = [m["iteration"] for m in im]
    attrs = list(im[-1]["p_rules"].keys())
    per_attr = {a: [m["p_rules"][a] for m in im] for a in attrs}
    return iters, attrs, per_attr


it_on,  attrs, pa_on  = load(ON_FILE)
it_off, _,     pa_off = load(OFF_FILE)

# Honour early stop: cut each run at the first iteration whose MIN across attributes
# reaches 80% (the actual stop condition), so the curves end where the run stopped.
THRESH = 80.0
def _stop(it, pa):
    mins = [min(pa[a][i] for a in attrs) for i in range(len(it))]
    for i, v in enumerate(mins):
        if v >= THRESH:
            return i + 1
    return len(it)
# Both curves end at the iteration where the run stopped (with-momentum hitting the
# target); there is no run past it, so the without-momentum curve is cut there too.
c_on = _stop(it_on, pa_on)
c_off = c_on
it_on  = it_on[:c_on];   pa_on  = {a: pa_on[a][:c_on]  for a in attrs}
it_off = it_off[:c_off]; pa_off = {a: pa_off[a][:c_off] for a in attrs}

fig, axes = plt.subplots(1, len(attrs), figsize=(16, 4.5), sharey=True)
for ax, a in zip(axes, attrs):
    ax.plot(it_on,  pa_on[a],  "-o", color="#1f77b4", lw=2, ms=4, label=LBL_ON)
    ax.plot(it_off, pa_off[a], "-s", color="#d62728", lw=2, ms=4, label="without momentum (β=0)")
    ax.axhline(80, ls="--", color="gray", lw=1)
    ax.axvline(it_on[-1], ls=":", color="#1f77b4", lw=1.2, alpha=0.7)   # early-stop iter
    ax.set_title(a)
    ax.set_xlabel("Iteration")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("P-rule (%)")
axes[0].legend(loc="lower right", fontsize=9)
fig.suptitle("HIMS-Tunisia — momentum ablation per sensitive attribute (grouped P-rule)")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"[plot] saved {OUT}")
for a in attrs:
    print(f"  {a:15s} with: end={pa_on[a][-1]:.1f} peak={max(pa_on[a]):.1f} | "
          f"without: end={pa_off[a][-1]:.1f} peak={max(pa_off[a]):.1f}")
