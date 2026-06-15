import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Always resolve CSVs next to THIS script, regardless of current directory.
HERE = os.path.dirname(os.path.abspath(__file__))
f07 = os.path.join(HERE, "lambda_log_beta07.csv")
f00 = os.path.join(HERE, "lambda_log_beta00.csv")
for f in (f07, f00):
    if not os.path.exists(f):
        raise SystemExit(
            f"Missing {os.path.basename(f)} — the beta run has not finished yet. "
            "Wait for both runs to complete before plotting."
        )

# Charger les logs (zero-init runs)
cols = ["iteration", "attr", "lambda", "beta"]
df07 = pd.read_csv(f07, names=cols)
df00 = pd.read_csv(f00, names=cols)

attrs = ["sex", "race"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, attr in zip(axes, attrs):
    d07 = df07[df07["attr"] == attr].sort_values("iteration")
    d00 = df00[df00["attr"] == attr].sort_values("iteration")

    # Prepend iteration 0 at lambda = 0 so both curves start from the origin.
    it07 = [0] + d07["iteration"].tolist()
    la07 = [0.0] + d07["lambda"].tolist()
    it00 = [0] + d00["iteration"].tolist()
    la00 = [0.0] + d00["lambda"].tolist()

    target = d07["lambda"].iloc[-1]
    ax.plot(it00, la00, "r--o", label="Without momentum (beta=0)")
    ax.plot(it07, la07, "b-^", label="With momentum (beta=0.7)")
    ax.axhline(target, color="green", linestyle=":",
               label=f"Operating point (lambda={target:.2f})")
    ax.set_title(f"Adult - attribute: {attr}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("lambda value")
    ax.set_xlim(left=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Lambda convergence: with vs. without momentum (Adult)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
out = os.path.join(HERE, "fig_momentum_convergence.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Figure saved: {out}")
