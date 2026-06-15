"""
Momentum ablation, fair version: run Adult TWICE with identical settings except
the momentum coefficient, BOTH forced to zero-init lambda (no fingerprint
warm-start), so the lambda curves start from the same point (0).

Controlled by env var AADA_MOMENTUM_BETA (0 or 0.7). Writes the lambda + p_rule
trajectory to a per-beta file. Does NOT touch the real long_term_memory.json:
it disables both warm-start (by clearing memory data) and saving (no-op save).

    $env:AADA_MOMENTUM_BETA=0   ; python run_momentum_ablation.py
    $env:AADA_MOMENTUM_BETA=0.7 ; python run_momentum_ablation.py
"""
import os
import sys
import json

# This script lives in adversarial_fairness/seeds/; put the package dir
# (adversarial_fairness/) on sys.path so orchestrator / state / memory import.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BETA = os.environ.get("AADA_MOMENTUM_BETA", "0.7")

# 1. Neutralise long-term memory: no warm-start in. For save_out, instead of a
#    no-op we INTERCEPT save_run to grab the iteration_metrics it is handed by
#    run_full_training (that is the real per-iteration trajectory), then discard.
import memory.long_term as ltmod
ltmod.LongTermMemory.find_warm_start = lambda self, *a, **k: (None, None)

_captured = {}
def _capture_save(self, *args, **kwargs):
    _captured["iteration_metrics"] = kwargs.get("iteration_metrics", [])
ltmod.LongTermMemory.save_run = _capture_save

from orchestrator import run_pipeline
from state import state

print(f"\n>>> Adult ablation: MOMENTUM_BETA={BETA}, zero-init forced <<<\n")

result = run_pipeline(
    dataset_path=os.path.join(_PARENT, "datasets/adult/raw/adult.data"),
    dataset_name=f"adult_ablation_b{BETA}",
    max_iterations=15,
    epochs_per_step=12,   # fast: fewer epochs/iter
    p_rule_threshold=80.0,
    initial_epochs=5,     # fast: shorter pretrain
)

ims = _captured.get("iteration_metrics")
if not ims:
    print("ERROR: no iteration_metrics captured; cannot save trajectory.")
    sys.exit(1)

traj = [{"iteration": m["iteration"], "lambda": m["lambda"],
         "p_rules": m["p_rules"], "accuracy": m["accuracy"]} for m in ims]
out = {"beta": float(BETA), "dataset": "adult", "zero_init": True, "trajectory": traj}
fname = f"ablation_adult_b{BETA}.json"
json.dump(out, open(fname, "w"), indent=2)
print(f"\n>>> saved {fname}  ({len(traj)} iterations)")
print("    lambda_sex :", [round(m["lambda"][1], 3) for m in traj])
print("    p_rule_sex :", [round(m["p_rules"].get("sex", 0), 1) for m in traj])
