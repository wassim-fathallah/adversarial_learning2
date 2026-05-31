"""
Lambda management — fully deterministic, no LLM.

decide_initial_lambda       : fingerprint warm-start from long-term memory,
                               falls back to random [0.05, 0.30] if no match.
decide_lambda_for_iteration : pure momentum formula + running-max guard.

Fingerprint similarity gates:
  Hard : |n_sensitive_attrs_past - current| <= 1   (must pass)
  Soft : weighted distance < 0.30                  (must pass)
  Weights: class_imbalance=0.40, group_size_ratio=0.35, size_bucket=0.25

Running-max guard:
  While P-rule < threshold → λ is not allowed to drop below its historical max.
  Once P-rule >= threshold → λ can decrease freely to recover accuracy.
"""

import json
import random
import numpy as np
from typing import List, Dict, Any, Optional

from langchain.tools import tool
from state import state


LAMBDA_LEARNING_RATE = 2.5
MOMENTUM_BETA        = 0.7

# Fingerprint similarity thresholds
_HARD_GATE_N_ATTRS = 1      # max difference in n_sensitive_attrs
_SOFT_GATE_DIST    = 0.30   # max weighted distance
_SAFETY_MARGIN     = 0.85   # scale matched lambda down slightly
_BUCKET_ORDER      = {"small": 0, "medium": 1, "large": 2}

# Similarity weights (must sum to 1.0)
_W_CLASS  = 0.40   # class_imbalance  — directly scales L_task
_W_GROUP  = 0.35   # group_size_ratio — adversary signal quality
_W_BUCKET = 0.25   # dataset size     — convergence speed


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint similarity
# ─────────────────────────────────────────────────────────────────────────────

def _fingerprint_distance(fp_a: dict, fp_b: dict) -> float:
    """
    Weighted distance between two fingerprints (0 = identical, 1 = maximally different).
    Only called after the hard gate passes.
    """
    d_class  = abs(fp_a["class_imbalance"]  - fp_b["class_imbalance"])   / 0.5
    d_group  = abs(fp_a["group_size_ratio"] - fp_b["group_size_ratio"])   / 0.5
    b_a = _BUCKET_ORDER.get(fp_a["dataset_size_bucket"], 1)
    b_b = _BUCKET_ORDER.get(fp_b["dataset_size_bucket"], 1)
    d_bucket = abs(b_a - b_b) / 2.0

    return _W_CLASS * d_class + _W_GROUP * d_group + _W_BUCKET * d_bucket


def retrieve_similar_run(current_fp: dict, long_term_memory) -> Optional[dict]:
    """
    Search all past runs for the closest structural match to current_fp.

    Returns the best matching run dict (with lambda_final, p_rules_final…)
    or None if no run passes both gates.
    """
    best_run  = None
    best_dist = float("inf")

    for key, runs in long_term_memory.data.items():
        for run in runs:
            past_fp = run.get("fingerprint")
            if not past_fp:
                continue

            # Hard gate — n_sensitive_attrs must be close
            if abs(past_fp.get("n_sensitive_attrs", 0) - current_fp["n_sensitive_attrs"]) > _HARD_GATE_N_ATTRS:
                continue

            dist = _fingerprint_distance(current_fp, past_fp)

            # Soft gate
            if dist >= _SOFT_GATE_DIST:
                continue

            if dist < best_dist:
                best_dist = dist
                best_run  = run

    return best_run


# ─────────────────────────────────────────────────────────────────────────────
# Core momentum update
# ─────────────────────────────────────────────────────────────────────────────

def decide_lambda_for_iteration(
    current_metrics: Dict[str, Any],
    lambda_max: float = 20.0,
) -> List[float]:
    """
    Momentum-based lambda update — no LLM involved.

    Per attribute:
      gap       = (threshold - P_rule) / 100
      increment = LAMBDA_LEARNING_RATE × gap
      momentum  = 0.7 × old_momentum + 0.3 × increment
      λ_new     = clamp(λ + momentum, 0, lambda_max)

    Running-max guard:
      If P-rule < threshold → λ_new = max(λ_new, best_λ_seen_so_far[attr])
      If P-rule >= threshold → λ_new can decrease freely (recover accuracy)
    """
    p_rules   = current_metrics.get("p_rules", {}) or {}
    threshold = float(state.p_rule_threshold or 80.0)
    current   = list(state.lambda_vector) if state.lambda_vector else [0.1] * len(state.sensitive_attrs)

    if not state.lambda_momentum or len(state.lambda_momentum) != len(current):
        state.lambda_momentum = [0.0] * len(current)

    if not hasattr(state, "best_lambda_seen") or len(state.best_lambda_seen) != len(current):
        state.best_lambda_seen = [0.0] * len(current)

    updated = []
    for i, attr in enumerate(state.sensitive_attrs):
        prule    = float(p_rules.get(attr, 0.0))
        lambda_i = current[i]

        gap       = (threshold - prule) / 100.0
        increment = LAMBDA_LEARNING_RATE * gap

        state.lambda_momentum[i] = (
            MOMENTUM_BETA * state.lambda_momentum[i]
            + (1 - MOMENTUM_BETA) * increment
        )

        lambda_i_new = float(np.clip(lambda_i + state.lambda_momentum[i], 0.0, lambda_max))

        if prule < threshold:
            lambda_i_new = max(lambda_i_new, state.best_lambda_seen[i])

        state.best_lambda_seen[i] = max(state.best_lambda_seen[i], lambda_i_new)
        updated.append(lambda_i_new)

        print(
            f"  [λ] {attr}: p_rule={prule:.2f}%  gap={gap*100:.2f}%"
            f"  momentum={state.lambda_momentum[i]:.4f}"
            f"  λ {lambda_i:.4f} -> {lambda_i_new:.4f}"
            + (f"  [max-guard]" if prule < threshold and lambda_i_new == state.best_lambda_seen[i] and lambda_i_new > lambda_i + state.lambda_momentum[i] else "")
        )

    return updated


# ─────────────────────────────────────────────────────────────────────────────
# LangChain Tool — initial lambda with fingerprint warm-start
# ─────────────────────────────────────────────────────────────────────────────

@tool
def decide_initial_lambda(n_sensitive: int = 2) -> str:
    """
    Sets the initial lambda vector before training starts.

    Strategy (in order):
      1. Fingerprint warm-start: find the most structurally similar past run
         in long-term memory and scale its final lambda by 0.85.
      2. Random fallback: uniform [0.05, 0.30] per attribute.

    The fingerprint encodes: n_sensitive_attrs, class_imbalance,
    group_size_ratio, dataset_size_bucket.
    """
    if state.lambda_vector:
        return json.dumps({"initial_lambda": state.lambda_vector, "source": "cached"}, indent=2)

    n = len(state.sensitive_attrs) if state.sensitive_attrs else n_sensitive

    # Try fingerprint warm-start if fingerprint has been computed
    current_fp = getattr(state, "fingerprint", None)
    if current_fp:
        try:
            from memory.long_term import LongTermMemory
            lt = LongTermMemory()
            match = retrieve_similar_run(current_fp, lt)

            if match:
                past_lambda  = match.get("lambda_final", [])
                past_n       = match.get("fingerprint", {}).get("n_sensitive_attrs", n)
                past_prules  = match.get("p_rules_final", {})

                if past_lambda:
                    # Scale by attribute ratio and apply safety margin
                    scale   = (n / max(past_n, 1)) * _SAFETY_MARGIN
                    lambdas = [round(float(np.clip(l * scale, 0.05, 5.0)), 3) for l in past_lambda[:n]]
                    # Pad if needed
                    while len(lambdas) < n:
                        lambdas.append(round(random.uniform(0.05, 0.30), 3))

                    state.lambda_vector = lambdas
                    print(f"[λ-init] fingerprint warm-start: {lambdas}")
                    print(f"  matched run: p_rules={past_prules}, dist≈{_fingerprint_distance(current_fp, match['fingerprint']):.3f}")
                    return json.dumps({"initial_lambda": lambdas, "source": "fingerprint_warmstart"}, indent=2)
        except Exception as e:
            print(f"[λ-init] fingerprint lookup failed: {e} — using random")

    # Random fallback
    lambdas = [round(random.uniform(0.05, 0.30), 3) for _ in range(n)]
    state.lambda_vector = lambdas
    print(f"[λ-init] random start: {lambdas}")
    return json.dumps({"initial_lambda": lambdas, "source": "random [0.05, 0.30]"}, indent=2)
