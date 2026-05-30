"""
Lambda management — fully deterministic, no LLM.

decide_initial_lambda : random warm start in [0.05, 0.30]
decide_lambda_for_iteration : pure momentum formula + running-max guard

Running-max guard:
  While P-rule < threshold → λ is not allowed to drop below its historical max
  (prevents momentum from oscillating λ down when fairness is still not met)
  Once P-rule >= threshold → λ can decrease freely to recover accuracy.
"""

import json
import numpy as np
from typing import List, Dict, Any

from langchain.tools import tool
from state import state


LAMBDA_LEARNING_RATE = 2.5
MOMENTUM_BETA        = 0.7


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

    # Running max: track highest λ ever used per attribute
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

        # Running-max guard: don't let λ fall while P-rule is still below target
        if prule < threshold:
            lambda_i_new = max(lambda_i_new, state.best_lambda_seen[i])

        # Update running max
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
# LangChain Tool — initial lambda (random start, no LLM, no history lookup)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def decide_initial_lambda(n_sensitive: int = 2) -> str:
    """
    Sets the initial lambda vector before training starts.
    Uses a random value in [0.05, 0.30] per sensitive attribute.
    No LLM call. No history lookup.
    Momentum will adjust from this starting point within the first few iterations.
    """
    if state.lambda_vector:
        return json.dumps({"initial_lambda": state.lambda_vector, "source": "cached"}, indent=2)

    n = len(state.sensitive_attrs) if state.sensitive_attrs else n_sensitive
    lambdas = [0.0] * n

    state.lambda_vector = lambdas

    print(f"[λ-init] start: {lambdas}")
    return json.dumps({"initial_lambda": lambdas, "source": "zero start"}, indent=2)
