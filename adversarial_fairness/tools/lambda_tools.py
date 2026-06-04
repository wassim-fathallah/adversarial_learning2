"""
Lambda management — fully deterministic, no LLM.

decide_initial_lambda       : zero-initialized lambda per attribute.
decide_lambda_for_iteration : pure momentum formula + running-max guard.

Running-max guard:
  While P-rule < threshold → λ is not allowed to drop below its historical max.
  Once P-rule >= threshold → λ can decrease freely to recover accuracy.
"""

import json
import numpy as np
from typing import List, Dict, Any

from langchain.tools import tool
from state import state
from memory.long_term import LongTermMemory


LAMBDA_LEARNING_RATE = 2.5
MOMENTUM_BETA        = 0.7


# Core momentum update

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


# LangChain Tool — initial lambda (zero start)

@tool
def decide_initial_lambda(n_sensitive: int = 2) -> str:
    """
    Sets the initial lambda vector before training starts.

    Strategy (in priority order):
      1. Already set this session (cached) — reuse.
      2. Fingerprint match in long-term memory — warm start at 50% of the
         lambda from the most structurally similar past successful run
         (similarity >= 0.75; cap at 5.0 per attribute).
      3. Zero init — no relevant history found.
    """
    if state.lambda_vector:
        return json.dumps({"initial_lambda": state.lambda_vector, "source": "cached"}, indent=2)

    n = len(state.sensitive_attrs) if state.sensitive_attrs else n_sensitive

    # Fingerprint-based warm start: requires data to be loaded into state
    if state.X_train is not None and state.sensitive_train is not None:
        try:
            lt = LongTermMemory()
            fp = LongTermMemory.compute_fingerprint(state)
            warm, info = lt.find_warm_start(fp, n)
            if warm is not None:
                state.lambda_vector = warm
                print(f"[λ-init] fingerprint warm start: {info}")
                return json.dumps(
                    {"initial_lambda": warm, "source": "fingerprint", "match_info": info},
                    indent=2,
                )
        except Exception as e:
            print(f"[λ-init] fingerprint search failed ({e}), falling back to zero init")

    lambdas = [0.0] * n
    state.lambda_vector = lambdas
    print(f"[λ-init] zero start: {lambdas}")
    return json.dumps({"initial_lambda": lambdas, "source": "zero"}, indent=2)
