"""
Short-term memory: tracks the lambda/metric trajectory WITHIN a single run.

The LLM reads this at every iteration to make an informed lambda decision.
It answers: "given where we've been in this run, what should lambda be next?"

Structure per entry:
{
    "iteration": 3,
    "lambda": [1.2, 0.8],
    "p_rules": {"sex": 61.3, "race": 74.2},
    "accuracy": 0.831,
    "adversary_loss": 0.412,
    "clf_task_loss": 0.298,
    "momentum": [0.15, 0.09]   ← kept for reference, LLM may override
}
"""

from typing import List, Dict, Any
import json


class ShortTermMemory:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def add(
        self,
        iteration: int,
        lambda_vector: List[float],
        p_rules: Dict[str, float],
        accuracy: float,
        adversary_loss: float,
        clf_task_loss: float,
        momentum: List[float] = None,
    ):
        entry = {
            "iteration": iteration,
            "lambda": [round(l, 4) for l in lambda_vector],
            "p_rules": {k: round(v, 2) for k, v in p_rules.items()},
            "accuracy": round(accuracy, 4),
            "adversary_loss": round(adversary_loss, 4),
            "clf_task_loss": round(clf_task_loss, 4),
            "momentum": [round(m, 4) for m in (momentum or [])]
        }
        self.history.append(entry)

    def last_n(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.history[-n:]

    def to_prompt_str(self, n: int = 6) -> str:
        """Compact JSON string for injection into LLM prompts."""
        recent = self.last_n(n)
        if not recent:
            return "No history yet (first iteration)."
        return json.dumps(recent, indent=2)

    def get_trend(self, attr: str) -> str:
        """
        Returns 'improving', 'stalling', or 'degrading' for a given
        sensitive attribute's p_rule trend over the last 3 iterations.
        """
        if len(self.history) < 2:
            return "unknown"
        recent = [h["p_rules"].get(attr, 0) for h in self.history[-3:]]
        delta = recent[-1] - recent[0]
        if delta > 2:
            return "improving"
        elif delta < -2:
            return "degrading"
        return "stalling"

    def reset(self):
        self.history = []
