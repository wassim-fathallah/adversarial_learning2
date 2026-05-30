"""
Long-term memory: persists successful runs across sessions as JSON.

Keyed by: "{dataset_name}|{target_col}|{sorted_sensitive_attrs}"
Each entry records the lambda trajectory and final outcome.

The LLM reads relevant past runs when deciding the INITIAL lambda and
when a run is stuck — past experience can suggest a proven strategy.
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


MEMORY_FILE = "long_term_memory.json"


class LongTermMemory:
    def __init__(self, path: str = MEMORY_FILE):
        self.path = path
        self.data: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.data = {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def _key(self, dataset_name: str, target_col: str, sensitive_attrs: List[str]) -> str:
        return f"{dataset_name}|{target_col}|{','.join(sorted(sensitive_attrs))}"

    def save_run(
        self,
        dataset_name: str,
        target_col: str,
        sensitive_attrs: List[str],
        lambda_final: List[float],
        p_rules_final: Dict[str, float],
        accuracy_final: float,
        total_epochs: int,
        iterations: int,
        success: bool,
        lambda_trajectory: List[List[float]] = None,
        iteration_metrics: List[Dict[str, Any]] = None,
    ):
        key = self._key(dataset_name, target_col, sensitive_attrs)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "lambda_final": lambda_final,
            "p_rules_final": p_rules_final,
            "accuracy_final": round(accuracy_final, 4),
            "total_epochs": total_epochs,
            "iterations": iterations,
            "success": success,
            "lambda_trajectory": lambda_trajectory or [],
            "iteration_metrics": iteration_metrics or [],
        }
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(entry)
        # Keep only last 10 runs per dataset/target/attrs combo
        self.data[key] = self.data[key][-10:]
        self._save()

    def get_relevant_runs(
        self,
        dataset_name: str,
        target_col: str,
        sensitive_attrs: List[str],
        successful_only: bool = False,
    ) -> List[Dict[str, Any]]:
        key = self._key(dataset_name, target_col, sensitive_attrs)
        runs = self.data.get(key, [])
        if successful_only:
            runs = [r for r in runs if r.get("success")]
        return runs

    def suggest_initial_lambda(
        self,
        dataset_name: str,
        target_col: str,
        sensitive_attrs: List[str],
        n_sensitive: int,
    ) -> List[float]:
        """
        Returns a warm-start lambda from past successful runs.
        Uses 50% of the best (lowest) successful lambda as starting point.
        Falls back to [0.1] * n_sensitive if no history.
        """
        runs = self.get_relevant_runs(dataset_name, target_col, sensitive_attrs, successful_only=True)
        if not runs:
            return [0.1] * n_sensitive
        # Pick run with highest p_rule achieved
        best = max(runs, key=lambda r: min(r["p_rules_final"].values()))
        warm = [max(0.05, l * 0.5) for l in best["lambda_final"]]
        return warm

    def to_prompt_str(
        self,
        dataset_name: str,
        target_col: str,
        sensitive_attrs: List[str],
    ) -> str:
        """Compact summary for LLM prompts — last 3 relevant runs."""
        runs = self.get_relevant_runs(dataset_name, target_col, sensitive_attrs)[-3:]
        if not runs:
            return "No past runs found for this dataset/attribute combination."
        summary = []
        for r in runs:
            summary.append({
                "success": r["success"],
                "lambda_final": r["lambda_final"],
                "p_rules_final": r["p_rules_final"],
                "accuracy_final": r["accuracy_final"],
                "iterations": r["iterations"],
            })
        return json.dumps(summary, indent=2)
