"""
Long-term memory: persists successful runs across sessions as JSON.

Keyed by: "{dataset_name}|{target_col}|{sorted_sensitive_attrs}"
Each entry records the lambda trajectory and final outcome.

Fingerprint-based warm-start: when starting a new run, the system searches
ALL stored runs (across datasets) for a structurally similar one — same number
of sensitive attributes with similar group imbalance and class balance. If a
match is found (similarity >= 0.75), the lambda from that run's best iteration
is used AS-IS as the starting point (full value — no halving, no cap).
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


MEMORY_FILE = "long_term_memory.json"

# Max runs kept per dataset key. Older runs beyond this are trimmed on save. Raised
# from 10 so seed sweeps / repeated runs accumulate instead of silently dropping the
# oldest; override with AADA_MAX_RUNS_PER_KEY (set very high to effectively keep all).
MAX_RUNS_PER_KEY = int(os.environ.get("AADA_MAX_RUNS_PER_KEY", "1000"))


class LongTermMemory:
    def __init__(self, path: str = MEMORY_FILE):
        self.path = path
        self.data: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _load(self):
        # utf-8-sig so a UTF-8 BOM (e.g. a file last written by PowerShell) is
        # stripped instead of crashing json.load. A read failure here must NOT
        # silently reset self.data to {} and then overwrite the file on the next
        # _save() — that wipes every other dataset's runs. If the file exists but
        # cannot be parsed, raise so the run aborts with the data still on disk.
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8-sig") as f:
                self.data = json.load(f)

    def _save(self):
        # Always UTF-8 without a BOM so the file is parsed identically on every
        # platform (Windows default cp1252 would otherwise corrupt non-ASCII).
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def _key(self, dataset_name: str, target_col: str, sensitive_attrs: List[str]) -> str:
        return f"{dataset_name}|{target_col}|{','.join(sorted(sensitive_attrs))}"

    # Fingerprint computation & similarity

    @staticmethod
    def compute_fingerprint(state) -> dict:
        """
        Compute a dataset fingerprint from loaded tensors (available after
        load_dataset). Used to find structurally similar past runs for
        lambda warm-start.

        Features (all scalar, normalized to a common scale):
          n_sensitive          : number of sensitive attributes
          sensitive_imbalance  : minority fraction per attr (sorted) — [0, 0.5]
          target_balance       : fraction of positive class — [0, 1]
          n_samples_log10      : log10 of training-set size
          n_features_log10     : log10 of feature count
        """
        sensitive = np.asarray(state.sensitive_train.cpu().numpy()
                               if hasattr(state.sensitive_train, "numpy")
                               else state.sensitive_train, dtype=float)
        if sensitive.ndim == 1:
            sensitive = sensitive.reshape(-1, 1)

        imbalances = []
        for i in range(sensitive.shape[1]):
            col = sensitive[:, i]
            imbalances.append(round(min(float(col.mean()), 1.0 - float(col.mean())), 4))
        imbalances.sort()   # order-independent: sort so attr ordering doesn't affect match

        y = np.asarray(state.y_train.cpu().numpy()
                       if hasattr(state.y_train, "numpy")
                       else state.y_train, dtype=float)
        target_balance = round(float(y.mean()), 4)

        X = np.asarray(state.X_train.cpu().numpy()
                       if hasattr(state.X_train, "numpy")
                       else state.X_train)
        return {
            "n_sensitive":       sensitive.shape[1],
            "sensitive_imbalance": imbalances,
            "target_balance":    target_balance,
            "n_samples_log10":   round(float(np.log10(max(X.shape[0], 1))), 3),
            "n_features_log10":  round(float(np.log10(max(X.shape[1], 1))), 3),
        }

    @staticmethod
    def _fingerprint_similarity(fp1: dict, fp2: dict) -> float:
        """
        Scalar similarity in [0, 1].  0 = incompatible (different n_sensitive),
        1 = identical.  Each feature is normalised to [0, 1] before averaging.
        """
        if fp1 is None or fp2 is None:
            return 0.0
        if fp1.get("n_sensitive") != fp2.get("n_sensitive"):
            return 0.0

        imb1 = sorted(fp1.get("sensitive_imbalance", []))
        imb2 = sorted(fp2.get("sensitive_imbalance", []))
        if len(imb1) != len(imb2):
            return 0.0

        diffs = []
        # imbalance is in [0, 0.5] -> normalise by 0.5
        for a, b in zip(imb1, imb2):
            diffs.append(abs(a - b) / 0.5)
        # target balance is in [0, 1]
        diffs.append(abs(fp1.get("target_balance", 0.5) - fp2.get("target_balance", 0.5)))
        # log10(samples): typical range 3–5 -> normalise by 3
        diffs.append(min(abs(fp1.get("n_samples_log10", 0) - fp2.get("n_samples_log10", 0)) / 3.0, 1.0))
        # log10(features): typical range 1–3 -> normalise by 2
        diffs.append(min(abs(fp1.get("n_features_log10", 0) - fp2.get("n_features_log10", 0)) / 2.0, 1.0))

        return float(1.0 - float(np.mean(diffs)))

    # Save / retrieve

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
        fairness_final: Dict[str, Dict[str, float]] = None,
        fairness_baseline: Dict[str, Dict[str, float]] = None,
        fingerprint: dict = None,
        lambda_at_best: List[float] = None,
        seed: int = None,
        report: Dict[str, Any] = None,
    ):
        key = self._key(dataset_name, target_col, sensitive_attrs)
        entry = {
            "timestamp":          datetime.now().isoformat(),
            "seed":               seed,
            "lambda_final":       lambda_final,
            "lambda_at_best":     lambda_at_best or lambda_final,
            "p_rules_final":      p_rules_final,
            "accuracy_final":     round(accuracy_final, 4),
            "total_epochs":       total_epochs,
            "iterations":         iterations,
            "success":            success,
            "lambda_trajectory":  lambda_trajectory or [],
            "iteration_metrics":  iteration_metrics or [],
            "fairness_final":     fairness_final or {},
            "fairness_baseline":  fairness_baseline or {},
            "fingerprint":        fingerprint or {},
            "report":             report or {},
        }
        # Optional isolated output: when AADA_SAVE_MEMORY_FILE is set, the run is
        # written ONLY to that file (loaded/accumulated independently), leaving the
        # real long_term_memory.json untouched. Warm-start still reads self.data
        # (the real file), so an isolated experiment runs under identical conditions
        # without polluting the main memory or racing a concurrent sweep's writes.
        save_path = os.environ.get("AADA_SAVE_MEMORY_FILE")
        if save_path and save_path != self.path:
            target: Dict[str, List[Dict[str, Any]]] = {}
            if os.path.exists(save_path):
                with open(save_path, "r", encoding="utf-8-sig") as f:
                    target = json.load(f)
            target.setdefault(key, []).append(entry)
            target[key] = target[key][-MAX_RUNS_PER_KEY:]
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(target, f, indent=2, ensure_ascii=False)
            return

        if key not in self.data:
            self.data[key] = []
        self.data[key].append(entry)
        self.data[key] = self.data[key][-MAX_RUNS_PER_KEY:]
        self._save()

    def find_warm_start(
        self,
        current_fingerprint: dict,
        n_sensitive: int,
        similarity_threshold: float = 0.75,
        safety_factor: float = 1.0,
        max_lambda: float = float("inf"),
    ) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Search ALL stored runs for a successful one whose fingerprint is
        structurally similar to the current dataset.  Returns the warm-start
        lambda vector and a human-readable info string, or (None, None) when no
        suitable match is found.

        Behaviour:
          - Only successful runs (all P-rules >= fairness threshold; accuracy
            is not limited and does not affect success).
          - lambda_at_best (lambda active at the best training iteration) is
            used, not lambda_final (which can still be climbing at run end).
          - The matched lambda is used AS-IS (safety_factor defaults to 1.0 — no
            halving — and no cap), so the new run starts from the full lambda of
            the closest reference dataset.
        """
        best_sim = -1.0
        best_run = None
        best_key = None

        for key, runs in self.data.items():
            for run in runs:
                if not run.get("success"):
                    continue
                fp = run.get("fingerprint")
                if not fp:
                    continue
                sim = self._fingerprint_similarity(current_fingerprint, fp)
                if sim > best_sim:
                    best_sim = sim
                    best_run = run
                    best_key = key

        if best_sim < similarity_threshold or best_run is None:
            return None, None

        # Prefer lambda_at_best (most conservative useful lambda);
        # fall back to lambda_final only if not stored.
        ref_lambda = best_run.get("lambda_at_best") or best_run.get("lambda_final", [])
        if not ref_lambda or len(ref_lambda) != n_sensitive:
            return None, None

        warm = [round(min(l * safety_factor, max_lambda), 4) for l in ref_lambda]
        info = (
            f"fingerprint match: src={best_key}, sim={best_sim:.3f}, "
            f"safety={safety_factor}, λ_ref={ref_lambda} -> λ_warm={warm}"
        )
        return warm, info

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
                "success":        r["success"],
                "lambda_final":   r["lambda_final"],
                "p_rules_final":  r["p_rules_final"],
                "accuracy_final": r["accuracy_final"],
                "iterations":     r["iterations"],
            })
        return json.dumps(summary, indent=2)
