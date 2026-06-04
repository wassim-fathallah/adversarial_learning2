"""
Training curves: P-rule, Equalized Odds/Opportunity, Accuracy, F1, ROC-AUC, Lambda.
Saved to PNG after training completes.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from typing import List, Dict
import os


class TrainingPlotter:
    def __init__(self, sensitive_attrs: List[str], save_dir: str = "."):
        self.sensitive_attrs = sensitive_attrs
        self.save_dir = save_dir

        self.iterations:  List[int]   = []
        self.accuracies:  List[float] = []
        self.precisions:  List[float] = []
        self.f1_scores:   List[float] = []
        self.roc_aucs:    List[float] = []
        self.adv_losses:  List[float] = []

        # Per sensitive attribute
        self.p_rules:      Dict[str, List[float]] = {a: [] for a in sensitive_attrs}
        self.eq_odds_tpr:  Dict[str, List[float]] = {a: [] for a in sensitive_attrs}
        self.eq_odds_fpr:  Dict[str, List[float]] = {a: [] for a in sensitive_attrs}
        self.eq_opp_tpr:   Dict[str, List[float]] = {a: [] for a in sensitive_attrs}
        self.lambdas:      Dict[str, List[float]] = {a: [] for a in sensitive_attrs}

    def update(
        self,
        iteration: int,
        accuracy: float,
        p_rules: Dict[str, float],
        lambdas: List[float],
        adv_loss: float,
        precision: float = 0.0,
        f1: float = 0.0,
        roc_auc: float = 0.0,
        fairness: Dict = None,
    ):
        self.iterations.append(iteration)
        self.accuracies.append(accuracy)
        self.precisions.append(precision)
        self.f1_scores.append(f1)
        self.roc_aucs.append(roc_auc)
        self.adv_losses.append(adv_loss)

        for i, attr in enumerate(self.sensitive_attrs):
            self.p_rules[attr].append(p_rules.get(attr, 0.0))
            if i < len(lambdas):
                self.lambdas[attr].append(lambdas[i])
            # Equalized odds/opportunity from nested fairness dict
            if fairness and attr in fairness:
                eo  = fairness[attr].get("equalized_odds", {})
                eop = fairness[attr].get("equalized_opportunity", {})
                self.eq_odds_tpr[attr].append(eo.get("tpr_gap", 0.0))
                self.eq_odds_fpr[attr].append(eo.get("fpr_gap", 0.0))
                self.eq_opp_tpr[attr].append(eop.get("tpr_gap", 0.0))
            else:
                self.eq_odds_tpr[attr].append(0.0)
                self.eq_odds_fpr[attr].append(0.0)
                self.eq_opp_tpr[attr].append(0.0)

    def save(self, filename: str = "training_curves.png"):
        COLORS = ["#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
        iters  = self.iterations
        n_rows = 5
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)
        fig.suptitle("Adversarial Fairness Training — All Metrics", fontsize=14, y=1.01)

        # Row 1: Performance
        ax = axes[0]
        ax.plot(iters, self.accuracies, "b-o", ms=4, label="Accuracy")
        ax.plot(iters, self.f1_scores,  "g-s", ms=4, label="F1 Score")
        ax.plot(iters, self.precisions, "m-^", ms=4, label="Precision")
        ax.plot(iters, self.roc_aucs,   "c-D", ms=4, label="ROC-AUC")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Performance Metrics")

        # Row 2: P-rule
        ax = axes[1]
        for i, attr in enumerate(self.sensitive_attrs):
            ax.plot(iters, self.p_rules[attr], color=COLORS[i], marker="s", ms=4, label=f"P-rule ({attr})")
        ax.axhline(y=80, color="gray", linestyle="--", alpha=0.7, label="80% threshold")
        ax.set_ylabel("P-rule (%)")
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Disparate Impact (P-rule) per Attribute")

        # Row 3: Equalized Odds (TPR gap)
        ax = axes[2]
        for i, attr in enumerate(self.sensitive_attrs):
            ax.plot(iters, self.eq_odds_tpr[attr], color=COLORS[i], marker="o", ms=4,
                    linestyle="-",  label=f"EqOdds TPR gap ({attr})")
            ax.plot(iters, self.eq_odds_fpr[attr], color=COLORS[i], marker="^", ms=4,
                    linestyle="--", label=f"EqOdds FPR gap ({attr})", alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel("Gap (lower = fairer)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_title("Equalized Odds — TPR & FPR Gap per Attribute")

        # Row 4: Equalized Opportunity (TPR gap only)
        ax = axes[3]
        for i, attr in enumerate(self.sensitive_attrs):
            ax.plot(iters, self.eq_opp_tpr[attr], color=COLORS[i], marker="D", ms=4,
                    label=f"EqOpp TPR gap ({attr})")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel("TPR gap (lower = fairer)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Equalized Opportunity — TPR Gap per Attribute")

        # Row 5: Lambda (decisions)
        ax = axes[4]
        for i, attr in enumerate(self.sensitive_attrs):
            if self.lambdas[attr]:
                ax.plot(iters, self.lambdas[attr], color=COLORS[i], marker="^", ms=4,
                        label=f"λ ({attr})")
        ax.set_ylabel("Lambda (λ)")
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Lambda Decisions by LLM Orchestrator")

        plt.tight_layout()
        out_path = os.path.join(self.save_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] Saved: {out_path}")
        return out_path
