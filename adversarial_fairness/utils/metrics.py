"""
Fairness and performance metrics.

Fairness metrics (per sensitive attribute):
  - P-rule (Disparate Impact)       → min(P(ŷ=1|z=0)/P(ŷ=1|z=1), reverse) * 100
  - Equalized Odds                  → max gap in TPR AND FPR across groups
  - Equalized Opportunity           → gap in TPR (recall) across groups
  - Demographic Parity difference   → |P(ŷ=1|z=0) - P(ŷ=1|z=1)|

Performance metrics:
  - Accuracy
  - Precision
  - F1 Score
  - ROC-AUC
"""

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, roc_auc_score, average_precision_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple


# Fairness metrics

def p_rule(y_pred: np.ndarray, sensitive: np.ndarray, threshold: float = 0.5) -> float:
    """Disparate impact ratio * 100. Target: >= 80."""
    y_bin = (y_pred >= threshold).astype(float)
    mask0 = sensitive == 0
    mask1 = sensitive == 1
    if mask0.sum() == 0 or mask1.sum() == 0:
        return 100.0
    p0 = y_bin[mask0].mean()
    p1 = y_bin[mask1].mean()
    if p0 == 0 and p1 == 0:
        return 100.0
    if p0 == 0 or p1 == 0:
        return 0.0
    return min(p0 / p1, p1 / p0) * 100.0


def _tpr_fpr(y_pred_bin: np.ndarray, y_true: np.ndarray, mask: np.ndarray):
    """True positive rate and false positive rate for a subgroup."""
    y_sub = y_true[mask]
    p_sub = y_pred_bin[mask]
    pos = y_sub == 1
    neg = y_sub == 0
    tpr = p_sub[pos].mean() if pos.sum() > 0 else 0.0
    fpr = p_sub[neg].mean() if neg.sum() > 0 else 0.0
    return float(tpr), float(fpr)


def equalized_odds(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Equalized Odds: both TPR and FPR should be equal across groups.
    Returns max gap in TPR and max gap in FPR (lower = fairer).
    """
    y_bin = (y_pred >= threshold).astype(int)
    mask0 = sensitive == 0
    mask1 = sensitive == 1
    tpr0, fpr0 = _tpr_fpr(y_bin, y_true, mask0)
    tpr1, fpr1 = _tpr_fpr(y_bin, y_true, mask1)
    return {
        "tpr_gap": round(abs(tpr0 - tpr1), 4),   # 0 = perfect
        "fpr_gap": round(abs(fpr0 - fpr1), 4),
        "tpr_group0": round(tpr0, 4),
        "tpr_group1": round(tpr1, 4),
        "fpr_group0": round(fpr0, 4),
        "fpr_group1": round(fpr1, 4),
    }


def equalized_opportunity(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Equalized Opportunity: TPR (recall for positive class) should be equal.
    Focuses only on the positive class — less strict than equalized odds.
    Returns TPR gap (lower = fairer).
    """
    y_bin = (y_pred >= threshold).astype(int)
    mask0 = sensitive == 0
    mask1 = sensitive == 1
    tpr0, _ = _tpr_fpr(y_bin, y_true, mask0)
    tpr1, _ = _tpr_fpr(y_bin, y_true, mask1)
    return {
        "tpr_gap": round(abs(tpr0 - tpr1), 4),
        "tpr_group0": round(tpr0, 4),
        "tpr_group1": round(tpr1, 4),
    }


def demographic_parity_diff(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """|P(ŷ=1|z=0) - P(ŷ=1|z=1)|. 0 = perfect parity."""
    y_bin = (y_pred >= threshold).astype(float)
    p0 = y_bin[sensitive == 0].mean() if (sensitive == 0).sum() > 0 else 0.0
    p1 = y_bin[sensitive == 1].mean() if (sensitive == 1).sum() > 0 else 0.0
    return round(abs(p0 - p1), 4)


# Multi-attribute fairness

def abcc(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Area Between Calibration Curves (ABCC).

    Measures how differently the model is calibrated between the two groups.
    A well-calibrated fair model should assign similar meaning to the same
    predicted probability regardless of which group a person belongs to.
    (e.g. a score of 0.7 should mean 70% chance of positive for BOTH groups)

    Lower ABCC = fairer calibration.
    ABCC = 0 means perfectly equal calibration across groups.
    """
    mask0 = sensitive == 0
    mask1 = sensitive == 1
    if mask0.sum() < n_bins or mask1.sum() < n_bins:
        return 0.0
    try:
        frac0, mean0 = calibration_curve(y_true[mask0], y_pred[mask0], n_bins=n_bins, strategy="uniform")
        frac1, mean1 = calibration_curve(y_true[mask1], y_pred[mask1], n_bins=n_bins, strategy="uniform")
        # Interpolate to same grid for comparison
        common = np.linspace(0, 1, n_bins)
        interp0 = np.interp(common, mean0, frac0)
        interp1 = np.interp(common, mean1, frac1)
        return round(float(np.mean(np.abs(interp0 - interp1))), 4)
    except Exception:
        return 0.0


def compute_all_fairness(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sensitive_matrix: np.ndarray,
    attr_names: List[str],
) -> Dict[str, Dict]:
    """
    Compute all fairness metrics for every sensitive attribute.
    Returns nested dict: {attr_name: {metric: value}}.
    """
    results = {}
    for i, name in enumerate(attr_names):
        s = sensitive_matrix[:, i]
        results[name] = {
            "p_rule":                  p_rule(y_pred, s),
            "equalized_odds":          equalized_odds(y_pred, y_true, s),
            "equalized_opportunity":   equalized_opportunity(y_pred, y_true, s),
            "demographic_parity_diff": demographic_parity_diff(y_pred, s),
            "abcc":                    abcc(y_pred, y_true, s),
        }
    return results


# Performance metrics

def compute_performance(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Performance metrics:
    - Accuracy   : fraction of correct predictions
    - Precision  : TP / (TP + FP)
    - F1 Score   : harmonic mean of precision and recall
    - ROC-AUC    : area under the ROC curve (threshold-independent)
    - AP         : Average Precision = area under Precision-Recall curve.
                   Better than ROC-AUC on imbalanced datasets because it
                   focuses on the positive class and penalizes false positives
                   more heavily.
    """
    y_bin = (y_pred >= threshold).astype(int)
    acc = float((y_bin == y_true).mean())
    try:
        auc = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        auc = 0.5
    try:
        ap = float(average_precision_score(y_true, y_pred))
    except ValueError:
        ap = 0.0
    try:
        prec = float(precision_score(y_true, y_bin, zero_division=0))
        f1   = float(f1_score(y_true, y_bin, zero_division=0))
    except Exception:
        prec = 0.0
        f1   = 0.0
    return {
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "f1_score":  round(f1, 4),
        "roc_auc":   round(auc, 4),
        "avg_precision": round(ap, 4),
    }


# Combined evaluation (called from training tools)

@torch.no_grad()
def evaluate(
    classifier,
    X: torch.Tensor,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    attr_names: List[str],
    device: str,
) -> Dict:
    """
    Full evaluation: performance + all fairness metrics per sensitive attribute.
    Returns a flat dict compatible with the rest of the code + nested fairness.
    """
    classifier.eval()
    preds = classifier(X.to(device))
    preds_np = preds.cpu().numpy().squeeze()
    y_np     = y.cpu().numpy()
    s_np     = sensitive.cpu().numpy()

    perf    = compute_performance(preds_np, y_np)
    fairness = compute_all_fairness(preds_np, y_np, s_np, attr_names)

    # Flat p_rules dict for backward compat with training loop
    p_rules = {name: fairness[name]["p_rule"] for name in attr_names}

    return {
        **perf,
        "p_rules":    p_rules,
        "min_p_rule": float(min(p_rules.values())),
        "fairness":   fairness,   # full nested fairness report
    }
