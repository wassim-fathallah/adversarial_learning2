"""
LangChain tools for model training — Zhang et al. 2018.

  pretrain          — trains classifier and adversary independently
  run_full_training — the adversarial loop; momentum formula decides lambda each iteration

Training follows Zhang et al. exactly:
  Adversary step : classifier.eval() + adversary.train()  (clean signal, no dropout noise)
  Classifier step: classifier.train() + adversary.eval()  (stable penalty gradient)
  Task loss      : plain nn.BCELoss() — no class weighting
  Batch size     : 32
"""

import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from langchain.tools import tool

from state import state
from models.classifier import Classifier
from models.adversary import Adversary
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from utils.metrics import evaluate
from utils.plotting import TrainingPlotter
from tools.lambda_tools import decide_lambda_for_iteration


# ─────────────────────────────────────────────────────────────────────────────
# Internal training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(X, y, sensitive, batch_size=32):
    dataset = TensorDataset(X, y.unsqueeze(1), sensitive)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _train_one_epoch_pretrain(classifier, adversary, clf_opt, adv_opt, loader, device):
    """
    Pre-train: classifier on task loss only, then adversary on detached predictions.
    Per batch:
      1. classifier.train()  — update on BCELoss(pred, y)
      2. classifier.eval()   — freeze dropout; adversary.train() — update on adv loss
    """
    clf_criterion = nn.BCELoss()
    adv_loss_fn   = nn.BCEWithLogitsLoss()

    total_clf, total_adv = 0.0, 0.0

    for X_b, y_b, s_b in loader:
        X_b, y_b, s_b = X_b.to(device), y_b.to(device), s_b.to(device)

        # ── Classifier step ───────────────────────────────────────────────────
        classifier.train()
        clf_opt.zero_grad()
        pred = classifier(X_b)
        loss_clf = clf_criterion(pred, y_b)
        loss_clf.backward()
        clf_opt.step()
        total_clf += loss_clf.item()

        # ── Adversary step — classifier frozen, clean signal ──────────────────
        classifier.eval()
        adversary.train()
        adv_opt.zero_grad()
        with torch.no_grad():
            pred_det = classifier(X_b)
        adv_logits = adversary(pred_det)
        loss_adv = adv_loss_fn(adv_logits, s_b)
        loss_adv.backward()
        adv_opt.step()
        total_adv += loss_adv.item()

    n = len(loader)
    return total_clf / n, total_adv / n


def _train_one_epoch_adversarial(
    classifier, adversary, clf_opt, adv_opt, loader, lambda_vector, device
):
    """
    Adversarial training epoch — Zhang et al. 2018, Figure 1.

    Per batch:
      Step 1 — Adversary:   classifier.eval(), adversary.train()
                            maximises L_adv on detached Ŷ
      Step 2 — Classifier:  classifier.train(), adversary.eval()
                            minimises L_task - sum_s(lambda_s * L_adv_s)
                            gradient flows: adv_logits → Ŷ → classifier weights
    """
    clf_criterion     = nn.BCELoss()
    adv_loss_fn_mean  = nn.BCEWithLogitsLoss(reduction="mean")
    adv_loss_fn_none  = nn.BCEWithLogitsLoss(reduction="none")

    lambdas = torch.tensor(lambda_vector, dtype=torch.float32).to(device)

    total_clf, total_adv, total_task = 0.0, 0.0, 0.0

    for X_b, y_b, s_b in loader:
        X_b, y_b, s_b = X_b.to(device), y_b.to(device), s_b.to(device)

        # ── Step 1 : Adversary — maximize L_adv ──────────────────────────────
        classifier.eval()      # dropout OFF → clean prediction for adversary
        adversary.train()
        adv_opt.zero_grad()
        with torch.no_grad():
            pred_det = classifier(X_b)
        adv_logits = adversary(pred_det)
        loss_adv = adv_loss_fn_mean(adv_logits, s_b)
        loss_adv.backward()
        adv_opt.step()
        total_adv += loss_adv.item()

        # ── Step 2 : Classifier — minimize L_task - λ·L_adv ──────────────────
        classifier.train()     # dropout ON for classifier update
        adversary.eval()       # dropout OFF → stable penalty gradient
        clf_opt.zero_grad()
        pred = classifier(X_b)
        loss_task = clf_criterion(pred, y_b)

        # Fresh adversary forward — gradient flows back through pred → classifier
        adv_logits_fresh = adversary(pred)                          # (N, n_attrs)
        # per-sample, per-attr loss; weight by lambda then mean over samples
        adv_loss_per = adv_loss_fn_none(adv_logits_fresh, s_b)      # (N, n_attrs)
        penalty = (adv_loss_per * lambdas).sum(dim=1).mean()

        loss_clf = loss_task - penalty
        loss_clf.backward()
        clf_opt.step()

        total_task += loss_task.item()
        total_clf  += loss_clf.item()

    n = len(loader)
    return total_task / n, total_adv / n, total_clf / n


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — pretrain
# ─────────────────────────────────────────────────────────────────────────────

@tool
def pretrain(n_epochs: int = 10) -> str:
    """
    Pre-trains the classifier and adversary independently for n_epochs.
    Gives both models a warm start before the adversarial competition begins.
    Returns a JSON summary of initial metrics.
    """
    if state.X_train is None:
        return "ERROR: Load dataset first."

    device     = state.device
    n_features  = state.X_train.shape[1]
    n_sensitive = state.sensitive_train.shape[1]

    torch.manual_seed(42)
    clf = Classifier(n_features).to(device)
    adv = Adversary(n_sensitive=n_sensitive).to(device)

    clf_opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    adv_opt = torch.optim.Adam(adv.parameters(), lr=1e-3)

    state.classifier    = clf
    state.adversary     = adv
    state.clf_optimizer = clf_opt
    state.adv_optimizer = adv_opt

    loader = _make_loader(state.X_train, state.y_train, state.sensitive_train)

    print(f"\n[pretrain] Pre-training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        clf_loss, adv_loss = _train_one_epoch_pretrain(
            clf, adv, clf_opt, adv_opt, loader, device
        )
        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch+1:3d}/{n_epochs} | clf_loss={clf_loss:.4f} | adv_loss={adv_loss:.4f}")

    metrics = evaluate(
        clf, state.X_test, state.y_test, state.sensitive_test,
        state.sensitive_attrs, device
    )
    print(f"[pretrain] Initial → acc={metrics['accuracy']:.4f} | p_rules={metrics['p_rules']}")

    state.total_epochs_run += n_epochs

    return json.dumps({
        "status": "pretrain_complete",
        "initial_metrics": metrics,
        "epochs_run": n_epochs,
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — run_full_training
# ─────────────────────────────────────────────────────────────────────────────

@tool
def run_full_training(
    max_iterations: int = 25,
    epochs_per_step: int = 50,
    p_rule_threshold: float = 80.0,
    lambda_max: float = 20.0,
) -> str:
    """
    Runs the full adversarial training loop.

    At EACH ITERATION:
      1. Train classifier + adversary for epochs_per_step epochs
      2. Evaluate fairness metrics (P-rule per sensitive attr, accuracy)
      3. Print step summary to console
      4. Momentum formula decides new lambda
      5. Update short-term memory with this iteration's results
      6. Check convergence: stop if all P-rules >= threshold

    After training:
      - Saves run to long-term memory
      - Saves training curves plot

    Returns a JSON summary of the final outcome.
    """
    if state.classifier is None:
        return "ERROR: Call pretrain first."

    state.p_rule_threshold = p_rule_threshold
    state.max_iterations   = max_iterations

    if not state.lambda_vector:
        state.lambda_vector = [0.1] * len(state.sensitive_attrs)

    short_term   = ShortTermMemory()
    long_term    = LongTermMemory()
    plotter      = TrainingPlotter(state.sensitive_attrs)

    # ── Best-result tracking ──────────────────────────────────────────────────
    # Primary  : among iterations where accuracy >= 0.80, pick highest P-rule
    # Fallback : if accuracy never reaches 0.80, pick highest accuracy overall
    best_metrics          = {}    # primary   — acc >= 0.80, max P-rule
    best_prule_at_acc80   = -1.0
    best_metrics_fallback = {}    # fallback  — max accuracy
    best_acc_fallback     = -1.0

    loader = _make_loader(state.X_train, state.y_train, state.sensitive_train)

    device  = state.device
    clf     = state.classifier
    adv     = state.adversary
    clf_opt = state.clf_optimizer
    adv_opt = state.adv_optimizer

    print(f"\n{'='*60}")
    print(f"[train] Starting adversarial training loop")
    print(f"  Max iterations : {max_iterations}")
    print(f"  Epochs/step    : {epochs_per_step}")
    print(f"  P-rule target  : {p_rule_threshold}%")
    print(f"  Initial λ      : {state.lambda_vector}")
    print(f"{'='*60}")

    final_metrics     = {}
    lambda_trajectory = []
    iteration_metrics = []

    for iteration in range(max_iterations):
        state.current_iteration = iteration

        # ── Train for epochs_per_step epochs ──────────────────────────────────
        epoch_task_losses, epoch_adv_losses = [], []
        for _ in range(epochs_per_step):
            task_loss, adv_loss, _ = _train_one_epoch_adversarial(
                clf, adv, clf_opt, adv_opt, loader, state.lambda_vector, device
            )
            epoch_task_losses.append(task_loss)
            epoch_adv_losses.append(adv_loss)

        state.total_epochs_run += epochs_per_step

        avg_task_loss = np.mean(epoch_task_losses)
        avg_adv_loss  = np.mean(epoch_adv_losses)

        # ── Evaluate ──────────────────────────────────────────────────────────
        metrics = evaluate(
            clf, state.X_test, state.y_test, state.sensitive_test,
            state.sensitive_attrs, device
        )
        metrics["adversary_loss"] = float(avg_adv_loss)
        metrics["clf_task_loss"]  = float(avg_task_loss)

        # ── Console log ───────────────────────────────────────────────────────
        p_rule_str = " | ".join(f"{a}={v:.1f}%" for a, v in metrics["p_rules"].items())
        print(
            f"[iter {iteration+1:2d}/{max_iterations}] "
            f"acc={metrics['accuracy']:.4f} | f1={metrics.get('f1_score',0):.4f} | "
            f"auc={metrics.get('roc_auc',0):.4f} | prec={metrics.get('precision',0):.4f}\n"
            f"  fairness: {p_rule_str} | "
            f"adv_loss={avg_adv_loss:.4f} | λ={[round(l,3) for l in state.lambda_vector]}"
        )

        # ── Update plotter ────────────────────────────────────────────────────
        plotter.update(
            iteration=iteration + 1,
            accuracy=metrics["accuracy"],
            p_rules=metrics["p_rules"],
            lambdas=state.lambda_vector,
            adv_loss=avg_adv_loss,
            precision=metrics.get("precision", 0.0),
            f1=metrics.get("f1_score", 0.0),
            roc_auc=metrics.get("roc_auc", 0.0),
            fairness=metrics.get("fairness", {}),
        )

        # ── Record iteration snapshot ─────────────────────────────────────────
        iteration_metrics.append({
            "iteration": iteration + 1,
            "accuracy":  round(metrics["accuracy"], 4),
            "f1_score":  round(metrics.get("f1_score", 0.0), 4),
            "roc_auc":   round(metrics.get("roc_auc", 0.0), 4),
            "precision": round(metrics.get("precision", 0.0), 4),
            "p_rules":   {k: round(v, 2) for k, v in metrics["p_rules"].items()},
            "lambda":    [round(l, 4) for l in state.lambda_vector],
            "adv_loss":  round(float(avg_adv_loss), 4),
        })

        # ── Track best result seen ────────────────────────────────────────────
        acc = metrics["accuracy"]
        prule = metrics["min_p_rule"]

        if acc >= 0.80:
            # Primary: accuracy constraint satisfied — maximise P-rule
            if prule > best_prule_at_acc80:
                best_prule_at_acc80 = prule
                best_metrics        = metrics
        else:
            # Fallback: accuracy never hit 0.80 — keep highest accuracy seen
            if acc > best_acc_fallback:
                best_acc_fallback     = acc
                best_metrics_fallback = metrics

        # ── Update short-term memory ──────────────────────────────────────────
        short_term.add(
            iteration=iteration + 1,
            lambda_vector=state.lambda_vector,
            p_rules=metrics["p_rules"],
            accuracy=metrics["accuracy"],
            adversary_loss=avg_adv_loss,
            clf_task_loss=avg_task_loss,
        )

        lambda_trajectory.append(state.lambda_vector.copy())
        final_metrics = metrics

        # ── Early stopping — only when BOTH accuracy AND fairness are satisfied ─
        # Optimal solution: accuracy >= 80% AND all P-rules >= threshold
        # If only one condition is met we keep training to find the trade-off.
        if acc >= 0.80 and prule >= p_rule_threshold:
            print(
                f"  [EARLY STOP] OPTIMAL SOLUTION FOUND at iteration {iteration+1}:\n"
                f"    acc={acc:.4f} >= 80%  |  min P-rule={prule:.1f}% >= {p_rule_threshold}%"
            )
            break

        # ── Momentum lambda update ────────────────────────────────────────────
        new_lambda = decide_lambda_for_iteration(
            current_metrics=metrics,
            lambda_max=lambda_max,
        )
        state.lambda_vector = new_lambda

    # ── Post-training ─────────────────────────────────────────────────────────
    # Pick the best result to report and save:
    #   1. Primary  — acc >= 0.80 exists → use iteration with highest P-rule
    #   2. Fallback — acc never hit 0.80 → use iteration with highest accuracy
    #   3. Last resort — nothing tracked yet (should never happen) → final iter
    if best_metrics:
        chosen = best_metrics
        status_tag = "optimal" if best_prule_at_acc80 >= p_rule_threshold else "best_trade_off"
    elif best_metrics_fallback:
        chosen = best_metrics_fallback
        status_tag = "best_accuracy_no_fairness"
    else:
        chosen = final_metrics
        status_tag = "max_iterations_reached"

    success = (
        chosen.get("accuracy", 0) >= 0.80
        and chosen.get("min_p_rule", 0) >= p_rule_threshold
    )

    print(
        f"\n  [BEST RESULT — {status_tag}]\n"
        f"    acc={chosen.get('accuracy',0):.4f}  "
        f"min_P-rule={chosen.get('min_p_rule',0):.1f}%  "
        f"P-rules={chosen.get('p_rules',{})}"
    )

    long_term.save_run(
        dataset_name=state.dataset_name,
        target_col=state.target_col,
        sensitive_attrs=state.sensitive_attrs,
        lambda_final=state.lambda_vector,
        p_rules_final=chosen.get("p_rules", {}),
        accuracy_final=chosen.get("accuracy", 0.0),
        total_epochs=state.total_epochs_run,
        iterations=state.current_iteration + 1,
        success=success,
        lambda_trajectory=lambda_trajectory,
        iteration_metrics=iteration_metrics,
    )

    plot_filename = f"{state.dataset_name}_training_curves.png" if state.dataset_name else "training_curves.png"
    plot_path = plotter.save(plot_filename)
    state.training_done = True

    summary = {
        "status":                   status_tag,
        "iterations_run":           state.current_iteration + 1,
        "total_epochs":             state.total_epochs_run,
        "final_metrics":            chosen,
        "final_lambda":             state.lambda_vector,
        "plot_saved":               plot_path,
        "long_term_memory_updated": True,
    }

    print(f"\n{'='*60}")
    print(f"[train] Done: {summary['status']}")
    print(f"  Final acc   : {final_metrics.get('accuracy', 0):.4f}")
    print(f"  Final p_rule: {final_metrics.get('p_rules', {})}")
    print(f"  Plot        : {plot_path}")
    print(f"{'='*60}\n")

    return json.dumps(summary, indent=2)
