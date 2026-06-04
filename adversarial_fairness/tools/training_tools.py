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


#
# Internal training helpers
#

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

        # Classifier step
        classifier.train()
        clf_opt.zero_grad()
        pred = classifier(X_b)
        loss_clf = clf_criterion(pred, y_b)
        loss_clf.backward()
        clf_opt.step()
        total_clf += loss_clf.item()

        # Adversary step — classifier frozen, clean signal
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

        # Step 1 : Adversary — maximize L_adv
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

        # Step 2 : Classifier — minimize L_task - λ·L_adv
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


#
# Tool 1 — pretrain
#

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

    # Record the clean baseline accuracy — the accuracy this dataset can actually
    # reach before adversarial fairness pressure. The accuracy floor used during
    # training is derived from this (not a hardcoded 80%), so each dataset is
    # judged against what IT can achieve.
    state.baseline_accuracy = float(metrics["accuracy"])

    state.total_epochs_run += n_epochs

    return json.dumps({
        "status": "pretrain_complete",
        "initial_metrics": metrics,
        "epochs_run": n_epochs,
    }, indent=2)


#
# Tool 2 — run_full_training
#

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

    # Accuracy is NOT limited. There is no accuracy floor / threshold: the model
    # reaches whatever accuracy it can, and selection is driven by fairness.
    # Baseline accuracy is still recorded (for reporting) but never gates anything.
    baseline = float(getattr(state, "baseline_accuracy", 0.0) or 0.0)

    if not state.lambda_vector:
        state.lambda_vector = [0.1] * len(state.sensitive_attrs)

    short_term   = ShortTermMemory()
    long_term    = LongTermMemory()
    plotter      = TrainingPlotter(state.sensitive_attrs)

    # Best-result tracking — fairness-first, no accuracy floor:
    #   Primary  : among iterations with min P-rule >= threshold, pick HIGHEST ACCURACY.
    #   Fallback : if the fairness target is never met, pick the highest min P-rule.
    best_metrics          = {}    # primary  — P-rule>=thr, max accuracy
    best_acc_at_fair      = -1.0
    best_metrics_fallback = {}    # fallback — target never met, max min P-rule
    best_prule_fallback   = -1.0

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
    print(f"  Baseline acc   : {baseline*100:.2f}% (clean, post-pretrain — reported, not a gate)")
    print(f"  Selection      : max accuracy among iters with min P-rule >= {p_rule_threshold}%")
    print(f"  Initial λ      : {state.lambda_vector}")
    print(f"{'='*60}")

    final_metrics     = {}
    lambda_trajectory = []
    iteration_metrics = []

    for iteration in range(max_iterations):
        state.current_iteration = iteration

        # Train for epochs_per_step epochs
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

        # Evaluate
        metrics = evaluate(
            clf, state.X_test, state.y_test, state.sensitive_test,
            state.sensitive_attrs, device
        )
        metrics["adversary_loss"] = float(avg_adv_loss)
        metrics["clf_task_loss"]  = float(avg_task_loss)

        # Console log
        p_rule_str = " | ".join(f"{a}={v:.1f}%" for a, v in metrics["p_rules"].items())
        print(
            f"[iter {iteration+1:2d}/{max_iterations}] "
            f"acc={metrics['accuracy']:.4f} | f1={metrics.get('f1_score',0):.4f} | "
            f"auc={metrics.get('roc_auc',0):.4f} | prec={metrics.get('precision',0):.4f}\n"
            f"  fairness: {p_rule_str} | "
            f"adv_loss={avg_adv_loss:.4f} | λ={[round(l,3) for l in state.lambda_vector]}"
        )

        # Update plotter
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

        # Record iteration snapshot
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

        # Track best result seen
        acc = metrics["accuracy"]
        prule = metrics["min_p_rule"]

        if prule >= p_rule_threshold:
            # Primary: fairness target met — maximise ACCURACY (no accuracy floor)
            if acc > best_acc_at_fair:
                best_acc_at_fair = acc
                best_metrics     = metrics
        else:
            # Fallback: fairness target not yet met — track highest min P-rule
            if prule > best_prule_fallback:
                best_prule_fallback   = prule
                best_metrics_fallback = metrics

        # Update short-term memory
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

        # Early stopping — fairness target only (accuracy is not limited).
        # Stop as soon as all P-rules meet the threshold; the best-accuracy
        # iteration among the fair ones is selected post-hoc.
        if prule >= p_rule_threshold:
            print(
                f"  [EARLY STOP] FAIRNESS TARGET REACHED at iteration {iteration+1}:\n"
                f"    min P-rule={prule:.1f}% >= {p_rule_threshold}%  |  acc={acc:.4f}"
            )
            break

        # Momentum lambda update
        new_lambda = decide_lambda_for_iteration(
            current_metrics=metrics,
            lambda_max=lambda_max,
        )
        state.lambda_vector = new_lambda

    # Post-training — accuracy is NOT limited; selection is fairness-first.
    # Pick the best result to report and save:
    #   1. Primary  — some iteration met min P-rule >= threshold → among those,
    #                 the one with the HIGHEST ACCURACY.
    #   2. Fallback — target never met → iteration with the highest min P-rule.
    #   3. Last resort — nothing tracked (should never happen) → final iteration.
    if best_metrics:
        chosen = best_metrics
        status_tag = "optimal"          # fairness target met; best accuracy among fair iters
    elif best_metrics_fallback:
        chosen = best_metrics_fallback
        status_tag = "best_fairness_target_not_met"
    else:
        chosen = final_metrics
        status_tag = "max_iterations_reached"

    # Success == fairness target met at the chosen iteration. Accuracy never gates.
    success = chosen.get("min_p_rule", 0) >= p_rule_threshold

    print(
        f"\n  [BEST RESULT — {status_tag}]\n"
        f"    acc={chosen.get('accuracy',0):.4f}  "
        f"min_P-rule={chosen.get('min_p_rule',0):.1f}%  "
        f"P-rules={chosen.get('p_rules',{})}"
    )

    # Fairness diffs per attribute (scaled to %, to match FFB benchmark)
    #   dp   = demographic parity difference        (FFB test/dp)
    #   eodd = equalized odds diff = mean(TPR gap, FPR gap)  (FFB test/eodd)
    #   eopp = equal opportunity diff = TPR gap      (FFB test/eopp)
    fairness_final = {}
    for attr, fm in chosen.get("fairness", {}).items():
        eo = fm.get("equalized_odds", {})
        fairness_final[attr] = {
            "dp":   round(fm.get("demographic_parity_diff", 0.0) * 100.0, 2),
            "eodd": round(0.5 * (eo.get("tpr_gap", 0.0) + eo.get("fpr_gap", 0.0)) * 100.0, 2),
            "eopp": round(fm.get("equalized_opportunity", {}).get("tpr_gap", 0.0) * 100.0, 2),
        }

    # Lambda at the best iteration (safer reference than lambda_final)
    # lambda_final is the momentum-updated value after the last iteration, which
    # may still be rising. lambda_at_best is the lambda that was actually ACTIVE
    # during the iteration that produced the chosen (best) result — a much more
    # meaningful reference for future warm-starts.
    # Mirror the fairness-first selection: among iterations meeting the P-rule
    # target, take the highest-accuracy one; otherwise the highest min P-rule.
    lambda_at_best = None
    if iteration_metrics:
        thr = float(p_rule_threshold)
        best_itr, best_acc, best_pr = None, -1.0, -1.0
        for m in iteration_metrics:
            pr = min(m["p_rules"].values()) if m["p_rules"] else 0.0
            if pr >= thr:
                if m["accuracy"] > best_acc:
                    best_acc, best_itr = m["accuracy"], m
            elif best_acc < 0 and pr > best_pr:   # only if no fair iter found yet
                best_pr, best_itr = pr, m
        if best_itr:
            lambda_at_best = best_itr.get("lambda")

    # Dataset fingerprint (for future cross-dataset warm-starts)
    run_fingerprint = {}
    try:
        run_fingerprint = LongTermMemory.compute_fingerprint(state)
    except Exception as e:
        print(f"[memory] fingerprint computation failed ({e}), saving without fingerprint")

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
        fairness_final=fairness_final,
        fingerprint=run_fingerprint,
        lambda_at_best=lambda_at_best,
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
