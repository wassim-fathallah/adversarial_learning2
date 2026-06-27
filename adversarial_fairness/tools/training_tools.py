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
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from langchain.tools import tool

import agent_log
from state import state
from models.agents import ClassifierAgent, ImageClassifierAgent, AdversaryAgent
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from utils.metrics import evaluate
from utils.plotting import TrainingPlotter
from tools.lambda_tools import decide_lambda_for_iteration


#
# Internal training helpers
#

DATASET_BATCH_SIZES = {
    "adult":   1024,
    "bank":    1024,
    "german":  32,
    "compas":  32,
    "kdd":     4096,
    "acs":     4096,
    "utkface": 128,
    "hims-tunisia": 32,
    "earnings_synth": 1024,   # Adult-sized synthetic (30k rows) — fingerprint demo
}

def _make_loader(X, y, sensitive, batch_size=32, seed=42):
    dataset = TensorDataset(X, y.unsqueeze(1), sensitive)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def _batch_size_for(dataset_name: str) -> int:
    return DATASET_BATCH_SIZES.get((dataset_name or "").lower(), 32)


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


def _fairness_diffs(fairness: dict) -> dict:
    """Per-attribute fairness diffs scaled to % (FFB-style), from an evaluate() fairness
    block. Used for BOTH the pre-training baseline and the final iteration so every run
    records equalized odds before and after debiasing.
      dp   = demographic parity difference
      eodd = equalized odds diff = mean(TPR gap, FPR gap)
      eopp = equal opportunity diff = TPR gap
    """
    out = {}
    for attr, fm in (fairness or {}).items():
        eo = fm.get("equalized_odds", {})
        out[attr] = {
            "dp":   round(fm.get("demographic_parity_diff", 0.0) * 100.0, 2),
            "eodd": round(0.5 * (eo.get("tpr_gap", 0.0) + eo.get("fpr_gap", 0.0)) * 100.0, 2),
            "eopp": round(fm.get("equalized_opportunity", {}).get("tpr_gap", 0.0) * 100.0, 2),
        }
    return out


def _parse_classifier_spec(spec: str):
    """Parse the classifier-architecture choice (AADA_CLASSIFIER / interface picker).

    Returns (kind, n_layers, n_hidden) where kind ∈ {"auto", "mlp", "cnn"}:
      auto          -> image→CNN, tabular→MLP (2×256); the historical default.
      mlp[:LxW]     -> force an MLP with L hidden layers of width W (e.g. mlp:3x256).
      cnn           -> force the ResNet-18 image predictor (needs reshapeable input).
    Unparseable values fall back to auto so a bad string can never break a run.
    """
    spec = (spec or "auto").strip().lower()
    if spec in ("", "auto"):
        return ("auto", None, None)
    if spec == "cnn":
        return ("cnn", None, None)
    if spec.startswith("mlp"):
        rest = spec[3:].lstrip(":").strip()
        if not rest:
            return ("mlp", 2, 256)
        try:
            l, w = rest.split("x")
            return ("mlp", int(l), int(w))
        except Exception:
            return ("mlp", 2, 256)
    return ("auto", None, None)


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

    # Default hidden width. Tabular MLP: 256 (FFB-matched, Appendix C).
    # Image CNN dense head: scale the width with the pixel count.
    if state.modality == "image":
        n_hidden = min(max(64, n_features // 8), 512) if n_features > 500 else 32
    else:
        n_hidden = 256

    # Architecture chosen in the interface (or AADA_CLASSIFIER). "auto" keeps the
    # historical behaviour exactly: image datasets use the ResNet-18 CNN predictor,
    # tabular datasets use the 2×256 MLP. The adversary, the λ/momentum logic and the
    # training loop are identical for every choice — only the predictor changes.
    kind, n_layers, n_hidden_override = _parse_classifier_spec(os.environ.get("AADA_CLASSIFIER"))
    use_image = (kind == "cnn") or (kind == "auto" and state.modality == "image")
    if use_image and not getattr(state, "image_shape", None):
        print("[model] CNN requested but this dataset has no image shape -> using an MLP instead")
        agent_log.orchestrator("CNN was requested but the data is tabular — falling back to an MLP.")
        use_image = False
    if n_hidden_override:
        n_hidden = n_hidden_override

    if use_image:
        clf = ImageClassifierAgent(state.image_shape, n_hidden=n_hidden).to(device)
        print(f"[model] IMAGE -> ResNet18 predictor | image_shape={state.image_shape}")
        agent_log.classifier(f"I am a CNN (ResNet-18) predictor — image input {state.image_shape}.")
    else:
        n_layers = n_layers or 2
        clf = ClassifierAgent(n_features, n_hidden=n_hidden, n_layers=n_layers).to(device)
        print(f"[model] TABULAR -> MLP predictor | n_features={n_features} "
              f"n_layers={n_layers} n_hidden={n_hidden}")
        agent_log.classifier(
            f"I am an MLP predictor — {n_layers} hidden layer(s) × {n_hidden}, {n_features} features.")
    adv = AdversaryAgent(n_sensitive=n_sensitive, n_hidden=n_hidden).to(device)
    agent_log.adversary(f"I will try to recover {n_sensitive} sensitive attribute(s) from the prediction.")

    clf_opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    adv_opt = torch.optim.Adam(adv.parameters(), lr=1e-3)

    state.classifier    = clf
    state.adversary     = adv
    state.clf_optimizer = clf_opt
    state.adv_optimizer = adv_opt

    seed = getattr(state, 'seed', 42)
    loader = _make_loader(state.X_train, state.y_train, state.sensitive_train,
                          batch_size=_batch_size_for(state.dataset_name), seed=seed)

    print(f"\n[pretrain] Pre-training for {n_epochs} epochs...")
    last_clf_loss = last_adv_loss = 0.0
    for epoch in range(n_epochs):
        clf_loss, adv_loss = _train_one_epoch_pretrain(
            clf, adv, clf_opt, adv_opt, loader, device
        )
        last_clf_loss, last_adv_loss = clf_loss, adv_loss
        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch+1:3d}/{n_epochs} | clf_loss={clf_loss:.4f} | adv_loss={adv_loss:.4f}")
    agent_log.classifier(f"Pretraining done — task loss {last_clf_loss:.3f}.")
    agent_log.adversary(f"Pretraining done — recovery loss {last_adv_loss:.3f}.")

    metrics = evaluate(
        clf, state.X_test, state.y_test, state.sensitive_test,
        state.sensitive_attrs, device, sensitive_raw=state.sensitive_test_raw
    )
    # Snapshot the baseline test predictions (before adversarial debiasing) so the
    # qualitative report can show the before→after distribution shift. The classifier
    # is trained in place, so these weights won't exist by report time.
    try:
        clf.eval()
        with torch.no_grad():
            state.baseline_probs = clf(state.X_test.to(device)).detach().cpu().numpy().reshape(-1)
    except Exception as e:
        print(f"[pretrain] baseline-prediction snapshot skipped ({e})")
        state.baseline_probs = None
    print(f"[pretrain] Initial -> acc={metrics['accuracy']:.4f} | p_rules={metrics['p_rules']}")
    _pr = " | ".join(f"{a}={v:.1f}%" for a, v in metrics["p_rules"].items())
    agent_log.orchestrator(
        f"Baseline before debiasing — accuracy {metrics['accuracy']*100:.1f}%, P-rule {_pr}.")

    # Record the clean baseline accuracy — the accuracy this dataset can actually
    # reach before adversarial fairness pressure. The accuracy floor used during
    # training is derived from this (not a hardcoded 80%), so each dataset is
    # judged against what IT can achieve.
    state.baseline_accuracy = float(metrics["accuracy"])
    state.initial_metrics   = metrics   # saved as iteration-0 by run_full_training

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

    seed = getattr(state, 'seed', 42)
    loader = _make_loader(state.X_train, state.y_train, state.sensitive_train,
                          batch_size=_batch_size_for(state.dataset_name), seed=seed)

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

    # Prepend iteration 0 — the post-pretrain baseline before any adversarial pressure.
    # Shown in the chart as the starting point so the viewer sees where p-rules and
    # accuracy were before debiasing began.
    initial = getattr(state, "initial_metrics", None)
    if initial:
        iteration_metrics.append({
            "iteration": 0,
            "accuracy":  round(initial["accuracy"], 4),
            "f1_score":  round(initial.get("f1_score", 0.0), 4),
            "roc_auc":   round(initial.get("roc_auc", 0.0), 4),
            "precision": round(initial.get("precision", 0.0), 4),
            "p_rules":   {k: round(v, 2) for k, v in initial["p_rules"].items()},
            "lambda":    [0.0] * len(state.sensitive_attrs),
            "adv_loss":  0.0,
        })

    # Leave-from-pretraining: if the post-pretrain baseline already clears the P-rule
    # target, the dataset is fair out of pretraining, so skip the adversarial phase
    # entirely (no debiasing needed). Set AADA_FORCE_ADV=1 to run the loop anyway.
    import os as _os_skip
    baseline_prule = float(initial.get("min_p_rule", 0.0)) if initial else 0.0
    if (initial is not None and baseline_prule >= p_rule_threshold
            and _os_skip.environ.get("AADA_FORCE_ADV") != "1"):
        print(f"\n  [LEAVE PRETRAINING] Baseline already fair: min P-rule="
              f"{baseline_prule:.1f}% >= {p_rule_threshold}% — skipping the adversarial "
              f"phase (no debiasing needed).\n")
        best_metrics     = initial
        best_acc_at_fair = initial.get("accuracy", 0.0)
        final_metrics    = initial
        lambda_trajectory.append([0.0] * len(state.sensitive_attrs))
        state.current_iteration = -1          # iterations_run = current_iteration + 1 = 0
        max_iterations = 0                     # the loop below becomes a no-op
        # Plot just the baseline point so the saved chart isn't empty.
        plotter.update(
            iteration=0,
            accuracy=initial["accuracy"],
            p_rules=initial["p_rules"],
            lambdas=[0.0] * len(state.sensitive_attrs),
            adv_loss=0.0,
            precision=initial.get("precision", 0.0),
            f1=initial.get("f1_score", 0.0),
            roc_auc=initial.get("roc_auc", 0.0),
            fairness=initial.get("fairness", {}),
        )

    for iteration in range(max_iterations):
        state.current_iteration = iteration

        os.environ["RUN_ITER"] = str(iteration + 1)

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
            state.sensitive_attrs, device, sensitive_raw=state.sensitive_test_raw
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

        # Per-iteration agent narration (the "agents in action" feed)
        agent_log.classifier(
            f"iter {iteration+1}: task loss {avg_task_loss:.3f}, accuracy {metrics['accuracy']*100:.1f}%.")
        agent_log.adversary(
            f"iter {iteration+1}: recovery loss {avg_adv_loss:.3f} "
            f"({'losing the signal — good for fairness' if avg_adv_loss > 0.6 else 'still recovering attributes'}).")
        _worst_attr = min(metrics["p_rules"], key=metrics["p_rules"].get) if metrics["p_rules"] else "—"
        agent_log.orchestrator(
            f"iter {iteration+1}: min P-rule {metrics['min_p_rule']:.1f}% "
            f"(worst: {_worst_attr}). {'Target met.' if metrics['min_p_rule'] >= p_rule_threshold else 'Raising λ on lagging attributes.'}")

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
        # Set AADA_NO_EARLYSTOP=1 to run the FULL max_iterations (used by the
        # momentum ablation so beta=0 keeps oscillating past the target instead
        # of stopping at the first hit).
        import os as _os_es
        if prule >= p_rule_threshold and _os_es.environ.get("AADA_NO_EARLYSTOP") != "1":
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

    # Fairness diffs per attribute (scaled to %, to match FFB benchmark), computed for
    # every dataset — including user-uploaded ones — at BOTH ends of training:
    #   fairness_baseline = before debiasing (post-pretrain), fairness_final = last/best.
    #   dp   = demographic parity difference        (FFB test/dp)
    #   eodd = equalized odds diff = mean(TPR gap, FPR gap)  (FFB test/eodd)
    #   eopp = equal opportunity diff = TPR gap      (FFB test/eopp)
    fairness_baseline = _fairness_diffs((getattr(state, "initial_metrics", None) or {}).get("fairness", {}))
    fairness_final    = _fairness_diffs(chosen.get("fairness", {}))

    # Equalized-odds before vs after, printed for every run.
    print("\n  [equalized odds]  (lower = fairer; % gap)")
    for attr in state.sensitive_attrs:
        b = fairness_baseline.get(attr, {}).get("eodd", float("nan"))
        f = fairness_final.get(attr, {}).get("eodd", float("nan"))
        print(f"    {attr:15s} before={b:6.2f}%  ->  after={f:6.2f}%")

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

    # Qualitative bias report (KDE distributions + written analysis). Generated on
    # the final classifier and stored in the run entry so the dashboard can show it
    # at the bottom of the run. Never allowed to break a training run; skip with
    # AADA_NO_REPORT=1 (e.g. ablation sweeps where the report is not needed).
    report = None
    if os.environ.get("AADA_NO_REPORT") != "1":
        try:
            from tools.report_tools import generate_report
            reports_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
            report = generate_report(state, reports_dir, language="en", context={
                "fairness_baseline": fairness_baseline,
                "fairness_final":    fairness_final,
                "baseline_metrics":  getattr(state, "initial_metrics", None) or {},
                "final_metrics":     chosen,
                "lambda_final":      list(state.lambda_vector),
                "iterations":        state.current_iteration + 1,
                "threshold":         float(p_rule_threshold),
            })
        except Exception as e:
            print(f"[report] qualitative report skipped ({e})")

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
        fairness_baseline=fairness_baseline,
        fingerprint=run_fingerprint,
        lambda_at_best=lambda_at_best,
        seed=getattr(state, "seed", None),
        report=report,
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
        "fairness_baseline":        fairness_baseline,
        "fairness_final":           fairness_final,
        "plot_saved":               plot_path,
        "report":                   report,
        "long_term_memory_updated": True,
    }

    print(f"\n{'='*60}")
    print(f"[train] Done: {summary['status']}")
    print(f"  Final acc   : {final_metrics.get('accuracy', 0):.4f}")
    print(f"  Final p_rule: {final_metrics.get('p_rules', {})}")
    print(f"  Plot        : {plot_path}")
    print(f"{'='*60}\n")

    return json.dumps(summary, indent=2)
