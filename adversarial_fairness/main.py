# -*- coding: utf-8 -*-
import sys, io
# line_buffering=True so output appears line-by-line as it's produced. Without it,
# TextIOWrapper defaults to block buffering (8 KB), so the pipeline looks "stuck"
# (nothing prints after the first flushed line) until it exits — and it even
# overrides `python -u`.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

"""
Entry point for the adversarial fairness pipeline.

Default (no args): runs ALL 7 datasets sequentially.
Single dataset   : python main.py --dataset adult

Usage examples:
    python main.py                          # all datasets
    python main.py --dataset adult
    python main.py --dataset adult --iterations 20 --epochs 50
"""

import argparse
import os
import traceback
import urllib.request
import urllib.error
from orchestrator import run_pipeline


# Ollama health check — must pass before any pipeline work starts

def _check_ollama(host: str = "http://localhost:11434", timeout: int = 5) -> None:
    """
    Ping the Ollama server. Exits the program immediately if it is not reachable.
    Call this once at startup before any LLM initialisation.
    """
    try:
        with urllib.request.urlopen(host, timeout=timeout) as resp:
            resp.read()
        print(f"[ollama] Server reachable at {host}", flush=True)
    except Exception as e:
        reason = getattr(e, "reason", e)
        print("\n" + "=" * 60, flush=True)
        print("  ERROR: Ollama is not reachable — cannot start.", flush=True)
        print(f"  Could not reach {host} ({reason})", flush=True)
        print("", flush=True)
        print("  Fix: open the Ollama app, or run:  ollama serve", flush=True)
        print("=" * 60 + "\n", flush=True)
        sys.exit(1)


DATASET_PRESETS = {
    "adult":   "datasets/adult/raw/adult.data",
    "german":  "datasets/german/raw/german_credit_risk.csv",
    "compas":  "datasets/compas/raw/compas-scores-two-years.csv",
    "utkface": "datasets/utkface/raw/age_gender.csv",
    "kdd":     "datasets/census_income_kdd/raw/census-income.data",
    "bank":    "datasets/bank_marketing/raw/bank-additional-full.csv",
    "acs":     "datasets/acs/raw/2018/1-Year/psam_p06.csv",
    "HIMS-Tunisia": "datasets/HIMS-Tunisia/HIMS-Tunisia.csv",
}

# Datasets to run when no --dataset flag is given
ALL_DATASETS = ["adult", "bank", "compas", "german", "kdd", "acs", "utkface"]


def _resolve_path(dataset_arg: str) -> tuple[str, str]:
    """Return (dataset_path, dataset_name) for a preset key or raw path.

    Preset matching is case-insensitive but the canonical-cased preset key is
    returned as the dataset name (e.g. 'HIMS-Tunisia' for any casing input)."""
    canon = {k.lower(): k for k in DATASET_PRESETS}
    key = dataset_arg.lower()
    if key in canon:
        name = canon[key]
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, DATASET_PRESETS[name]), name
    return dataset_arg, os.path.splitext(os.path.basename(dataset_arg))[0]


def _run_one(name: str, path: str, args) -> dict:
    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  Dataset : {name.upper()}")
    print(f"  Path    : {path}")
    print(f"  Seed    : {args.seed}")
    print(SEP)

    result = run_pipeline(
        dataset_path=path,
        dataset_name=name,
        max_iterations=args.iterations,
        epochs_per_step=args.epochs,
        p_rule_threshold=args.threshold,
        initial_epochs=args.pretrain,
        device=args.device,
        target_override=args.target,
        seed=args.seed,
    )
    return result


def _print_result(name: str, result: dict):
    SEP = "-" * 60
    print(f"\n{SEP}")
    print(f"  RESULT — {name.upper()} — {result.get('status', 'unknown')}")
    print(SEP)
    fm = result.get("final_metrics", {})
    if fm:
        print(f"  Accuracy : {fm.get('accuracy', 'n/a')}")
        print(f"  F1       : {fm.get('f1_score', 'n/a')}")
        print(f"  ROC-AUC  : {fm.get('roc_auc', 'n/a')}")
        print(f"  P-rules  : {fm.get('p_rules', {})}")
    if result.get("plot_saved"):
        print(f"  Plot     : {result['plot_saved']}")


def main():
    parser = argparse.ArgumentParser(description="Adversarial Fairness Pipeline")
    parser.add_argument("--dataset",    type=str, default=None,
                        help="Dataset path or preset name. Omit to run all 7 datasets.")
    parser.add_argument("--name",       type=str, default="",
                        help="Override dataset name used for memory key")
    parser.add_argument("--target",     type=str, default=None,
                        help="Force the target column (overrides LLM pick).")
    parser.add_argument("--iterations", type=int, default=25,
                        help="Max adversarial iterations (default: 25)")
    parser.add_argument("--epochs",     type=int, default=50,
                        help="Epochs per adversarial step (default: 50)")
    parser.add_argument("--threshold",  type=float, default=80.0,
                        help="P-rule target %% (default: 80). Accuracy is NOT "
                             "limited — among iterations meeting this P-rule, the "
                             "highest-accuracy one is selected.")
    parser.add_argument("--pretrain",   type=int, default=10,
                        help="Pre-training epochs (default: 10)")
    parser.add_argument("--device",     type=str, default=None,
                        help="cpu or cuda (auto-detected if omitted)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    _check_ollama()

    # Single dataset mode
    if args.dataset:
        path, name = _resolve_path(args.dataset)
        if args.name:
            name = args.name
        result = _run_one(name, path, args)
        _print_result(name, result)
        return

    # Multi-dataset mode — run ALL_DATASETS sequentially
    base = os.path.dirname(os.path.abspath(__file__))
    results_summary = []

    print("\n" + "=" * 60)
    print("  ADVERSARIAL FAIRNESS — ALL DATASETS")
    print(f"  Datasets: {', '.join(ALL_DATASETS)}")
    print("=" * 60)

    for ds_name in ALL_DATASETS:
        rel_path = DATASET_PRESETS.get(ds_name)
        if rel_path is None:
            print(f"\n[skip] No preset path for '{ds_name}'")
            continue

        abs_path = os.path.join(base, rel_path)
        if not os.path.exists(abs_path):
            print(f"\n[skip] File not found for '{ds_name}': {abs_path}")
            results_summary.append({"name": ds_name, "status": "skipped — file not found"})
            continue

        try:
            result = _run_one(ds_name, abs_path, args)
            _print_result(ds_name, result)
            fm = result.get("final_metrics", {})
            results_summary.append({
                "name":     ds_name,
                "status":   result.get("status", "unknown"),
                "accuracy": fm.get("accuracy", "n/a"),
                "p_rules":  fm.get("p_rules", {}),
            })
        except Exception as e:
            print(f"\n[ERROR] {ds_name} failed: {e}")
            traceback.print_exc()
            results_summary.append({"name": ds_name, "status": f"ERROR: {e}"})

    # Final summary table
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY — ALL DATASETS")
    print("=" * 60)
    for r in results_summary:
        name   = r["name"].upper().ljust(12)
        status = str(r.get("status", "?")).ljust(30)
        acc    = r.get("accuracy", "")
        pr     = r.get("p_rules", {})
        acc_str = f"acc={acc}" if acc != "n/a" else ""
        pr_str  = "  ".join(f"{k}={v:.1f}%" for k, v in pr.items()) if pr else ""
        print(f"  {name} {status}  {acc_str}  {pr_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()
