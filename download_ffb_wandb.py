"""
Download FFB published results from public WandB projects.

Fetches all tabular runs from the FFB paper's WandB projects and saves them
as JSON files compatible with fair_fairness_benchmark/results/ format.

Usage:
    python download_ffb_wandb.py              # all tabular projects
    python download_ffb_wandb.py --quick      # first 20 runs per project
    python download_ffb_wandb.py --project exp1.erm   # single project

Requirements: pip install wandb
No WandB account needed — these are public projects.
"""

import json
import os
import argparse
import time

try:
    import wandb
except ImportError:
    print("wandb not installed. Run: pip install wandb")
    raise

ENTITY = "fair_benchmark"

# Tabular datasets to include
TARGET_TABULAR = {"adult", "german", "compas", "bank_marketing", "HIMS-Tunisia", "kdd", "acs"}
# Image datasets to include
TARGET_IMAGE   = {"utkface"}

# Save only the final step instead of full history — ~5 KB vs ~5 MB per file
FINAL_ONLY = True

# Tabular: ERM, AdvDebias, PR, HSIC, LAFTR, DiffDP, DiffEopp, DiffEodd
TABULAR_PROJECTS = {
    "exp1.erm":      "erm",
    "exp1.adv_gr":   "adv",
    "exp1.pr":       "pr",
    "exp1.hsic":     "hsic",
    "exp1.laftr":    "laftr",
    "exp1.diffdp":   "diffdp",
    "exp1.diffeopp": "diffeopp",
    "exp1.diffeodd": "diffeodd",
}

# Image: ERM, PR, HSIC only — FFB has no AdvDebias or LAFTR for image data
IMAGE_PROJECTS = {
    "exp2.erm":  "erm",
    "exp2.pr":   "pr",
    "exp2.hsic": "hsic",
}

OUT_DIR = os.path.join(os.path.dirname(__file__),
                       "fair_fairness_benchmark", "results")


def safe_get(d, key, default=None):
    try:
        return d[key]
    except Exception:
        return default


def fetch_project(project_name, method_name, target_datasets, max_runs=None):
    api  = wandb.Api(timeout=60)
    path = f"{ENTITY}/{project_name}"

    print(f"\n[{method_name}] Fetching {path} ...")
    try:
        runs = api.runs(path, per_page=100)
    except Exception as e:
        print(f"  ERROR: could not access {path} — {e}")
        return 0

    saved = 0
    for i, run in enumerate(runs):
        if max_runs and i >= max_runs:
            break

        cfg = run.config or {}

        # Extract metadata from run config or name
        dataset      = safe_get(cfg, "dataset",        "unknown")
        sensitive    = safe_get(cfg, "sensitive_attr",  safe_get(cfg, "sens_col", "unknown"))
        # LAFTR's fairness coefficient is logged as "A_z" (not "lam"); fall back to it
        # so LAFTR sweeps keep a real lambda label instead of collapsing to 0.0.
        lam          = safe_get(cfg, "lam",             safe_get(cfg, "A_z", safe_get(cfg, "alpha", 0.0)))
        seed         = safe_get(cfg, "seed",            0)
        target_attr  = safe_get(cfg, "target_attr",    None)   # image datasets only

        # Skip datasets not in the target set for this project type
        if target_datasets and dataset not in target_datasets:
            continue

        # Try to get history (step-level metrics)
        try:
            if FINAL_ONLY:
                # Use run summary (final values only) — much faster, ~5 KB per file
                summary = dict(run.summary or {})
                if not summary:
                    continue
                entry = {"step": int(safe_get(summary, "_step", 0))}
                for k, v in summary.items():
                    if k.startswith("_"):
                        continue
                    try:
                        entry[k] = float(v)
                    except (ValueError, TypeError):
                        pass
                history = [entry]
            else:
                hist_df = run.history(samples=200, pandas=True)
                if hist_df is None or hist_df.empty:
                    continue
                history = []
                for _, row in hist_df.iterrows():
                    entry = {"step": int(row.get("_step", 0))}
                    for col in hist_df.columns:
                        if col.startswith("_"):
                            continue
                        val = row.get(col)
                        if val is not None and str(val) not in ("nan", "None"):
                            try:
                                entry[col] = float(val)
                            except (ValueError, TypeError):
                                entry[col] = val
                    history.append(entry)

            if not history:
                continue

        except Exception as e:
            print(f"  WARN: could not fetch history for run {run.name}: {e}")
            continue

        meta = {
            "method":         method_name,
            "dataset":        dataset,
            "sensitive_attr": sensitive,
            "lam":            lam,
            "seed":           seed,
            "source":         "wandb",
            "wandb_run":      run.name,
            "project":        project_name,
        }
        if target_attr:
            meta["target_attr"] = target_attr

        record = {"metadata": meta, "history": history}

        # For image datasets the target_attr is part of the experimental setup and
        # must be in the filename so runs with different targets don't overwrite each other.
        target_suffix = f"_{target_attr}" if target_attr else ""
        fname = f"{dataset}_{method_name}_{sensitive}{target_suffix}_lam{lam}_seed{seed}.json"
        fpath = os.path.join(OUT_DIR, fname)

        # Skip if already downloaded
        if os.path.exists(fpath):
            continue

        with open(fpath, "w") as f:
            json.dump(record, f, indent=2)

        saved += 1
        if saved % 10 == 0:
            print(f"  ... {saved} runs saved")

        time.sleep(0.1)   # be polite to the API

    print(f"  [done] {saved} runs saved from {project_name}")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default=None,
                        help="Single WandB project to download (e.g. exp1.erm)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Optional dataset filter (e.g. acs kdd). Defaults to all supported datasets.")
    parser.add_argument("--quick",   action="store_true",
                        help="Download only first 20 runs per project (for testing)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    max_runs = 20 if args.quick else None
    total    = 0
    dataset_filter = set(args.datasets) if args.datasets else None

    def filtered_targets(targets):
        if not dataset_filter:
            return targets
        return {dataset for dataset in targets if dataset in dataset_filter}

    # Single project override
    if args.project:
        all_projects = {**TABULAR_PROJECTS, **IMAGE_PROJECTS}
        if args.project in all_projects:
            is_image    = args.project in IMAGE_PROJECTS
            target_ds   = filtered_targets(TARGET_IMAGE if is_image else TARGET_TABULAR)
            total += fetch_project(args.project, all_projects[args.project],
                                   target_datasets=target_ds, max_runs=max_runs)
    else:
        print("\nTabular methods")
        for project_name, method_name in TABULAR_PROJECTS.items():
            total += fetch_project(project_name, method_name,
                                   target_datasets=filtered_targets(TARGET_TABULAR), max_runs=max_runs)

        print("\nImage methods (UTKFace)")
        for project_name, method_name in IMAGE_PROJECTS.items():
            total += fetch_project(project_name, method_name,
                                   target_datasets=filtered_targets(TARGET_IMAGE), max_runs=max_runs)

    print(f"\nDone. {total} total runs saved to:\n  {OUT_DIR}")
    print("\nRefresh the Streamlit dashboard to see the new results.")


if __name__ == "__main__":
    main()
