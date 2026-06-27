# -*- coding: utf-8 -*-
import sys, io
# Line-buffered UTF-8 so the agent-activity lines stream to the interface as they
# are produced (same rationale as main.py).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

"""
Pre-run schema suggestion for the interface.

Runs ONLY the orchestrator's perception step: it asks the LLM (or uses the curated
config for a known dataset) which columns are sensitive attributes and what the
target is, then prints the suggestion as JSON between sentinels so the dashboard
can parse it while still showing the streamed "[AGENT:*]" activity lines.

    python suggest_schema.py --dataset <path> --name <name> --model llama3.1

Output (one line):
    @@SCHEMA@@{"sensitive_attrs": [...], "target_col": "...", "columns": [...], ...}
"""

import argparse
import json
import os

import agent_log


def _read_columns(path: str):
    """Best-effort column list for the file (used to populate the picker)."""
    try:
        import pandas as pd
        sep = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
        df = pd.read_csv(path, nrows=50, sep=sep, engine="python")
        return [str(c) for c in df.columns]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if args.model:
        os.environ["AADA_LLM_MODEL"] = args.model
    # Never apply a sensitive override here — we want the raw suggestion.
    os.environ.pop("AADA_SENSITIVE_OVERRIDE", None)

    name = args.name or os.path.splitext(os.path.basename(args.dataset))[0]
    columns = _read_columns(args.dataset)

    out = {"sensitive_attrs": [], "target_col": None, "columns_to_drop": [],
           "modality": "tabular", "columns": columns, "error": None}
    try:
        agent_log.orchestrator(f"Inspecting '{name}' to suggest a target and sensitive attributes.")
        from tools.data_tools import identify_sensitive
        schema = json.loads(identify_sensitive.invoke(
            {"dataset_path": args.dataset, "dataset_name": name}))
        out["sensitive_attrs"]  = schema.get("sensitive_attrs", [])
        out["target_col"]       = schema.get("target_col")
        out["columns_to_drop"]  = schema.get("columns_to_drop", [])
        out["modality"]         = schema.get("modality", "tabular")
        # Make sure suggested/known columns appear as options even if the quick read missed them.
        for c in out["sensitive_attrs"] + ([out["target_col"]] if out["target_col"] else []):
            if c and c not in out["columns"]:
                out["columns"].append(c)
        agent_log.orchestrator(
            f"Suggestion ready — protect {out['sensitive_attrs']}, predict '{out['target_col']}'. "
            f"You can confirm or change this.")
    except Exception as e:
        out["error"] = str(e)
        agent_log.orchestrator(f"Could not auto-suggest ({e}); please choose the columns manually.")

    print("@@SCHEMA@@" + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
