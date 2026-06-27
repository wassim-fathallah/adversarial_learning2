# -*- coding: utf-8 -*-
"""
Re-render the qualitative-report figures WITHOUT re-running training.

generate_report() saves the raw plot inputs next to the PNGs
(report_<name>_plotdata.npz + report_<name>_plotmeta.json). This script loads
those and re-draws the figures, so styling changes (legend position, colours,
labels, sizes) are instant — edit tools/report_tools.py panel helpers, then:

    python replot_report.py HIMS-Tunisia
    python replot_report.py                 # defaults to HIMS-Tunisia

The PNGs are overwritten in place, so any run whose stored report points at them
picks up the new figures with no memory change.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools.report_tools import build_figures   # noqa: E402


def replot(name: str, reports_dir: str = None):
    reports_dir = reports_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    safe = name.replace(os.sep, "_")
    npz_path  = os.path.join(reports_dir, f"report_{safe}_plotdata.npz")
    meta_path = os.path.join(reports_dir, f"report_{safe}_plotmeta.json")
    if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
        raise SystemExit(
            f"No saved plot data for '{name}' in {reports_dir}.\n"
            f"Run the pipeline once (it now saves {os.path.basename(npz_path)} + "
            f"{os.path.basename(meta_path)}); after that this script needs no training.")

    d    = np.load(npz_path)
    meta = json.load(open(meta_path, encoding="utf-8"))

    attrs      = meta["attrs"]
    target_col = meta["target_col"]
    ds_name    = meta["dataset_name"]
    probs      = d["probs"]
    base_probs = d["base_probs"]
    codes      = d["codes"]                      # (n_attrs, N)

    have_before = bool(meta.get("have_before")) and base_probs.size > 0
    code_by_attr = {a: codes[i].astype(int) for i, a in enumerate(attrs)}

    # Rebuild the in-memory stats shape build_figures expects (int group codes).
    stats = {}
    for a in attrs:
        sj = meta["stats"][a]
        stats[a] = {
            "groups":        {int(c): dict(g) for c, g in sj["groups"].items()},
            "labels":        {int(c): lbl for c, lbl in sj["labels"].items()},
            "p_rule":        sj["p_rule"],
            "p_rule_before": sj.get("p_rule_before"),
            "n_groups":      sj["n_groups"],
        }

    plots = build_figures(
        reports_dir, ds_name, target_col, attrs, stats, code_by_attr,
        probs, base_probs if have_before else None, have_before,
        meta.get("eodd_before", [None] * len(attrs)),
        meta.get("eodd_after",  [None] * len(attrs)),
    )
    print(f"Re-plotted {len(plots)} figure(s) for {ds_name} (no training):")
    for p in plots:
        print("  -", p["plot"])


if __name__ == "__main__":
    replot(sys.argv[1] if len(sys.argv) > 1 else "HIMS-Tunisia")
