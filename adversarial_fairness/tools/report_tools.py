# -*- coding: utf-8 -*-
"""
Qualitative bias report — the end-of-run "explainability" artefact.

After the adversarial loop finishes, this module looks at the predictions the
trained classifier makes on held-out data and produces, per sensitive attribute:

  * a KDE plot of the predicted P(target = 1) for each demographic group, with
    the positive-prediction rate and the grouped P-rule annotated,
  * a BEFORE → AFTER view (post-pretrain baseline vs final model) so the closing
    gap is visible,
  * a base-rate vs predicted-rate view (the true outcome rate per group next to
    the model's predicted rate — separates real-world base rates from model bias),
  * per-group sizes, fairness-metric before/after, the accuracy cost + final λ,
    a pass/short verdict per attribute, and
  * a short written analysis of the residual bias (English), produced by the
    selected Ollama model and falling back to a deterministic, stats-based
    paragraph when the LLM is unavailable — so the report ALWAYS renders.

Everything is returned to the caller, which stores it in the run's long-term
memory entry so the dashboard can render the report at the bottom of that run.
Generation is wrapped by the caller in try/except and gated by AADA_NO_REPORT=1,
so a report failure can never break a training run.
"""

import os
import json
import numpy as np
import torch

import agent_log

import matplotlib
matplotlib.use("Agg")          # headless: never needs a display in the subprocess
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Group labelling
# ----------------------------------------------------------------------------

def _binary_labels(attr: str, rule: dict) -> dict:
    """Human label for each code of a BINARY attribute, from its binarization rule."""
    rule = rule or {}
    if "positive_value" in rule:
        pv = rule["positive_value"]
        return {1: f"{attr} = {pv}", 0: f"{attr} ≠ {pv}"}
    if "threshold" in rule:
        t = rule["threshold"]
        return {1: f"{attr} ≥ {t:g}", 0: f"{attr} < {t:g}"}
    return {1: f"{attr} = 1", 0: f"{attr} = 0"}


# ----------------------------------------------------------------------------
# Prediction distributions + statistics
# ----------------------------------------------------------------------------

def _predict_probs(clf, X_test) -> np.ndarray:
    """P(target = 1) for every test row, from the trained classifier."""
    clf.eval()
    device = next(clf.parameters()).device
    with torch.no_grad():
        out = clf(X_test.to(device))
    return out.detach().cpu().numpy().reshape(-1)


def _group_stats(probs: np.ndarray, codes: np.ndarray) -> dict:
    """Per-group prediction stats + the grouped P-rule (min/max selection rate)."""
    groups = {}
    for c in np.unique(codes):
        p = probs[codes == c]
        if p.size == 0:
            continue
        groups[int(c)] = {
            "n":          int(p.size),
            "mean":       float(np.mean(p)),
            "median":     float(np.median(p)),
            "std":        float(np.std(p)),
            "sel_rate":   float(np.mean(p >= 0.5) * 100.0),   # positive-prediction rate %
        }
    rates = [g["sel_rate"] for g in groups.values()]
    mx = max(rates) if rates else 0.0
    p_rule = (min(rates) / mx * 100.0) if mx > 0 else 100.0
    return {"groups": groups, "p_rule": p_rule, "n_groups": len(groups)}


# ----------------------------------------------------------------------------
# Per-attribute panel drawing (shared by the three figures)
# ----------------------------------------------------------------------------

def _extremes(s):
    if not s["groups"]:
        return None, None
    lo = min(s["groups"], key=lambda c: s["groups"][c]["sel_rate"])
    hi = max(s["groups"], key=lambda c: s["groups"][c]["sel_rate"])
    return lo, hi


def _panel_distribution(ax, attr, s, codes, probs, target_col, cmap, sns):
    """Final predicted-outcome KDE, one curve per group."""
    labels = s["labels"]
    lo_c, hi_c = _extremes(s)
    for j, (c, g) in enumerate(sorted(s["groups"].items())):
        p = probs[codes == c]
        emphasise = (s["n_groups"] <= 5) or (c in (lo_c, hi_c))
        color = cmap(j % 10)
        lbl = f"{labels.get(c, f'group {c}')} — {g['sel_rate']:.0f}%"
        if sns is not None and p.size >= 5 and np.std(p) > 1e-4:
            sns.kdeplot(x=p, ax=ax, fill=emphasise, alpha=0.35 if emphasise else 0.0,
                        bw_adjust=1.35, clip=(0.0, 1.0), common_norm=False,
                        linewidth=2 if emphasise else 1,
                        color=color if emphasise else "0.6",
                        label=lbl if emphasise else None)
        else:
            ax.axvline(g["mean"], color=color, lw=2 if emphasise else 1,
                       label=lbl if emphasise else None)
    lo = s["groups"][lo_c]["sel_rate"] if lo_c is not None else 0.0
    hi = s["groups"][hi_c]["sel_rate"] if hi_c is not None else 0.0
    txt = (f"Positive-prediction rate\nlowest  = {lo:.1f}%\nhighest = {hi:.1f}%\n"
           f"P-rule ({s['n_groups']} grp) = {s['p_rule']:.1f}%"
           f"  {'(< 80)' if s['p_rule'] < 80 else '(≥ 80)'}")
    # These distributions pile up near P=1, so the legend goes top-LEFT (empty there)
    # and the P-rule box bottom-left, keeping both off the curves.
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round",
                      facecolor="#fff0f0" if s["p_rule"] < 80 else "#f0fff0",
                      edgecolor="#d62728" if s["p_rule"] < 80 else "#2ca02c", alpha=0.9))
    title_attr = f"{attr} ({s['n_groups']} groups)" if s["n_groups"] > 2 else attr
    ax.set_title(f"Sensitive attribute: {title_attr}", fontsize=12)
    ax.set_xlabel(f"P({target_col} = 1 | group)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, 1); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)


def _panel_before_after(ax, attr, s, codes, probs, base_probs, target_col, cmap, sns):
    """Baseline (dashed) vs final (solid) KDE, one pair per group."""
    labels = s["labels"]
    lo_c, hi_c = _extremes(s)
    ba_groups = (sorted(s["groups"]) if s["n_groups"] <= 5
                 else [g for g in (lo_c, hi_c) if g is not None])
    do_fill = len(ba_groups) <= 2
    for j, c in enumerate(ba_groups):
        if c is None:
            continue
        col = cmap(j % 10)
        pf, pb = probs[codes == c], base_probs[codes == c]
        if sns is not None and pf.size >= 5 and np.std(pf) > 1e-4:
            sns.kdeplot(x=pb, ax=ax, color=col, ls="--", lw=1.5, clip=(0, 1),
                        label=f"{labels.get(c)} — before")
            # NB: with fill=False, seaborn's `alpha` dims the LINE — passing 0.0 made
            # the "after" curve invisible. Fill only in the binary case; otherwise draw
            # an opaque solid line so every group's "after" curve shows.
            if do_fill:
                sns.kdeplot(x=pf, ax=ax, color=col, ls="-", lw=2.2, clip=(0, 1),
                            fill=True, alpha=0.2, label=f"{labels.get(c)} — after")
            else:
                sns.kdeplot(x=pf, ax=ax, color=col, ls="-", lw=2.2, clip=(0, 1),
                            label=f"{labels.get(c)} — after")
        else:
            ax.axvline(np.mean(pb), color=col, ls="--", lw=1.5)
            ax.axvline(np.mean(pf), color=col, ls="-", lw=2.2)
    pr_b = s.get("p_rule_before")
    box = (f"P-rule\nbefore = {pr_b:.0f}%\nafter  = {s['p_rule']:.0f}%"
           if pr_b is not None else f"P-rule after = {s['p_rule']:.0f}%")
    # P-rule box top-center; legend top-left so they don't overlap.
    ax.text(0.5, 0.98, box, transform=ax.transAxes, fontsize=9, va="top", ha="center",
            bbox=dict(boxstyle="round", facecolor="#eef3ff", edgecolor="#1f77b4", alpha=0.9))
    ax.set_title(f"{attr}: before → after debiasing", fontsize=12)
    ax.set_xlabel(f"P({target_col} = 1)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, 1); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)


# ----------------------------------------------------------------------------
# Narrative (LLM with deterministic fallback)
# ----------------------------------------------------------------------------

def _severity(delta_abs: float) -> str:
    if delta_abs < 0.05:
        return "low"
    if delta_abs <= 0.15:
        return "moderate"
    return "severe"


def _templated_narrative(stats: dict, target_col: str, summary: dict) -> str:
    """Deterministic English summary from the computed stats (no LLM needed)."""
    parts = []
    for attr, s in stats.items():
        groups = s["groups"]
        if len(groups) < 2:
            continue
        lo_c = min(groups, key=lambda c: groups[c]["sel_rate"])
        hi_c = max(groups, key=lambda c: groups[c]["sel_rate"])
        lo, hi = groups[lo_c]["sel_rate"], groups[hi_c]["sel_rate"]
        lo_lbl = s["labels"].get(lo_c, f"group {lo_c}")
        hi_lbl = s["labels"].get(hi_c, f"group {hi_c}")
        sev = _severity((hi - lo) / 100.0)
        line = (f"For {attr}, '{lo_lbl}' is predicted '{target_col}' "
                f"{lo:.0f}% of the time versus {hi:.0f}% for '{hi_lbl}' "
                f"(P-rule {s['p_rule']:.0f}%{', below the 80% bar' if s['p_rule'] < 80 else ', within the 80% bar'}). "
                f"The disadvantaged group is '{lo_lbl}'; severity is {sev}.")
        if s.get("p_rule_before") is not None:
            line += f" Debiasing moved the P-rule from {s['p_rule_before']:.0f}% to {s['p_rule']:.0f}%."
        # Base-rate amplification note when true rates are available.
        tr_lo = groups[lo_c].get("true_rate")
        tr_hi = groups[hi_c].get("true_rate")
        if tr_lo is not None and tr_hi is not None and not (np.isnan(tr_lo) or np.isnan(tr_hi)):
            true_gap = tr_hi - tr_lo
            pred_gap = hi - lo
            rel = "amplifies" if pred_gap > true_gap + 2 else ("mirrors" if abs(pred_gap - true_gap) <= 2 else "reduces")
            line += (f" The actual outcome gap is {true_gap:.0f} pts; the model {rel} it "
                     f"(predicted gap {pred_gap:.0f} pts).")
        parts.append(line)
    if summary and summary.get("acc_before") is not None and summary.get("acc_after") is not None:
        parts.append(f"Accuracy went from {summary['acc_before']:.1f}% to {summary['acc_after']:.1f}% "
                     f"over {summary.get('iterations', '?')} iterations.")
    return " ".join(parts) if parts else "No multi-group disparity could be summarised."


def _llm_narrative(stats: dict, target_col: str, summary: dict, llm_model: str) -> str:
    """Ask the selected Ollama model for a concise English bias analysis."""
    summary_lines = []
    for attr, s in stats.items():
        gl = []
        for c, g in s["groups"].items():
            tr = g.get("true_rate")
            tr_s = f", actual outcome rate {tr:.1f}%" if (tr is not None and not np.isnan(tr)) else ""
            gl.append(f"    {s['labels'].get(c, f'group {c}')}: "
                      f"predicted-positive rate {g['sel_rate']:.1f}%{tr_s}, n={g['n']}")
        head = f"Attribute '{attr}' — grouped P-rule {s['p_rule']:.1f}%"
        if s.get("p_rule_before") is not None:
            head += f" (was {s['p_rule_before']:.1f}% before debiasing)"
        summary_lines.append(head + ":\n" + "\n".join(gl))
    data_block = "\n\n".join(summary_lines)
    acc_line = ""
    if summary and summary.get("acc_before") is not None:
        acc_line = (f"\nAccuracy: {summary['acc_before']:.1f}% before -> "
                    f"{summary['acc_after']:.1f}% after, {summary.get('iterations','?')} iterations.")

    prompt = f"""You are an expert in algorithmic fairness. A classifier predicts '{target_col}'.
Below is, per demographic group on held-out test data, the model's predicted-positive
rate and (where given) the actual outcome rate in the data:

{data_block}{acc_line}

Write a concise English analysis (4-6 sentences) that:
1. names the disadvantaged group for each sensitive attribute,
2. quantifies the gap in percentage points and as a ratio (the P-rule),
3. says whether the model amplifies, mirrors or reduces the actual outcome-rate gap,
4. notes whether debiasing improved the P-rule and at what accuracy cost,
5. notes the gap is not designed but flows through correlated proxies.
Do not use bullet points or headings. Plain prose only."""

    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model=llm_model, temperature=0.2, num_gpu=0)
    return str(llm.invoke(prompt)).strip()


# ----------------------------------------------------------------------------
# Figure builder + plot-data persistence (shared by training-time generation and
# the offline replot_report.py — restyle figures with NO re-training).
# ----------------------------------------------------------------------------

def _stats_to_json(stats: dict) -> dict:
    """JSON-safe stats (int group keys -> str), for storage + replotting."""
    return {
        a: {"p_rule": round(s["p_rule"], 2),
            "p_rule_before": round(s["p_rule_before"], 2) if s.get("p_rule_before") is not None else None,
            "n_groups": s["n_groups"],
            "groups": {str(c): {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in g.items()} for c, g in s["groups"].items()},
            "labels": {str(c): lbl for c, lbl in s["labels"].items()}}
        for a, s in stats.items()
    }


def save_plotdata(out_dir, safe, attrs, stats, code_by_attr, probs, base_probs,
                  have_before, target_col, dataset_name, eodd_before, eodd_after):
    """Persist the raw inputs the figures are drawn from, so they can be regenerated
    or restyled later WITHOUT re-running training (consumed by replot_report.py)."""
    os.makedirs(out_dir, exist_ok=True)
    codes_mat = np.stack([np.asarray(code_by_attr[a]) for a in attrs], axis=0)
    np.savez_compressed(
        os.path.join(out_dir, f"report_{safe}_plotdata.npz"),
        probs=np.asarray(probs),
        base_probs=np.asarray(base_probs) if have_before else np.zeros(0),
        codes=codes_mat,
    )
    meta = {
        "dataset_name": dataset_name, "target_col": target_col, "attrs": list(attrs),
        "have_before": bool(have_before),
        "stats": _stats_to_json(stats),
        "eodd_before": list(eodd_before), "eodd_after": list(eodd_after),
    }
    with open(os.path.join(out_dir, f"report_{safe}_plotmeta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def build_figures(out_dir, dataset_name, target_col, attrs, stats, code_by_attr,
                  probs, base_probs, have_before, eodd_before, eodd_after):
    """Render the report PNGs from already-computed stats + raw arrays. Shared by
    generate_report (training time) and replot_report.py (offline restyling).
    Returns the plots list [{title, plot}] (paths relative to the package dir)."""
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        sns_arg = sns
    except Exception:
        sns_arg = None
    n = len(attrs)
    cmap = plt.get_cmap("tab10")
    os.makedirs(out_dir, exist_ok=True)
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    safe = (dataset_name or "run").replace(os.sep, "_")

    def _save_fig(fig, suffix):
        ap = os.path.join(out_dir, f"report_{safe}_{suffix}.png")
        fig.savefig(ap, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return os.path.relpath(ap, pkg_dir).replace(os.sep, "/")

    def _make(suffix, suptitle, draw):
        fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 4.8), squeeze=False)
        for i, attr in enumerate(attrs):
            draw(axes[0][i], attr, stats[attr], code_by_attr[attr])
        fig.suptitle(suptitle, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return _save_fig(fig, suffix)

    plots = []
    plots.append({"title": "Predicted-outcome distribution by group", "plot": _make(
        "distributions", f"{dataset_name} — predicted P({target_col} = 1) by sensitive group",
        lambda ax, attr, s, codes: _panel_distribution(ax, attr, s, codes, probs, target_col, cmap, sns_arg))})
    if have_before:
        plots.append({"title": "Before → after debiasing", "plot": _make(
            "before_after", f"{dataset_name} — before → after debiasing",
            lambda ax, attr, s, codes: _panel_before_after(ax, attr, s, codes, probs, base_probs, target_col, cmap, sns_arg))})
    if any(v is not None for v in list(eodd_before) + list(eodd_after)):
        fig, ax = plt.subplots(1, 1, figsize=(max(6.0, 3.0 * n), 4.8))
        xs = np.arange(n); w = 0.38
        b1 = ax.bar(xs - w / 2, [v or 0.0 for v in eodd_before], w, label="before", color="#7f7f7f")
        b2 = ax.bar(xs + w / 2, [v or 0.0 for v in eodd_after],  w, label="after",  color="#1f77b4")
        ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
        ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)
        ax.set_xticks(xs); ax.set_xticklabels(attrs)
        ax.set_ylabel("Equalized-odds gap (%) — lower is fairer", fontsize=11)
        ax.set_title("Equalized odds: before vs after debiasing", fontsize=12)
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
        fig.suptitle(f"{dataset_name} — equalized odds before vs after", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        plots.append({"title": "Equalized odds — before vs after", "plot": _save_fig(fig, "eodd")})
    return plots


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def generate_report(state, out_dir: str, language: str = "en",
                    llm_model: str = None, context: dict = None) -> dict:
    """Build the figure + written narrative + numeric extras for the trained model.

    Returns a dict with: plot (path rel. to the package dir), narrative, stats,
    fairness (before/after dp/eodd/eopp/p-rule), summary (accuracy cost, λ, verdicts),
    language. Raises on hard failure; the caller decides whether to swallow it.
    """
    context = context or {}
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        _have_sns = True
    except Exception:
        _have_sns = False

    clf   = state.classifier
    attrs = list(state.sensitive_attrs)
    rules = getattr(state, "binarization_rules", {}) or {}
    target_col = getattr(state, "target_col", "target")

    agent_log.report(f"Analysing predictions on {state.X_test.shape[0]} held-out rows for {attrs}.")
    probs = _predict_probs(clf, state.X_test)
    y = (np.asarray(state.y_test.cpu().numpy()).reshape(-1)
         if getattr(state, "y_test", None) is not None else None)
    base_probs = getattr(state, "baseline_probs", None)
    base_probs = np.asarray(base_probs).reshape(-1) if base_probs is not None else None
    have_before = base_probs is not None and base_probs.shape == probs.shape

    raw  = getattr(state, "sensitive_test_raw", None)
    binz = state.sensitive_test
    raw_np  = raw.cpu().numpy()  if raw  is not None else None
    binz_np = binz.cpu().numpy() if binz is not None else None

    # ---- per-attribute statistics (final, baseline, true base rate) -------------
    stats, code_by_attr = {}, {}
    for i, attr in enumerate(attrs):
        codes = (raw_np[:, i] if raw_np is not None else binz_np[:, i]).astype(int)
        code_by_attr[attr] = codes
        s = _group_stats(probs, codes)
        # Prefer real group names captured at load time (region: Center-West / Greater
        # Tunis / Others; educ: Low / Mid / Higher; Gender: Female / Male). Fall back to
        # binarization-rule labels for binary attrs, then to a generic "group k".
        gl = (getattr(state, "group_labels", {}) or {}).get(attr)
        if gl:
            labels = {c: gl.get(c, gl.get(str(c), f"group {c}")) for c in s["groups"]}
        elif s["n_groups"] <= 2 and set(s["groups"].keys()) <= {0, 1}:
            labels = _binary_labels(attr, rules.get(attr))
        else:
            labels = {c: f"group {c}" for c in s["groups"]}
        s["labels"] = labels
        if y is not None:
            for c in s["groups"]:
                yc = y[codes == c]
                s["groups"][c]["true_rate"] = float(np.mean(yc >= 0.5) * 100.0) if yc.size else float("nan")
        if have_before:
            sb = _group_stats(base_probs, codes)
            s["p_rule_before"] = sb["p_rule"]
            for c in s["groups"]:
                if c in sb["groups"]:
                    s["groups"][c]["sel_rate_before"] = sb["groups"][c]["sel_rate"]
        stats[attr] = s

    # ---- equalized-odds (before/after) from the run context --------------------
    fb_ctx = context.get("fairness_baseline", {}) or {}
    ff_ctx = context.get("fairness_final", {}) or {}
    eodd_b = [fb_ctx.get(a, {}).get("eodd") for a in attrs]
    eodd_a = [ff_ctx.get(a, {}).get("eodd") for a in attrs]

    # ---- persist the raw plot inputs, then render the figures ------------------
    # Saving first means the figures can be restyled offline with replot_report.py —
    # no re-training to move a legend or tweak a colour. Never fatal.
    safe = (state.dataset_name or "run").replace(os.sep, "_")
    try:
        save_plotdata(out_dir, safe, attrs, stats, code_by_attr, probs, base_probs,
                      have_before, target_col, state.dataset_name, eodd_b, eodd_a)
    except Exception as e:
        print(f"[report] plotdata save skipped ({e})")

    plots = build_figures(out_dir, state.dataset_name, target_col, attrs, stats,
                          code_by_attr, probs, base_probs if have_before else None,
                          have_before, eodd_b, eodd_a)
    agent_log.report(f"Saved {len(plots)} report figure(s).")

    # ---- fairness before/after + run summary (for the dashboard tables) --------
    fb = context.get("fairness_baseline", {}) or {}
    ff = context.get("fairness_final", {}) or {}
    fairness = {}
    for attr in attrs:
        s = stats[attr]
        fairness[attr] = {
            "prule_before": round(s["p_rule_before"], 2) if s.get("p_rule_before") is not None else None,
            "prule_after":  round(s["p_rule"], 2),
            "dp_before":   fb.get(attr, {}).get("dp"),   "dp_after":   ff.get(attr, {}).get("dp"),
            "eodd_before": fb.get(attr, {}).get("eodd"), "eodd_after": ff.get(attr, {}).get("eodd"),
            "eopp_before": fb.get(attr, {}).get("eopp"), "eopp_after": ff.get(attr, {}).get("eopp"),
        }

    thr = float(context.get("threshold", 80.0))
    bm  = context.get("baseline_metrics", {}) or {}
    fm  = context.get("final_metrics", {}) or {}
    verdicts = {}
    for attr in attrs:
        pr = stats[attr]["p_rule"]
        verdicts[attr] = (f"PASS — P-rule {pr:.0f}% ≥ {thr:.0f}%" if pr >= thr
                          else f"SHORT — P-rule {pr:.0f}% < {thr:.0f}% (raise λ or add iterations)")
    summary = {
        "acc_before":   round(bm["accuracy"] * 100, 2) if bm.get("accuracy") is not None else None,
        "acc_after":    round(fm["accuracy"] * 100, 2) if fm.get("accuracy") is not None else None,
        "lambda_final": [round(float(l), 3) for l in context.get("lambda_final", [])],
        "iterations":   context.get("iterations"),
        "threshold":    thr,
        "attrs":        attrs,
        "verdicts":     verdicts,
    }

    # ---- written analysis: LLM (English). AADA_REPORT_REQUIRE_LLM=1 forbids the
    #      deterministic fallback, so the narrative is GUARANTEED LLM-written or the
    #      report aborts (used when you must be sure the LLM produced it).
    narrative = ""
    llm_used = False
    model = llm_model or os.environ.get("AADA_LLM_MODEL", "llama3.1")
    require_llm = os.environ.get("AADA_REPORT_REQUIRE_LLM") == "1"
    if os.environ.get("AADA_REPORT_NO_LLM") == "1" and not require_llm:
        narrative = _templated_narrative(stats, target_col, summary)
        agent_log.report("Composed a templated bias analysis (LLM disabled).")
    else:
        try:
            agent_log.report(f"Asking {model} to write the qualitative bias analysis…")
            narrative = _llm_narrative(stats, target_col, summary, model)
            llm_used = True
            agent_log.report(f"Received the LLM analysis (model: {model}).")
        except Exception as e:
            if require_llm:
                agent_log.report(f"LLM REQUIRED but failed ({e}); aborting report — NO fallback.")
                raise RuntimeError(f"AADA_REPORT_REQUIRE_LLM=1 but the LLM failed: {e}")
            narrative = _templated_narrative(stats, target_col, summary)
            agent_log.report(f"LLM unavailable ({e}); used the templated analysis instead.")

    stats_json = _stats_to_json(stats)

    result = {"plots": plots,                       # list of {title, plot} — one PNG per view
              "plot": plots[0]["plot"] if plots else None,   # back-compat: the distributions PNG
              "narrative": narrative, "stats": stats_json,
              "fairness": fairness, "summary": summary, "language": language,
              "llm_used": llm_used, "llm_model": model if llm_used else None}

    # ---- assemble the PDF report (charts + LLM narrative + metrics table) -------
    try:
        try:
            from tools.pdf_report import generate_pdf_report
        except Exception:
            from pdf_report import generate_pdf_report
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = generate_pdf_report(result, out_dir, state.dataset_name, pkg_dir,
                                       llm_model=(model if llm_used else None))
        result["pdf"] = os.path.relpath(pdf_path, pkg_dir).replace(os.sep, "/")
        agent_log.report(f"Saved the PDF report ({os.path.basename(pdf_path)}).")
    except Exception as e:
        print(f"[report] PDF generation skipped ({e})")

    return result
