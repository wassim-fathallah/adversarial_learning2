"""
Unified Fairness Dashboard
--------------------------
Tab 1 — Agentic Adversarial Debiasing  : reads adversarial_fairness/long_term_memory.json
Tab 2 — FFB Benchmark                  : reads fair_fairness_benchmark/results/*.json
Tab 3 — Comparison — HIMS-Tunisia      : both systems side by side
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from glob import glob

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px

# Paths
ROOT        = os.path.dirname(os.path.abspath(__file__))
MY_MEMORY   = os.path.join(ROOT, "adversarial_fairness", "long_term_memory.json")
FFB_RESULTS = os.path.join(ROOT, "fair_fairness_benchmark", "results")

COLORS = ["#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#3498db", "#e67e22"]

st.set_page_config(page_title="Fairness Dashboard", page_icon="⚖️", layout="wide")
st.title("⚖️ Agentic Adversarial Fairness Algorithm Dashboard")


# Data loaders

@st.cache_data(ttl=10)
def load_my_memory():
    if not os.path.exists(MY_MEMORY):
        return {}
    with open(MY_MEMORY, encoding="utf-8-sig") as f:
        return json.load(f)


@st.cache_data(ttl=30)
def load_ffb_results():
    files = glob(os.path.join(FFB_RESULTS, "*.json"))
    results = []
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8-sig") as f:
                data = json.load(f)
            if data.get("history"):
                results.append(data)
        except Exception:
            pass
    return results


# Agents-in-action — live activity feed + model discovery

AF_DIR = os.path.join(ROOT, "adversarial_fairness")

# Icon per agent key (must match adversarial_fairness/agent_log.py).
AGENT_ICONS = {
    "orchestrator": "🧭",
    "classifier":   "🏋️",
    "adversary":    "🎯",
    "llm":          "🧠",
    "report":       "📊",
}
_AGENT_RE = re.compile(r"^\[AGENT:([a-z_]+)\]\s?(.*)$")
_SCHEMA_RE = re.compile(r"^@@SCHEMA@@(.*)$")


def _render_agent_feed(placeholder, events):
    """Render the last ~22 agent events as a compact live feed."""
    if not events:
        placeholder.markdown("_waiting for the agents…_")
        return
    rows = []
    for ag, msg in events[-22:]:
        icon = AGENT_ICONS.get(ag, "•")
        rows.append(f"{icon} **{ag.capitalize()}** — {msg}")
    placeholder.markdown("\n\n".join(rows))


def stream_with_agents(cmd, env=None, cwd=AF_DIR, spinner="Working…"):
    """Run `cmd`, streaming a live 'Agents in action' feed + a raw-log expander.

    Returns (returncode, schema_dict_or_None, all_log_lines). `[AGENT:*]` lines
    drive the feed; an `@@SCHEMA@@{...}` line (from suggest_schema.py) is parsed
    out and returned.
    """
    run_env = {**os.environ, **(env or {})}
    st.markdown("##### 🤖 Agents in action")
    feed_box = st.empty()
    events = []
    schema = None
    logs = []
    with st.expander("Raw log", expanded=False):
        log_box = st.empty()
    with st.spinner(spinner):
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", cwd=cwd, env=run_env,
        )
        for line in proc.stdout:
            line = line.rstrip()
            m = _AGENT_RE.match(line)
            if m:
                events.append((m.group(1), m.group(2)))
                _render_agent_feed(feed_box, events)
                continue
            ms = _SCHEMA_RE.match(line)
            if ms:
                try:
                    schema = json.loads(ms.group(1))
                except Exception:
                    schema = None
                continue
            logs.append(line)
            log_box.text("\n".join(logs[-40:]))
        proc.wait()
    return proc.returncode, schema, logs


# Helpers — My System

def group_by_dataset(memory: dict):
    groups = defaultdict(list)
    for key, runs in memory.items():
        dataset_name = key.split("|")[0]
        target_col   = key.split("|")[1] if "|" in key else ""
        attrs        = key.split("|")[2] if key.count("|") >= 2 else ""
        for idx, run in enumerate(runs):
            groups[dataset_name].append({
                **run,
                "_key": key, "_run_index": idx,
                "_target": target_col, "_attrs": attrs,
            })
    return dict(groups)


def _sensitive_attrs(run):
    attrs_str = run.get("_attrs", "")
    if attrs_str:
        return [a.strip() for a in attrs_str.split(",") if a.strip()]
    return list(run.get("p_rules_final", {}).keys())


def plot_my_run(run):
    iters = run.get("iteration_metrics", [])
    attrs = _sensitive_attrs(run)

    if not iters:
        fig = go.Figure()
        for i, (attr, val) in enumerate(run.get("p_rules_final", {}).items()):
            fig.add_trace(go.Bar(name=f"P-rule ({attr})", x=[attr], y=[val],
                                 marker_color=COLORS[i % len(COLORS)]))
        fig.add_hline(y=80, line_dash="dash", line_color="gray", annotation_text="80%")
        fig.update_layout(title="Final P-rules (no iteration data)", height=300)
        return fig

    # Separate iteration-0 (pretrain baseline) from adversarial iterations for
    # distinct visual treatment — star marker, no line connecting to iter 1.
    has_iter0 = iters and iters[0]["iteration"] == 0
    adv_iters = iters[1:] if has_iter0 else iters
    adv_xs    = [m["iteration"] for m in adv_iters]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Accuracy / F1 / ROC-AUC",
                                        "P-rule per Attribute",
                                        "Lambda (λ)",
                                        "Adversary Loss"),
                        vertical_spacing=0.07)

    # Accuracy / F1 / AUC — adversarial iterations
    fig.add_trace(go.Scatter(x=adv_xs, y=[m["accuracy"] for m in adv_iters],
                             name="Accuracy", line=dict(color="royalblue"), mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=adv_xs, y=[m["f1_score"] for m in adv_iters],
                             name="F1", line=dict(color="seagreen"), mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=adv_xs, y=[m["roc_auc"] for m in adv_iters],
                             name="AUC", line=dict(color="darkorchid"), mode="lines+markers"), row=1, col=1)
    fig.add_hline(y=0.80, line_dash="dash", line_color="gray", row=1, col=1)

    # Pretrain baseline markers (iteration 0) — star + value label
    if has_iter0:
        m0 = iters[0]
        fig.add_trace(go.Scatter(
            x=[0], y=[m0["accuracy"]], name="Pretrain (acc)",
            mode="markers+text",
            marker=dict(symbol="star", size=14, color="royalblue"),
            text=[f"{m0['accuracy']*100:.1f}%"],
            textposition="top center",
            showlegend=True), row=1, col=1)

    # P-rule — adversarial iterations
    for i, attr in enumerate(attrs):
        fig.add_trace(go.Scatter(x=adv_xs, y=[m["p_rules"].get(attr, 0) for m in adv_iters],
                                 name=f"P-rule ({attr})",
                                 line=dict(color=COLORS[i % len(COLORS)]), mode="lines+markers"), row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="gray", row=2, col=1)

    # Pretrain p-rule baseline markers (iteration 0) — star + value label
    if has_iter0:
        m0 = iters[0]
        for i, attr in enumerate(attrs):
            val = m0["p_rules"].get(attr, 0)
            fig.add_trace(go.Scatter(
                x=[0], y=[val],
                name=f"Pretrain P-rule ({attr})",
                mode="markers+text",
                marker=dict(symbol="star", size=14, color=COLORS[i % len(COLORS)]),
                text=[f"{val:.1f}%"],
                textposition="top center",
                showlegend=True), row=2, col=1)

    for i, attr in enumerate(attrs):
        fig.add_trace(go.Scatter(x=adv_xs,
                                 y=[m["lambda"][i] if i < len(m.get("lambda", [])) else None for m in adv_iters],
                                 name=f"λ ({attr})",
                                 line=dict(color=COLORS[i % len(COLORS)], dash="dot"), mode="lines+markers"), row=3, col=1)

    fig.add_trace(go.Scatter(x=adv_xs, y=[m.get("adv_loss", 0) for m in adv_iters],
                             name="Adv. Loss", line=dict(color="tomato"), mode="lines+markers"), row=4, col=1)

    fig.update_yaxes(title_text="Score", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="P-rule (%)", range=[0, 110], row=2, col=1)
    fig.update_yaxes(title_text="λ", row=3, col=1)
    fig.update_yaxes(title_text="Loss", row=4, col=1)
    fig.update_xaxes(title_text="Iteration", row=4, col=1)
    fig.update_layout(height=900, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1),
                      margin=dict(t=80))
    return fig


def show_my_run_summary(run):
    p_rules = run.get("p_rules_final", {})
    acc     = run.get("accuracy_final", 0)
    success = run.get("success", False)
    ts      = run.get("timestamp", "")[:19].replace("T", " ")
    iters   = run.get("iterations", "?")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc*100:.2f}%",
                delta="✓ ≥80%" if acc >= 0.80 else "✗ <80%",
                delta_color="normal" if acc >= 0.80 else "inverse")
    for i, (attr, val) in enumerate(p_rules.items()):
        if i < 3:
            [col2, col3, col4][i].metric(f"P-rule ({attr})", f"{val:.1f}%",
                                         delta="✓ ≥80%" if val >= 80 else "✗ <80%",
                                         delta_color="normal" if val >= 80 else "inverse")
    st.caption(f"{ts} | Iterations: {iters} | {'✅ SUCCESS' if success else '🔄 Not yet'}")

    # Equalized odds before (post-pretrain baseline) vs after (final iteration), per
    # attribute. Lower = fairer; a negative Δ means the gap shrank with debiasing.
    fb = run.get("fairness_baseline", {})
    ff = run.get("fairness_final", {})
    if ff:
        st.markdown("**Equalized odds** — % gap, lower is fairer")
        eo_rows = []
        for attr in p_rules.keys():
            before = fb.get(attr, {}).get("eodd")
            after  = ff.get(attr, {}).get("eodd")
            eo_rows.append({
                "Attribute":         attr,
                "Before (baseline)": f"{before:.2f}%" if before is not None else "—",
                "After (final)":     f"{after:.2f}%"  if after  is not None else "—",
                "Δ":                 f"{after - before:+.2f}" if (before is not None and after is not None) else "—",
            })
        st.dataframe(pd.DataFrame(eo_rows), hide_index=True, use_container_width=True)


def _fmt_ba(before, after, pct=True):
    """'before → after' with a sign-aware delta, robust to missing values."""
    def f(v):
        if v is None:
            return "—"
        return f"{v:.1f}%" if pct else f"{v:.2f}"
    s = f"{f(before)} → {f(after)}"
    if before is not None and after is not None:
        s += f"  (Δ {after - before:+.1f})" if pct else f"  (Δ {after - before:+.2f})"
    return s


def render_run_report(run):
    """Qualitative bias report (distributions, before→after, base-rate, fairness,
    verdicts + written analysis) at the bottom of a run, if one was saved."""
    report = run.get("report") or {}
    plot_rel  = report.get("plot")
    plots     = report.get("plots")
    narrative = report.get("narrative")
    summary   = report.get("summary") or {}
    fairness  = report.get("fairness") or {}
    stats     = report.get("stats") or {}
    if not plot_rel and not plots and not narrative:
        return

    st.divider()
    st.markdown("#### 🔍 Qualitative bias report")

    # Download the full PDF report (charts + LLM qualitative analysis), if generated.
    pdf_rel = report.get("pdf")
    if pdf_rel:
        pdf_abs = os.path.join(AF_DIR, pdf_rel)
        if os.path.exists(pdf_abs):
            with open(pdf_abs, "rb") as _f:
                # Unique key per run — render_run_report runs in a loop, and several runs
                # can each expose a PDF button; without a key Streamlit auto-IDs collide
                # (StreamlitDuplicateElementId) and the later run's tab fails to render.
                _dlkey = ("dlpdf_"
                          f"{run.get('timestamp','')}_{run.get('_run_index','')}_"
                          f"{os.path.basename(pdf_abs)}")
                st.download_button("⬇️ Download PDF report", _f.read(),
                                   file_name=os.path.basename(pdf_abs),
                                   mime="application/pdf", use_container_width=True,
                                   key=_dlkey)
            if report.get("llm_used"):
                st.caption(f"Qualitative analysis written by LLM: {report.get('llm_model', '')}")

    # Accuracy cost + convergence summary.
    ab, aa = summary.get("acc_before"), summary.get("acc_after")
    if ab is not None or aa is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy (before → after)",
                  f"{aa:.1f}%" if aa is not None else "—",
                  delta=f"{aa - ab:+.1f} pts" if (ab is not None and aa is not None) else None,
                  delta_color="off")
        c2.metric("Iterations", summary.get("iterations", "—"))
        lam = summary.get("lambda_final") or []
        c3.metric("Final λ", ", ".join(f"{l:g}" for l in lam) if lam else "—")

    # Per-attribute pass / short verdict.
    for attr, v in (summary.get("verdicts") or {}).items():
        (st.success if str(v).startswith("PASS") else st.warning)(f"**{attr}** — {v}")

    # The figures — one PNG per view (distributions / before→after / equalized odds).
    figs = plots if plots else ([{"title": None, "plot": plot_rel}] if plot_rel else [])
    for fg in figs:
        rel, title = fg.get("plot"), fg.get("title")
        if title:
            st.markdown(f"**{title}**")
        ap = os.path.join(AF_DIR, rel) if rel else None
        if ap and os.path.exists(ap):
            st.image(ap, use_container_width=True)
        elif rel:
            st.caption(f"(figure not found at {rel})")

    # Fairness metrics before → after.
    if fairness:
        rows = []
        for attr, fv in fairness.items():
            rows.append({
                "Attribute": attr,
                "P-rule":    _fmt_ba(fv.get("prule_before"), fv.get("prule_after")),
                "DP gap":    _fmt_ba(fv.get("dp_before"),   fv.get("dp_after")),
                "EqOdds":    _fmt_ba(fv.get("eodd_before"), fv.get("eodd_after")),
                "EqOpp":     _fmt_ba(fv.get("eopp_before"), fv.get("eopp_after")),
            })
        st.markdown("**Fairness metrics — before → after** (P-rule higher is fairer; gaps lower is fairer)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Per-group breakdown: size, actual vs predicted positive rate.
    if stats:
        grows = []
        for attr, s in stats.items():
            labels = s.get("labels", {})
            for code, g in (s.get("groups") or {}).items():
                grows.append({
                    "Attribute":      attr,
                    "Group":          labels.get(code, f"group {code}"),
                    "N":              g.get("n"),
                    "Actual +%":      f"{g['true_rate']:.1f}%" if g.get("true_rate") is not None else "—",
                    "Predicted +%":   f"{g.get('sel_rate', float('nan')):.1f}%",
                    "Predicted +% (before)": f"{g['sel_rate_before']:.1f}%" if g.get("sel_rate_before") is not None else "—",
                })
        if grows:
            st.markdown("**Group breakdown** — size, actual outcome rate vs model's predicted rate")
            st.dataframe(pd.DataFrame(grows), hide_index=True, use_container_width=True)

    if narrative:
        st.markdown("**Analysis**")
        st.markdown(
            "<div style='background:#fff3f3;border-left:4px solid #d62728;"
            "padding:0.8rem 1rem;border-radius:4px'>" +
            narrative.replace("\n", "<br>") + "</div>",
            unsafe_allow_html=True,
        )


# Helpers — FFB

UTILITY_METRICS  = ["acc", "ap", "auc", "f1"]
FAIRNESS_METRICS = ["dp", "eopp", "eodd", "abcc", "prule"]
ALL_METRICS      = UTILITY_METRICS + FAIRNESS_METRICS


def ffb_final_df(results_list):
    rows = []
    for r in results_list:
        last = r["history"][-1]
        meta = r["metadata"]
        row  = {"method": meta["method"], "sensitive": meta["sensitive_attr"],
                "lam": meta["lam"], "seed": meta["seed"]}
        for m in ALL_METRICS:
            row[m] = round(last.get(f"test/{m}", float("nan")), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "sensitive", "lam", "seed"]).reset_index(drop=True)


def ffb_history_df(results_list):
    rows = []
    for r in results_list:
        meta  = r["metadata"]
        label = f"{meta['method']} | {meta['sensitive_attr']} | lam={meta['lam']} | seed={meta['seed']}"
        for entry in r["history"]:
            row = {"method": meta["method"], "sensitive": meta["sensitive_attr"],
                   "lam": meta["lam"], "seed": meta["seed"], "run": label,
                   "step": entry["step"]}
            row.update({k: v for k, v in entry.items() if k != "step"})
            rows.append(row)
    return pd.DataFrame(rows)


# Max-Acc / Max-Prule extraction (mirrors compare_ffb.py)

# How to read "Ours" out of long_term_memory.json, per FFB dataset name.
#   key      : memory key to use   attr_map: {ours_attr_in_memory : ffb_attr}
OURS_CONFIG = {
    "adult":          {"key": "adult|income|race,sex",
                       "attr_map": {"race": "race", "sex": "sex"}},
    "compas":         {"key": "compas|two_year_recid|race,sex",
                       "attr_map": {"race": "race", "sex": "sex"}},
    "german":         {"key": "german|Class|Age,Sex",
                       "attr_map": {"Age": "age", "Sex": "sex"}},
    "bank_marketing": {"key": "bank|y|age,marital",
                       "attr_map": {"age": "age"}},
    "utkface":        {"key": "utkface|age|ethnicity,gender",
                       "attr_map": {"gender": "Gender", "ethnicity": "Race"}},
    "HIMS-Tunisia":   {"key": "HIMS-Tunisia|legal_entry|Gender,coastal_origin,educ_level",
                       "attr_map": {"Gender": "sex",
                                    "coastal_origin": "coastal_origin",
                                    "educ_level": "educ_level"}},
}


def selection_table(df_final, my_memory, ffb_dataset, ffb_attr):
    """
    For the given (dataset, attribute) and a per-(method,lam,seed) metrics frame,
    return a DataFrame with, per method, the Max-Acc and Max-Prule operating
    points (mean over seeds) plus DP/EOdd/EOpp at those points, and an Ours row.
    """
    def _num(v):
        return round(float(v), 2) if pd.notna(v) else None

    rows = []
    for method in sorted(df_final["method"].unique()):
        sub = df_final[df_final["method"] == method]
        g = (sub.groupby("lam")[["acc", "prule", "dp", "eodd", "eopp"]]
                .mean().reset_index().dropna(subset=["acc"]))
        if g.empty:
            continue
        # Drop degenerate collapse points (P-rule~100 with all gaps ~0 = the
        # majority-class predictor). Keep all only if nothing else remains.
        degenerate = ((g["prule"] >= 99.0) & (g["dp"] < 0.5)
                      & (g["eodd"] < 0.5) & (g["eopp"] < 0.5))
        gg = g[~degenerate] if (~degenerate).any() else g
        la = gg.loc[gg["acc"].idxmax()]
        gp = gg.dropna(subset=["prule"])
        lp = gp.loc[gp["prule"].idxmax()] if not gp.empty else la
        # Trade-off: highest min(acc, prule) — best balanced point (both>=80 if
        # attainable, else closest to the 80/80 corner)
        if not gp.empty:
            lt = gp.loc[gp[["acc", "prule"]].min(axis=1).idxmax()]
        else:
            lt = la

        def mk(label, r):
            return {"Method": method, "Select": label,
                    "Acc": _num(r["acc"]), "P-rule": _num(r["prule"]),
                    "ΔDP": _num(r["dp"]), "ΔEOdd": _num(r["eodd"]),
                    "ΔEOpp": _num(r["eopp"])}

        # Always 3 rows per method: Max-Acc, Max-Prule, Trade-off (no merging)
        rows.append(mk("Max-Acc", la))
        rows.append(mk("Max-Prule", lp))
        rows.append(mk("Trade-off", lt))

    # Ours row for this dataset/attribute 
    cfg = OURS_CONFIG.get(ffb_dataset)
    if cfg:
        runs = my_memory.get(cfg["key"])
        ours_attr = next((oa for oa, fa in cfg["attr_map"].items() if fa == ffb_attr), None)
        if runs and ours_attr:
            run = runs[-1]
            prules = run.get("p_rules_final", {})
            if ours_attr in prules:
                fair = run.get("fairness_final", {}).get(ours_attr, {})
                rows.append({
                    "Method": "Ours",
                    "Select": "Final" if run.get("success") else "Best",
                    "Acc": _num(run.get("accuracy_final", 0) * 100),
                    "P-rule": _num(prules[ours_attr]),
                    "ΔDP": _num(fair.get("dp")) if fair.get("dp") is not None else None,
                    "ΔEOdd": _num(fair.get("eodd")) if fair.get("eodd") is not None else None,
                    "ΔEOpp": _num(fair.get("eopp")) if fair.get("eopp") is not None else None,
                })
    return pd.DataFrame(rows)


# TABS

tab_my, tab_ffb = st.tabs([
    "⚙️ Agentic Adversarial Debiasing",
    "📊 FFB Benchmark",
])


# TAB 1 : Agentic Adversarial Debiasing

with tab_my:

    # Upload → orchestrator suggests → you confirm → train (agents shown live)
    with st.expander("Apply AAD on a new dataset", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload dataset file(s) — .csv, .data, .tsv, .txt or any tabular format. "
            "Upload multiple files if your dataset is split across files (e.g. train + test).",
            type=None,
            accept_multiple_files=True,
        )

        classifier_choices = {
            "Auto — image→CNN, tabular→MLP (2×256)": "auto",
            "MLP — 2 layers × 256 (default)":         "mlp:2x256",
            "MLP — 3 layers × 256":                   "mlp:3x256",
            "MLP — 2 layers × 128":                   "mlp:2x128",
            "MLP — 3 layers × 512":                   "mlp:3x512",
            "CNN — ResNet-18 (image data)":           "cnn",
        }
        m1, _ = st.columns([2, 2])
        clf_label = m1.selectbox(
            "🧠 Classifier model", list(classifier_choices),
            help="The predictor architecture. 'Auto' uses a CNN for images and a 2×256 MLP "
                 "for tabular data; or force an MLP depth/width, or a CNN.")
        classifier_spec = classifier_choices[clf_label]

        c1, c2, c3, c4 = st.columns(4)
        iterations = c1.slider("Iterations",  5,  50, 25)
        epochs     = c2.slider("Epochs/step", 10, 100, 50)
        threshold  = c3.slider("P-rule target %", 50, 95, 80)
        pretrain   = c4.slider("Pretrain epochs", 5, 30, 10)

        # When multiple files: let user pick which one is the main entry point
        main_file = None
        if len(uploaded_files) == 1:
            main_file = uploaded_files[0].name
        elif len(uploaded_files) > 1:
            main_file = st.selectbox(
                "Which file is the main dataset file?",
                [f.name for f in uploaded_files],
            )

        # Signature of the current upload set — a changed upload invalidates a stale suggestion.
        upload_sig = "|".join(sorted(f.name for f in uploaded_files)) if uploaded_files else ""

        # --- Step 1 : the orchestrator perceives the schema and SUGGESTS ----------
        if uploaded_files and st.button(
                "🔍 Step 1 — Analyze (let the orchestrator suggest)", type="secondary"):
            tmp_dir   = tempfile.mkdtemp()
            tmp_paths = {}
            for uf in uploaded_files:
                dest = os.path.join(tmp_dir, uf.name)
                with open(dest, "wb") as f:
                    f.write(uf.read())
                tmp_paths[uf.name] = dest
            entry_path = tmp_paths[main_file]
            ds_name    = os.path.splitext(main_file)[0]

            cmd = [sys.executable, os.path.join(AF_DIR, "suggest_schema.py"),
                   "--dataset", entry_path, "--name", ds_name]
            rc, schema, _ = stream_with_agents(cmd, spinner="Reading the schema…")
            schema = schema or {}
            st.session_state["analysis"] = {
                "sig":        upload_sig,
                "tmp_dir":    tmp_dir,
                "entry":      entry_path,
                "name":       ds_name,
                "columns":    schema.get("columns") or [],
                "suggested_sensitive": schema.get("sensitive_attrs", []),
                "suggested_target":    schema.get("target_col"),
                "modality":   schema.get("modality", "tabular"),
                "error":      schema.get("error"),
            }
            st.rerun()

        # --- Step 2 : you confirm / edit, then train ------------------------------
        analysis = st.session_state.get("analysis")
        if analysis and analysis.get("sig") == upload_sig and uploaded_files:
            cols = analysis["columns"]
            mod  = analysis.get("modality", "tabular")
            banner = (st.success if mod != "image" else st.info)
            banner(
                f"Orchestrator analyzed **{analysis['name']}** → detected a "
                f"**{'TABULAR (MLP)' if mod != 'image' else 'IMAGE (CNN)'}** dataset. "
                f"Confirm or change the target and the sensitive attributes below.")
            if analysis.get("error"):
                st.warning(f"Auto-suggestion was partial ({analysis['error']}); pick the columns manually.")
            if not cols:
                st.warning("Couldn't read the columns automatically — re-run Step 1 or check the file.")

            tgt_default = analysis.get("suggested_target")
            tcol = st.selectbox(
                "🎯 Target column (what to predict)",
                cols or ([tgt_default] if tgt_default else [""]),
                index=(cols.index(tgt_default) if (cols and tgt_default in cols) else 0))
            sens_opts    = [c for c in cols if c != tcol]
            sens_default = [s for s in analysis.get("suggested_sensitive", []) if s in sens_opts]
            sattrs = st.multiselect(
                "🛡️ Sensitive attributes to protect", sens_opts, default=sens_default,
                help="The orchestrator's suggestion is pre-filled — add or remove as you like.")

            if st.button("▶ Step 2 — Start Training", type="primary"):
                if not sattrs:
                    st.error("Pick at least one sensitive attribute before training.")
                else:
                    cmd = [
                        sys.executable, os.path.join(AF_DIR, "main.py"),
                        "--dataset",    analysis["entry"],
                        "--name",       analysis["name"],
                        "--target",     tcol,
                        "--sensitive",  ",".join(sattrs),
                        "--classifier", classifier_spec,
                        "--iterations", str(iterations),
                        "--epochs",     str(epochs),
                        "--threshold",  str(threshold),
                        "--pretrain",   str(pretrain),
                    ]
                    st.info(
                        f"Training **{analysis['name']}** — predict '{tcol}', "
                        f"protect {sattrs}, classifier `{clf_label}`")
                    rc, _, _ = stream_with_agents(cmd, spinner="Training in progress…")
                    shutil.rmtree(analysis["tmp_dir"], ignore_errors=True)
                    st.session_state.pop("analysis", None)
                    if rc == 0:
                        st.success("✅ Training complete! Scroll down for results and the "
                                   "qualitative bias report.")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("❌ Training failed. Open the raw log above to see why.")

    st.divider()

    if st.button("🔄 Refresh", key="refresh_my"):
        st.cache_data.clear()

    memory = load_my_memory()

    if not memory:
        st.warning(f"No data found at `{MY_MEMORY}`.")
        st.info("Run: `.venv\\Scripts\\python adversarial_fairness/main.py --dataset adult`")
    else:
        datasets    = group_by_dataset(memory)
        ds_names    = sorted(datasets.keys())
        ds_tabs     = st.tabs([f"📊 {n.upper()}" for n in ds_names])

        for ds_tab, ds_name in zip(ds_tabs, ds_names):
            with ds_tab:
                runs = datasets[ds_name]
                st.subheader(f"**{ds_name.upper()}** — {len(runs)} run(s)")
                run_labels = [f"Run {r['_run_index']+1}  {r.get('timestamp','')[:10]}" for r in runs]
                run_tabs   = st.tabs(run_labels)
                for i, (rt, run) in enumerate(zip(run_tabs, runs)):
                    with rt:
                        show_my_run_summary(run)
                        st.plotly_chart(plot_my_run(run), use_container_width=True,
                                        key=f"my_{ds_name}_{i}")
                        render_run_report(run)


# TAB 2 : FFB Benchmark

with tab_ffb:
    col_ref, col_dl, _ = st.columns([1, 2, 6])
    if col_ref.button("🔄 Refresh", key="refresh_ffb"):
        st.cache_data.clear()
        st.rerun()

    col_dl.info("To load more methods: `python download_ffb_wandb.py`", icon="💡")

    # Upload a dataset and run the FFB methods on it
    with st.expander("🧪 Run FFB methods on your own dataset (upload)", expanded=False):
        up = st.file_uploader("Upload a tabular CSV", type=["csv"], key="ffb_gen_upload")
        if up is not None:
            try:
                head = pd.read_csv(up, nrows=200)
                cols = list(head.columns)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                cols = []

            if cols:
                gc1, gc2 = st.columns(2)
                gen_target = gc1.selectbox("Target column (what to predict)", cols, key="ffb_gen_target")
                gen_sens   = gc2.multiselect(
                    "Sensitive column(s) — one FFB sweep is run per attribute",
                    [c for c in cols if c != gen_target], key="ffb_gen_sens")

                mc1, mc2, mc3 = st.columns(3)
                gen_methods = mc1.multiselect("Methods", ["erm", "adv", "laftr", "hsic", "pr"],
                                              default=["erm", "adv", "laftr", "hsic", "pr"],
                                              key="ffb_gen_methods")
                gen_seeds = mc2.selectbox("Seeds", ["Full (10)", "Quick (1 — smoke test)"],
                                          index=0, key="ffb_gen_seeds")
                run_gen = mc3.button("▶ Run FFB sweep", type="primary",
                                     disabled=not (gen_sens and gen_methods))

                st.caption("⚠️ A full sweep (10 seeds, all λ) can take **hours** on CPU. "
                           "Use 'Quick' first to verify it works.")

                if run_gen and gen_sens and gen_methods:
                    gdir = os.path.join(ROOT, "fair_fairness_benchmark", "datasets", "generic")
                    os.makedirs(gdir, exist_ok=True)
                    up.seek(0)
                    with open(os.path.join(gdir, "data.csv"), "wb") as fh:
                        fh.write(up.read())
                    json.dump({"csv_name": "data.csv", "target_attr": gen_target,
                               "sensitive_attrs": gen_sens, "drop_cols": []},
                              open(os.path.join(gdir, "config.json"), "w"), indent=2)

                    python = sys.executable
                    cmd = [python, "run_generic_sweep.py", "--methods", ",".join(gen_methods),
                           "--sensitive_attrs", ",".join(gen_sens)]
                    if gen_seeds.startswith("Quick"):
                        cmd += ["--seeds", "42"]

                    st.info(f"Running {len(gen_methods)} method(s) on {len(gen_sens)} "
                            f"attribute(s). Results stream below.")
                    log_box, logs = st.empty(), []
                    with st.spinner("FFB sweep running…"):
                        proc = subprocess.Popen(
                            cmd, cwd=os.path.join(ROOT, "fair_fairness_benchmark"),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "WANDB_MODE": "disabled"})
                        for line in proc.stdout:
                            logs.append(line.rstrip())
                            log_box.code("\n".join(logs[-30:]))
                        proc.wait()
                    if proc.returncode == 0:
                        st.success("✅ FFB sweep complete. Select dataset **generic** in the "
                                   "filters below to see the results.")
                        st.cache_data.clear()
                    else:
                        st.error("❌ Sweep failed — check the log above.")

    ffb_all = load_ffb_results()

    if not ffb_all:
        st.warning(f"No FFB results found in `{FFB_RESULTS}`.")
        st.code("python download_ffb_wandb.py --quick")
    else:
        # Filters inside the tab (not sidebar)
        st.markdown("#### Filters")
        fc1, fc2, fc3, fc4 = st.columns(4)

        datasets_ffb = sorted(set(r["metadata"]["dataset"] for r in ffb_all))
        sel_ds       = fc1.selectbox("Dataset", datasets_ffb, key="ffb_ds")

        sens_ffb     = sorted(set(r["metadata"]["sensitive_attr"] for r in ffb_all
                                  if r["metadata"]["dataset"] == sel_ds))
        sel_sens     = fc2.selectbox("Sensitive Attribute", sens_ffb, key="ffb_sens")

        filtered_base = [r for r in ffb_all
                         if r["metadata"]["dataset"] == sel_ds
                         and r["metadata"]["sensitive_attr"] == sel_sens]

        # Hide the diff* variants from the FFB tab (not part of the comparison)
        _HIDE_FFB = {"diffdp", "diffeopp", "diffeodd"}
        methods_ffb  = sorted({r["metadata"]["method"] for r in filtered_base
                               if r["metadata"]["method"].lower() not in _HIDE_FFB})
        seeds_ffb    = sorted(set(r["metadata"]["seed"]   for r in filtered_base))

        # Default to ALL — never show empty results by accident
        sel_methods = fc3.multiselect("Methods", methods_ffb,
                                      default=methods_ffb, key=f"ffb_methods_{sel_ds}_{sel_sens}")
        sel_seeds   = fc4.multiselect("Seeds",   seeds_ffb,
                                      default=seeds_ffb,   key=f"ffb_seeds_{sel_ds}_{sel_sens}")

        # Fall back to all if user cleared the selection
        if not sel_methods:
            sel_methods = methods_ffb
        if not sel_seeds:
            sel_seeds = seeds_ffb

        filtered = [r for r in filtered_base
                    if r["metadata"]["method"] in sel_methods
                    and r["metadata"]["seed"]   in sel_seeds]

        if not filtered:
            st.warning(f"No results for **{sel_ds} / {sel_sens}**. "
                       f"Run `python download_ffb_wandb.py` to fetch more data.")
        else:
            st.caption(f"{len(filtered)} runs loaded")
            ft1, ft2, ft3, ft4 = st.tabs([
                "Metrics Table", "Training Curves",
                "Utility-Fairness Scatter", "Max-Acc / Max-Prule",
            ])

            with ft1:
                st.subheader("Final Test Metrics")
                df_final = ffb_final_df(filtered)
                avail    = [m for m in ALL_METRICS if df_final[m].notna().any()]
                st.dataframe(
                    df_final[["method", "lam"] + avail].style
                    .highlight_max(subset=[m for m in UTILITY_METRICS  if m in avail], color="lightgreen")
                    .highlight_min(subset=[m for m in FAIRNESS_METRICS if m in avail], color="lightgreen"),
                    use_container_width=True,
                )
                n_cols = min(3, len(avail))
                if n_cols:
                    cols = st.columns(n_cols)
                    for i, metric in enumerate(avail):
                        fig = px.bar(df_final, x="method", y=metric, color="method",
                                     title=metric.upper())
                        fig.update_layout(showlegend=False, height=260, margin=dict(t=40, b=20))
                        cols[i % n_cols].plotly_chart(fig, use_container_width=True)

            with ft2:
                st.subheader("Training Curves")
                df_hist  = ffb_history_df(filtered)
                avail_m  = sorted([c for c in df_hist.columns if "/" in c and df_hist[c].notna().any()])
                if avail_m:
                    c1, c2 = st.columns(2)
                    split  = c1.selectbox("Split", ["test", "val", "train"], key="ffb_split")
                    base_m = sorted(set(m.split("/")[1] for m in avail_m))
                    metric = c2.selectbox("Metric", base_m, key="ffb_metric")
                    col    = f"{split}/{metric}"
                    if col in df_hist.columns:
                        fig = px.line(df_hist, x="step", y=col, color="method",
                                      title=f"{col} over training")
                        fig.update_layout(height=420)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No per-step metrics available.")

            with ft3:
                st.subheader("Utility-Fairness Trade-off")
                df_hist = ffb_history_df(filtered)
                f_opts  = [c for c in df_hist.columns if any(c.endswith(f"/{m}") for m in FAIRNESS_METRICS)]
                u_opts  = [c for c in df_hist.columns if any(c.endswith(f"/{m}") for m in UTILITY_METRICS)]
                if f_opts and u_opts:
                    c1, c2, c3 = st.columns(3)
                    xc   = c1.selectbox("Fairness (X)", f_opts, key="ffb_x")
                    yc   = c2.selectbox("Utility (Y)",  u_opts, key="ffb_y")
                    spl  = c3.selectbox("Split", ["test", "val", "train"], key="ffb_spl")
                    xcol = f"{spl}/{xc.split('/')[-1]}"
                    ycol = f"{spl}/{yc.split('/')[-1]}"
                    df_p = df_hist[[c for c in [xcol, ycol, "method", "step"] if c in df_hist.columns]].dropna()
                    if not df_p.empty and xcol in df_p and ycol in df_p:
                        fig = px.scatter(df_p, x=xcol, y=ycol, color="method",
                                         hover_data=["step"], title=f"{ycol} vs {xcol}")
                        fig.update_layout(height=480)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough metric columns for scatter plot.")

            with ft4:
                st.subheader(f"Max-Acc / Max-Prule — {sel_ds} / {sel_sens}")
                st.caption(
                    "Per method: the λ that maximizes **accuracy** and the λ that "
                    "maximizes **P-rule** (mean over the selected seeds). ΔDP / ΔEOdd / "
                    "ΔEOpp are the fairness gaps at that operating point (lower is "
                    "fairer). Single-λ methods (ERM/LAFTR) show one **Single** row. "
                    "**Ours** is our adaptive method's single operating point "
                    "(Final if the stop condition was reached, else Best). All values in %."
                )
                my_mem  = load_my_memory()
                df_sel  = selection_table(ffb_final_df(filtered), my_mem, sel_ds, sel_sens)
                if df_sel.empty:
                    st.info("No operating points to show for this selection.")
                else:
                    def _hl_ours(row):
                        return ["background-color: #fff3cd" if row["Method"] == "Ours" else ""
                                for _ in row]
                    st.dataframe(
                        df_sel.style.apply(_hl_ours, axis=1)
                              .format({c: "{:.2f}" for c in ["Acc", "P-rule", "ΔDP", "ΔEOdd", "ΔEOpp"]},
                                      na_rep="—"),
                        use_container_width=True, hide_index=True,
                    )
                    if (df_sel["Method"] == "Ours").any() and \
                       df_sel.loc[df_sel["Method"] == "Ours", "ΔDP"].isna().all():
                        st.caption("ℹ️ Ours ΔDP/ΔEOdd/ΔEOpp are blank — re-run that dataset "
                                   "to persist them (older runs only stored Acc + P-rule).")


