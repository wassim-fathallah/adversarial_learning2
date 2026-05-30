"""
Unified Fairness Dashboard
--------------------------
Tab 1 — Agentic Adversarial Debiasing  : reads adversarial_fairness/long_term_memory.json
Tab 2 — FFB Benchmark                  : reads fair_fairness_benchmark/results/*.json
Tab 3 — Comparison — Migration         : both systems side by side
"""

import json
import os
from collections import defaultdict
from glob import glob

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
MY_MEMORY   = os.path.join(ROOT, "adversarial_fairness", "long_term_memory.json")
FFB_RESULTS = os.path.join(ROOT, "fair_fairness_benchmark", "results")

COLORS = ["#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#3498db", "#e67e22"]

st.set_page_config(page_title="Fairness Dashboard", page_icon="⚖️", layout="wide")
st.title("⚖️ Adversarial Fairness — Thesis Dashboard")


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_my_memory():
    if not os.path.exists(MY_MEMORY):
        return {}
    with open(MY_MEMORY) as f:
        return json.load(f)


@st.cache_data(ttl=30)
def load_ffb_results():
    files = glob(os.path.join(FFB_RESULTS, "*.json"))
    results = []
    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            if data.get("history"):
                results.append(data)
        except Exception:
            pass
    return results


# ── Helpers — My System ───────────────────────────────────────────────────────

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

    xs = [m["iteration"] for m in iters]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Accuracy / F1 / ROC-AUC",
                                        "P-rule per Attribute",
                                        "Lambda (λ)",
                                        "Adversary Loss"),
                        vertical_spacing=0.07)

    fig.add_trace(go.Scatter(x=xs, y=[m["accuracy"] for m in iters],
                             name="Accuracy", line=dict(color="royalblue"), mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[m["f1_score"] for m in iters],
                             name="F1", line=dict(color="seagreen"), mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[m["roc_auc"] for m in iters],
                             name="AUC", line=dict(color="darkorchid"), mode="lines+markers"), row=1, col=1)
    fig.add_hline(y=0.80, line_dash="dash", line_color="gray", row=1, col=1)

    for i, attr in enumerate(attrs):
        fig.add_trace(go.Scatter(x=xs, y=[m["p_rules"].get(attr, 0) for m in iters],
                                 name=f"P-rule ({attr})",
                                 line=dict(color=COLORS[i % len(COLORS)]), mode="lines+markers"), row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="gray", row=2, col=1)

    for i, attr in enumerate(attrs):
        fig.add_trace(go.Scatter(x=xs,
                                 y=[m["lambda"][i] if i < len(m.get("lambda", [])) else None for m in iters],
                                 name=f"λ ({attr})",
                                 line=dict(color=COLORS[i % len(COLORS)], dash="dot"), mode="lines+markers"), row=3, col=1)

    fig.add_trace(go.Scatter(x=xs, y=[m.get("adv_loss", 0) for m in iters],
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


# ── Helpers — FFB ─────────────────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_my, tab_ffb = st.tabs([
    "⚙️ Agentic Adversarial Debiasing",
    "📊 FFB Benchmark",
])


# ── TAB 1 : Agentic Adversarial Debiasing ────────────────────────────────────

with tab_my:
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


# ── TAB 2 : FFB Benchmark ─────────────────────────────────────────────────────

with tab_ffb:
    col_ref, col_dl, _ = st.columns([1, 2, 6])
    if col_ref.button("🔄 Refresh", key="refresh_ffb"):
        st.cache_data.clear()
        st.rerun()

    col_dl.info("To load more methods: `python download_ffb_wandb.py`", icon="💡")

    ffb_all = load_ffb_results()

    if not ffb_all:
        st.warning(f"No FFB results found in `{FFB_RESULTS}`.")
        st.code("python download_ffb_wandb.py --quick")
    else:
        # ── Filters inside the tab (not sidebar) ──────────────────────────────
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

        methods_ffb  = sorted(set(r["metadata"]["method"] for r in filtered_base))
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
            ft1, ft2, ft3 = st.tabs(["Metrics Table", "Training Curves", "Utility-Fairness Scatter"])

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


