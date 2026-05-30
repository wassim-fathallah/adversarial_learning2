"""
Adversarial Fairness Dashboard
-------------------------------
- One tab per dataset (adult, german, …)
- Within each tab: all runs shown as sub-tabs (Run 1, Run 2, …)
- Each run tab shows: summary metrics + interactive training curves
"""

import json
import os
from collections import defaultdict

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "long_term_memory.json")
COLORS = ["#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE) as f:
        return json.load(f)


def group_by_dataset(memory: dict) -> dict[str, list[dict]]:
    """Returns {dataset_name: [enriched_run_dict, ...]}"""
    groups: dict[str, list] = defaultdict(list)
    for key, runs in memory.items():
        dataset_name = key.split("|")[0]
        target_col   = key.split("|")[1] if "|" in key else ""
        attrs        = key.split("|")[2] if key.count("|") >= 2 else ""
        for idx, run in enumerate(runs):
            groups[dataset_name].append({
                **run,
                "_key": key,
                "_run_index": idx,
                "_target": target_col,
                "_attrs": attrs,
            })
    return dict(groups)


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sensitive_attrs(run: dict) -> list[str]:
    attrs_str = run.get("_attrs", "")
    if attrs_str:
        return [a.strip() for a in attrs_str.split(",") if a.strip()]
    # Infer from p_rules_final keys
    return list(run.get("p_rules_final", {}).keys())


def plot_run(run: dict) -> go.Figure:
    iters = run.get("iteration_metrics", [])
    attrs = _sensitive_attrs(run)

    if not iters:
        # No per-iteration data — show a simple bar chart of final values
        fig = go.Figure()
        p_rules = run.get("p_rules_final", {})
        for i, (attr, val) in enumerate(p_rules.items()):
            fig.add_trace(go.Bar(
                name=f"P-rule ({attr})",
                x=[attr],
                y=[val],
                marker_color=COLORS[i % len(COLORS)],
            ))
        fig.add_hline(y=80, line_dash="dash", line_color="gray",
                      annotation_text="80% threshold")
        fig.update_layout(
            title="Final P-rules (no iteration data available — re-run to get curves)",
            yaxis_title="P-rule (%)", yaxis_range=[0, 110],
            height=350,
        )
        return fig

    # ── Build figure with 4 subplots ──────────────────────────────────────────
    xs = [m["iteration"] for m in iters]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Performance (Accuracy / F1 / ROC-AUC)",
            "Disparate Impact — P-rule per Attribute",
            "Lambda (λ) per Attribute",
            "Adversary Loss",
        ),
        vertical_spacing=0.07,
    )

    # Row 1 — performance
    fig.add_trace(go.Scatter(x=xs, y=[m["accuracy"]  for m in iters],
                             name="Accuracy",  line=dict(color="royalblue"), mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[m["f1_score"]  for m in iters],
                             name="F1 Score",  line=dict(color="seagreen"),  mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[m["roc_auc"]   for m in iters],
                             name="ROC-AUC",   line=dict(color="darkorchid"), mode="lines+markers"), row=1, col=1)
    # 80% accuracy reference line
    fig.add_hline(y=0.80, line_dash="dash", line_color="gray", row=1, col=1)

    # Row 2 — P-rule
    for i, attr in enumerate(attrs):
        p_vals = [m["p_rules"].get(attr, 0) for m in iters]
        fig.add_trace(go.Scatter(x=xs, y=p_vals, name=f"P-rule ({attr})",
                                 line=dict(color=COLORS[i % len(COLORS)]),
                                 mode="lines+markers"), row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="gray", row=2, col=1)

    # Row 3 — Lambda
    for i, attr in enumerate(attrs):
        lam_vals = [m["lambda"][i] if i < len(m.get("lambda", [])) else None for m in iters]
        fig.add_trace(go.Scatter(x=xs, y=lam_vals, name=f"λ ({attr})",
                                 line=dict(color=COLORS[i % len(COLORS)], dash="dot"),
                                 mode="lines+markers"), row=3, col=1)

    # Row 4 — Adversary loss
    fig.add_trace(go.Scatter(x=xs, y=[m.get("adv_loss", 0) for m in iters],
                             name="Adv. Loss", line=dict(color="tomato"),
                             mode="lines+markers"), row=4, col=1)

    fig.update_yaxes(title_text="Score (0–1)",  range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="P-rule (%)",   range=[0, 110],  row=2, col=1)
    fig.update_yaxes(title_text="λ",                              row=3, col=1)
    fig.update_yaxes(title_text="Loss",                           row=4, col=1)
    fig.update_xaxes(title_text="Iteration", row=4, col=1)

    fig.update_layout(
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Run summary card
# ──────────────────────────────────────────────────────────────────────────────

def show_run_summary(run: dict):
    p_rules = run.get("p_rules_final", {})
    acc     = run.get("accuracy_final", 0)
    success = run.get("success", False)
    ts      = run.get("timestamp", "")[:19].replace("T", " ")
    iters   = run.get("iterations", "?")
    epochs  = run.get("total_epochs", "?")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc*100:.2f}%",
                delta="✓ ≥80%" if acc >= 0.80 else "✗ <80%",
                delta_color="normal" if acc >= 0.80 else "inverse")
    for i, (attr, val) in enumerate(p_rules.items()):
        cols = [col2, col3, col4]
        if i < len(cols):
            cols[i].metric(
                f"P-rule ({attr})", f"{val:.1f}%",
                delta="✓ ≥80%" if val >= 80 else "✗ <80%",
                delta_color="normal" if val >= 80 else "inverse",
            )

    st.caption(
        f"Timestamp: {ts} | Iterations: {iters} | Total epochs: {epochs} | "
        f"Status: {'✅ SUCCESS' if success else '🔄 Not yet'}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Adversarial Fairness Dashboard",
        page_icon="⚖️",
        layout="wide",
    )
    st.title("⚖️ Adversarial Fairness Training Dashboard")

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()

    memory = load_memory()

    if not memory:
        st.warning(f"No training data found at `{MEMORY_FILE}`.")
        st.info("Run `python main.py --dataset adult` (or german/compas) to generate data.")
        return

    datasets = group_by_dataset(memory)
    dataset_names = sorted(datasets.keys())

    # ── One tab per dataset ───────────────────────────────────────────────────
    tabs = st.tabs([f"📊 {name.upper()}" for name in dataset_names])

    for tab, dataset_name in zip(tabs, dataset_names):
        with tab:
            runs = datasets[dataset_name]
            st.subheader(f"Dataset: **{dataset_name.upper()}**")
            st.caption(f"{len(runs)} run(s) recorded")

            if not runs:
                st.info("No runs yet.")
                continue

            # ── Sub-tabs: one per run ─────────────────────────────────────────
            run_labels = [
                f"Run {r['_run_index']+1}  {r.get('timestamp','')[:10]}"
                for r in runs
            ]
            run_tabs = st.tabs(run_labels)

            for i, (run_tab, run) in enumerate(zip(run_tabs, runs)):
                with run_tab:
                    show_run_summary(run)
                    st.plotly_chart(plot_run(run), use_container_width=True, key=f"plot_{dataset_name}_{i}")


if __name__ == "__main__":
    main()
