import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
from glob import glob

st.set_page_config(page_title="FFB Fairness Benchmark", layout="wide")
st.title("FFB: Fairness Benchmark Results")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


@st.cache_data
def load_results():
    files = glob(os.path.join(RESULTS_DIR, "*.json"))
    all_results = []
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            if data.get("history"):
                all_results.append(data)
        except Exception:
            pass
    return all_results


def reload():
    load_results.clear()
    st.rerun()


col_refresh, _ = st.columns([1, 8])
if col_refresh.button("Refresh results"):
    reload()

results = load_results()

if not results:
    st.warning(f"No results found in `{RESULTS_DIR}`. Run the benchmark methods first.")
    st.info("Example:\n```\ncd src\npython ffb_tabular_erm.py --dataset migration --sensitive_attr sex\n```")
    st.stop()

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")

datasets = sorted(set(r["metadata"]["dataset"] for r in results))
selected_dataset = st.sidebar.selectbox("Dataset", datasets)

sens_attrs = sorted(set(
    r["metadata"]["sensitive_attr"] for r in results
    if r["metadata"]["dataset"] == selected_dataset
))
selected_sens = st.sidebar.selectbox("Sensitive Attribute", sens_attrs)

filtered = [
    r for r in results
    if r["metadata"]["dataset"] == selected_dataset
    and r["metadata"]["sensitive_attr"] == selected_sens
]

methods = sorted(set(r["metadata"]["method"] for r in filtered))
selected_methods = st.sidebar.multiselect("Methods", methods, default=methods)

seeds = sorted(set(r["metadata"]["seed"] for r in filtered))
selected_seeds = st.sidebar.multiselect("Seeds", seeds, default=seeds)

filtered = [
    r for r in filtered
    if r["metadata"]["method"] in selected_methods
    and r["metadata"]["seed"] in selected_seeds
]

if not filtered:
    st.warning("No results match the selected filters.")
    st.stop()

# ── Helper: build DataFrames ─────────────────────────────────────────────────
UTILITY_METRICS  = ["acc", "ap", "auc", "f1"]
FAIRNESS_METRICS = ["dp", "eopp", "eodd", "abcc", "prule"]
ALL_METRICS      = UTILITY_METRICS + FAIRNESS_METRICS


def final_metrics_df(results_list):
    rows = []
    for r in results_list:
        last = r["history"][-1]
        meta = r["metadata"]
        row = {
            "method":       meta["method"],
            "sensitive":    meta["sensitive_attr"],
            "lam":          meta["lam"],
            "seed":         meta["seed"],
        }
        for m in ALL_METRICS:
            row[m] = round(last.get(f"test/{m}", float("nan")), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "sensitive", "lam", "seed"]).reset_index(drop=True)


def history_df(results_list):
    rows = []
    for r in results_list:
        meta = r["metadata"]
        label = f"{meta['method']} | {meta['sensitive_attr']} | lam={meta['lam']} | seed={meta['seed']}"
        for entry in r["history"]:
            row = {
                "method":    meta["method"],
                "sensitive": meta["sensitive_attr"],
                "lam":       meta["lam"],
                "seed":      meta["seed"],
                "run":       label,
                "step":      entry["step"],
            }
            row.update({k: v for k, v in entry.items() if k not in ("step",)})
            rows.append(row)
    return pd.DataFrame(rows)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Metrics Table", "Training Curves", "Utility-Fairness Scatter"])

# ── Tab 1: Metrics Table ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Final Test Metrics (last training step)")
    df_final = final_metrics_df(filtered)

    available_cols = [m for m in ALL_METRICS if df_final[m].notna().any()]
    st.dataframe(
        df_final[["method", "lam"] + available_cols].style.highlight_max(
            subset=[m for m in UTILITY_METRICS if m in available_cols], color="lightgreen"
        ).highlight_min(
            subset=[m for m in FAIRNESS_METRICS if m in available_cols], color="lightgreen"
        ),
        use_container_width=True,
    )

    st.subheader("Bar Charts")
    plot_cols = [m for m in available_cols if df_final[m].notna().any()]
    n_cols = min(3, len(plot_cols))
    cols = st.columns(n_cols)
    for i, metric in enumerate(plot_cols):
        fig = px.bar(
            df_final, x="method", y=metric,
            color="method", title=metric.upper(),
            labels={"method": "Method", metric: metric},
        )
        fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=20))
        cols[i % n_cols].plotly_chart(fig, use_container_width=True)

# ── Tab 2: Training Curves ────────────────────────────────────────────────────
with tab2:
    st.subheader("Training Curves")
    df_hist = history_df(filtered)

    available_metrics = sorted([
        c for c in df_hist.columns
        if "/" in c and df_hist[c].notna().any()
    ])

    if not available_metrics:
        st.info("No per-step metrics found.")
    else:
        c1, c2 = st.columns(2)
        split = c1.selectbox("Split", ["test", "val", "train"], key="split_curves")
        base_metrics = sorted(set(m.split("/")[1] for m in available_metrics))
        metric_name = c2.selectbox("Metric", base_metrics, key="metric_curves")
        col_name = f"{split}/{metric_name}"

        if col_name in df_hist.columns:
            fig = px.line(
                df_hist, x="step", y=col_name, color="method",
                title=f"{col_name} over training",
                labels={"step": "Step", col_name: metric_name, "method": "Method"},
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"`{col_name}` not available for the selected methods.")

# ── Tab 3: Utility-Fairness Scatter ──────────────────────────────────────────
with tab3:
    st.subheader("Utility-Fairness Trade-off")
    df_hist = history_df(filtered)

    fairness_options = [c for c in df_hist.columns if any(c.endswith(f"/{m}") for m in FAIRNESS_METRICS)]
    utility_options  = [c for c in df_hist.columns if any(c.endswith(f"/{m}") for m in UTILITY_METRICS)]

    if not fairness_options or not utility_options:
        st.info("Not enough metric columns to plot. Run more methods.")
    else:
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("Fairness (X)", fairness_options, key="x_scatter")
        y_col = c2.selectbox("Utility (Y)", utility_options, key="y_scatter")
        split_scatter = c3.selectbox("Split", ["test", "val", "train"], key="split_scatter")

        x_col_s = f"{split_scatter}/{x_col.split('/')[-1]}"
        y_col_s = f"{split_scatter}/{y_col.split('/')[-1]}"

        valid_cols = [c for c in [x_col_s, y_col_s, "method", "step"] if c in df_hist.columns]
        plot_df = df_hist[valid_cols].dropna()

        if plot_df.empty or x_col_s not in plot_df.columns or y_col_s not in plot_df.columns:
            st.info("Not enough data for this combination.")
        else:
            fig = px.scatter(
                plot_df, x=x_col_s, y=y_col_s, color="method",
                hover_data=["step"],
                title=f"{y_col_s} vs {x_col_s}",
                labels={x_col_s: x_col_s, y_col_s: y_col_s, "method": "Method"},
            )
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)
