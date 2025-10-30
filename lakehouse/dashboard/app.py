from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

from lakehouse.config import (
    SCHEMA_GOLD,
    TBL_GLD_DAILY_METRICS,
    MODELS_DIR,
)
from lakehouse.utils.db import get_connection

st.set_page_config(page_title="Local Lakehouse Dashboard", layout="wide")
st.title("Local Lakehouse: Marketing Performance")

# Plotly styling
PX_TEMPLATE = "plotly_white"
CURRENCY_PREFIX = "£"

# Optional local LLM (LM Studio / OpenAI-compatible) config
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "")  # e.g., http://localhost:1234/v1/chat/completions
LLM_API_KEY = os.getenv("LLM_API_KEY", "")     # optional for LM Studio
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def llm_summarise(title: str, bullets: list[str]) -> str:
    """Summarise bullets into 2 concise UK-English sentences: 1 summary + 1 'So what?' action.
    Uses local LLM if configured; otherwise a rule-based fallback.
    """
    if LLM_ENDPOINT:
        try:
            headers = {"Content-Type": "application/json"}
            if LLM_API_KEY:
                headers["Authorization"] = f"Bearer {LLM_API_KEY}"
            prompt = (
                f"You are a marketing analytics assistant. For '{title}', write exactly two UK-English sentences: "
                f"(1) a clear summary of the metrics; (2) a 'So what?' action recommendation for a non-technical decision maker. "
                f"Avoid jargon; be factual and concise.\n\nMetrics:\n- " + "\n- ".join(bullets)
            )
            payload = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 180,
            }
            resp = requests.post(LLM_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=10)
            if resp.ok:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return content.strip()
        except Exception:
            pass
    # Fallback
    if not bullets:
        return "Summary unavailable. Consider reallocating budget to top‑ROAS ad sets."
    summary = bullets[0]
    action = "Review pacing and shift small budget from low‑ROAS to high‑ROAS ad sets to improve returns."
    if len(bullets) > 1:
        action = bullets[1]
    return summary + " " + action

# Load data
con = get_connection()
df: pd.DataFrame = con.execute(f"SELECT * FROM {SCHEMA_GOLD}.{TBL_GLD_DAILY_METRICS}").df()
try:
    dq_df: pd.DataFrame = con.execute(f"SELECT * FROM {SCHEMA_GOLD}.data_quality_report").df()
except Exception:
    dq_df = pd.DataFrame()
con.close()

if df.empty:
    st.warning("No gold data found. Please run the pipeline first (python -m lakehouse.run_all).")
    st.stop()

# Sidebar filters – Ad sets and date range
with st.sidebar:
    st.markdown("### Filters")
    ad_sets = sorted(df["ad_set_id"].unique())
    colA, colB = st.columns(2)
    with colA:
        if st.button("Select all"):
            st.session_state["sel_ad_sets"] = ad_sets
    with colB:
        if st.button("Clear"):
            st.session_state["sel_ad_sets"] = []
    sel_default = st.session_state.get("sel_ad_sets", ad_sets)
    sel_ad_sets = st.multiselect("Ad sets", options=ad_sets, default=sel_default)

    min_date, max_date = pd.to_datetime(df["date"]).min(), pd.to_datetime(df["date"]).max()
    date_range = st.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

# Apply filters
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_d, end_d = min_date, max_date

mask = (df["ad_set_id"].isin(sel_ad_sets)) & (pd.to_datetime(df["date"]).between(start_d, end_d))
df_f = df.loc[mask].copy()

# Top KPIs (context)
st.caption("Overview of core outcomes for the selected ad sets and dates.")
kp1, kp2, kp3, kp4 = st.columns(4)
kp1.metric("Total spend", f"{CURRENCY_PREFIX}{df_f['spend'].sum():,.0f}")
kp2.metric("Total revenue", f"{CURRENCY_PREFIX}{df_f['revenue'].sum():,.0f}")
kp3.metric("Total bookings", f"{int(df_f['bookings'].sum()):,}")
roas = (df_f["revenue"].sum() / df_f["spend"].sum()) if df_f["spend"].sum() > 0 else 0
kp4.metric("Overall ROAS", f"{roas:,.2f}x")

# Tabs
tab_overview, tab_dq, tab_models, tab_whatif = st.tabs(["Overview", "Data Quality", "Models", "What‑if"])

with tab_overview:
    st.caption("Trends and mix to understand how spend converts into revenue and bookings.")

    # Dual-axis: Spend (bar) vs Revenue (line) per day
    st.subheader("Daily spend vs revenue")
    df_day = df_f.groupby("date").agg({"spend": "sum", "revenue": "sum"}).reset_index()
    fig_combo = go.Figure()
    fig_combo.add_trace(go.Bar(x=df_day["date"], y=df_day["spend"], name="Spend", marker_color="#1f77b4", yaxis="y1"))
    fig_combo.add_trace(go.Scatter(x=df_day["date"], y=df_day["revenue"], name="Revenue", mode="lines+markers", line=dict(color="#ff7f0e", width=2), yaxis="y2"))
    fig_combo.update_layout(
        template=PX_TEMPLATE,
        yaxis=dict(title=f"Spend ({CURRENCY_PREFIX})", side="left", showgrid=True),
        yaxis2=dict(title=f"Revenue ({CURRENCY_PREFIX})", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=40, r=10, l=10, b=10),
    )
    st.plotly_chart(fig_combo, use_container_width=True)

    # ROAS bar per ad set (sorted)
    st.subheader("ROAS by ad set (sorted)")
    agg = df_f.groupby("ad_set_id").agg({"spend": "sum", "revenue": "sum"}).reset_index()
    agg["roas"] = np.where(agg["spend"] > 0, agg["revenue"] / agg["spend"], 0.0)
    roas_sorted = agg.sort_values("roas", ascending=False)
    fig_roas = px.bar(roas_sorted, x="roas", y="ad_set_id", orientation="h", title="Average ROAS (higher is better)", template=PX_TEMPLATE, color="roas", color_continuous_scale="Blues")
    fig_roas.update_xaxes(title="ROAS (x)")
    fig_roas.update_yaxes(title="Ad set")
    st.plotly_chart(fig_roas, use_container_width=True)

    # Pacing status stacked bar by day
    st.subheader("Pacing status by day")
    pace = df_f.groupby(["date", "pacing_status"]).size().reset_index(name="count")
    pace["pacing_status"] = pace["pacing_status"].astype(str)
    fig_pace = px.bar(pace, x="date", y="count", color="pacing_status", barmode="stack", title="Under/On/Over pacing mix", template=PX_TEMPLATE, color_discrete_map={
        "under_pacing": "#d62728",
        "on_pace": "#2ca02c",
        "over_pacing": "#ff7f0e",
        "unknown": "#7f7f7f",
    })
    fig_pace.update_yaxes(title="Ad set count")
    st.plotly_chart(fig_pace, use_container_width=True)

    # Top/bottom ROAS table
    st.subheader("Top/bottom ROAS")
    top = roas_sorted.head(5)[["ad_set_id", "spend", "revenue", "roas"]].copy()
    bottom = roas_sorted.tail(5)[["ad_set_id", "spend", "revenue", "roas"]].copy()
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Top 5 ad sets by ROAS")
        top_display = top.rename(columns={"ad_set_id": "Ad set", "spend": f"Spend ({CURRENCY_PREFIX})", "revenue": f"Revenue ({CURRENCY_PREFIX})", "roas": "ROAS (x)"})
        st.dataframe(top_display, use_container_width=True)
    with c2:
        st.caption("Bottom 5 ad sets by ROAS")
        bottom_display = bottom.rename(columns={"ad_set_id": "Ad set", "spend": f"Spend ({CURRENCY_PREFIX})", "revenue": f"Revenue ({CURRENCY_PREFIX})", "roas": "ROAS (x)"})
        st.dataframe(bottom_display, use_container_width=True)

    # Summary
    st.subheader("Summary")
    if not df_f.empty:
        df_sorted = df_f.sort_values("date")
        spend_chg = (df_sorted["spend"].iloc[-1] - df_sorted["spend"].iloc[0]) / max(df_sorted["spend"].iloc[0], 1e-9) * 100
        rev_chg = (df_sorted["revenue"].iloc[-1] - df_sorted["revenue"].iloc[0]) / max(df_sorted["revenue"].iloc[0], 1e-9) * 100
        bullets = [
            f"Spend changed by {spend_chg:,.1f}% and revenue by {rev_chg:,.1f}%.",
            f"Focus budget on the top ROAS ad sets ({', '.join(top_display['Ad set'])}); trim or test creatives for the bottom group.",
        ]
        st.caption(llm_summarise("Overview", bullets))

with tab_dq:
    st.caption("Cleansed silver data is validated; issues appear here for quick triage.")
    st.subheader("Data Quality Report")
    if dq_df.empty:
        st.info("Run the pipeline to generate the data quality report.")
    else:
        checks_total = len(dq_df)
        checks_ok = int((dq_df.get("null_fraction", 0) == 0).sum()) if "null_fraction" in dq_df.columns else checks_total
        cA, cB = st.columns(2)
        cA.metric("Checks passed", f"{checks_ok}/{checks_total}")
        if "null_fraction" in dq_df.columns:
            dq_issues = dq_df[dq_df["null_fraction"] > 0].sort_values("null_fraction", ascending=False)
        else:
            dq_issues = dq_df
        if dq_issues.empty:
            cB.success("No nulls detected in key columns.")
        else:
            cB.warning("Some columns have nulls – see details below.")
            pretty = dq_issues.rename(columns={
                "column_name": "Column",
                "dtype": "Type",
                "null_fraction": "Null fraction",
                "mean": "Mean",
                "std": "Std dev",
                "min": "Min",
                "max": "Max",
                "generated_at": "Generated at",
            })
            st.dataframe(pretty, use_container_width=True)
        with st.expander("Show full DQ table"):
            st.dataframe(dq_df, use_container_width=True)

with tab_models:
    st.caption("How well we predict outcomes and what drives the predictions.")
    st.subheader("Bookings regression (XGBoost)")
    reg_meta_path = MODELS_DIR / "bookings_xgb.json"
    if reg_meta_path.exists():
        with open(reg_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        st.json(meta.get("metrics", {}))
        if meta.get("features") and meta.get("importances"):
            imp = np.array(meta["importances"], dtype=float)
            imp_pct = (imp / imp.sum() * 100.0).tolist() if imp.sum() > 0 else imp.tolist()
            imp_df = pd.DataFrame({"feature": meta["features"], "importance_%": imp_pct}).sort_values("importance_%", ascending=False)
            fig_imp = px.bar(imp_df, x="importance_%", y="feature", orientation="h", title="Feature importances (regression)", template=PX_TEMPLATE, color="importance_%", color_continuous_scale="Tealgrn")
            fig_imp.update_xaxes(title="Importance (%)")
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Regression artefacts not found. Train the model after building gold tables.")

    st.subheader("Under‑pacing classification (XGBoost)")
    clf_meta_path = MODELS_DIR / "underpacing_xgb.json"
    clf_pkl_path = MODELS_DIR / "underpacing_xgb.pkl"
    if clf_meta_path.exists() and clf_pkl_path.exists():
        with open(clf_meta_path, "r", encoding="utf-8") as f:
            clf_meta = json.load(f)
        st.json(clf_meta.get("metrics", {}))
        if clf_meta.get("features") and clf_meta.get("importances"):
            cimp = np.array(clf_meta["importances"], dtype=float)
            cimp_pct = (cimp / cimp.sum() * 100.0).tolist() if cimp.sum() > 0 else cimp.tolist()
            cimp_df = pd.DataFrame({"feature": clf_meta["features"], "importance_%": cimp_pct}).sort_values("importance_%", ascending=False)
            fig_cimp = px.bar(cimp_df, x="importance_%", y="feature", orientation="h", title="Feature importances (classification)", template=PX_TEMPLATE, color="importance_%", color_continuous_scale="Purpor")
            fig_cimp.update_xaxes(title="Importance (%)")
            st.plotly_chart(fig_cimp, use_container_width=True)

        st.markdown("### AUC before/after with label noise")
        noise = st.slider("Label noise (flip % of labels)", 0.0, 0.3, 0.1, 0.01)
        try:
            import pickle
            with open(clf_pkl_path, "rb") as f:
                clf_pipe = pickle.load(f)
            feat_cols = clf_meta.get("features", [])
            df_score = df_f.dropna(subset=feat_cols + ["pacing_status"]).copy()
            if not df_score.empty:
                X = df_score[feat_cols]
                y_true = (df_score["pacing_status"].astype(str) == "under_pacing").astype(int).values
                proba = clf_pipe.predict_proba(X)[:, 1]
                auc_base = roc_auc_score(y_true, proba)
                rng = np.random.default_rng(42)
                y_noisy = y_true.copy()
                n_flip = int(len(y_true) * noise)
                if n_flip > 0:
                    idx = rng.choice(len(y_true), size=n_flip, replace=False)
                    y_noisy[idx] = 1 - y_noisy[idx]
                auc_noisy = roc_auc_score(y_noisy, proba)
                fpr0, tpr0, _ = roc_curve(y_true, proba)
                fpr1, tpr1, _ = roc_curve(y_noisy, proba)
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr0, y=tpr0, mode="lines", name=f"Baseline ROC (AUC {auc_base:.3f})"))
                roc_fig.add_trace(go.Scatter(x=fpr1, y=tpr1, mode="lines", name=f"Noisy ROC (AUC {auc_noisy:.3f})"))
                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
                roc_fig.update_layout(template=PX_TEMPLATE, title="ROC curves: baseline vs noisy labels")
                st.plotly_chart(roc_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute AUC comparison: {e}")
    else:
        st.info("Classification artefacts not found. Train the classifier.")

with tab_whatif:
    st.caption("Budget scenarios to explore reallocation and scaling trade‑offs.")
    st.subheader("Budget reallocation (what‑if)")
    with st.expander("Reallocate budget from low‑ROAS to high‑ROAS ad sets"):
        period = st.selectbox("Aggregation period", ["All time", "Last 7 days", "Last 14 days"], index=1)
        if period == "Last 7 days":
            df_w = df_f[df_f["date"] >= (df_f["date"].max() - pd.Timedelta(days=6))]
        elif period == "Last 14 days":
            df_w = df_f[df_f["date"] >= (df_f["date"] .max() - pd.Timedelta(days=13))]
        else:
            df_w = df_f

        agg2 = df_w.groupby("ad_set_id").agg({"spend": "sum", "revenue": "sum"}).reset_index()
        agg2["roas"] = np.where(agg2["spend"] > 0, agg2["revenue"] / agg2["spend"], 0.0)

        shift_pct = st.slider("Shift % of total spend from bottom‑quartile to top‑quartile", 0.0, 0.3, 0.1, 0.01)
        elasticity = st.slider("Revenue elasticity (0.5 conservative … 1.0 proportional)", 0.5, 1.0, 0.8, 0.05)

        if len(agg2) >= 4 and agg2["spend"].sum() > 0:
            q = agg2["roas"].quantile([0.25, 0.75]).values
            bottom2 = agg2[agg2["roas"] <= q[0]].copy()
            top2 = agg2[agg2["roas"] >= q[1]].copy()
            mid2 = agg2[(agg2["roas"] > q[0]) & (agg2["roas"] < q[1])].copy()

            total_spend = agg2["spend"].sum()
            shift_amount = total_spend * shift_pct
            if not bottom2.empty and not top2.empty:
                bottom2["spend_new"] = bottom2["spend"] * (1 - shift_pct)
                top_weight = top2["spend"].sum()
                add = shift_amount
                top2["spend_new"] = top2["spend"] + add * (top2["spend"] / max(top_weight, 1e-9))
                mid2["spend_new"] = mid2["spend"]

                proj = pd.concat([bottom2, mid2, top2], ignore_index=True)
                proj["revenue_proj"] = proj["roas"] * proj["spend_new"] ** elasticity / (proj["spend"].replace(0, 1e-9) ** (elasticity - 1))

                base_rev = agg2["revenue"].sum()
                proj_rev = proj["revenue_proj"].sum()
                delta = proj_rev - base_rev

                st.write(f"Base revenue: {CURRENCY_PREFIX}{base_rev:,.0f} → Projected: {CURRENCY_PREFIX}{proj_rev:,.0f} (Δ {CURRENCY_PREFIX}{delta:,.0f})")
                st.dataframe(proj[["ad_set_id", "spend", "spend_new", "roas"]].sort_values("roas", ascending=False))
            else:
                st.info("Not enough dispersion in ROAS to form quartiles. Try a wider period.")
        else:
            st.info("Need at least 4 ad sets with non-zero spend for the reallocation demo.")

    st.subheader("Simple spend scaling")
    spend_scale = st.slider("Spend scale", 0.5, 1.5, 1.0, 0.05)
    adj_spend = df_f["spend"].sum() * spend_scale
    st.write(f"Adjusted spend: {CURRENCY_PREFIX}{adj_spend:,.0f}")
