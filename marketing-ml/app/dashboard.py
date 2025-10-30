from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import os
import numpy as np
import pandas as pd
import duckdb
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.formatting import fmt_gbp, kpi_tile
from utils.charts import add_ma, add_roas_target
from utils.insight import so_what
from utils.lm import call_lm_with_status, SYSTEM_PROMPT, health_check
from utils.exports import export_csv, export_pdf
from utils.insight import sanitize_lm_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "lakehouse.duckdb"

st.set_page_config(page_title="Marketing ML", layout="wide")
st.title("Marketing ML – Local Lakehouse Dashboard")
st.caption("Local, offline decisioning: filtered KPIs, models, and 'So what' actions.")
# Presenter mode toggle per spec (default off); safe to reference later
presenter_mode = st.sidebar.checkbox("Presenter mode", value=False)
st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Connect once per app run
con = duckdb.connect(str(DB_PATH))


def load_gold() -> pd.DataFrame:
    """Load training dataset from gold layer for visualisation and what-if."""
    return con.execute("SELECT * FROM gold.training_dataset").df()


df = load_gold()
if df.empty:
    st.warning("Run the pipeline first: python scripts/run_all.py")
    st.stop()

# Sidebar filters: ad sets and date range with quick actions
st.sidebar.header("Filters")
# ROAS target slider for visuals
roas_target = st.sidebar.slider("ROAS target (×)", 1.0, 10.0, 4.0, 0.5)
ad_sets = sorted(df["ad_set_id"].unique())
colSA, colSB = st.sidebar.columns(2)
if colSA.button("Select all"):
    st.session_state["sel_ads"] = ad_sets
if colSB.button("Reset filters"):
    st.session_state["sel_ads"] = ad_sets
sel_default = st.session_state.get("sel_ads", ad_sets)
sel_ad_sets = st.sidebar.multiselect("Ad sets", ad_sets, default=sel_default)
# removed AOV slider now that real revenue is available
min_d, max_d = pd.to_datetime(df["date"]).min(), pd.to_datetime(df["date"]).max()
colA, colB = st.sidebar.columns(2)
start_d = colA.date_input("Start", min_d.date(), min_value=min_d.date(), max_value=max_d.date())
end_d = colB.date_input("End", max_d.date(), min_value=min_d.date(), max_value=max_d.date())

# Apply filters; short-circuit empty states
mask = (df["ad_set_id"].isin(sel_ad_sets)) & (pd.to_datetime(df["date"]).between(pd.to_datetime(start_d), pd.to_datetime(end_d)))
dff = df.loc[mask].copy()
if dff.empty:
    st.info("No data for current filters. Click 'Reset filters' in the sidebar.")
    st.stop()

# KPI tiles with previous-period comparisons
st.caption(f"Data last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")
period_days = (pd.to_datetime(end_d) - pd.to_datetime(start_d)).days + 1
# Keep bookings proxy as before for continuity (training labels are forward-looking); revenue now real
mean_conv7 = float(dff.get("conv_7d", pd.Series(0, index=dff.index)).mean()) if "conv_7d" in dff.columns else 0.0
if np.isnan(mean_conv7):
	mean_conv7 = 0.0
est_daily_bookings = mean_conv7 / 7.0
bookings_total = int(round(est_daily_bookings * period_days))
# Previous period
prev_start = (pd.to_datetime(start_d) - pd.Timedelta(days=period_days)).date()
prev_end = (pd.to_datetime(start_d) - pd.Timedelta(days=1)).date()
mask_prev = (df["ad_set_id"].isin(sel_ad_sets)) & (pd.to_datetime(df["date"]).between(pd.to_datetime(prev_start), pd.to_datetime(prev_end)))
df_prev = df.loc[mask_prev].copy()
mean_conv7_prev = float(df_prev.get("conv_7d", pd.Series(0, index=df_prev.index)).mean()) if "conv_7d" in df_prev.columns else 0.0
if np.isnan(mean_conv7_prev):
	mean_conv7_prev = 0.0
bookings_prev = int(round((mean_conv7_prev/7.0) * period_days))

spend_total = float(dff["spend"].sum())
spend_prev = float(df_prev["spend"].sum()) or 0.0
# Real revenue and ROAS
rev_total = float(dff.get("revenue", pd.Series(0, index=dff.index)).sum())
rev_prev = float(df_prev.get("revenue", pd.Series(0, index=df_prev.index)).sum())
roas_overall = (rev_total / spend_total) if spend_total > 0 else 0.0
roas_prev = (rev_prev / spend_prev) if spend_prev > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
kpi_tile("Total spend", fmt_gbp(spend_total), None if spend_prev == 0 else f"{((spend_total-spend_prev)/max(spend_prev,1e-9))*100:.1f}%")
kpi_tile("Total revenue", fmt_gbp(rev_total), None)
kpi_tile("Total bookings", f"{bookings_total:,}", None if bookings_prev == 0 else f"{((bookings_total-bookings_prev)/max(bookings_prev,1))*100:.1f}%")
kpi_tile("Overall ROAS", f"{roas_overall:.2f}x", None if roas_prev == 0 else f"{((roas_overall-roas_prev)/max(roas_prev,1e-9))*100:.1f}%")

# After KPI tiles, add concise on-screen definitions when in presenter mode
# Always show concise definitions
st.caption("Definitions: ROAS = revenue ÷ spend. Pacing bands: under <0.90, on 0.90–1.10, over >1.10.")

# Build LM payload globally so the panel appears on all tabs
# ROAS by ad set (eligible only)
agg_lm = dff.groupby("ad_set_id").agg({"spend": "sum", "impressions": "sum", "ctr": "mean", "revenue": "sum"}).reset_index()
agg_lm = agg_lm[agg_lm["impressions"] >= 5000] if not agg_lm.empty else agg_lm
agg_lm["roas"] = np.where(agg_lm["spend"] > 0, agg_lm["revenue"]/agg_lm["spend"], 0.0)

# Pacing mix for current filters
pacing_lm = con.execute("SELECT date, ad_set_id, CASE WHEN pacing_ratio < 0.9 THEN 'under' WHEN pacing_ratio <= 1.1 THEN 'on' ELSE 'over' END AS pacing_status FROM gold.adset_pacing").df()
pacing_lm = pacing_lm[pacing_lm["ad_set_id"].isin(sel_ad_sets) & (pd.to_datetime(pacing_lm["date"]).between(pd.to_datetime(start_d), pd.to_datetime(end_d)))]
mix_lm = pacing_lm.groupby(["date","pacing_status"]).size().reset_index(name="count")

ok, status = health_check()
st.sidebar.caption(f"LM Studio: {'online' if ok else 'offline'} ({status})")
st.sidebar.caption(f"Endpoint: {os.getenv('LM_ENDPOINT', 'http://localhost:1234/v1/chat/completions')}")
st.sidebar.caption(f"Model: {os.getenv('LM_MODEL', 'local')}")
with st.sidebar:
	st.subheader("Next steps (LM Studio)")
	lm_payload = {
		"period": f"{start_d} to {end_d}",
		"kpis": {
			"spend_total": float(spend_total),
			"revenue_total": float(rev_total),
			"bookings_total": int(bookings_total),
			"roas_overall": float(roas_overall),
		},
		"pacing_mix": mix_lm.groupby("pacing_status")["count"].sum().to_dict() if not mix_lm.empty else {"under":0,"on":0,"over":0},
		"roas_by_adset": agg_lm[["ad_set_id","roas","spend"]].to_dict(orient="records") if not agg_lm.empty else [],
	}
	if ok:
		with st.spinner("Generating actions..."):
			lm_text, lm_status = call_lm_with_status(SYSTEM_PROMPT, lm_payload)
		st.caption(f"LM Studio status: {lm_status}")
		st.markdown(sanitize_lm_text(lm_text))
	else:
		st.caption("No suggestions (LM offline)")
	st.caption("Generated locally via LM Studio")

# Tabs organise the narrative
ov, dq, mdl, wf = st.tabs(["Overview", "Data Quality", "Models", "What‑if"]) 

with ov:
    st.caption("Trends and mix: how spend converts into outcomes across the selected period.")

    # Daily spend vs revenue (7‑day MA overlay)
    st.subheader("Daily spend vs revenue (7‑day MA)")
    daily = dff.groupby("date").agg({"spend": "sum", "revenue": "sum"}).reset_index()
    daily = daily.sort_values("date")
    daily["revenue_ma7"] = daily["revenue"].rolling(7, min_periods=1).mean()
    daily_ma = add_ma(daily, ["spend"], 7)
    fig = go.Figure()
    # Bars muted; lines blue/orange per spec
    fig.add_trace(go.Bar(x=daily_ma["date"], y=daily_ma["spend"], name="Spend (£)", marker_color="#B0C4DE", yaxis="y1"))
    fig.add_trace(go.Scatter(x=daily_ma["date"], y=daily_ma["spend_ma7"], name="Spend 7‑day MA", line=dict(color="#1f77b4"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=daily_ma["date"], y=daily_ma["revenue"], name="Revenue (£)", line=dict(color="#ff7f0e"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=daily_ma["date"], y=daily_ma["revenue_ma7"], name="Revenue 7‑day MA", line=dict(color="#ff7f0e", dash="dash"), yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="Spend (£)"),
        yaxis2=dict(title="Revenue (£)", overlaying="y", side="right")
    )
    st.plotly_chart(fig, width='stretch')
    st.caption("Context: bars/MA show spend; orange shows revenue (dual axis).")
    st.markdown("### Summary – Daily spend vs revenue")
    so_what("Spend trend", "Spend is stable with minor variance.", "Smooth pacing improves efficiency.", "Keep timing aligned; avoid end‑of‑week surges.")
    exp1 = {
        "chart": "daily_spend",
        "period": f"{start_d} to {end_d}",
        "spend_sum": float(daily_ma["spend"].sum()),
        "spend_mean": float(daily_ma["spend"].mean()),
        "spend_std": float(daily_ma["spend"].std(ddof=0)),
        "spend_ma7_last": float(daily_ma["spend_ma7"].iloc[-1]) if len(daily_ma) else 0.0,
        "n_days": int(len(daily_ma)),
    }
    _txt, _ = ("", "")
    if ok:
        _txt, _ = call_lm_with_status(SYSTEM_PROMPT, exp1)
        st.caption(sanitize_lm_text(_txt))
    else:
        st.caption("LM offline — actions available when LM Studio is running.")

    # ROAS by ad set (sorted) with target helper; include concise definition
    st.subheader("ROAS by ad set (sorted)")
    st.caption("ROAS = revenue / spend. Target line configurable in sidebar.")
    agg = dff.groupby(["ad_set_id"]).agg({"spend": "sum", "impressions": "sum", "revenue": "sum"}).reset_index()
    MIN_IMPR = 5000  # exclude low‑data ad sets for clarity
    agg = agg[agg["impressions"] >= MIN_IMPR]
    agg["roas"] = np.where(agg["spend"] > 0, agg["revenue"] / agg["spend"], 0.0)
    fig2 = px.bar(agg.sort_values("roas", ascending=False), x="roas", y="ad_set_id", orientation="h", labels={"roas": "ROAS (x)", "ad_set_id": "Ad set"})
    fig2 = add_roas_target(fig2, roas_target)
    fig2.update_traces(hovertemplate="Ad set: %{y}<br>ROAS: %{x:.2f}×<br>Spend: £%{customdata[0]:,.0f}<br>Impressions: %{customdata[1]:,}", customdata=agg[["spend","impressions"]].to_numpy())
    st.plotly_chart(fig2, width='stretch')
    st.caption("Context: sorted by ROAS; dotted line marks target from sidebar.")
    # Top/Bottom ROAS tables
    top5 = agg.sort_values("roas", ascending=False).head(5)[["ad_set_id","spend","revenue","roas"]]
    bot5 = agg.sort_values("roas", ascending=True).head(5)[["ad_set_id","spend","revenue","roas"]]
    cta, ctb = st.columns(2)
    cta.markdown("#### Top 5 ad sets by ROAS")
    cta.dataframe(top5, hide_index=True)
    ctb.markdown("#### Bottom 5 ad sets by ROAS")
    ctb.dataframe(bot5, hide_index=True)
    st.caption("Use this as the action list: trim bottom set, re‑invest in top set.")
    st.markdown("### Summary – ROAS by ad set")
    so_what("ROAS levels", "Some ad sets exceed target.", "Shift to winners increases efficiency.", "Move 5–10% from bottom quartile to top quartile.")
    exp2 = {
        "chart": "roas_by_adset",
        "adsets": int(len(agg)),
        "min_impressions": int(MIN_IMPR),
        "top3": agg.sort_values("roas", ascending=False).head(3)[["ad_set_id", "roas", "spend", "impressions"]].to_dict(orient="records"),
        "bottom3": agg.sort_values("roas", ascending=True).head(3)[["ad_set_id", "roas", "spend", "impressions"]].to_dict(orient="records"),
    }
    _txt2, _ = ("", "")
    if ok:
        _txt2, _ = call_lm_with_status(SYSTEM_PROMPT, exp2)
        st.caption(sanitize_lm_text(_txt2))
    else:
        st.caption("LM offline — actions available when LM Studio is running.")

    # Pacing status by day; clarify band thresholds
    st.subheader("Pacing status by day")
    st.caption("Bands: under < 0.90, on 0.90–1.10, over > 1.10")
    pacing = con.execute("SELECT date, ad_set_id, CASE WHEN pacing_ratio < 0.9 THEN 'under' WHEN pacing_ratio <= 1.1 THEN 'on' ELSE 'over' END AS pacing_status FROM gold.adset_pacing").df()
    pacing = pacing[pacing["ad_set_id"].isin(sel_ad_sets) & (pd.to_datetime(pacing["date"]).between(pd.to_datetime(start_d), pd.to_datetime(end_d)))]
    mix = pacing.groupby(["date", "pacing_status"]).size().reset_index(name="count")
    fig3 = px.bar(mix, x="date", y="count", color="pacing_status", barmode="stack", title="Under/On/Over pacing mix")
    st.plotly_chart(fig3, width='stretch')
    st.markdown("### Summary – Pacing status")
    so_what("Pacing mix", "Under‑pacing appears mid‑period.", "Under‑pacing risks missed demand.", "Increase caps and broaden targeting on affected days.")
    exp3 = {
        "chart": "pacing_mix",
        "days": int(mix["date"].nunique()) if not mix.empty else 0,
        "mix_totals": mix.groupby("pacing_status")["count"].sum().to_dict() if not mix.empty else {"under":0,"on":0,"over":0},
        "bands": {"under": "<0.90", "on": "0.90–1.10", "over": ">1.10"},
    }
    _txt3, _ = ("", "")
    if ok:
        _txt3, _ = call_lm_with_status(SYSTEM_PROMPT, exp3)
        st.caption(sanitize_lm_text(_txt3))
    else:
        st.caption("LM offline — actions available when LM Studio is running.")

with dq:
    st.subheader("Data Quality (silver.dq_summary)")
    try:
        dqdf = con.execute("SELECT * FROM silver.dq_summary").df()
        checks_total = len(dqdf)
        checks_ok = int((dqdf.get("null_fraction", 0) == 0).sum()) if "null_fraction" in dqdf.columns else checks_total
        cA, cB = st.columns(2)
        cA.metric("Checks passed", f"{checks_ok}/{checks_total}")
        st.caption("Validated columns and null rates after silver‑layer cleansing.")
        st.dataframe(dqdf)
    except Exception:
        st.info("DQ table not available.")

with mdl:
    st.subheader("Model (under‑pacing next day)")
    st.caption("Time‑based CV (k folds) with calibrated probabilities. See pipeline logs for AUC mean±std.")
    try:
        import joblib  # type: ignore
        from datetime import datetime as _dt
        cls_info = joblib.load(PROJECT_ROOT / "data/gold/model_xgb.pkl")
        m = cls_info.get("metrics", {})
        c1, c2 = st.columns([1,2])
        with c1:
            st.markdown("#### Model summary")
            st.metric("Train window", f"{m.get('train_start')} → {m.get('train_end')}")
            st.metric("Valid window", f"{m.get('valid_start')} → {m.get('valid_end')}")
            st.metric("Label positive rate", f"{(m.get('label_rate_train', 0)*100):.1f}%")
            st.metric("CV AUC (5‑fold)", f"{m.get('cv_auc_mean', float('nan')):.3f} ± {m.get('cv_auc_std', float('nan')):.3f}")
            # Compute holdout AUC on-the-fly for display
            try:
                import numpy as _np
                from sklearn.metrics import roc_auc_score as _auc
                # Load data and time split similar to training (80/20 by date)
                df_all = con.execute("SELECT * FROM gold.training_dataset").df()
                dts = pd.to_datetime(df_all["date"]).sort_values().unique()
                cutoff_idx = int(len(dts) * 0.8)
                cutoff = dts[cutoff_idx-1] if cutoff_idx>0 else dts[-1]
                train_mask = pd.to_datetime(df_all["date"]) <= cutoff
                valid_mask = pd.to_datetime(df_all["date"]) > cutoff
                feats = list(cls_info.get("feature_importances", {}).keys())
                if not feats:
                    feats = ["ctr","conv_7d","clicks","impressions","spend","aud_daily_budget","pacing_ratio"]
                Xv = df_all.loc[valid_mask, feats]
                yv = df_all.loc[valid_mask, "label_under_pacing_next"].astype(int).values if "label_under_pacing_next" in df_all.columns else None
                if yv is not None and len(_np.unique(yv))>=2 and len(Xv)>0:
                    proba = cls_info["model"].predict_proba(Xv)[:,1]
                    hold = _auc(yv, proba)
                    st.metric("Holdout AUC", f"{hold:.3f}")
            except Exception:
                pass
            st.caption("Probabilities calibrated | No look‑ahead")
        with c2:
            imps = cls_info.get("feature_importances", {})
            if imps:
                imp_df = pd.DataFrame({"feature": list(imps.keys()), "importance": list(imps.values())}).sort_values("importance", ascending=False)
                st.markdown("#### Feature importance (classifier)")
                st.bar_chart(imp_df.set_index("feature"))
    except Exception:
        st.info("Classifier model info unavailable.")

    # Conversion probability model block
    st.subheader("Model (conversion probability next 7 days)")
    st.caption("XGBRegressor predicting conv_next7d_rate with time‑aware split.")
    try:
        import joblib  # type: ignore
        reg_info = joblib.load(PROJECT_ROOT / "data/gold/model_conv_xgb.pkl")
        rimps = reg_info.get("feature_importances", {})
        if rimps:
            rimp_df = pd.DataFrame({"feature": list(rimps.keys()), "importance": list(rimps.values())}).sort_values("importance", ascending=False)
            st.markdown("#### Feature importance (regressor)")
            st.bar_chart(rimp_df.set_index("feature"))
            st.caption("Shows which inputs most influence predicted conversion rate.")
        st.json(reg_info.get("metrics", {}))
    except Exception:
        st.info("Conversion model info unavailable.")

with wf:
    st.subheader("What‑if: budget reallocation")
    period = st.selectbox("Aggregation period", ["All time", "Last 7 days", "Last 14 days"], index=1)
    dfw = dff.copy()
    if period == "Last 7 days":
        dfw = dfw[pd.to_datetime(dfw["date"]) >= (pd.to_datetime(end_d) - pd.Timedelta(days=6))]
    elif period == "Last 14 days":
        dfw = dfw[pd.to_datetime(dfw["date"]) >= (pd.to_datetime(end_d) - pd.Timedelta(days=13))]

    agg2 = dfw.groupby("ad_set_id").agg({"spend": "sum", "impressions": "sum", "revenue": "sum"}).reset_index()
    agg2 = agg2[agg2["impressions"] >= 5000]
    agg2["roas"] = np.where(agg2["spend"] > 0, agg2["revenue"] / agg2["spend"], 0.0)
    shift_pct = st.slider("Shift % from bottom→top quartile", 0.0, 0.3, 0.1, 0.01)
    elasticity = st.slider("Revenue elasticity", 0.5, 1.0, 0.8, 0.05)

    if len(agg2) >= 4 and agg2["spend"].sum() > 0:
        q = agg2["roas"].quantile([0.25, 0.75]).values
        bottom = agg2[agg2["roas"] <= q[0]].copy()
        top = agg2[agg2["roas"] >= q[1]].copy()
        mid = agg2[(agg2["roas"] > q[0]) & (agg2["roas"] < q[1])].copy()
        total_spend = agg2["spend"].sum()
        shift_amount = total_spend * shift_pct
        if not bottom.empty and not top.empty:
            bottom["spend_new"] = bottom["spend"] * (1 - shift_pct)
            top_weight = top["spend"].sum()
            top["spend_new"] = top["spend"] + shift_amount * (top["spend"] / max(top_weight, 1e-9))
            mid["spend_new"] = mid["spend"]
            proj = pd.concat([bottom, mid, top], ignore_index=True)
            # Project revenue with constant ROAS scaled by elasticity
            proj["revenue_proj"] = proj["spend_new"] * proj["roas"] * elasticity
            uplift = float(proj["revenue_proj"].sum() - agg2["revenue"].sum())
            uplift_low, uplift_high = uplift * 0.8, uplift * 1.2
            st.caption(f"Projected uplift range: £{uplift_low:,.0f}–£{uplift_high:,.0f} (base £{uplift:,.0f})")
            st.dataframe(proj[["ad_set_id", "spend", "spend_new", "roas", "revenue", "revenue_proj"]].sort_values("roas", ascending=False))
            export_csv(proj, "reallocation_plan")
            export_pdf("<h2>What‑if summary</h2>", "what_if_summary")
        else:
            st.info("Not enough dispersion in ROAS to form quartiles. Try a wider period.")
    else:
        st.info("Need at least 4 ad sets with non‑zero spend for the reallocation demo.")

# Footer note in presenter mode
if ok:
	st.markdown("— Data source: local CSVs; revenue from booking events; models trained with time‑based CV.")
st.caption("Generated locally with DuckDB + LM Studio | v1.0  © Matthew Paver 2025")

# Chart-specific Explain buttons
with ov:
    if st.button("Explain this chart (Spend vs Revenue)"):
        if ok:
            payload = {
                "n_days": int(len(daily_ma)),
                "spend_sum": float(daily_ma["spend"].sum()),
                "spend_mean": float(daily_ma["spend"].mean()),
                "revenue_sum": float(daily_ma["revenue"].sum()),
            }
            txt, _ = call_lm_with_status(SYSTEM_PROMPT, payload)
            st.caption(sanitize_lm_text(txt))
        else:
            st.caption("LM offline — start LM Studio to generate a summary.")

    if st.button("Explain this chart (ROAS by ad set)"):
        if ok:
            payload = {
                "adsets": int(len(agg)),
                "top_roas": float(agg["roas"].max()) if len(agg) else 0.0,
                "bottom_roas": float(agg["roas"].min()) if len(agg) else 0.0,
                "median_roas": float(agg["roas"].median()) if len(agg) else 0.0,
                "target": float(roas_target),
            }
            txt, _ = call_lm_with_status(SYSTEM_PROMPT, payload)
            st.caption(sanitize_lm_text(txt))
        else:
            st.caption("LM offline — start LM Studio to generate a summary.")

    if st.button("Explain this chart (Pacing mix)"):
        if ok:
            payload = {
                "days": int(mix["date"].nunique()) if not mix.empty else 0,
                "under": int(mix[mix["pacing_status"]=="under"]["count"].sum()) if not mix.empty else 0,
                "on": int(mix[mix["pacing_status"]=="on"]["count"].sum()) if not mix.empty else 0,
                "over": int(mix[mix["pacing_status"]=="over"]["count"].sum()) if not mix.empty else 0,
            }
            txt, _ = call_lm_with_status(SYSTEM_PROMPT, payload)
            st.caption(sanitize_lm_text(txt))
        else:
            st.caption("LM offline — start LM Studio to generate a summary.")

# Cleanly close the shared connection
con.close()
