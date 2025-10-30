# Presentation Script (3–5 slides, ~20 minutes)

## Slide 1 – Outcome and KPIs (non‑technical)
- What we built: a local, cloud‑free pipeline that mirrors Databricks (raw → bronze → silver → gold → model → dashboard).
- What it delivers: live KPIs (Spend, Revenue, ROAS, Bookings) and pacing risk insights.
- Headline: ROAS ≈ 9× across the sample period; clear path to budget efficiency.

## Slide 2 – Architecture (technical)
- Storage/compute: DuckDB schemas (`bronze`, `silver`, `gold`) with pandas transforms.
- Data quality: negative spend→0, type casting, enum normalisation, duplicates removed.
- Modelling: XGBoost (regression for bookings, classification for under‑pacing) with time‑based evaluation.
- Serving: Streamlit dashboard; Docker for reproducibility; Makefile for local speed.

## Slide 3 – Features and Attribution (both)
- Features: performance (impressions/clicks/spend/ctr/cpm/frequency), budget (planned/actual/utilisation), conversions (bookings/revenue/soft conversions), economics (ROAS, CPA).
- Attribution: demo aggregates; production approach via 7‑day lookback joins/windows.
- Data Quality report to surface nulls and anomalies.

## Slide 4 – Results and Explainability (both)
- Bookings model: MAE ~0.26, R² ~0.72; key drivers from feature importances.
- Under‑pacing classifier: AUC baseline vs noisy labels; ROC curves to show robustness.
- Baselines: 7‑day moving average; demonstrates value vs naïve forecasting.

## Slide 5 – Actions and Next Steps (both)
- What‑if: reallocate budget from low‑ROAS to high‑ROAS ad sets; show projected uplift.
- Next: weekly/lag features, backtesting, drift monitoring, CI/CD, real‑time API.
- Cost/pragmatism: local DuckDB is fast and inexpensive; expandable to object storage and schedulers.
