# Local Lakehouse (DuckDB + pandas + Sklearn + XGBoost + Streamlit)

This project mirrors a Databricks-style pipeline locally: raw → bronze → silver → gold → model → dashboard.

## Tech
- DuckDB (schemas: `bronze`, `silver`, `gold`)
- pandas for ingestion/transforms (UK date parsing)
- scikit-learn + XGBoost for ML
- Streamlit + Plotly for dashboard

## Quick start

1) Create and activate a virtual environment (Python 3.10–3.13):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r lakehouse/requirements.txt
```

2) Run the full pipeline locally:
```bash
python -m lakehouse.run_all
```
This will:
- Create `lakehouse/lakehouse.duckdb` and schemas (`bronze`, `silver`, `gold`)
- Ingest CSVs from `marketing-ml/data/raw/` into `bronze`
- Clean/standardise into `silver`
- Aggregate features into `gold`
- Compute a Data Quality report into `gold.data_quality_report`
- Train an XGBoost regression model and store artefacts under `lakehouse/models/`
- Train an under‑pacing classifier and store artefacts

3) Start the dashboard:
```bash
streamlit run lakehouse/dashboard/app.py
```

## Makefile (convenience)
```bash
make install
make run
make train-clf
make dashboard
```

## Docker usage

Build the image (from repo root):
```bash
docker build -t local-lakehouse -f lakehouse/Dockerfile .
```
Run the pipeline (bind-mount to persist DB and models to your folder):
```bash
docker run --rm -v "$(pwd)":/app local-lakehouse \
  bash -lc "python -m lakehouse.run_all"
```
Run the dashboard on http://localhost:8501:
```bash
docker run --rm -p 8501:8501 -v "$(pwd)":/app local-lakehouse \
  bash -lc "streamlit run lakehouse/dashboard/app.py --server.port 8501 --server.address 0.0.0.0"
```
Multi-arch push (Intel + Apple Silicon):
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t YOURORG/local-lakehouse:latest -f lakehouse/Dockerfile --push .
```
Troubleshooting Docker on macOS:
- Ensure Docker Desktop is running (search “Docker”, wait for “Docker is running”), then:
```bash
docker info
```
- If context is wrong:
```bash
docker context ls
docker context use default   # or 'desktop' / 'colima' if you use Colima
```
- Lightweight alternative: Colima
```bash
brew install colima docker
colima start
docker context use colima
```

## AI summaries (optional)
The dashboard can summarise each visualisation in 1–2 UK‑English sentences using a local LLM via an OpenAI‑compatible API (e.g., LM Studio).

- Launch LM Studio server on your host and set env vars before starting the app:
```bash
export LLM_ENDPOINT=http://YOUR_HOST:1234/v1/chat/completions
export LLM_MODEL=meta-llama-3.1-8b-instruct   # or llama-3.2-3b-instruct for lighter CPU runs
# export LLM_API_KEY=anything                  # optional; LM Studio typically doesn't need it
streamlit run lakehouse/dashboard/app.py
```
- If not configured, the app falls back to a concise rule‑based summary.

## Presenting to technical and non‑technical audiences
- Non‑technical focus:
  - What changed: spend, revenue, ROAS, bookings; top/bottom ad sets; clear trends.
  - Why it matters: budget efficiency, pacing risk, revenue upside from reallocation.
  - Simple visuals: KPIs, time series, 7‑day baseline, what‑if sliders with plain language.
- Technical focus:
  - How it works: lakehouse layers, data quality rules, attribution approach.
  - Models: features, time‑based evaluation, artefacts, importances, AUC/MAE/R².
  - Ops: idempotent writes, Docker build/run, reproducibility, next steps (drift, CI/CD).

## Architecture (for presentation)

```
Raw CSVs (repo root)
   │
   ▼
Bronze (DuckDB): raw copies
   │  (schema: bronze.*)
   ▼
Silver (DuckDB): type fixes, DQ rules, dedupe
   │  (schema: silver.*)
   ▼
Gold (DuckDB): daily features, training set, DQ report
   │  (schema: gold.*)
   ├─► Model: XGBoost (sklearn pipeline)
   │       └─ artefacts: lakehouse/models/*
   ▼
Dashboard (Streamlit): gold metrics + model + DQ
```

- Batch cadence (daily). Real‑time extension: Kafka → micro-batches to DuckDB; orchestration via Airflow/Prefect.

## Feature engineering

- Silver rules:
  - Negative `spend` → 0.0; cast numeric types
  - Recompute `ctr` and `cpm` where safe; drop duplicates
  - Normalise `pacing_status` to {under_pacing, on_pace, over_pacing, unknown}
  - Conversions: null `value` → 0.0; lowercase types/windows
- Gold features per (date, ad_set_id): performance, budget, conversions, economics; plus DQ report.
- Attribution window: aggregated pragmatically here; production approach would implement lookback windows via SQL window functions.

## ML models

- Regression: predict `target_bookings` with XGBoost; sklearn Pipeline (impute+scale); time-based split.
- Classification: under‑pacing risk; XGBoost; accuracy/F1/AUC metrics.
- Artefacts: `lakehouse/models/*.pkl|*.json`.

## Data quality & validation

- `gold.data_quality_report`: null fractions and basic stats.
- Silver enforces types and enums; safe metric recomputation.

## Alignment with the brief (checklist)
- End‑to‑end pipeline: ingestion → feature engineering → training → serving (dashboard).
- Technology choices: DuckDB, pandas, sklearn/XGBoost, Streamlit; Docker for reproducibility.
- Batch vs real‑time: batch implemented; real‑time extension described.
- Scalability: clear path to object storage + scheduler; multi‑arch container build.
- Feature engineering targets: pacing risk (classification), conversion/bookings (regression), attribution notes.
- Training pipeline: time‑based split, metrics logged, artefacts persisted.
- Deployment options: batch predictions + dashboard; real‑time API discussed.
- Monitoring: DQ table, metrics JSON, guidance for drift checks.

## Over‑deliver ideas
- Add lag/rolling features and weekly aggregates to a `gold_plus` table.
- Backtesting: rolling/expanding window evaluation with per‑fold metrics.
- Baselines: 7‑day moving average vs XGBoost for delta context.
- Explainability: feature importances/SHAP saved and exposed in dashboard.
- Drift snapshot: compare latest feature stats vs training; alert thresholds.
- Tests: lightweight `pytest` for SQL/DQ expectations.
- CI: GitHub Actions to run `make run` and publish image to GHCR.
