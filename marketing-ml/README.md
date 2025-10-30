# marketing-ml-lakehouse (local lakehouse demo)

Local, offline lakehouse mirroring Databricks layers (bronze→silver→gold) with DuckDB, Pandas ETL, XGBoost, and a Streamlit app. Optional LM Studio provides “Next steps” actions.

## Project layout
```
marketing-ml/
  data/{raw,bronze,silver,gold}/
  scripts/
  utils/
  app/
  lakehouse.duckdb (created by scripts)
```

Place CSVs in `data/raw/`:
- meta_campaign_performance.csv
- budget_pacing.csv
- conversion_events.csv
- audience_segments.csv

## Run (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
streamlit run app/dashboard.py
```

## Docker
```bash
cd marketing-ml
# build image in this folder
docker build -t marketing-ml -f Dockerfile .
# run pipeline inside container (writes lakehouse.duckdb in /app)
docker run --rm -v "$(pwd)":/app marketing-ml bash -lc "python scripts/run_all.py"
# run app (http://localhost:8501)
docker run --rm -p 8501:8501 -v "$(pwd)":/app marketing-ml
```

## LM Studio (optional)
- Start server at `http://localhost:1234` (or your LAN host) and set `LM_ENDPOINT`/`LM_MODEL`.
- Sidebar shows status and three concise action bullets (grounded in current filters). Caption: “AI summary generated locally”.

## Architecture (Mermaid)
```mermaid
flowchart LR
  RAW[raw CSVs] --> BZ[(bronze tables)] --> SL[(silver tables\nDQ+uniqueness)] --> GL[(gold features\nlabels+scores)]
  GL --> ML[(XGBoost models\ncalibrated)] --> SCORED[(gold.scored_*)]
  GL --> APP[Streamlit app\nPresenter mode]
```

## Quality gates
- Deterministic outputs; no forward leakage
- Time-aware validation (train ≤ T, validate > T)
- Charts with targets/bands and So‑what context
- “Next steps” grounded in on‑screen numbers
- Fully runnable offline (local CSVs, DuckDB, LM Studio)

## Notes
- Revenue derived from booking_completed events → `silver.revenue_by_day` → `gold.training_dataset` (`revenue`, `roas`).
- Probabilities calibrated (isotonic). Feature importances included for both models.
- Footer in app: “Generated locally with DuckDB + LM Studio | v1.0”.

## 5‑minute presenter script
1. Filters (00:30) – Pick ad sets and period; Presenter mode on.
2. KPIs (00:45) – Spend, revenue, bookings, ROAS with deltas.
3. Daily trend (00:45) – Spend bars + MA7; revenue line; “Explain this chart”.
4. ROAS by ad set (01:15) – Sorted bars; target slider; top/bottom tables; So‑what.
5. Pacing mix (00:45) – Under/on/over bands; So‑what.
6. Models (00:45) – AUC±std, label rate, windows; feature importances; calibrated note.
7. What‑if (00:30) – Shift % + elasticity; uplift ±20%; CSV/PDF export.

Tips
- Keep the sidebar LM panel open; it updates with the current filters.
- Use the definitions caption to remind non‑technical viewers of ROAS and pacing bands.
- If data looks sparse, hit “Reset filters”.
