# Marketing ML – Local Lakehouse Dashboard

End‑to‑end local lakehouse: DuckDB (bronze→silver→gold), pandas transforms, XGBoost models, and a Streamlit dashboard.

## Getting started

- Python 3.10–3.13 on macOS or Linux
- From the repository root, create a virtual environment and install requirements:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r lakehouse/requirements.txt
```

Run the full pipeline and launch the dashboard:

```bash
python -m lakehouse.run_all
PYTHONPATH=$(pwd) streamlit run lakehouse/dashboard/app.py
```

Data note: raw CSVs are read from `marketing-ml/data/raw/` (kept in-repo for convenience):
- `audience_segments.csv`
- `budget_pacing.csv`
- `conversion_events.csv`
- `meta_campaign_performance.csv`

Large artefacts (models, DuckDB files, datasets) are ignored by Git; Git LFS tracks `*.pkl` and `*.duckdb` if you choose to include them.

## Repo hygiene

- .gitignore excludes caches, OS files, raw/intermediate data, and model artefacts.
- Git LFS is configured for *.pkl and *.duckdb via .gitattributes.
- Include only tiny, anonymised sample data if needed.

## Publish to GitHub

After you create a GitHub repository, push with:

```bash
git remote add origin https://github.com/YOUR_ORG/marketing-ml-lakehouse.git
git branch -M main
git push -u origin main
```
