# Marketing ML Lakehouse (Monorepo)

This repository contains two runnable, local projects:

- marketing-ml/: a self-contained demo with data folders, scripts, and a Streamlit dashboard.
- lakehouse/: a modular package with pipeline, models, and a dashboard app.

## Getting started

- Python 3.10–3.13 on macOS or Linux
- Create a virtual environment and install requirements for the project you want to run:

```bash
# Option A: run the marketing-ml demo
cd marketing-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
streamlit run app/dashboard.py

# Option B: run the lakehouse package
cd ..
python -m venv .venv && source .venv/bin/activate
pip install -r lakehouse/requirements.txt
python -m lakehouse.run_all
streamlit run lakehouse/dashboard/app.py
```

Data note: raw CSVs live under marketing-ml/data/raw/ (see that README for file names). Large artefacts (models, databases, and datasets) are intentionally ignored by Git; Git LFS tracks *.pkl and *.duckdb if you decide to include them.

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
