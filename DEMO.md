# DEMO — Marketing ML Lakehouse

A sample-data path with no ad account or API keys. Installation time depends on your computer and connection.

## What this proves

Raw marketing CSVs can rebuild a trusted local analytics product from scratch:

1. Load fixtures into DuckDB (bronze → silver → gold).
2. Train an example XGBoost model from constructed features.
3. Run deterministic quality / leakage checks.
4. Open a Streamlit dashboard on the same artefacts.

It does **not** prove live campaign performance, ROAS from a connected ad platform, or that the model would generalise to your account.

## Run it

Use Python 3.11 for the tested setup. Before installing the ML packages, you can check the included input files using only Python's standard library:

```bash
python3.11 -m lakehouse.contracts
```

This checks the four configured CSV sources without training a model or downloading data. It writes `lakehouse/artifacts/contract_report.json`. A `warn` result needs inspection; it is not a clean bill of health. An empty file, duplicate column name, missing required column or a row with the wrong number of fields fails before ingestion, with the source name and reason recorded in that report.

### Full local workflow

```bash
git clone https://github.com/MatthewPaver/marketing-ml-lakehouse.git
cd marketing-ml-lakehouse
make install
make run          # rebuild lakehouse + train
make test         # rebuild from fixtures, then pytest
make dashboard    # stays running at http://localhost:8501; stop with Ctrl+C
```

### Windows / no Make

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m lakehouse.run_all
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m streamlit run lakehouse/dashboard/app.py
```

### Where your result goes

- Data checks: `lakehouse/artifacts/contract_report.json` (source names, hashes, row counts and warnings).
- Rebuilt tables: `lakehouse/lakehouse.duckdb`, in the `bronze`, `silver` and `gold` schemas.
- Models and evaluation files: `lakehouse/models/` and `lakehouse/artifacts/`.
- Local dashboard: `http://localhost:8501`; stop it with Ctrl+C.

Run from the repository root. If Python 3.11 is missing, install that version before the full dependency setup. If a source contract fails, fix the named input first; do not train on the remaining files. Re-running the pipeline replaces generated local tables/models and the local `docs/generated/` evidence snapshot, so keep your own source files and any results you need outside those generated folders. It does not publish them to GitHub Pages.

Browser evidence console (no install): https://matthewpaver.github.io/marketing-ml-lakehouse/

The browser console is a fixed demonstration snapshot, not a live view of the files you rebuild. Use the local Streamlit dashboard to inspect your current pipeline output. Do not expect local input changes to appear on the public website.

**First useful result:** open the local data-quality view, inspect the duplicate campaign/day keys, then compare the next-day model with the simple baseline. A data-quality failure or a model that does not improve on the baseline is a useful finding, not a reason to hide the result.

## What to look at

| Surface | Why it matters |
| --- | --- |
| Gold tables in DuckDB | The medallion path is inspectable, not a notebook side-effect |
| Dashboard pacing / ROAS panels | Metrics come from committed CSVs — labelled as demo |
| `make test` | CI rebuilds the same artefacts a recruiter can reproduce |

## Boundaries

- Demo data under `data/raw/` (August 2024 sample travel marketing).
- Browser console reviews aggregates; full DuckDB rebuild and training run locally.
- Canonical path is the root Makefile + `lakehouse/` package (legacy `marketing-ml/` tree removed).

## For the portfolio conversation

Useful talking point: “analytics demos often stop at a chart; this one packages ingestion, quality gates, training and a dashboard as one rebuildable loop.”
