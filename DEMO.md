# DEMO — Marketing ML Lakehouse

One path a stranger can finish in about ten minutes. No ad account. No API keys.

## What this proves

Raw marketing CSVs can rebuild a trusted local analytics product from scratch:

1. Load fixtures into DuckDB (bronze → silver → gold).
2. Train an example XGBoost model from constructed features.
3. Run deterministic quality / leakage checks.
4. Open a Streamlit dashboard on the same artefacts.

It does **not** prove live campaign performance, ROAS from a connected ad platform, or that the model would generalise to your account.

## Run it

```bash
git clone https://github.com/MatthewPaver/marketing-ml-lakehouse.git
cd marketing-ml-lakehouse
make install
make run          # rebuild lakehouse + train
make dashboard    # Streamlit on http://localhost:8501
make test         # rebuild from fixtures, then pytest
```

Browser evidence console (no install): https://matthewpaver.github.io/marketing-ml-lakehouse/

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
