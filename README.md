# Marketing ML Lakehouse

<div align="center">

### DuckDB lakehouse, ML training pipeline, and Streamlit dashboard

![Python](https://img.shields.io/badge/Python-3.10--3.13-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![DuckDB](https://img.shields.io/badge/DuckDB-Lakehouse-FFF700?style=flat-square&logo=duckdb&logoColor=000000)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6B00?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
[![Validate](https://github.com/MatthewPaver/marketing-ml-lakehouse/actions/workflows/validate.yml/badge.svg)](https://github.com/MatthewPaver/marketing-ml-lakehouse/actions/workflows/validate.yml)

</div>

---

![Marketing ML Lakehouse dashboard](docs/assets/dashboard.png)

**[Open the no-setup evidence console](https://matthewpaver.github.io/marketing-ml-lakehouse/)** — a complete browser review of the committed campaign, pacing, quality and lineage evidence. The Python engine remains the canonical way to rebuild the lakehouse and models.

## Portfolio Quick Read

| Section | Where to look |
|:---|:---|
| What it solves | Turns raw marketing CSVs into a repeatable local analytics and ML workflow |
| Quick start | [`make install`](#canonical-setup), [`make run`](#run-the-pipeline), [`make dashboard`](#run-the-dashboard) |
| Screenshot | [Portfolio Store](https://matthewpaver.github.io/MatthewPaver/store/) |
| Architecture | [System Shape](#system-shape) |
| Tests | `make test` rebuilds the local lakehouse and runs validation |
| Tech stack | `Python` `DuckDB` `pandas` `XGBoost` `Streamlit` |

## Status

`Runnable application`

This repository contains a local-first marketing analytics stack built around:

- a DuckDB bronze → silver → gold pipeline
- ML training for performance and pacing
- a Streamlit dashboard for exploration and presentation

## Reviewer Pack

| Area | Details |
|:---|:---|
| What it solves | Raw campaign CSVs become an inspectable bronze, silver, and gold lakehouse with model training and dashboard review. |
| Screenshot | [Portfolio Store preview](https://matthewpaver.github.io/MatthewPaver/store/preview.html?app=lakehouse) |
| Run locally | `make install && make run` |
| Dashboard | `make dashboard` then open `http://localhost:8501` |
| Tests | `make test` rebuilds DuckDB/models from demo data, then runs pytest |
| Demo data | Included under `data/raw/` |
| Architecture | Raw CSVs -> DuckDB bronze/silver/gold -> XGBoost models -> data-quality checks -> Streamlit dashboard |
| Limitations | Local-first demonstration with sample marketing data; not connected to a live ad-platform API. |

## Practical Test

Can raw campaign CSVs rebuild a trusted analytics product from scratch?

The useful check is the full path:

1. Load raw files into DuckDB.
2. Build bronze, silver, and gold tables.
3. Train the model from generated features.
4. Run quality and risk checks.
5. Open the Streamlit dashboard and inspect the result.

That is the point of the repo: prove the local analytics loop is repeatable, not just show a finished chart.

## Reviewer Notes

- **Reproducible path:** root `Makefile` and `requirements.txt` are the canonical entry points.
- **Data engineering signal:** bronze, silver, and gold layers make the pipeline auditable rather than a one-off notebook.
- **ML signal:** model training sits after feature construction, with dashboard consumption separated from pipeline execution.
- **Agent signal:** `lakehouse/agents.py` adds deterministic data-quality, feature-drift, campaign-insight, and model-risk reviewers around the pipeline.
- **Verification path:** run `make test` after setup; it rebuilds the local DuckDB/models from demo data before running pytest. Use `make run` and `make dashboard` for the full local flow.

## System Shape

![Marketing ML Lakehouse architecture](docs/assets/architecture.svg)

```mermaid
flowchart LR
    A["Raw marketing CSVs"] --> B["Bronze tables"]
    B --> C["Silver cleaned data"]
    C --> D["Gold features"]
    D --> E["XGBoost models"]
    D --> F["Data quality checks"]
    E --> G["Streamlit dashboard"]
    F --> G
```

The project is designed to show a full local analytics workflow: ingestion, transformation, feature building, model training, quality checks, and dashboard consumption.

## Canonical Entry Point

The implementation lives under [`lakehouse/`](lakehouse), with demo inputs in
[`data/raw/`](data/raw) and active tests in [`tests/`](tests).

## Canonical Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or use the Makefile:

```bash
make install
```

## Run The Pipeline

```bash
python -m lakehouse.run_all
```

Or:

```bash
make run
```

## Run The Dashboard

```bash
streamlit run lakehouse/dashboard/app.py
```

Or:

```bash
make dashboard
```

The dashboard is served at `http://localhost:8501`.

## Data Expectations

The current pipeline reads raw inputs from `data/raw/`:

- `audience_segments.csv`
- `budget_pacing.csv`
- `conversion_events.csv`
- `meta_campaign_performance.csv`

## Repository Layout

```text
lakehouse/      active pipeline, models, and dashboard
data/raw/       small reproducible demo inputs
tests/          active pipeline and reviewer tests
requirements.txt
Makefile
```

## Notes

- Root `requirements.txt` is canonical.
- `lakehouse/requirements.txt` is a compatibility shim for the root dependency list.
- If you land inside the subdirectories directly, prefer coming back to the repository root for setup.

## Docker Compose

Run the complete pipeline in a disposable container:

```bash
make compose-pipeline
```

Or rebuild the pipeline and start the dashboard on
`http://localhost:8501`:

```bash
make compose-dashboard
```

## License

MIT. See [`LICENSE`](LICENSE).
