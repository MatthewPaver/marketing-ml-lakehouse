# Marketing ML Lakehouse

<div align="center">

### DuckDB lakehouse, ML training pipeline, and Streamlit dashboard

![Python](https://img.shields.io/badge/Python-3.10--3.13-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![DuckDB](https://img.shields.io/badge/DuckDB-Lakehouse-FFF700?style=flat-square&logo=duckdb&logoColor=000000)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6B00?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## Status

`Runnable application`

This repository contains a local-first marketing analytics stack built around:

- a DuckDB bronze → silver → gold pipeline
- ML training for performance and pacing
- a Streamlit dashboard for exploration and presentation

## Canonical Entry Point

The active implementation lives under [`lakehouse/`](lakehouse).

The older [`marketing-ml/`](marketing-ml) subtree is retained as a legacy reference, but the root README, root `requirements.txt`, and root Makefile now point to the current `lakehouse/` flow.

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

The current pipeline reads raw inputs from `marketing-ml/data/raw/`:

- `audience_segments.csv`
- `budget_pacing.csv`
- `conversion_events.csv`
- `meta_campaign_performance.csv`

## Repository Layout

```text
lakehouse/      active pipeline, models, and dashboard
marketing-ml/   older reference implementation
requirements.txt
Makefile
```

## Notes

- Root `requirements.txt` is canonical.
- `lakehouse/requirements.txt` and `marketing-ml/requirements.txt` are compatibility shims.
- If you land inside the subdirectories directly, prefer coming back to the repository root for setup.

## License

MIT. See [`LICENSE`](LICENSE).
