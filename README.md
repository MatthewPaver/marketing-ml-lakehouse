# 🏗️ Marketing ML – Local Lakehouse Dashboard

<div align="center">

### End-to-End Marketing Analytics Lakehouse | 🦆 DuckDB | 🤖 ML-Driven Insights | 📊 Streamlit Dashboard

**Automated data ingestion, ML-driven pacing and conversion modelling, and LM Studio-powered insight generation**

[![Python](https://img.shields.io/badge/Python-3.10--3.13-3670A0?style=flat-square&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-Lakehouse-FFF700?style=flat-square&logo=duckdb&logoColor=000000)](https://duckdb.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6B00?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📋 Overview

An end-to-end local marketing analytics lakehouse built with Python, DuckDB, and Streamlit. This project demonstrates a complete data engineering and machine learning pipeline, implementing a bronze→silver→gold data transformation architecture with automated data ingestion, ML-driven pacing and conversion modelling, and LM Studio-powered insight generation.

### ✨ Key Features

- **🦆 DuckDB Lakehouse Architecture** — Bronze→silver→gold data transformation pipeline
- **🤖 Machine Learning Models** — XGBoost models for conversion prediction and campaign pacing optimisation
- **📊 Interactive Dashboard** — Streamlit dashboard for real-time analytics and visualisation
- **🔄 Automated Data Ingestion** — Pipeline for processing marketing data from multiple sources
- **🧠 LLM-Powered Insights** — LM Studio integration for automated insight generation
- **📈 Campaign Analytics** — Performance tracking, pacing analysis, and conversion modelling

---

## 🏗️ Architecture

### Data Pipeline

```
Raw Data (Bronze) → Transformed Data (Silver) → Analytics-Ready (Gold) → ML Models → Dashboard
```

- **Bronze Layer:** Raw CSV data from marketing sources
- **Silver Layer:** Cleaned and standardised data using pandas transformations
- **Gold Layer:** Aggregated and feature-engineered data ready for ML and analytics
- **ML Layer:** XGBoost models for conversion prediction and pacing optimisation
- **Presentation Layer:** Streamlit dashboard for interactive exploration

---

## 🚀 Getting Started

### Prerequisites

- **Python:** 3.10–3.13
- **Operating System:** macOS or Linux
- **Memory:** Recommended 4GB+ RAM for processing larger datasets

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/MatthewPaver/marketing-ml-lakehouse.git
cd marketing-ml-lakehouse
```

2. **Create a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r lakehouse/requirements.txt
```

### Running the Pipeline

1. **Run the full data pipeline:**

```bash
python -m lakehouse.run_all
```

This will:
- Ingest raw data from `marketing-ml/data/raw/`
- Transform data through bronze→silver→gold layers
- Train XGBoost models for conversion and pacing
- Generate insights using LM Studio (if configured)

2. **Launch the Streamlit dashboard:**

```bash
PYTHONPATH=$(pwd) streamlit run lakehouse/dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

---

## 📊 Data Sources

Raw CSV files are read from `marketing-ml/data/raw/`:

- `audience_segments.csv` — Audience segmentation data
- `budget_pacing.csv` — Budget pacing and spend tracking
- `conversion_events.csv` — Conversion event tracking
- `meta_campaign_performance.csv` — Campaign performance metrics

---

## 🛠️ Tech Stack

### Core Technologies

- **Python** — Primary programming language
- **DuckDB** — In-process analytical database for lakehouse architecture
- **pandas** — Data manipulation and transformation
- **XGBoost** — Gradient boosting for ML models
- **Streamlit** — Interactive dashboard framework

### Additional Tools

- **LM Studio** — Local LLM integration for insight generation
- **Git LFS** — Large file storage for models and datasets

---

## 📁 Repository Structure

```
marketing-ml-lakehouse/
├── lakehouse/              # Main lakehouse pipeline code
│   ├── dashboard/          # Streamlit dashboard application
│   ├── requirements.txt    # Python dependencies
│   └── run_all.py          # Main pipeline execution script
├── marketing-ml/           # Marketing data and ML components
│   └── data/
│       └── raw/            # Raw CSV data files
├── LICENSE                 # MIT License
└── README.md              # This file
```

---

## 🔧 Configuration

### Git LFS

Large artefacts (models, DuckDB files, datasets) are tracked using Git LFS:

- `*.pkl` — Serialised model files
- `*.duckdb` — DuckDB database files

### Data Management

- Raw and intermediate data files are excluded from Git via `.gitignore`
- Only anonymised sample data should be included in the repository
- Model artefacts and large datasets are managed through Git LFS

---

## 📈 Features & Capabilities

### Data Processing

- **Automated Ingestion** — Process multiple CSV sources into unified format
- **Data Transformation** — Bronze→silver→gold pipeline with pandas
- **Feature Engineering** — Temporal features and aggregations for ML

### Machine Learning

- **Conversion Prediction** — XGBoost models to predict conversion likelihood
- **Campaign Pacing** — ML-driven pacing optimisation models
- **Forecast Accuracy** — Improved forecast reliability through feature engineering

### Analytics & Visualisation

- **Interactive Dashboard** — Real-time analytics in Streamlit
- **Campaign Performance** — Visualise campaign metrics and trends
- **ML-Driven Recommendations** — Actionable insights from model outputs
- **Automated Insights** — LLM-generated summaries and recommendations

---

## 🎯 Use Cases

- **Marketing Analytics** — Track and analyse campaign performance
- **Conversion Optimisation** — Predict and improve conversion rates
- **Budget Management** — Optimise campaign pacing and spend allocation
- **Performance Forecasting** — ML-driven forecasting for campaign planning
- **Data-Driven Insights** — Automated insight generation from campaign data

---

## 📝 Notes

- **Data Privacy:** Only anonymised sample data is included in the repository
- **Model Artefacts:** Large model files are managed through Git LFS
- **Local Processing:** Designed for local development and experimentation
- **Extensibility:** Architecture supports additional data sources and ML models

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🔗 Related Projects

- [Marketing ML Lakehouse](https://github.com/MatthewPaver/marketing-ml-lakehouse) — This repository
- [Profile](https://github.com/MatthewPaver) — View all my projects

---

<div align="center">

**Built with ❤️ using Python, DuckDB, and Streamlit**

[← Back to Profile](https://github.com/MatthewPaver)

</div>
