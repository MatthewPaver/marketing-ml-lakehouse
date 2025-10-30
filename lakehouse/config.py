from __future__ import annotations

from pathlib import Path

# As a CS undergraduate style, we keep configuration centralised for maintainability.
# We resolve the repository root based on this file's location to keep things portable.

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
LAKEHOUSE_DIR: Path = PROJECT_ROOT / "lakehouse"
DB_FILE: Path = LAKEHOUSE_DIR / "lakehouse.duckdb"
MODELS_DIR: Path = LAKEHOUSE_DIR / "models"

# Raw CSV file inputs (kept at repository root as per brief)
RAW_AUDIENCE_SEGMENTS: Path = PROJECT_ROOT / "audience_segments.csv"
RAW_BUDGET_PACING: Path = PROJECT_ROOT / "budget_pacing.csv"
RAW_META_PERF: Path = PROJECT_ROOT / "meta_campaign_performance.csv"
RAW_CONVERSIONS: Path = PROJECT_ROOT / "conversion_events.csv"

# DuckDB schema names for each layer
SCHEMA_BRONZE: str = "bronze"
SCHEMA_SILVER: str = "silver"
SCHEMA_GOLD: str = "gold"

# Table names (namespaced by schema at runtime)
TBL_AUDIENCE_SEGMENTS: str = "audience_segments"
TBL_BUDGET_PACING: str = "budget_pacing"
TBL_META_PERF: str = "meta_campaign_performance"
TBL_CONVERSIONS: str = "conversion_events"

# Silver table names
TBL_SLV_META_PERF: str = "slv_meta_campaign_performance"
TBL_SLV_BUDGET_PACING: str = "slv_budget_pacing"
TBL_SLV_CONVERSIONS: str = "slv_conversion_events"

# Gold table names
TBL_GLD_DAILY_METRICS: str = "gld_daily_metrics"
TBL_GLD_TRAINING_SET: str = "gld_training_set"

# Misc settings
DATE_FORMAT_DMY: str = "%d/%m/%Y"  # Dates in inputs are in day/month/year
RANDOM_SEED: int = 42  # Reproducibility for ML
