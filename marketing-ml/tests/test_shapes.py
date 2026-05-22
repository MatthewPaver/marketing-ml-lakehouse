from __future__ import annotations

from pathlib import Path
import duckdb
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DB_PATH = REPO_ROOT / "lakehouse" / "lakehouse.duckdb"


def test_active_lakehouse_tables_exist():
    con = duckdb.connect(str(DB_PATH))
    tables = con.execute(
        """
        SELECT table_schema || '.' || table_name AS table_id
        FROM information_schema.tables
        WHERE table_schema IN ('bronze', 'silver', 'gold')
        """
    ).df()
    con.close()
    expected = {
        "bronze.meta_campaign_performance",
        "bronze.budget_pacing",
        "bronze.conversion_events",
        "silver.slv_meta_campaign_performance",
        "silver.slv_budget_pacing",
        "silver.slv_conversion_events",
        "gold.gld_daily_metrics",
        "gold.gld_training_set",
        "gold.data_quality_report",
    }
    assert expected.issubset(set(tables["table_id"])), "active lakehouse tables missing"


def test_row_alignment_silver_gold():
    con = duckdb.connect(str(DB_PATH))
    silver_rows = con.execute("SELECT COUNT(*) FROM silver.slv_meta_campaign_performance").fetchone()[0]
    gold_rows = con.execute("SELECT COUNT(*) FROM gold.gld_training_set").fetchone()[0]
    con.close()
    assert gold_rows <= silver_rows, "gold rows should not exceed silver base"


def test_roas_calculation_tolerance():
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT revenue, spend, roas FROM gold.gld_daily_metrics WHERE spend>0 LIMIT 100").df()
    con.close()
    if not df.empty:
        approx = (df["revenue"] / df["spend"]).values
        assert np.allclose(approx, df["roas"].values, rtol=1e-6, atol=1e-6)


def test_model_artifacts_are_created():
    model_dir = REPO_ROOT / "lakehouse" / "models"
    expected = [
        model_dir / "bookings_xgb.json",
        model_dir / "bookings_xgb.pkl",
        model_dir / "underpacing_xgb.json",
        model_dir / "underpacing_xgb.pkl",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)


def test_no_future_aware_features():
    metadata_files = [
        REPO_ROOT / "lakehouse" / "models" / "bookings_xgb.json",
        REPO_ROOT / "lakehouse" / "models" / "underpacing_xgb.json",
    ]
    for metadata_file in metadata_files:
        data = json.loads(metadata_file.read_text())
        feats = data.get("features", data.get("feature_cols", []))
        assert all(not str(f).startswith("next_") for f in feats), "feature list contains future-aware fields"


def test_silver_daily_unique_keys_and_roas():
    con = duckdb.connect(str(DB_PATH))
    dup = con.execute("SELECT COUNT(*) FROM (SELECT ad_set_id, date, COUNT(*) c FROM silver.slv_meta_campaign_performance GROUP BY 1,2 HAVING c>1)").fetchone()[0]
    assert dup == 0
    df = con.execute("SELECT revenue, spend, roas FROM gold.gld_daily_metrics WHERE spend>0 LIMIT 100").df()
    con.close()
    if not df.empty:
        import numpy as np
        assert np.allclose((df["revenue"]/df["spend"]).values, df["roas"].values, rtol=1e-6, atol=1e-6)
