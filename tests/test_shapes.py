from __future__ import annotations

from pathlib import Path
import duckdb
import json
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
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
        assert all(not str(f).startswith(("next_", "target_")) for f in feats), "feature list contains future-aware fields"
        assert data["leakage_checks"]["target_date_after_as_of_date"] is True
        assert data["leakage_checks"]["dates_disjoint"] is True
        assert "baseline" in data
        assert any("95pct" in metric for metric in data["metrics"])


def test_training_rows_have_a_strict_next_day_target():
    con = duckdb.connect(str(DB_PATH))
    invalid = con.execute(
        """
        SELECT COUNT(*)
        FROM gold.gld_training_set
        WHERE target_date <= as_of_date
           OR date_diff('day', as_of_date, target_date) != 1
        """
    ).fetchone()[0]
    con.close()
    assert invalid == 0


def test_temporal_split_never_shares_a_date():
    from lakehouse.ml.validation import split_by_date

    frame = pd.DataFrame(
        {
            "as_of_date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03"],
            "value": [1, 2, 3, 4],
        }
    )
    train, test = split_by_date(frame, test_fraction=0.34)
    assert set(train["as_of_date"]).isdisjoint(set(test["as_of_date"]))


def test_silver_daily_unique_keys_and_roas():
    con = duckdb.connect(str(DB_PATH))
    dup = con.execute("SELECT COUNT(*) FROM (SELECT ad_set_id, date, COUNT(*) c FROM silver.slv_meta_campaign_performance GROUP BY 1,2 HAVING c>1)").fetchone()[0]
    assert dup == 0
    df = con.execute("SELECT revenue, spend, roas FROM gold.gld_daily_metrics WHERE spend>0 LIMIT 100").df()
    con.close()
    if not df.empty:
        import numpy as np
        assert np.allclose((df["revenue"]/df["spend"]).values, df["roas"].values, rtol=1e-6, atol=1e-6)


def test_quality_report_contains_executable_integrity_checks():
    report = json.loads((REPO_ROOT / "lakehouse" / "artifacts" / "quality_report.json").read_text())
    names = {check["name"] for check in report["checks"]}
    assert {"daily_key_unique", "spend_reconciles", "conversion_revenue_reconciles"}.issubset(names)
    assert all(check["status"] in {"pass", "fail"} for check in report["checks"])
    assert report["provenance"]["contract_report_sha256"]
    assert report["status"] == "pass"


def test_pages_evidence_is_generated_from_pipeline_artifacts():
    evidence = json.loads((REPO_ROOT / "docs" / "generated" / "evidence.json").read_text())
    assert evidence["schema_version"] == 1
    assert evidence["models"]["bookings"]["metrics"]["baseline_mae"] >= 0
    assert evidence["models"]["bookings"]["split"]["test_rows"] > 0
    assert evidence["quality"]["status"] == "pass"
    assert evidence["provenance"]["generator"] == "lakehouse.publish_pages"
