"""Compute and persist data quality report for gold.daily_metrics.

The report includes per-column null fraction and basic descriptive stats.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import pandas as pd

from lakehouse.config import PROJECT_ROOT, SCHEMA_GOLD, TBL_GLD_DAILY_METRICS
from lakehouse.utils.db import get_connection, ensure_schemas

REPORT_TABLE = "data_quality_report"


def compute_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        total = len(series)
        nulls = int(series.isna().sum())
        null_fraction = float(nulls) / total if total > 0 else 0.0
        stats = {
            "column_name": col,
            "dtype": dtype,
            "null_fraction": null_fraction,
            "mean": float(series.mean()) if pd.api.types.is_numeric_dtype(series) else None,
            "std": float(series.std()) if pd.api.types.is_numeric_dtype(series) else None,
            "min": float(series.min()) if pd.api.types.is_numeric_dtype(series) else None,
            "max": float(series.max()) if pd.api.types.is_numeric_dtype(series) else None,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        rows.append(stats)
    return pd.DataFrame(rows)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict[str, object]:
    con = get_connection()
    ensure_schemas(con)
    df = con.execute(f"SELECT * FROM {SCHEMA_GOLD}.{TBL_GLD_DAILY_METRICS}").df()
    report = compute_report(df)
    con.register("v_dq", report)
    con.execute(
        f"CREATE OR REPLACE TABLE {SCHEMA_GOLD}.{REPORT_TABLE} AS SELECT * FROM v_dq"
    )
    daily_duplicates = con.execute(
        f"SELECT COUNT(*) FROM (SELECT date, ad_set_id FROM {SCHEMA_GOLD}.{TBL_GLD_DAILY_METRICS} GROUP BY 1,2 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    spend, actual = con.execute(
        f"SELECT SUM(spend), SUM(actual_spend) FROM {SCHEMA_GOLD}.{TBL_GLD_DAILY_METRICS}"
    ).fetchone()
    source_revenue = con.execute(
        "SELECT SUM(value) FROM silver.slv_conversion_events WHERE conversion_type = 'booking_completed'"
    ).fetchone()[0]
    gold_revenue = con.execute(
        f"SELECT SUM(revenue) FROM {SCHEMA_GOLD}.{TBL_GLD_DAILY_METRICS}"
    ).fetchone()[0]
    checks = [
        {"name": "daily_key_unique", "status": "pass" if daily_duplicates == 0 else "fail", "observed": int(daily_duplicates), "expected": 0},
        {"name": "spend_reconciles", "status": "pass" if abs(float(spend) - float(actual)) < 0.01 else "fail", "observed": float(spend), "expected": float(actual), "tolerance": 0.01},
        {"name": "conversion_revenue_reconciles", "status": "pass" if abs(float(source_revenue) - float(gold_revenue)) < 0.01 else "fail", "observed": float(gold_revenue), "expected": float(source_revenue), "tolerance": 0.01},
    ]
    con.close()
    contract_path = PROJECT_ROOT / "lakehouse" / "artifacts" / "contract_report.json"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status": "fail" if any(check["status"] == "fail" for check in checks) else "pass",
        "rows": int(len(df)),
        "checks": checks,
        "column_profile": json.loads(report.to_json(orient="records")),
        "provenance": {"contract_report_sha256": _sha256(contract_path)},
    }
    output = PROJECT_ROOT / "lakehouse" / "artifacts" / "quality_report.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["status"] == "fail":
        raise ValueError("Data-quality integrity checks failed")
    return payload


if __name__ == "__main__":
    run()
