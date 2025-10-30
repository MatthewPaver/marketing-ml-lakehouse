"""Compute and persist data quality report for gold.daily_metrics.

The report includes per-column null fraction and basic descriptive stats.
"""

from __future__ import annotations

from datetime import datetime
import pandas as pd

from lakehouse.config import SCHEMA_GOLD, TBL_GLD_DAILY_METRICS
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


def run() -> None:
    con = get_connection()
    ensure_schemas(con)
    df = con.execute(f"SELECT * FROM {SCHEMA_GOLD}.{TBL_GLD_DAILY_METRICS}").df()
    report = compute_report(df)
    con.register("v_dq", report)
    con.execute(
        f"CREATE OR REPLACE TABLE {SCHEMA_GOLD}.{REPORT_TABLE} AS SELECT * FROM v_dq"
    )
    con.close()


if __name__ == "__main__":
    run()
