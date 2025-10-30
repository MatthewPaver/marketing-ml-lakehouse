"""Raw → Bronze ingestion using pandas into DuckDB.

This module reads the provided CSV files with UK date parsing and writes
idempotent bronze tables under the DuckDB `bronze` schema.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from lakehouse.config import (
    RAW_AUDIENCE_SEGMENTS,
    RAW_BUDGET_PACING,
    RAW_META_PERF,
    RAW_CONVERSIONS,
    SCHEMA_BRONZE,
    TBL_AUDIENCE_SEGMENTS,
    TBL_BUDGET_PACING,
    TBL_META_PERF,
    TBL_CONVERSIONS,
)
from lakehouse.utils.db import get_connection, ensure_schemas, register_dataframe, create_or_replace_from_view

# As a CS undergraduate style, we separate ingestion (raw→bronze) from transformations.


def read_csv(path: Path, parse_date_col: str | None = None) -> pd.DataFrame:
    """Read a CSV into a pandas DataFrame with optional UK date parsing."""
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")
    if parse_date_col is None:
        return pd.read_csv(path)
    return pd.read_csv(path, parse_dates=[parse_date_col], dayfirst=True)


def ingest() -> None:
    con = get_connection()
    ensure_schemas(con)

    # Read raw CSVs with UK-style dates
    df_audience = read_csv(RAW_AUDIENCE_SEGMENTS)
    df_budget = read_csv(RAW_BUDGET_PACING, parse_date_col="date")
    df_meta = read_csv(RAW_META_PERF, parse_date_col="date")
    df_conv = read_csv(RAW_CONVERSIONS, parse_date_col="date")

    # Register and write to bronze schema (create/replace keeps runs idempotent)
    register_dataframe(con, df_audience, "v_audience")
    register_dataframe(con, df_budget, "v_budget")
    register_dataframe(con, df_meta, "v_meta")
    register_dataframe(con, df_conv, "v_conv")

    create_or_replace_from_view(con, f"{SCHEMA_BRONZE}.{TBL_AUDIENCE_SEGMENTS}", "v_audience")
    create_or_replace_from_view(con, f"{SCHEMA_BRONZE}.{TBL_BUDGET_PACING}", "v_budget")
    create_or_replace_from_view(con, f"{SCHEMA_BRONZE}.{TBL_META_PERF}", "v_meta")
    create_or_replace_from_view(con, f"{SCHEMA_BRONZE}.{TBL_CONVERSIONS}", "v_conv")

    con.close()


if __name__ == "__main__":
    ingest()
