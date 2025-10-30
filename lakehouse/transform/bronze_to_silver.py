"""Bronze → Silver cleansing and conformance.

- Enforces numeric types and safe metric recomputation
- Normalises enums (e.g., pacing_status)
- Removes duplicates and missing identifiers
"""

from __future__ import annotations

import pandas as pd

from lakehouse.config import (
    SCHEMA_BRONZE,
    SCHEMA_SILVER,
    TBL_META_PERF,
    TBL_BUDGET_PACING,
    TBL_CONVERSIONS,
    TBL_SLV_META_PERF,
    TBL_SLV_BUDGET_PACING,
    TBL_SLV_CONVERSIONS,
)
from lakehouse.utils.db import get_connection, ensure_schemas

# In silver, we correct data quality issues and enforce types/ranges.


def clean_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Cast numerics
    for col in ["impressions", "clicks"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["spend", "ctr", "cpm", "frequency"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix negative spend → 0
    df.loc[df["spend"] < 0, "spend"] = 0.0

    # Drop rows missing identifiers
    df = df.dropna(subset=["ad_set_id", "date"])

    # Recompute metrics where possible (divide-by-zero safe)
    with pd.option_context("mode.use_inf_as_na", True):
        df["ctr"] = (df["clicks"].astype("float") / df["impressions"].astype("float") * 100.0).where(
            df["impressions"].fillna(0) > 0, df["ctr"]
        )
        df["cpm"] = (df["spend"].astype("float") / df["impressions"].astype("float") * 1000.0).where(
            df["impressions"].fillna(0) > 0, df["cpm"]
        )

    # Drop exact duplicates
    df = df.drop_duplicates()
    return df


def clean_budget(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["planned_spend", "actual_spend", "budget_utilization"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["pacing_status"] = df["pacing_status"].astype(str).str.lower()
    df.loc[~df["pacing_status"].isin(["under_pacing", "on_pace", "over_pacing"]), "pacing_status"] = "unknown"
    df = df.drop_duplicates()
    return df


def clean_conversions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["conversion_type"] = df["conversion_type"].astype(str).str.lower()
    df["attribution_window"] = df["attribution_window"].astype(str).str.lower()
    df = df.dropna(subset=["ad_set_id", "date"]).copy()
    df["value"] = df["value"].fillna(0.0)
    df = df.drop_duplicates()
    return df


def transform() -> None:
    con = get_connection()
    ensure_schemas(con)

    # Load bronze into pandas via DuckDB
    df_meta = con.execute(f"SELECT * FROM {SCHEMA_BRONZE}.{TBL_META_PERF}").df()
    df_budget = con.execute(f"SELECT * FROM {SCHEMA_BRONZE}.{TBL_BUDGET_PACING}").df()
    df_conv = con.execute(f"SELECT * FROM {SCHEMA_BRONZE}.{TBL_CONVERSIONS}").df()

    df_meta_c = clean_meta(df_meta)
    df_budget_c = clean_budget(df_budget)
    df_conv_c = clean_conversions(df_conv)

    # Write to silver
    con.register("v_meta_c", df_meta_c)
    con.register("v_budget_c", df_budget_c)
    con.register("v_conv_c", df_conv_c)

    con.execute(
        f"CREATE OR REPLACE TABLE {SCHEMA_SILVER}.{TBL_SLV_META_PERF} AS SELECT * FROM v_meta_c"
    )
    con.execute(
        f"CREATE OR REPLACE TABLE {SCHEMA_SILVER}.{TBL_SLV_BUDGET_PACING} AS SELECT * FROM v_budget_c"
    )
    con.execute(
        f"CREATE OR REPLACE TABLE {SCHEMA_SILVER}.{TBL_SLV_CONVERSIONS} AS SELECT * FROM v_conv_c"
    )

    con.close()


if __name__ == "__main__":
    transform()
