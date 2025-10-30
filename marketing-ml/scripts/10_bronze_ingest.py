from __future__ import annotations

from pathlib import Path
import polars as pl
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "lakehouse.duckdb"

TABLES = {
    "meta_campaign_performance": RAW_DIR / "meta_campaign_performance.csv",
    "budget_pacing": RAW_DIR / "budget_pacing.csv",
    "conversion_events": RAW_DIR / "conversion_events.csv",
    "audience_segments": RAW_DIR / "audience_segments.csv",
}


def log(msg: str) -> None:
    print(f"[bronze] {msg}")


def read_csv_safe(path: Path) -> pl.DataFrame:
    if not path.exists():
        log(f"missing: {path}")
        return pl.DataFrame()
    return pl.read_csv(path, infer_schema_length=2000)


def cast_df(name: str, df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    if name == "meta_campaign_performance":
        return df.with_columns([
            pl.col("date").cast(pl.Utf8, strict=False),
            pl.col("ad_set_id").cast(pl.Utf8, strict=False),
            pl.col("ad_set_name").cast(pl.Utf8, strict=False),
            pl.col("impressions").cast(pl.Int64, strict=False),
            pl.col("clicks").cast(pl.Int64, strict=False),
            pl.col("spend").cast(pl.Float64, strict=False),
            pl.col("ctr").cast(pl.Float64, strict=False),
            pl.col("cpm").cast(pl.Float64, strict=False),
            pl.col("frequency").cast(pl.Float64, strict=False),
        ])
    if name == "budget_pacing":
        return df.with_columns([
            pl.col("date").cast(pl.Utf8, strict=False),
            pl.col("ad_set_id").cast(pl.Utf8, strict=False),
            pl.all().exclude(["date", "ad_set_id"]).cast(pl.Float64, strict=False),
        ])
    if name == "conversion_events":
        return df.with_columns([
            pl.col("date").cast(pl.Utf8, strict=False),
            pl.col("ad_set_id").cast(pl.Utf8, strict=False),
            pl.col("value").cast(pl.Float64, strict=False),
        ])
    if name == "audience_segments":
        return df.with_columns([
            pl.col("ad_set_id").cast(pl.Utf8, strict=False),
            pl.col("daily_budget").cast(pl.Float64, strict=False),
        ])
    return df


def write_bronze(con: duckdb.DuckDBPyConnection, name: str, df: pl.DataFrame) -> None:
    con.register(f"view_{name}", df.to_arrow())
    con.execute(f"CREATE OR REPLACE TABLE bronze.{name} AS SELECT * FROM view_{name}")


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    for name, path in TABLES.items():
        log(f"ingesting {name}")
        df = read_csv_safe(path)
        df = cast_df(name, df)
        write_bronze(con, name, df)
    con.close()
    log("done.")


if __name__ == "__main__":
    main()
