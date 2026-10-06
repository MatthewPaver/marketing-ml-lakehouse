"""Prepare a leakage-safe GA4 daily forecasting table from a local export.

The BigQuery query in ``public_data/ga4/export_daily.sql`` is intentionally a
separate, explicit step.  This module never needs cloud credentials and records
the local input checksum in its output metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "event_date",
    "active_users",
    "sessions",
    "page_views",
    "add_to_carts",
    "purchases",
    "purchase_revenue",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_export(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("GA4 export must be CSV or Parquet")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"GA4 export is missing columns: {sorted(missing)}")
    return frame


def build_prospective_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one row per as-of day with only current/past features."""
    daily = frame.copy()
    daily["event_date"] = pd.to_datetime(daily["event_date"], format="%Y%m%d", errors="coerce")
    if daily["event_date"].isna().any():
        # Accept ISO dates as well as the native GA4 YYYYMMDD representation.
        daily["event_date"] = pd.to_datetime(frame["event_date"], errors="raise")
    daily = daily.sort_values("event_date").drop_duplicates("event_date", keep="last")
    for column in sorted(REQUIRED_COLUMNS - {"event_date"}):
        daily[column] = pd.to_numeric(daily[column], errors="raise")
        if (daily[column] < 0).any():
            raise ValueError(f"{column} contains negative values")

    daily = daily.rename(columns={"event_date": "as_of_date", "purchases": "purchases_asof"})
    daily["purchases_trailing_7d"] = daily["purchases_asof"].rolling(7, min_periods=1).mean()
    daily["sessions_trailing_7d"] = daily["sessions"].rolling(7, min_periods=1).mean()
    daily["target_date"] = daily["as_of_date"].shift(-1)
    daily["target_next_day_purchases"] = daily["purchases_asof"].shift(-1)
    daily = daily[
        (daily["target_date"] - daily["as_of_date"]).dt.days.eq(1)
    ].copy()
    return daily


def prepare(input_path: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_prospective_table(load_export(input_path))
    table_path = output_dir / "ga4_next_day_purchases.parquet"
    table.to_parquet(table_path, index=False)
    metadata = {
        "source": "bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*",
        "source_documentation": "https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset",
        "input_file": input_path.name,
        "input_sha256": _sha256(input_path),
        "rows": int(len(table)),
        "prediction_contract": {
            "as_of": "end of as_of_date",
            "horizon": "next calendar day",
            "target": "target_next_day_purchases",
            "baseline": "purchases_asof persistence",
        },
        "limitations": [
            "Google describes the sample as obfuscated and internally inconsistent in places.",
            "This adapter demonstrates engineering and temporal evaluation, not current business performance.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="CSV or Parquet produced by export_daily.sql")
    parser.add_argument("--output", type=Path, default=Path("data/public/ga4/derived"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
