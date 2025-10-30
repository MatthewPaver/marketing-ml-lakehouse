from __future__ import annotations

from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "lakehouse.duckdb"


def log(msg: str) -> None:
    print(f"[labels] {msg}")


def main() -> None:
    con = duckdb.connect(str(DB_PATH))

    # Ensure 7-day conversions exist
    con.execute(
        """
        CREATE OR REPLACE TABLE gold.adset_conversions_7d AS
        WITH dates AS (
            SELECT DISTINCT date, ad_set_id FROM silver.meta_campaign_performance
        )
        SELECT d.ad_set_id, d.date,
               COALESCE((SELECT COUNT(*) FROM silver.conversion_events c WHERE c.ad_set_id=d.ad_set_id AND c.conv_date BETWEEN d.date AND d.date + INTERVAL 7 DAY), 0) AS conv_7d
        FROM dates d;
        """
    )

    # Build labels view joining silver meta for impressions
    con.execute(
        """
        CREATE OR REPLACE TABLE gold.adset_conv_labels AS
        SELECT m.ad_set_id, m.date, c.conv_7d,
               CASE WHEN m.impressions > 0 THEN c.conv_7d::DOUBLE / m.impressions ELSE 0.0 END AS conv_next7d_rate,
               CASE WHEN c.conv_7d > 0 THEN 1 ELSE 0 END AS conv_next7d_label
        FROM silver.meta_campaign_performance m
        LEFT JOIN gold.adset_conversions_7d c USING(ad_set_id, date);
        """
    )

    log("labels created: gold.adset_conv_labels")
    con.close()


if __name__ == "__main__":
    main()
