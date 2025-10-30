from __future__ import annotations

import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "lakehouse.duckdb"


def log(msg: str) -> None:
    print(f"[gold] {msg}")


def main() -> None:
    con = duckdb.connect(str(DB_PATH))

    # 7d conversions forward-looking window [date, date+7]
    con.execute(
        """
        CREATE OR REPLACE TABLE gold.adset_conversions_7d AS
        WITH dates AS (
            SELECT DISTINCT date, ad_set_id FROM silver.meta_campaign_performance
        )
        SELECT d.ad_set_id, d.date,
               COALESCE(
                 (SELECT COUNT(*) FROM silver.conversion_events c
                  WHERE c.ad_set_id=d.ad_set_id AND c.conv_date BETWEEN d.date AND d.date + INTERVAL 7 DAY), 0) AS conv_7d
        FROM dates d;
        """
    )
    log(f"adset_conversions_7d rows: {con.execute('SELECT COUNT(*) FROM gold.adset_conversions_7d').fetchone()[0]}")

    # pacing with LEAD next day
    con.execute(
        """
        CREATE OR REPLACE TABLE gold.adset_pacing AS
        SELECT ad_set_id, date,
               pacing_ratio,
               LEAD(pacing_ratio, 1) OVER (PARTITION BY ad_set_id ORDER BY date) AS next_pacing_ratio,
               CASE WHEN pacing_ratio < 0.9 THEN 'under'
                    WHEN pacing_ratio <= 1.1 THEN 'on'
                    ELSE 'over' END AS pacing_status
        FROM silver.budget_pacing;
        """
    )
    log(f"adset_pacing rows: {con.execute('SELECT COUNT(*) FROM gold.adset_pacing').fetchone()[0]}")

    # labels table
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
    log(f"adset_conv_labels rows: {con.execute('SELECT COUNT(*) FROM gold.adset_conv_labels').fetchone()[0]}")

    # training dataset join
    con.execute(
        """
        CREATE OR REPLACE TABLE gold.training_dataset AS
        SELECT m.date, m.ad_set_id,
               m.ctr, m.impressions, m.clicks, m.spend,
               a.daily_budget AS aud_daily_budget,
               p.pacing_ratio,
               c.conv_7d,
               COALESCE(r.revenue, 0.0) AS revenue,
               CASE WHEN m.spend > 0 THEN COALESCE(r.revenue, 0.0)/m.spend ELSE 0.0 END AS roas,
               lbl.conv_next7d_rate,
               lbl.conv_next7d_label,
               CASE WHEN p.next_pacing_ratio < 0.9 THEN 1 ELSE 0 END AS label_under_pacing_next
        FROM silver.meta_campaign_performance m
        LEFT JOIN silver.audience_segments a USING(ad_set_id)
        LEFT JOIN gold.adset_pacing p USING(ad_set_id, date)
        LEFT JOIN gold.adset_conversions_7d c USING(ad_set_id, date)
        LEFT JOIN gold.adset_conv_labels lbl USING(ad_set_id, date)
        LEFT JOIN silver.revenue_by_day r USING(ad_set_id, date);
        """
    )
    log(f"training_dataset rows: {con.execute('SELECT COUNT(*) FROM gold.training_dataset').fetchone()[0]}")
    pos_rate = con.execute("SELECT AVG(CASE WHEN label_under_pacing_next=1 THEN 1.0 ELSE 0.0 END) FROM gold.training_dataset").fetchone()[0]
    log(f"label_under_pacing_next positive rate: {pos_rate:.3f}")

    log("done.")
    con.close()


if __name__ == "__main__":
    main()
