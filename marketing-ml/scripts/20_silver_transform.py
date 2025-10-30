from __future__ import annotations

import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "lakehouse.duckdb"


def log(msg: str) -> None:
    print(f"[silver] {msg}")


def main() -> None:
    con = duckdb.connect(str(DB_PATH))

    # meta aggregated to daily (sum sums, mean for rate-like fields)
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.meta_campaign_performance AS
        SELECT
            STRPTIME(date, '%d/%m/%Y')::DATE AS date,
            ad_set_id,
            SUM(impressions) AS impressions,
            SUM(clicks) AS clicks,
            SUM(GREATEST(spend, 0.0)) AS spend,
            AVG(ctr) AS ctr,
            AVG(cpm) AS cpm,
            AVG(frequency) AS frequency
        FROM bronze.meta_campaign_performance
        GROUP BY 1,2;
        """
    )
    rows_m = con.execute("SELECT COUNT(*) FROM silver.meta_campaign_performance").fetchone()[0]

    # conversion events → revenue daily per ad set
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.revenue_by_day AS
        SELECT
            STRPTIME(date, '%d/%m/%Y')::DATE AS date,
            ad_set_id,
            SUM(CASE WHEN conversion_type='booking_completed' AND value IS NOT NULL THEN value ELSE 0 END) AS revenue
        FROM bronze.conversion_events
        GROUP BY 1,2;
        """
    )
    rows_r = con.execute("SELECT COUNT(*) FROM silver.revenue_by_day").fetchone()[0]

    # silver_daily join with ROAS
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.silver_daily AS
        SELECT m.ad_set_id, m.date, m.impressions, m.clicks, m.spend, m.ctr, m.cpm, m.frequency,
               COALESCE(r.revenue, 0.0) AS revenue,
               CASE WHEN m.spend > 0 THEN COALESCE(r.revenue,0.0)/m.spend ELSE 0.0 END AS roas
        FROM silver.meta_campaign_performance m
        LEFT JOIN silver.revenue_by_day r USING(ad_set_id, date);
        """
    )
    rows_sd = con.execute("SELECT COUNT(*) FROM silver.silver_daily").fetchone()[0]
    uniq_keys = con.execute("SELECT COUNT(*) FROM (SELECT ad_set_id, date FROM silver.silver_daily GROUP BY 1,2)").fetchone()[0]
    log(f"meta rows: {rows_m}, revenue rows: {rows_r}, silver_daily rows: {rows_sd}, unique (ad_set_id,date): {uniq_keys}")

    # budget pacing: use budget_utilization directly
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.budget_pacing AS
        SELECT
            STRPTIME(b.date, '%d/%m/%Y')::DATE AS date,
            b.ad_set_id,
            COALESCE(a.daily_budget, 0.0) AS daily_budget,
            CAST(b.budget_utilization AS DOUBLE) AS pacing_ratio
        FROM bronze.budget_pacing b
        LEFT JOIN bronze.audience_segments a USING(ad_set_id);
        """
    )

    # audience segments passthrough
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.audience_segments AS
        SELECT ad_set_id, ad_set_name, target_age, target_interests, lookalike_source,
               COALESCE(daily_budget, 0.0) AS daily_budget
        FROM bronze.audience_segments;
        """
    )

    # DQ summary
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.dq_summary AS
        SELECT 'silver_daily' AS table_name, 'spend' AS column_name, 'DOUBLE' AS dtype,
               (COUNT(*) FILTER(WHERE spend IS NULL))::DOUBLE/NULLIF(COUNT(*),0) AS null_fraction,
               MIN(spend) AS min, MAX(spend) AS max, AVG(spend) AS mean, STDDEV(spend) AS std, now() AS generated_at
        FROM silver.silver_daily
        UNION ALL
        SELECT 'silver_daily','impressions','HUGEINT', (COUNT(*) FILTER(WHERE impressions IS NULL))::DOUBLE/NULLIF(COUNT(*),0), MIN(impressions), MAX(impressions), AVG(impressions), STDDEV(impressions), now()
        FROM silver.silver_daily
        UNION ALL
        SELECT 'silver_daily','revenue','DOUBLE', (COUNT(*) FILTER(WHERE revenue IS NULL))::DOUBLE/NULLIF(COUNT(*),0), MIN(revenue), MAX(revenue), AVG(revenue), STDDEV(revenue), now()
        FROM silver.silver_daily
        UNION ALL
        SELECT 'budget_pacing','pacing_ratio','DOUBLE', (COUNT(*) FILTER(WHERE pacing_ratio IS NULL))::DOUBLE/NULLIF(COUNT(*),0), MIN(pacing_ratio), MAX(pacing_ratio), AVG(pacing_ratio), STDDEV(pacing_ratio), now()
        FROM silver.budget_pacing;
        """
    )

    # uniqueness guard
    dup = con.execute("SELECT COUNT(*) FROM (SELECT ad_set_id, date, COUNT(*) c FROM silver.silver_daily GROUP BY 1,2 HAVING c>1)").fetchone()[0]
    if dup:
        raise RuntimeError("Uniqueness violation in silver.silver_daily (ad_set_id,date)")

    log("done.")
    con.close()


if __name__ == "__main__":
    main()
