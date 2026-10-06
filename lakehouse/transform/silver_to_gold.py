"""Silver → Gold feature aggregation and training set creation.

This module joins cleansed performance, budget, and conversions to produce
analytics-friendly metrics and a supervised learning training table.
"""

from __future__ import annotations

import duckdb

from lakehouse.config import (
    SCHEMA_SILVER,
    SCHEMA_GOLD,
    TBL_SLV_META_PERF,
    TBL_SLV_BUDGET_PACING,
    TBL_SLV_CONVERSIONS,
    TBL_GLD_DAILY_METRICS,
    TBL_GLD_TRAINING_SET,
)
from lakehouse.utils.db import get_connection, ensure_schemas

# In gold, we prepare analytics-friendly aggregates and features for modelling.


def transform() -> None:
    con = get_connection()
    ensure_schemas(con)

    # Daily metrics joined view (per date, ad_set_id)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {SCHEMA_GOLD}.{TBL_GLD_DAILY_METRICS} AS
        WITH meta AS (
            SELECT
                date,
                ad_set_id,
                ad_set_name,
                COALESCE(impressions, 0) AS impressions,
                COALESCE(clicks, 0) AS clicks,
                COALESCE(spend, 0.0) AS spend,
                COALESCE(ctr, 0.0) AS ctr,
                COALESCE(cpm, 0.0) AS cpm,
                COALESCE(frequency, 0.0) AS frequency
            FROM {SCHEMA_SILVER}.{TBL_SLV_META_PERF}
        ),
        budget AS (
            SELECT
                date,
                ad_set_id,
                COALESCE(planned_spend, 0.0) AS planned_spend,
                COALESCE(actual_spend, 0.0) AS actual_spend,
                COALESCE(budget_utilization, 0.0) AS budget_utilization,
                COALESCE(pacing_status, 'unknown') AS pacing_status
            FROM {SCHEMA_SILVER}.{TBL_SLV_BUDGET_PACING}
        ),
        conv AS (
            SELECT
                date,
                ad_set_id,
                -- Sum value for booking_completed only as revenue. Other events are auxiliary.
                SUM(CASE WHEN conversion_type = 'booking_completed' THEN value ELSE 0.0 END) AS revenue,
                COUNT(CASE WHEN conversion_type = 'booking_completed' THEN 1 END) AS bookings,
                COUNT(CASE WHEN conversion_type IN ('event_registration', 'newsletter_signup', 'hotel_inquiry', 'flight_search', 'destination_guide_download') THEN 1 END) AS soft_conversions
            FROM {SCHEMA_SILVER}.{TBL_SLV_CONVERSIONS}
            GROUP BY 1,2
        )
        SELECT
            COALESCE(meta.date, budget.date, conv.date) AS date,
            COALESCE(meta.ad_set_id, budget.ad_set_id, conv.ad_set_id) AS ad_set_id,
            meta.ad_set_name AS ad_set_name,
            COALESCE(impressions, 0) AS impressions,
            COALESCE(clicks, 0) AS clicks,
            COALESCE(spend, 0.0) AS spend,
            COALESCE(ctr, 0.0) AS ctr,
            COALESCE(cpm, 0.0) AS cpm,
            COALESCE(frequency, 0.0) AS frequency,
            COALESCE(planned_spend, 0.0) AS planned_spend,
            COALESCE(actual_spend, 0.0) AS actual_spend,
            COALESCE(budget_utilization, 0.0) AS budget_utilization,
            COALESCE(pacing_status, 'unknown') AS pacing_status,
            COALESCE(revenue, 0.0) AS revenue,
            COALESCE(bookings, 0) AS bookings,
            COALESCE(soft_conversions, 0) AS soft_conversions,
            CASE WHEN COALESCE(spend,0) > 0 THEN COALESCE(revenue,0) / spend ELSE NULL END AS roas,
            CASE WHEN COALESCE(bookings,0) > 0 THEN COALESCE(spend,0) / bookings ELSE NULL END AS cpa
        FROM meta
        FULL OUTER JOIN budget USING(date, ad_set_id)
        FULL OUTER JOIN conv USING(date, ad_set_id)
        ORDER BY 1,2;
        """
    )

    # Prospective training set.  Every feature is known at the end of
    # ``as_of_date``; both targets describe the following calendar day.  The
    # explicit target date makes the prediction horizon testable rather than a
    # convention hidden in model code.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {SCHEMA_GOLD}.{TBL_GLD_TRAINING_SET} AS
        WITH sequenced AS (
            SELECT
                date AS as_of_date,
                LEAD(date) OVER (PARTITION BY ad_set_id ORDER BY date) AS target_date,
                ad_set_id,
                ad_set_name,
                impressions AS impressions_asof,
                clicks AS clicks_asof,
                spend AS spend_asof,
                ctr AS ctr_asof,
                cpm AS cpm_asof,
                frequency AS frequency_asof,
                planned_spend AS planned_spend_asof,
                actual_spend AS actual_spend_asof,
                budget_utilization AS budget_utilization_asof,
                CASE pacing_status
                    WHEN 'under_pacing' THEN 0
                    WHEN 'on_pace' THEN 1
                    WHEN 'over_pacing' THEN 2
                    ELSE 3
                END AS pacing_status_asof,
                CASE WHEN pacing_status = 'under_pacing' THEN 1 ELSE 0 END AS under_pacing_asof,
                soft_conversions AS soft_conversions_asof,
                revenue AS revenue_asof,
                bookings AS bookings_asof,
                AVG(bookings) OVER (
                    PARTITION BY ad_set_id ORDER BY date
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS bookings_trailing_3d,
                AVG(bookings) OVER (
                    PARTITION BY ad_set_id ORDER BY date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) AS bookings_trailing_7d,
                AVG(budget_utilization) OVER (
                    PARTITION BY ad_set_id ORDER BY date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) AS budget_utilization_trailing_7d,
                LEAD(bookings) OVER (PARTITION BY ad_set_id ORDER BY date) AS target_next_day_bookings,
                LEAD(CASE WHEN pacing_status = 'under_pacing' THEN 1 ELSE 0 END)
                    OVER (PARTITION BY ad_set_id ORDER BY date) AS target_next_day_under_pacing
            FROM {SCHEMA_GOLD}.{TBL_GLD_DAILY_METRICS}
        )
        SELECT
            *
        FROM sequenced
        WHERE target_date IS NOT NULL
          AND date_diff('day', as_of_date, target_date) = 1
        ORDER BY as_of_date, ad_set_id;
        """
    )

    con.close()


if __name__ == "__main__":
    transform()
