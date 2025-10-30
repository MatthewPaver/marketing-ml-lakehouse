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

    # Training set for supervised learning
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {SCHEMA_GOLD}.{TBL_GLD_TRAINING_SET} AS
        SELECT
            date,
            ad_set_id,
            ad_set_name,
            impressions,
            clicks,
            spend,
            ctr,
            cpm,
            frequency,
            planned_spend,
            actual_spend,
            budget_utilization,
            CASE pacing_status
                WHEN 'under_pacing' THEN 0
                WHEN 'on_pace' THEN 1
                WHEN 'over_pacing' THEN 2
                ELSE 3
            END AS pacing_status_idx,
            soft_conversions,
            COALESCE(revenue, 0.0) AS revenue,
            COALESCE(bookings, 0) AS target_bookings
        FROM {SCHEMA_GOLD}.{TBL_GLD_DAILY_METRICS};
        """
    )

    con.close()


if __name__ == "__main__":
    transform()
