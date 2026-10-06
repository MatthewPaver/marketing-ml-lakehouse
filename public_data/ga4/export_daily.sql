-- Standard SQL. Replace YOUR_PROJECT with a project you can write to, or use
-- the BigQuery UI's Save results action to export this result as CSV/Parquet.
-- Source: Google's GA4 obfuscated ecommerce public sample, 2020-11-01 to 2021-01-31.
SELECT
  event_date,
  COUNT(DISTINCT user_pseudo_id) AS active_users,
  COUNT(DISTINCT CONCAT(
    user_pseudo_id,
    '-',
    COALESCE(CAST((SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS STRING), 'unknown')
  )) AS sessions,
  COUNTIF(event_name = 'page_view') AS page_views,
  COUNTIF(event_name = 'add_to_cart') AS add_to_carts,
  COUNTIF(event_name = 'purchase') AS purchases,
  COALESCE(SUM(IF(event_name = 'purchase', ecommerce.purchase_revenue, 0)), 0) AS purchase_revenue
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
GROUP BY event_date
ORDER BY event_date;
