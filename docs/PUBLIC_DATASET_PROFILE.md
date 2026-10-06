# Optional public-data profile: GA4 ecommerce

The committed CSVs stay small so the complete pipeline is fast in CI. For a more realistic nested event source, use Google's official `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` dataset as an optional profile.

Google describes it as three months of obfuscated Google Merchandise Store GA4 ecommerce export data. It includes placeholder and null values and may have limited internal consistency because of obfuscation. That makes it useful for schema evolution, nested-field extraction, partition pruning, null handling and session-level gold tables. It should not be used to claim current campaign performance or compared with the separate Google Analytics Demo Account.

## Reproducible profile

1. Run [`public_data/ga4/export_daily.sql`](../public_data/ga4/export_daily.sql) in BigQuery Sandbox or the free usage tier.
2. Save the result as `data/public/ga4/export.parquet` (this local path is ignored).
3. Run `make ga4-profile INPUT=data/public/ga4/export.parquet`.
4. Inspect `data/public/ga4/derived/metadata.json` and `ga4_next_day_purchases.parquet`.

The derived table asks a concrete prospective question: **given daily ecommerce activity known at the end of day _t_, how well can we predict purchases on day _t+1_?** It records the input checksum, uses a one-calendar-day horizon, and makes same-day purchases available as an explicit persistence baseline. The final day is excluded because its target is unknowable inside the sample.

The adapter is implemented in [`lakehouse/public_data/ga4.py`](../lakehouse/public_data/ga4.py) and is covered by a small credential-free test. Raw public-data exports are not committed.

Source: https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset

The public profile is an extension exercise, not a hidden dependency. The default demo remains credential-free and reproducible.
