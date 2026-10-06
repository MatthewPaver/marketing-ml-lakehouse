# Reusable lakehouse template

This repository is both a marketing analytics example and a small reference template. The reusable part is the contract and layer boundary, not the marketing column names.

## The invariant

```text
source files
  -> structural contract and checksums
  -> bronze source-shaped tables
  -> silver typed, deduplicated and conformed tables
  -> gold decision and model tables
  -> quality and model evidence
  -> dashboard or downstream consumer
```

## Adapt it to another domain

1. Put small, rights-cleared fixtures in `data/raw/`.
2. Replace `contracts/raw_sources.json` with the expected files, columns, non-null fields, and natural keys.
3. Keep bronze source-shaped. Do not hide cleaning inside ingestion.
4. Put type repair, deduplication, rejected values, and business conformance in silver.
5. Build each gold table for a named decision or model, not because a metric is available.
6. Add a contract test for every new source and a business-rule test for every gold metric.
7. Treat generated DuckDB files and models as build artefacts. Rebuild them in CI.
8. Record a production path for storage, orchestration, secrets, access control, observability, retention, and backfill.

## What this local version demonstrates

- deterministic rebuild from committed inputs;
- source checksums and structural contracts before ingestion;
- bronze, silver and gold responsibilities;
- duplicate and invalid-value handling;
- data-quality evidence beside model evidence;
- a browser evidence console plus a local dashboard;
- a CI run that rebuilds before testing.

## What a production version still needs

- object storage and an open table format such as Iceberg, Delta, or DuckLake;
- partition and incremental-merge strategy;
- an orchestrator with retries, backfills and run metadata;
- a catalogue, ownership and column-level lineage;
- secrets, roles and environment isolation;
- data retention, deletion and cost controls;
- model registry, monitoring and promotion rules if ML remains in scope.

The template is deliberately vendor-neutral. The DuckDB implementation keeps it runnable on a laptop; the boundary notes show where platform choices belong.
