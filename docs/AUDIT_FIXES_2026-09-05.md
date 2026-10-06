# Audit fixes — 2026-09-05

This repository is a **data-engineering and prospective-evaluation teaching case**. Its synthetic fixture does not establish commercial campaign performance.

## Changes

- Both targets are next-calendar-day outcomes with an explicit end-of-day decision timestamp. Features use only the as-of row and trailing history; target dates must be strictly later.
- Whole as-of dates are isolated between train and holdout partitions.
- Booking and pacing models are compared with prior-day persistence baselines. Artifacts report row counts, date ranges, model/baseline metrics and deterministic bootstrap uncertainty intervals.
- Model results remain negative when they fail to beat baseline; no positive result is assumed by the presentation layer.
- Raw source contracts execute required-column/non-null/key checks and record SHA-256 provenance.
- Data-quality generation now executes daily-key uniqueness plus spend and conversion-revenue reconciliations, and writes a versioned JSON artifact.
- `lakehouse.publish_pages` builds the Pages evidence payload from the rebuilt DuckDB tables, contract report, quality report and model metadata. The browser fetches that generated file instead of embedding campaign/check results in JavaScript.
- An optional GA4 public-data profile is documented separately; it is not required for the offline fixture.

## Verification

- Existing local Python 3.11 environment: full pipeline rebuilt successfully; regression MAE `0.5249` versus persistence `0.6667` (skill `0.2126`); pacing balanced accuracy `0.9091` versus persistence `0.8706`.
- Python suite: `18 passed`.
- Node Pages contract: `2 passed`.
- Generated quality status: `pass`; 126 gold daily rows, unique daily keys, and both monetary reconciliations passed.

## Remaining limitations

- The synthetic fixture is small (21–22 observed days per ad set), so intervals and point estimates are unstable and are not evidence of business impact.
- The contract report contains one source warning that remains visible rather than being silently repaired.
- A second verification used a fresh Python 3.11 environment with wheel-only installation from `requirements.txt`: full pipeline and 18 tests passed with the same model/baseline values. Key resolved versions: DuckDB 1.5.5, pandas 2.3.3, NumPy 2.4.6, scikit-learn 1.9.0, XGBoost 2.1.4. This checks the current macOS resolution, not every platform or future dependency version.
- Pages serves committed generated evidence and does not execute DuckDB or model training in the browser. Rebuild it with `make run` before publishing changed inputs or code.
