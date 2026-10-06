"""Prospective next-day bookings model using XGBoost.

- Decision point: end of ``as_of_date``
- Target: bookings on ``target_date`` (the next calendar day)
- Baseline: today's bookings persist into tomorrow
- Artefacts: pickled pipeline + metrics JSON under `lakehouse/models/`
"""

from __future__ import annotations

import json
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from lakehouse.config import (
    SCHEMA_GOLD,
    TBL_GLD_TRAINING_SET,
    MODELS_DIR,
    RANDOM_SEED,
)
from lakehouse.utils.db import get_connection, ensure_schemas
from lakehouse.ml.validation import (
    assert_prospective_features,
    bootstrap_interval,
    split_by_date,
    split_metadata,
)


def load_training_dataframe() -> pd.DataFrame:
    """Load the supervised learning table from DuckDB into pandas."""
    con = get_connection()
    ensure_schemas(con)
    df = con.execute(f"SELECT * FROM {SCHEMA_GOLD}.{TBL_GLD_TRAINING_SET}").df()
    con.close()
    return df


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    return split_by_date(df, test_fraction=test_fraction)


def train() -> dict:
    """Train the model and persist artefacts and basic metrics."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_dataframe()

    target_col = "target_next_day_bookings"
    feature_cols = [
        "impressions_asof",
        "clicks_asof",
        "spend_asof",
        "ctr_asof",
        "cpm_asof",
        "frequency_asof",
        "planned_spend_asof",
        "actual_spend_asof",
        "budget_utilization_asof",
        "pacing_status_asof",
        "soft_conversions_asof",
        "revenue_asof",
        "bookings_asof",
        "bookings_trailing_3d",
        "bookings_trailing_7d",
        "budget_utilization_trailing_7d",
    ]
    assert_prospective_features(feature_cols)

    if df.empty:
        raise RuntimeError("Training set is empty. Ensure gold transformations have run.")

    # Time-based split rather than random split for a more realistic eval
    train_df, test_df = time_based_split(df, test_fraction=0.2)
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(float)
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(float)

    numeric_features = feature_cols
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        objective="reg:squarederror",
        n_jobs=4,
    )

    pipeline = Pipeline(steps=[("pre", preprocessor), ("model", model)])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    baseline_preds = X_test["bookings_asof"].astype(float)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    skill_over_baseline = None if baseline_mae == 0 else 1 - (mae / baseline_mae)

    artefact_prefix = MODELS_DIR / "bookings_xgb"
    with open(f"{artefact_prefix}.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    importances = pipeline.named_steps["model"].feature_importances_.tolist()

    meta = {
        "model": "XGBRegressor",
        "target": target_col,
        "prediction_contract": {
            "as_of": "end of as_of_date",
            "horizon": "next calendar day",
            "unit": "ad set x day",
        },
        "features": feature_cols,
        "importances": importances,
        "baseline": "persistence: bookings_asof",
        "metrics": {
            "mae": float(mae),
            "mae_95pct_bootstrap": bootstrap_interval(abs(y_test.to_numpy() - preds)),
            "baseline_mae": float(baseline_mae),
            "baseline_mae_95pct_bootstrap": bootstrap_interval(abs(y_test.to_numpy() - baseline_preds.to_numpy())),
            "skill_over_baseline": None if skill_over_baseline is None else float(skill_over_baseline),
        },
        "split": split_metadata(train_df, test_df),
        "leakage_checks": {
            "target_date_after_as_of_date": bool((pd.to_datetime(df["target_date"]) > pd.to_datetime(df["as_of_date"])).all()),
            "target_columns_excluded": True,
            "dates_disjoint": True,
        },
    }
    with open(f"{artefact_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    results = train()
    print(json.dumps(results, indent=2))
