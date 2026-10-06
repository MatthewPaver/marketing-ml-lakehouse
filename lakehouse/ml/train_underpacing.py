"""Prospective classification model for next-day under-pacing risk.

- Decision point: end of ``as_of_date``
- Label: next calendar day's under-pacing state
- Baseline: today's state persists into tomorrow
- Artefacts: pickled pipeline + metrics/metadata JSON under `lakehouse/models/`
"""

from __future__ import annotations

import json
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from lakehouse.config import (
    SCHEMA_GOLD,
    TBL_GLD_TRAINING_SET,
    MODELS_DIR,
    RANDOM_SEED,
)
from lakehouse.utils.db import get_connection, ensure_schemas
from lakehouse.ml.validation import assert_prospective_features, bootstrap_interval, split_by_date, split_metadata


def load_dataframe() -> pd.DataFrame:
    con = get_connection()
    ensure_schemas(con)
    df = con.execute(f"SELECT * FROM {SCHEMA_GOLD}.{TBL_GLD_TRAINING_SET}").df()
    con.close()
    return df


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    return split_by_date(df, test_fraction=test_fraction)


def train() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataframe()
    if df.empty:
        raise RuntimeError("Gold metrics empty. Run transforms first.")

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
        "under_pacing_asof",
        "soft_conversions_asof",
        "revenue_asof",
        "bookings_asof",
        "bookings_trailing_3d",
        "bookings_trailing_7d",
        "budget_utilization_trailing_7d",
    ]
    target_col = "target_next_day_under_pacing"
    assert_prospective_features(feature_cols)

    train_df, test_df = time_based_split(df, test_fraction=0.2)
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)]
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=4,
        eval_metric="logloss",
    )

    pipeline = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    baseline_preds = X_test["under_pacing_asof"].astype(int)
    auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else None
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "auc": auc,
        "baseline_balanced_accuracy": float(balanced_accuracy_score(y_test, baseline_preds)),
        "baseline_f1": float(f1_score(y_test, baseline_preds, zero_division=0)),
        "positive_prevalence": float(y_test.mean()),
        "accuracy_95pct_bootstrap": bootstrap_interval((preds == y_test.to_numpy()).astype(float)),
        "baseline_accuracy_95pct_bootstrap": bootstrap_interval((baseline_preds.to_numpy() == y_test.to_numpy()).astype(float)),
    }

    # Persist artefacts
    artefact_prefix = MODELS_DIR / "underpacing_xgb"
    with open(f"{artefact_prefix}.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    # Feature importances from fitted XGB model
    importances = pipeline.named_steps["model"].feature_importances_.tolist()

    with open(f"{artefact_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "XGBClassifier",
                "target": target_col,
                "prediction_contract": {
                    "as_of": "end of as_of_date",
                    "horizon": "next calendar day",
                    "unit": "ad set x day",
                },
                "baseline": "persistence: under_pacing_asof",
                "metrics": metrics,
                "features": feature_cols,
                "importances": importances,
                "split": split_metadata(train_df, test_df),
                "leakage_checks": {
                    "target_date_after_as_of_date": bool((pd.to_datetime(df["target_date"]) > pd.to_datetime(df["as_of_date"])).all()),
                    "target_columns_excluded": True,
                    "dates_disjoint": True,
                },
            },
            f,
            indent=2,
        )

    return metrics


if __name__ == "__main__":
    results = train()
    print(json.dumps(results, indent=2))
