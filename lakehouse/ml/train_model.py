"""Model training on gold training set using XGBoost in an sklearn Pipeline.

- Target: daily bookings (`target_bookings`)
- Features: performance, budget, pacing index, soft conversions, revenue
- Artefacts: pickled pipeline + metrics JSON under `lakehouse/models/`
"""

from __future__ import annotations

import json
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
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


def load_training_dataframe() -> pd.DataFrame:
    """Load the supervised learning table from DuckDB into pandas."""
    con = get_connection()
    ensure_schemas(con)
    df = con.execute(f"SELECT * FROM {SCHEMA_GOLD}.{TBL_GLD_TRAINING_SET}").df()
    con.close()
    return df


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    df_sorted = df.sort_values("date")
    split_idx = int(len(df_sorted) * (1 - test_fraction))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    return train_df, test_df


def train() -> dict:
    """Train the model and persist artefacts and basic metrics."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_dataframe()

    target_col = "target_bookings"
    feature_cols = [
        "impressions",
        "clicks",
        "spend",
        "ctr",
        "cpm",
        "frequency",
        "planned_spend",
        "actual_spend",
        "budget_utilization",
        "pacing_status_idx",
        "soft_conversions",
        "revenue",
    ]

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
    r2 = r2_score(y_test, preds)

    artefact_prefix = MODELS_DIR / "bookings_xgb"
    with open(f"{artefact_prefix}.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    importances = pipeline.named_steps["model"].feature_importances_.tolist()

    meta = {
        "model": "XGBRegressor",
        "target": target_col,
        "features": feature_cols,
        "importances": importances,
        "metrics": {"mae": float(mae), "r2": float(r2)},
    }
    with open(f"{artefact_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    results = train()
    print(json.dumps(results, indent=2))
