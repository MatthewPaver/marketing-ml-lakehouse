"""Classification model for under-pacing risk.

- Label: 1 if pacing_status == 'under_pacing', else 0
- Features: non-leaky numeric features (exclude planned/actual/budget_utilization)
- Time-based split (train early dates, test later dates)
- Artefacts: pickled pipeline + metrics/metadata JSON under `lakehouse/models/`
"""

from __future__ import annotations

import json
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from lakehouse.config import (
    SCHEMA_GOLD,
    TBL_GLD_DAILY_METRICS,
    MODELS_DIR,
    RANDOM_SEED,
)
from lakehouse.utils.db import get_connection, ensure_schemas


def load_dataframe() -> pd.DataFrame:
    con = get_connection()
    ensure_schemas(con)
    df = con.execute(f"SELECT * FROM {SCHEMA_GOLD}.{TBL_GLD_DAILY_METRICS}").df()
    con.close()
    return df


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    df_sorted = df.sort_values("date")
    split_idx = int(len(df_sorted) * (1 - test_fraction))
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def train() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataframe()
    if df.empty:
        raise RuntimeError("Gold metrics empty. Run transforms first.")

    df["label_under_pacing"] = (df["pacing_status"].astype(str) == "under_pacing").astype(int)

    # Non-leaky features: exclude planned_spend, actual_spend, budget_utilization
    feature_cols = [
        "impressions",
        "clicks",
        "spend",
        "ctr",
        "cpm",
        "frequency",
        "soft_conversions",
        "revenue",
    ]
    target_col = "label_under_pacing"

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

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, proba)),
    }

    # Persist artefacts
    artefact_prefix = MODELS_DIR / "underpacing_xgb"
    with open(f"{artefact_prefix}.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    # Feature importances from fitted XGB model
    importances = pipeline.named_steps["model"].feature_importances_.tolist()

    with open(f"{artefact_prefix}.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "features": feature_cols, "importances": importances}, f, indent=2)

    return metrics


if __name__ == "__main__":
    results = train()
    print(json.dumps(results, indent=2))
