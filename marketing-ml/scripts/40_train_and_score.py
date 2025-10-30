from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import duckdb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from utils.timeutils import time_based_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "lakehouse.duckdb"
MODEL_DIR = PROJECT_ROOT / "data" / "gold"
MODEL_PATH = MODEL_DIR / "model_xgb.pkl"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.json"

BASE_FEATURES = ["ctr", "conv_7d", "clicks", "impressions", "spend", "aud_daily_budget", "pacing_ratio"]
TARGET = "label_under_pacing_next"


def log(msg: str) -> None:
    print(f"[train] {msg}")


def load_df() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM gold.training_dataset").df()
    con.close()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=[TARGET])


def build_pipeline(features: list[str], pos_weight: float | None) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), features)
    ])
    clf_base = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
        eval_metric="logloss",
        scale_pos_weight=pos_weight if pos_weight and pos_weight > 0 else 1.0,
    )
    clf = CalibratedClassifierCV(clf_base, method="isotonic", cv=3)
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    df = load_df()

    # Features: include revenue if present
    features = BASE_FEATURES.copy()
    if "revenue" in df.columns:
        features = [*features, "revenue"]

    # Time-based split for holdout validation
    train_mask, valid_mask, cutoff = time_based_split(df, "date", train_frac=0.8)
    log(f"cutoff date (train ≤ T, valid > T): {cutoff.date()}")

    df_train, df_valid = df.loc[train_mask].copy(), df.loc[valid_mask].copy()

    X_train, y_train = df_train[features], df_train[TARGET].astype(int).values
    X_valid, y_valid = df_valid[features], df_valid[TARGET].astype(int).values

    # Class imbalance handling
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    label_rate = float(pos / max(pos + neg, 1.0))
    pos_weight = (neg / max(pos, 1.0)) if pos > 0 else 1.0
    log(f"class ratio (train): pos={pos:.0f}, neg={neg:.0f}, scale_pos_weight={pos_weight:.2f}")

    # CV on train only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs: list[float] = []
    for tr_idx, te_idx in skf.split(X_train, y_train):
        pipe = build_pipeline(features, pos_weight)
        pipe.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        proba = pipe.predict_proba(X_train.iloc[te_idx])[:, 1]
        if len(np.unique(y_train[te_idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_train[te_idx], proba))
    auc_mean = float(np.mean(aucs)) if aucs else float("nan")
    auc_std = float(np.std(aucs)) if aucs else float("nan")
    log(f"CV AUC mean±std (train): {auc_mean:.3f}±{auc_std:.3f} over {len(aucs)} folds")

    # Fit final calibrated model on full train
    final_pipe = build_pipeline(features, pos_weight)
    final_pipe.fit(X_train, y_train)

    # Evaluate on holdout
    if len(np.unique(y_valid)) >= 2 and len(df_valid) > 0:
        holdout_auc = roc_auc_score(y_valid, final_pipe.predict_proba(X_valid)[:, 1])
        log(f"holdout AUC (valid): {holdout_auc:.3f} on {len(df_valid)} rows")

    # Persist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Derive simple importances from a direct XGB fit (for presentation)
    imp_model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
        eval_metric="logloss",
        scale_pos_weight=pos_weight if pos_weight and pos_weight > 0 else 1.0,
    )
    imp_model.fit(X_train, y_train)
    importances = {f: float(w) for f, w in zip(features, getattr(imp_model, 'feature_importances_', np.zeros(len(features))))}
    joblib.dump({
        "model": final_pipe,
        "metrics": {
            "cv_auc_mean": auc_mean,
            "cv_auc_std": auc_std,
            "label_rate_train": label_rate,
            "train_start": str(df_train["date"].min()),
            "train_end": str(df_train["date"].max()),
            "valid_start": str(df_valid["date"].min()) if len(df_valid)>0 else None,
            "valid_end": str(df_valid["date"].max()) if len(df_valid)>0 else None,
            "calibrated": True
        },
        "feature_importances": importances
    }, MODEL_PATH)
    FEATURE_LIST_PATH.write_text(json.dumps({"features": features}, indent=2))
    log(f"model saved: {MODEL_PATH}")

    # Score all rows
    con = duckdb.connect(str(DB_PATH))
    feat_cols = ", ".join(features)
    df_all = con.execute(f"SELECT ad_set_id, date, {feat_cols} FROM gold.training_dataset").df()
    proba = final_pipe.predict_proba(df_all[features])[:, 1]
    out = pd.DataFrame({"ad_set_id": df_all["ad_set_id"], "date": df_all["date"], "under_pacing_risk": proba})
    con.register("v_scores", out)
    con.execute("CREATE OR REPLACE TABLE gold.scored_adsets AS SELECT * FROM v_scores")
    con.close()
    log("scored_adsets written.")


if __name__ == "__main__":
    main()
