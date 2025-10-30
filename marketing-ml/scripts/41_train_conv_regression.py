from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import duckdb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

from utils.timeutils import time_based_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "lakehouse.duckdb"
MODEL_DIR = PROJECT_ROOT / "data" / "gold"
MODEL_PATH = MODEL_DIR / "model_conv_xgb.pkl"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list_conv.json"

FEATURES = ["ctr", "conv_7d", "clicks", "impressions", "spend", "aud_daily_budget", "pacing_ratio"]
TARGET = "conv_next7d_rate"


def log(msg: str) -> None:
    print(f"[train_conv] {msg}")


def load_df() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM gold.training_dataset").df()
    con.close()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET])
    return df


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), FEATURES)
    ])
    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=4,
    )
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    df = load_df()

    # time-based split
    train_mask, valid_mask, cutoff = time_based_split(df, "date", train_frac=0.8)
    log(f"cutoff date (train ≤ T, valid > T): {cutoff.date()}")
    df_train, df_valid = df.loc[train_mask].copy(), df.loc[valid_mask].copy()

    X_train, y_train = df_train[FEATURES], df_train[TARGET].astype(float).values
    X_valid, y_valid = df_valid[FEATURES], df_valid[TARGET].astype(float).values

    # CV on train with folds by stratifying binned target to stabilise
    bins = np.clip(np.digitize(y_train, bins=np.quantile(y_train, [0, .25, .5, .75, 1])), 1, 4)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    r2s: list[float] = []
    maes: list[float] = []
    for tr_idx, te_idx in skf.split(X_train, bins):
        pipe = build_pipeline()
        pipe.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        preds = pipe.predict(X_train.iloc[te_idx])
        r2s.append(r2_score(y_train[te_idx], preds))
        maes.append(mean_absolute_error(y_train[te_idx], preds))
    log(f"CV R² mean±std: {np.mean(r2s):.3f}±{np.std(r2s):.3f}; MAE mean±std: {np.mean(maes):.4f}±{np.std(maes):.4f}")

    # fit final
    final_pipe = build_pipeline()
    final_pipe.fit(X_train, y_train)

    # holdout eval
    preds_valid = final_pipe.predict(X_valid)
    log(f"holdout R²: {r2_score(y_valid, preds_valid):.3f}; MAE: {mean_absolute_error(y_valid, preds_valid):.4f} on {len(df_valid)} rows")

    # persist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Derive simple importances from a direct XGB fit (for presentation)
    imp_reg = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=4,
    )
    imp_reg.fit(X_train, y_train)
    importances = {f: float(w) for f, w in zip(FEATURES, getattr(imp_reg, 'feature_importances_', np.zeros(len(FEATURES))))}
    joblib.dump({
        "model": final_pipe,
        "metrics": {"cv_r2_mean": float(np.mean(r2s)), "cv_r2_std": float(np.std(r2s)), "cv_mae_mean": float(np.mean(maes)), "cv_mae_std": float(np.std(maes))},
        "feature_importances": importances
    }, MODEL_PATH)
    FEATURE_LIST_PATH.write_text(json.dumps({"features": FEATURES}, indent=2))

    # score
    con = duckdb.connect(str(DB_PATH))
    df_all = con.execute(f"SELECT ad_set_id, date, {', '.join(FEATURES)} FROM gold.training_dataset").df()
    conv_prob = final_pipe.predict(df_all[FEATURES])
    out = pd.DataFrame({"ad_set_id": df_all["ad_set_id"], "date": df_all["date"], "conv_prob": conv_prob})
    con.register("v_conv_scores", out)
    con.execute("CREATE OR REPLACE TABLE gold.scored_conversions AS SELECT * FROM v_conv_scores")
    con.close()
    log("scored_conversions written.")


if __name__ == "__main__":
    main()
