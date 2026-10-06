"""Shared temporal-validation helpers for the prospective models."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd
import numpy as np


TARGET_COLUMNS = {
    "target_date",
    "target_next_day_bookings",
    "target_next_day_under_pacing",
}


def bootstrap_interval(values: Sequence[float], *, seed: int = 42, draws: int = 2000) -> list[float] | None:
    """Return a deterministic percentile interval for a sample mean."""
    sample = np.asarray(values, dtype=float)
    if sample.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(sample, size=(draws, sample.size), replace=True).mean(axis=1)
    return [float(bound) for bound in np.quantile(means, [0.025, 0.975])]


def assert_prospective_features(feature_cols: Sequence[str]) -> None:
    """Fail fast if a target or future-looking field reaches the feature set."""
    leaked = TARGET_COLUMNS.intersection(feature_cols)
    future_named = {name for name in feature_cols if name.startswith(("target_", "next_"))}
    if leaked or future_named:
        raise ValueError(f"Future/target fields cannot be features: {sorted(leaked | future_named)}")


def split_by_date(
    df: pd.DataFrame,
    *,
    date_col: str = "as_of_date",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on whole dates so a date can never appear in both partitions."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    dates = sorted(pd.to_datetime(df[date_col]).dropna().unique())
    if len(dates) < 2:
        raise ValueError("at least two distinct as-of dates are required")
    test_date_count = min(len(dates) - 1, max(1, math.ceil(len(dates) * test_fraction)))
    test_dates = set(dates[-test_date_count:])
    train = df[~pd.to_datetime(df[date_col]).isin(test_dates)].copy()
    test = df[pd.to_datetime(df[date_col]).isin(test_dates)].copy()
    return train.sort_values(date_col), test.sort_values(date_col)


def split_metadata(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    return {
        "strategy": "walk-forward holdout by unique as-of date",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": str(pd.to_datetime(train_df["as_of_date"]).min().date()),
        "train_end": str(pd.to_datetime(train_df["as_of_date"]).max().date()),
        "test_start": str(pd.to_datetime(test_df["as_of_date"]).min().date()),
        "test_end": str(pd.to_datetime(test_df["as_of_date"]).max().date()),
        "dates_disjoint": bool(
            set(pd.to_datetime(train_df["as_of_date"]))
            .isdisjoint(set(pd.to_datetime(test_df["as_of_date"])))
        ),
    }
