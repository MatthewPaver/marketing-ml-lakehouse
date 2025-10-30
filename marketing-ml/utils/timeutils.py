from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def time_based_split(df: pd.DataFrame, date_col: str, train_frac: float = 0.8) -> Tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Return boolean index arrays for train/valid by sorted unique dates.

    - Ensures no leakage by splitting on dates, not rows.
    - Returns (train_mask, valid_mask, cutoff_date).
    """
    dates = pd.to_datetime(df[date_col])
    uniq = np.array(sorted(dates.unique()))
    if len(uniq) < 2:
        cutoff = uniq[0]
        mask_train = dates <= cutoff
        mask_valid = ~mask_train
        return mask_train.values, mask_valid.values, pd.to_datetime(cutoff)
    split_idx = max(1, int(len(uniq) * train_frac))
    cutoff = uniq[split_idx - 1]
    mask_train = dates <= cutoff
    mask_valid = dates > cutoff
    return mask_train.values, mask_valid.values, pd.to_datetime(cutoff)
