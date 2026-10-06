from __future__ import annotations

import pandas as pd

from lakehouse.public_data.ga4 import build_prospective_table


def test_ga4_profile_is_strictly_prospective():
    frame = pd.DataFrame(
        {
            "event_date": ["20210101", "20210102", "20210103"],
            "active_users": [10, 12, 8],
            "sessions": [11, 14, 9],
            "page_views": [30, 40, 21],
            "add_to_carts": [3, 4, 2],
            "purchases": [1, 2, 0],
            "purchase_revenue": [15.0, 40.0, 0.0],
        }
    )
    result = build_prospective_table(frame)
    assert len(result) == 2
    assert (result["target_date"] > result["as_of_date"]).all()
    assert result["target_next_day_purchases"].tolist() == [2.0, 0.0]
    assert result["purchases_asof"].tolist() == [1, 2]
