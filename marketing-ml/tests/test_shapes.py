from __future__ import annotations

from pathlib import Path
import duckdb
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "lakehouse.duckdb"


def test_scored_adsets_exists():
    con = duckdb.connect(str(DB_PATH))
    tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='gold'").df()
    con.close()
    assert (tables["table_name"] == "scored_adsets").any(), "gold.scored_adsets missing"


def test_row_alignment_silver_gold():
    con = duckdb.connect(str(DB_PATH))
    silver_rows = con.execute("SELECT COUNT(*) FROM silver.meta_campaign_performance").fetchone()[0]
    gold_rows = con.execute("SELECT COUNT(*) FROM gold.training_dataset").fetchone()[0]
    con.close()
    assert gold_rows <= silver_rows, "gold rows should not exceed silver base"


def test_roas_calculation_tolerance():
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT revenue, spend, roas FROM gold.training_dataset WHERE spend>0 LIMIT 100").df()
    con.close()
    if not df.empty:
        approx = (df["revenue"] / df["spend"]).values
        assert np.allclose(approx, df["roas"].values, rtol=1e-6, atol=1e-6)


def test_probabilities_in_unit_interval():
    con = duckdb.connect(str(DB_PATH))
    if (con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='gold' AND table_name='scored_adsets'").fetchone()[0]):
        probs = con.execute("SELECT under_pacing_risk FROM gold.scored_adsets LIMIT 100").df()
        con.close()
        if not probs.empty:
            p = probs["under_pacing_risk"].values
            assert ((p >= 0) & (p <= 1)).all()
    else:
        con.close()


def test_no_future_aware_features():
    feat_file = PROJECT_ROOT / "data/gold/feature_list.json"
    if feat_file.exists():
        feats = json.loads(feat_file.read_text()).get("features", [])
        assert all(not f.startswith("next_") for f in feats), "feature list contains future-aware fields"


def test_silver_daily_unique_keys_and_roas():
    con = duckdb.connect(str(DB_PATH))
    # uniqueness
    dup = con.execute("SELECT COUNT(*) FROM (SELECT ad_set_id, date, COUNT(*) c FROM silver.silver_daily GROUP BY 1,2 HAVING c>1)").fetchone()[0]
    assert dup == 0
    # roas tolerance on sample where spend>0
    df = con.execute("SELECT revenue, spend, roas FROM silver.silver_daily WHERE spend>0 LIMIT 100").df()
    con.close()
    if not df.empty:
        import numpy as np
        assert np.allclose((df["revenue"]/df["spend"]).values, df["roas"].values, rtol=1e-6, atol=1e-6)
