from lakehouse.agents import (
    campaign_insight_agent,
    data_quality_agent,
    feature_drift_agent,
    model_risk_reviewer,
    run_marketing_agent_review,
)


ROWS = [
    {"campaign_id": "c1", "date": "2026-05-01", "spend": 100, "revenue": 410, "budget": 300},
    {"campaign_id": "c2", "date": "2026-05-01", "spend": 250, "revenue": 500, "budget": 300},
]


def test_data_quality_agent_flags_duplicates_and_missing_values():
    signal = data_quality_agent([*ROWS, {"campaign_id": "c1", "date": "2026-05-01", "spend": None}])

    assert signal.status == "review"
    assert "duplicate" in " ".join(signal.evidence)


def test_feature_drift_agent_flags_large_changes():
    signal = feature_drift_agent({"cpc": 1.5, "ctr": 0.03}, {"cpc": 1.0, "ctr": 0.029})

    assert signal.status == "review"
    assert any("cpc" in item for item in signal.evidence)


def test_campaign_insight_agent_identifies_underpacing():
    signal = campaign_insight_agent(ROWS)

    assert signal.status == "review"
    assert "c1" in signal.headline


def test_model_risk_reviewer_blocks_weak_metrics():
    signal = model_risk_reviewer({"auc": 0.58, "mae": 0.18})

    assert signal.status == "review"
    assert signal.actions


def test_agent_review_returns_four_specialists():
    review = run_marketing_agent_review(ROWS, {"cpc": 1.0}, {"cpc": 1.0}, {"auc": 0.72, "mae": 0.2})

    assert [signal.agent for signal in review] == [
        "data_quality_agent",
        "feature_drift_agent",
        "campaign_insight_agent",
        "model_risk_reviewer",
    ]
