from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class AgentSignal:
    agent: str
    status: str
    headline: str
    evidence: list[str]
    actions: list[str]


def _pct_delta(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if current == 0 else 1.0
    return (current - baseline) / abs(baseline)


def data_quality_agent(rows: Iterable[dict]) -> AgentSignal:
    records = list(rows)
    total = len(records)
    missing_cells = sum(1 for row in records for value in row.values() if value in {None, ""})
    duplicate_keys = total - len({(row.get("campaign_id"), row.get("date")) for row in records})
    status = "pass" if missing_cells == 0 and duplicate_keys == 0 else "review"

    return AgentSignal(
        agent="data_quality_agent",
        status=status,
        headline="Marketing input quality is clean" if status == "pass" else "Marketing input needs review",
        evidence=[
            f"{total} campaign rows inspected",
            f"{missing_cells} missing cells",
            f"{duplicate_keys} duplicate campaign/date keys",
        ],
        actions=[] if status == "pass" else ["Fix missing values or duplicate campaign/date keys before model training"],
    )


def feature_drift_agent(current: dict[str, float], baseline: dict[str, float], threshold: float = 0.25) -> AgentSignal:
    drifted = []
    for feature, current_value in current.items():
        if feature not in baseline:
            continue
        delta = _pct_delta(float(current_value), float(baseline[feature]))
        if abs(delta) >= threshold:
            drifted.append((feature, delta))

    status = "review" if drifted else "pass"
    return AgentSignal(
        agent="feature_drift_agent",
        status=status,
        headline="Feature drift detected" if drifted else "No material feature drift detected",
        evidence=[f"{name}: {delta:+.0%}" for name, delta in drifted] or ["All tracked features are within threshold"],
        actions=["Rebuild training set and compare model metrics before trusting new scores"] if drifted else [],
    )


def campaign_insight_agent(rows: Iterable[dict]) -> AgentSignal:
    records = list(rows)
    if not records:
        return AgentSignal("campaign_insight_agent", "review", "No campaign rows available", [], ["Run the pipeline first"])

    def roas(row: dict) -> float:
        spend = float(row.get("spend") or 0)
        return float(row.get("revenue") or 0) / spend if spend else 0.0

    ranked = sorted(records, key=roas, reverse=True)
    best = ranked[0]
    avg_roas = mean(roas(row) for row in records)
    underpacing = [row for row in records if float(row.get("spend") or 0) < float(row.get("budget") or 0) * 0.7]

    return AgentSignal(
        agent="campaign_insight_agent",
        status="review" if underpacing else "pass",
        headline=f"Best campaign is {best.get('campaign_id', 'unknown')} at {roas(best):.2f} ROAS",
        evidence=[
            f"Average ROAS: {avg_roas:.2f}",
            f"{len(underpacing)} campaigns are materially underpacing budget",
        ],
        actions=["Investigate underpacing campaigns before scaling budget"] if underpacing else ["Use best performers as creative/audience benchmarks"],
    )


def model_risk_reviewer(metrics: dict[str, float], minimum_auc: float = 0.65, maximum_error: float = 0.35) -> AgentSignal:
    auc = float(metrics.get("auc", metrics.get("roc_auc", 0.0)))
    error = float(metrics.get("error", metrics.get("mae", 0.0)))
    issues = []
    if auc and auc < minimum_auc:
        issues.append(f"AUC {auc:.2f} is below {minimum_auc:.2f}")
    if error and error > maximum_error:
        issues.append(f"Error {error:.2f} is above {maximum_error:.2f}")

    return AgentSignal(
        agent="model_risk_reviewer",
        status="review" if issues else "pass",
        headline="Model risk needs review" if issues else "Model metrics are within release thresholds",
        evidence=issues or [f"AUC {auc:.2f}", f"Error {error:.2f}"],
        actions=["Hold recommendations until model drift and error are explained"] if issues else ["Publish scores with monitoring notes"],
    )


def run_marketing_agent_review(
    rows: Iterable[dict],
    current_features: dict[str, float],
    baseline_features: dict[str, float],
    metrics: dict[str, float],
) -> list[AgentSignal]:
    campaign_rows = list(rows)
    return [
        data_quality_agent(campaign_rows),
        feature_drift_agent(current_features, baseline_features),
        campaign_insight_agent(campaign_rows),
        model_risk_reviewer(metrics),
    ]
