"""Generate the static Pages evidence payload from rebuilt pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from lakehouse.config import PROJECT_ROOT, SCHEMA_GOLD, TBL_GLD_DAILY_METRICS
from lakehouse.utils.db import get_connection


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate() -> dict[str, object]:
    artifacts = PROJECT_ROOT / "lakehouse" / "artifacts"
    models = PROJECT_ROOT / "lakehouse" / "models"
    quality_path = artifacts / "quality_report.json"
    contract_path = artifacts / "contract_report.json"
    bookings_path = models / "bookings_xgb.json"
    pacing_path = models / "underpacing_xgb.json"
    con = get_connection()
    campaigns = con.execute(
        f"""SELECT ad_set_id AS id, MAX(ad_set_name) AS name, SUM(impressions)::BIGINT AS impressions,
        SUM(clicks)::BIGINT AS clicks, SUM(spend) AS spend, SUM(planned_spend) AS planned,
        SUM(revenue) AS revenue, SUM(bookings)::BIGINT AS conversions,
        SUM(CASE WHEN pacing_status='under_pacing' THEN 1 ELSE 0 END)::BIGINT AS under,
        COUNT(*)::BIGINT AS days FROM {SCHEMA_GOLD}.{TBL_GLD_DAILY_METRICS}
        GROUP BY ad_set_id ORDER BY ad_set_id"""
    ).df().to_dict(orient="records")
    con.close()
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaigns": campaigns,
        "quality": _read_json(quality_path),
        "contracts": _read_json(contract_path),
        "models": {"bookings": _read_json(bookings_path), "underpacing": _read_json(pacing_path)},
        "provenance": {
            "generator": "lakehouse.publish_pages",
            "quality_report_sha256": _sha256(quality_path),
            "contract_report_sha256": _sha256(contract_path),
            "bookings_model_sha256": _sha256(bookings_path),
            "underpacing_model_sha256": _sha256(pacing_path),
        },
    }
    output = PROJECT_ROOT / "docs" / "generated" / "evidence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(generate()["provenance"], indent=2))
