from __future__ import annotations

import json
from pathlib import Path

import pytest

from lakehouse.contracts import validate_contracts


def test_committed_sources_satisfy_structural_contract(tmp_path: Path):
    report = validate_contracts(report_path=tmp_path / "report.json")
    assert report["status"] in {"pass", "warn"}
    assert len(report["sources"]) == 4
    assert all(source["sha256"] for source in report["sources"])
    assert all(source["rows"] > 0 for source in report["sources"])


def test_missing_column_fails_before_bronze(tmp_path: Path):
    data_file = tmp_path / "bad.csv"
    data_file.write_text("id,value\n1,2\n", encoding="utf-8")
    contract = {
        "version": "test",
        "sources": [
            {
                "name": "bad",
                "path": str(data_file),
                "required_columns": ["id", "event_time"],
                "required_non_null": ["id"],
                "key": ["id"],
            }
        ],
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="missing_column:event_time"):
        validate_contracts(contract_path=contract_path, report_path=tmp_path / "report.json")
