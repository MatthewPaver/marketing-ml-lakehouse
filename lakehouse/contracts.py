"""File-level data contracts for the raw landing zone."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lakehouse.config import PROJECT_ROOT


DEFAULT_CONTRACT = PROJECT_ROOT / "contracts" / "raw_sources.json"
DEFAULT_REPORT = PROJECT_ROOT / "lakehouse" / "artifacts" / "contract_report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(spec: dict[str, Any]) -> dict[str, Any]:
    configured = Path(spec["path"])
    path = configured if configured.is_absolute() else PROJECT_ROOT / configured
    if not path.exists():
        return {"name": spec["name"], "path": spec["path"], "status": "fail", "fatal": ["file_missing"]}

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)

    required = set(spec.get("required_columns") or [])
    missing_columns = sorted(required - set(columns))
    non_null = spec.get("required_non_null") or []
    null_counts = {
        column: sum(1 for row in rows if row.get(column) in (None, ""))
        for column in non_null
    }
    key = spec.get("key") or []
    key_counts = Counter(tuple(row.get(column) for column in key) for row in rows) if key else Counter()
    duplicate_key_rows = sum(count - 1 for count in key_counts.values() if count > 1)
    fatal = [f"missing_column:{column}" for column in missing_columns]
    if not rows:
        fatal.append("empty_source")
    fatal.extend(f"duplicate_column:{column}" for column, count in Counter(columns).items() if count > 1)
    fatal.extend(
        f"malformed_row:{index}"
        for index, row in enumerate(rows, start=2)
        if None in row or any(value is None for value in row.values())
    )
    warnings = [f"null_required:{column}:{count}" for column, count in null_counts.items() if count]
    if duplicate_key_rows:
        warnings.append(f"duplicate_key_rows:{duplicate_key_rows}")
    return {
        "name": spec["name"],
        "path": spec["path"],
        "status": "fail" if fatal else ("warn" if warnings else "pass"),
        "rows": len(rows),
        "columns": columns,
        "key": key,
        "sha256": _sha256(path),
        "fatal": fatal,
        "warnings": warnings,
    }


def validate_contracts(
    contract_path: Path = DEFAULT_CONTRACT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    sources = [validate_source(spec) for spec in contract["sources"]]
    report = {
        "contract_version": contract["version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "fail" if any(source.get("fatal") for source in sources) else (
            "warn" if any(source.get("warnings") for source in sources) else "pass"
        ),
        "sources": sources,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] == "fail":
        failures = [f"{source['name']}: {item}" for source in sources for item in source.get("fatal", [])]
        raise ValueError("Raw data contract failed: " + ", ".join(failures))
    return report


if __name__ == "__main__":
    result = validate_contracts()
    print(json.dumps(result, indent=2))
