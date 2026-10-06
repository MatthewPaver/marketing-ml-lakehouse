"""Pipeline orchestrator for local lakehouse run (contracts→layers→dq→models)."""

from __future__ import annotations

# As a CS undergraduate style, we provide a simple orchestrator for local runs.

from lakehouse.contracts import validate_contracts
from lakehouse.ingest.raw_to_bronze import ingest as step_ingest
from lakehouse.transform.bronze_to_silver import transform as step_bronze_to_silver
from lakehouse.transform.silver_to_gold import transform as step_silver_to_gold
from lakehouse.quality.compute_dq import run as step_dq
from lakehouse.ml.train_model import train as step_train_reg
from lakehouse.ml.train_underpacing import train as step_train_clf
from lakehouse.publish_pages import generate as step_publish_pages


def main() -> None:
    print("[1/7] Validating source contracts …")
    contract_report = validate_contracts()
    warning_count = sum(len(source["warnings"]) for source in contract_report["sources"])
    print(f"Contracts passed with {warning_count} warning(s).")
    print("[2/7] Ingesting raw → bronze …")
    step_ingest()
    print("[3/7] Transforming bronze → silver …")
    step_bronze_to_silver()
    print("[4/7] Transforming silver → gold …")
    step_silver_to_gold()
    print("[5/7] Computing data quality report …")
    step_dq()
    print("[6/7] Training regression model (bookings) …")
    reg_meta = step_train_reg()
    print("Regression complete. Metrics:")
    print(reg_meta["metrics"])
    print("[7/7] Training classification model (under-pacing) …")
    clf_metrics = step_train_clf()
    print("Classification complete. Metrics:")
    print(clf_metrics)
    print("Publishing generated Pages evidence …")
    step_publish_pages()


if __name__ == "__main__":
    main()
