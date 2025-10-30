"""Pipeline orchestrator for local lakehouse run (raw→bronze→silver→gold→dq→models)."""

from __future__ import annotations

# As a CS undergraduate style, we provide a simple orchestrator for local runs.

from lakehouse.ingest.raw_to_bronze import ingest as step_ingest
from lakehouse.transform.bronze_to_silver import transform as step_bronze_to_silver
from lakehouse.transform.silver_to_gold import transform as step_silver_to_gold
from lakehouse.quality.compute_dq import run as step_dq
from lakehouse.ml.train_model import train as step_train_reg
from lakehouse.ml.train_underpacing import train as step_train_clf


def main() -> None:
    print("[1/6] Ingesting raw → bronze …")
    step_ingest()
    print("[2/6] Transforming bronze → silver …")
    step_bronze_to_silver()
    print("[3/6] Transforming silver → gold …")
    step_silver_to_gold()
    print("[4/6] Computing data quality report …")
    step_dq()
    print("[5/6] Training regression model (bookings) …")
    reg_meta = step_train_reg()
    print("Regression complete. Metrics:")
    print(reg_meta["metrics"])
    print("[6/6] Training classification model (under-pacing) …")
    clf_metrics = step_train_clf()
    print("Classification complete. Metrics:")
    print(clf_metrics)


if __name__ == "__main__":
    main()
