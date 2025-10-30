from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    "scripts/00_init_lakehouse.py",
    "scripts/10_bronze_ingest.py",
    "scripts/20_silver_transform.py",
    "scripts/30_gold_features.py",
    "scripts/35_gold_labels.py",
    "scripts/40_train_and_score.py",
    "scripts/41_train_conv_regression.py",
]


def run(step: str) -> None:
    print(f"[run] {step}")
    res = subprocess.run([sys.executable, str(PROJECT_ROOT / step)])
    if res.returncode != 0:
        raise SystemExit(f"Step failed: {step}")


def main() -> None:
    for s in STEPS:
        run(s)
    print("[run] all steps complete")


if __name__ == "__main__":
    main()
