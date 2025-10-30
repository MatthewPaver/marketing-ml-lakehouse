from __future__ import annotations

import os
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
DB_PATH = PROJECT_ROOT / "lakehouse.duckdb"

SCHEMAS = ["bronze", "silver", "gold"]


def log(msg: str) -> None:
    print(f"[init] {msg}")


def ensure_dirs() -> None:
    for d in [DATA_DIR, RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").write_text("") if not any(d.iterdir()) else None
    log("folders ensured: data/{raw,bronze,silver,gold}")


def ensure_db_and_schemas() -> None:
    con = duckdb.connect(str(DB_PATH))
    for s in SCHEMAS:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
    con.close()
    log(f"duckdb at {DB_PATH}")
    log("schemas ensured: bronze, silver, gold")


def main() -> None:
    log("initialising lakehouse…")
    ensure_dirs()
    ensure_db_and_schemas()
    log("done.")


if __name__ == "__main__":
    main()
