from __future__ import annotations

import duckdb
from pathlib import Path
from typing import Iterable, Any

from lakehouse.config import DB_FILE, SCHEMA_BRONZE, SCHEMA_SILVER, SCHEMA_GOLD

# As a CS undergraduate style, we create small, focused helper functions.


def get_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection to the lakehouse file, creating it as needed."""
    database_path: Path = db_path if db_path is not None else DB_FILE
    database_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database_path))
    return con


def ensure_schemas(con: duckdb.DuckDBPyConnection) -> None:
    """Create schemas if they do not exist. Idempotent operation."""
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_BRONZE}")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_SILVER}")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_GOLD}")


def register_dataframe(con: duckdb.DuckDBPyConnection, df: Any, view_name: str) -> None:
    """Register a pandas DataFrame as a DuckDB view for SQL operations."""
    con.register(view_name, df)


def create_or_replace_from_view(
    con: duckdb.DuckDBPyConnection,
    qualified_table: str,
    view_name: str,
) -> None:
    """Create or replace a table from a registered view. Safe to re-run."""
    con.execute(f"CREATE OR REPLACE TABLE {qualified_table} AS SELECT * FROM {view_name}")


def execute_many(con: duckdb.DuckDBPyConnection, statements: Iterable[str]) -> None:
    """Execute multiple SQL statements sequentially for convenience."""
    for stmt in statements:
        con.execute(stmt)
