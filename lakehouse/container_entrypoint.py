"""Check local output access before starting a non-root container command.

This never repairs permissions or changes ownership on a caller's bind mount.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys


OUTPUTS = (
    "lakehouse/lakehouse.duckdb",
    "lakehouse/artifacts/contract_report.json",
    "lakehouse/artifacts/quality_report.json",
    "lakehouse/models/bookings_xgb.pkl",
    "lakehouse/models/bookings_xgb.json",
    "lakehouse/models/underpacing_xgb.pkl",
    "lakehouse/models/underpacing_xgb.json",
    "docs/generated/evidence.json",
)


def output_errors(root: Path) -> list[str]:
    """Check existing outputs and nearest existing parents without writing."""
    errors = []
    root = root.resolve()
    for relative in OUTPUTS:
        path = root / relative
        dangling = False
        for component in (path, *path.parents):
            if component == root:
                break
            if component.is_symlink() and not component.exists():
                dangling = True
                break
        if dangling:
            errors.append(f"{relative}: dangling symlink needs operator attention")
            continue
        if not path.resolve().is_relative_to(root):
            errors.append(f"{relative}: output resolves outside project")
            continue
        if path.exists() and (
            not path.is_file() or not os.access(path, os.R_OK | os.W_OK)
        ):
            errors.append(f"{relative}: existing output is not a readable, writable file")
        parent = path.parent
        while not parent.exists() and parent != root:
            parent = parent.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            errors.append(f"{relative}: parent {parent} does not allow output creation")
    return errors


def main() -> int:
    if os.geteuid() == 0:
        print("Refusing root. Rebuild with your non-root LOCAL_UID/LOCAL_GID.", file=sys.stderr)
        return 2
    errors = output_errors(Path(__file__).resolve().parents[1])
    if errors:
        print("Output permissions need operator attention; no ownership was changed:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 2
    if len(sys.argv) < 2:
        print("A container command is required.", file=sys.stderr)
        return 2
    os.execvp(sys.argv[1], sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
