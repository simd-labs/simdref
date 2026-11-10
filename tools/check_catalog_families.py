#!/usr/bin/env python3
"""Fail when a catalog family lost every row.

A lost source (a vendor step that stopped running, a fetch that fell
back to an empty bundle) still builds a catalog that passes row-total
floors. This reads the built catalog and fails on any empty family.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

INTRINSIC_FAMILIES = ("x86", "arm", "riscv", "cuda")
INSTRUCTION_FAMILIES = ("ptx",)


def main() -> int:
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/derived/catalog.db")
    empty: list[str] = []
    with sqlite3.connect(db) as conn:
        for table, families in (
            ("intrinsics_data", INTRINSIC_FAMILIES),
            ("instructions_data", INSTRUCTION_FAMILIES),
        ):
            for family in families:
                count = conn.execute(
                    f"select count(*) from {table} where architecture = ?", (family,)
                ).fetchone()[0]
                print(f"{table} {family}: {count}")
                if not count:
                    empty.append(f"{table} {family}")
    if empty:
        print(f"empty catalog families: {', '.join(empty)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
