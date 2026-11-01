"""Issue #53 and #54: the form pick must weigh operand order and widths.

AT&T lists the immediate first while catalog form tokens are in Intel
order. The picker matched tokens by position, so every form scored 0 and
the first row won. Immediate width, register width and memory operands
must pick the row instead. The byte-identity path can mask this in the
full pipeline, so each repro asserts the picked record key directly.

Each test reads one repro fixture (the issue body, plus an Intel-syntax
twin) and checks the picked catalog row per line.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from simdref.annotate import LineKind, lookup, parse_asm_line, pick_record
from simdref.storage import SQLITE_PATH, open_db

_USER_CATALOG = Path.home() / ".local" / "share" / "simdref" / "catalog.db"
_catalog_env = os.environ.get("SIMDREF_CATALOG", "")
_missing = Path(tempfile.gettempdir()) / "simdref-test-no-catalog.db"
CATALOG = (
    Path(_catalog_env)
    if _catalog_env and Path(_catalog_env).is_file()
    else (_missing if _catalog_env else (SQLITE_PATH if SQLITE_PATH.is_file() else _USER_CATALOG))
)

FIXTURES = Path(__file__).parent / "fixtures" / "issue_53"

# The row each issue line must join, in fixture order. JMP (Rel32) is the
# near relative jump; the catalog has no rel-width hint in the text.
ISSUE53_ROWS = [
    "ADD (R64, I8)",
    "ADD (R32, I8)",
    "SUB (R32, I8)",
    "MOV (R32, I32)",
    "JMP (Rel32)",
]
ISSUE54_ROWS = [
    "VFMADD231PS (YMM, YMM, M256)",
    "VFMADD231PS (YMM, YMM, YMM)",
    "VPERMD (YMM, YMM, M256)",
    "VPERMD (YMM, YMM, YMM)",
]


class Issue5354FormPickTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CATALOG.is_file():
            raise unittest.SkipTest(f"simdref catalog missing at {CATALOG}")
        cls._conn = open_db(path=CATALOG)
        cls._conn.execute("PRAGMA query_only=1")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_conn", None) is not None:
            cls._conn.close()

    def _picked_rows(self, fixture_name: str) -> list[str]:
        lines = (FIXTURES / fixture_name).read_text().splitlines()
        rows: list[str] = []
        for line in lines:
            parsed = parse_asm_line(line)
            self.assertEqual(parsed.kind, LineKind.INSTRUCTION, line)
            record = pick_record(lookup(parsed.mnemonic, self._conn), operands=parsed.operands)
            self.assertIsNotNone(record, line)
            rows.append(record.key)
        return rows

    def test_issue53_att_immediate_widths(self):
        self.assertEqual(self._picked_rows("issue53_att.s"), ISSUE53_ROWS)

    def test_issue53_intel_twins(self):
        self.assertEqual(self._picked_rows("issue53_intel.s"), ISSUE53_ROWS)

    def test_issue54_att_memory_operands(self):
        self.assertEqual(self._picked_rows("issue54_att.s"), ISSUE54_ROWS)

    def test_issue54_intel_twins(self):
        self.assertEqual(self._picked_rows("issue54_intel.s"), ISSUE54_ROWS)


if __name__ == "__main__":
    unittest.main()
