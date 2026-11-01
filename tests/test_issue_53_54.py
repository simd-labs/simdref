"""Issue #53 and #54: the form pick must weigh operand order and widths.

AT&T lists the immediate first while catalog form tokens are in Intel
order. The picker matched tokens by position, so every form scored 0 and
the first row won. Immediate width, register width and memory operands
must pick the row instead. The byte-identity path can mask this in the
full pipeline, so each repro asserts the picked record key directly.

Each test reads one repro fixture (the issue body, plus an Intel-syntax
twin) and checks the picked catalog row per line. The catalog is a small
fixture DB: the tests once read the ambient user or worktree catalog at
import time, and which catalog that was depended on whether an earlier
test had bootstrapped ``data/derived/catalog.db``. That made the suite
order-dependent (JMP (Rel8) beat JMP (Rel32) on the stale user catalog).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simdref.annotate import LineKind, lookup, parse_asm_line, pick_record
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.storage import build_sqlite, open_db

FIXTURES = Path(__file__).parent / "fixtures" / "issue_53"


def _record(mnemonic: str, form: str) -> InstructionRecord:
    return InstructionRecord(
        mnemonic=mnemonic, form=form, summary=f"{form} row.", architecture="x86"
    )


# The candidate sets each repro line must discriminate, mirroring the live
# catalog rows. The JMP decoy is the Rel8 row that won on the stale user
# catalog when no run had bootstrapped the worktree catalog yet.
ROWS = [
    _record("ADD", "ADD (R64, I8)"),
    _record("ADD", "ADD (R64, R64)"),
    _record("ADD", "ADD (R64, I32)"),
    _record("ADD", "ADD (R32, I8)"),
    _record("SUB", "SUB (R32, I8)"),
    _record("SUB", "SUB (R32, R32)"),
    _record("MOV", "MOV (R32, I32)"),
    _record("MOV", "MOV (R32, R32)"),
    _record("JMP", "JMP (Rel8)"),
    _record("JMP", "JMP (Rel32)"),
    _record("VFMADD231PS", "VFMADD231PS (YMM, YMM, M256)"),
    _record("VFMADD231PS", "VFMADD231PS (YMM, YMM, YMM)"),
    _record("VPERMD", "VPERMD (YMM, YMM, M256)"),
    _record("VPERMD", "VPERMD (YMM, YMM, YMM)"),
]

# The row each issue line must join, in fixture order. JMP (Rel32) is the
# near relative jump; the text carries no rel-width hint, so the common
# near row wins over Rel8.
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
        cls._tmp = tempfile.TemporaryDirectory(prefix="simdref-issue53-")
        db = Path(cls._tmp.name) / "catalog.db"
        build_sqlite(
            Catalog(
                intrinsics=[],
                instructions=ROWS,
                sources=[
                    SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")
                ],
                generated_at="2025-01-01",
            ),
            db,
        )
        cls._conn = open_db(path=db)

    @classmethod
    def tearDownClass(cls):
        cls._conn.close()
        cls._tmp.cleanup()

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
