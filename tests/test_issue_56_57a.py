"""Issues #56 and #57a: A64 summaries from AARCHMRS data, B.cond rows.

The fixture carries real nodes from AARCHMRS 2025-09 (MOVI, FMLA, LD4, LDP,
B.cond, B) plus one operation title for MOVI, the only narrative field the
payload can carry.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from simdref.annotate import AnnotateOptions, annotate_stream
from simdref.arm_instructions import parse_arm_instruction_payload
from simdref.models import Catalog
from simdref.storage import build_sqlite, load_instructions_by_mnemonic_from_db, open_db

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "issue_56" / "arm_a64_instructions.json"

EXPECTED_SUMMARIES = {
    "MOVI": "Move Immediate.",
    "FMLA": "Floating-point fused Multiply-Add.",
    "LD4": "Load Multiple Structures.",
    "LDP": "Load Pair of Registers.",
}

ISSUE_57_FIVE = {"eq", "ne", "lt", "hs", "lo"}
ALL_CONDITION_CODES = {
    "eq",
    "ne",
    "hs",
    "cs",
    "lo",
    "cc",
    "mi",
    "pl",
    "vs",
    "vc",
    "hi",
    "ls",
    "ge",
    "lt",
    "gt",
    "le",
    "al",
    "nv",
}


def _records() -> list:
    return parse_arm_instruction_payload(FIXTURE.read_text())


_TMPDIRS: list = []


def _memory_db(records) -> sqlite3.Connection:
    """Records in an in-memory catalog db, so annotate uses the mnemonic
    matcher the way a schema-13 install does (no byte identity, issue #51)."""
    catalog = Catalog(
        intrinsics=[],
        instructions=sorted(
            records, key=lambda item: (item.architecture, item.mnemonic, item.form)
        ),
        sources=[],
        generated_at="2025-01-01T00:00:00+00:00",
    )
    tmpdir = tempfile.TemporaryDirectory()
    db_path = Path(tmpdir.name) / "catalog.db"
    build_sqlite(catalog, path=db_path)
    file_conn = open_db(path=db_path)
    mem = sqlite3.connect(":memory:")
    file_conn.backup(mem)
    file_conn.close()
    mem.row_factory = sqlite3.Row
    _TMPDIRS.append(tmpdir)  # keep the temp db alive for the process
    return mem


class A64SummaryTest(unittest.TestCase):
    """Issue #56: rows take the Arm ARM title, not a placeholder."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = _records()

    def test_issue_mnemonics_carry_arm_titles(self) -> None:
        seen = {record.mnemonic for record in self.records}
        self.assertTrue(set(EXPECTED_SUMMARIES) <= seen)
        for record in self.records:
            expected = EXPECTED_SUMMARIES.get(record.mnemonic)
            if expected:
                self.assertEqual(record.summary, expected, record.form)

    def test_operation_title_beats_the_mnemonic_table(self) -> None:
        # The fixture gives MOVI_advsimd an operation title; the walker takes
        # the data title when the payload carries one.
        movi = next(record for record in self.records if record.mnemonic == "MOVI")
        self.assertEqual(movi.summary, "Move Immediate.")

    def test_fallback_is_a_real_sentence(self) -> None:
        # A mnemonic without a data title or a table entry keeps a generated
        # sentence, not the old "<Title> instruction." placeholder.
        record = next(
            record for record in self.records if record.mnemonic == "B" and record.form == "B (#0)"
        )
        self.assertEqual(record.summary, "B is an Arm instruction.")
        for other in self.records:
            self.assertNotRegex(other.summary, r"^[A-Z][a-z0-9]* instruction\.$")


class BCondRowTest(unittest.TestCase):
    """Issue #57a: one B.cond row per condition code."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = _records()

    def test_one_row_per_condition_code(self) -> None:
        dotted = {record.mnemonic for record in self.records if record.mnemonic.startswith("B.")}
        self.assertEqual(dotted, {f"B.{code.upper()}" for code in ALL_CONDITION_CODES})
        self.assertTrue(ISSUE_57_FIVE <= {m.split(".")[1].lower() for m in dotted})
        for record in self.records:
            if record.mnemonic.startswith("B."):
                self.assertEqual(record.form, f"{record.mnemonic} (#0)")
                self.assertEqual(record.summary, "Branch conditionally.")
                self.assertEqual(record.architecture, "arm")
                self.assertEqual(record.enc_bits, 32)
                self.assertEqual((record.enc_mask, record.enc_match), (0xFF000010, 0x54000000))

    def test_unconditional_branch_row_survives(self) -> None:
        forms = [record.form for record in self.records if record.mnemonic == "B"]
        self.assertIn("B (#0)", forms)

    def test_mnemonic_lookup_is_case_insensitive(self) -> None:
        conn = _memory_db(self.records)
        try:
            rows = load_instructions_by_mnemonic_from_db(conn, "b.eq")
            self.assertEqual([row.mnemonic for row in rows], ["B.EQ"])
        finally:
            conn.close()


class A64AnnotateTest(unittest.TestCase):
    """End-to-end annotate on the issue repro lines."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.conn = _memory_db(_records())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn.close()

    def _annotate_json(self, lines: list[str]) -> dict[str, dict]:
        out = "".join(annotate_stream(lines, opts=AnnotateOptions(fmt="json"), conn=self.conn))
        records = json.loads(out)
        return {record["mnemonic"]: record for record in records}

    def test_issue_56_repro_lines_get_arm_summaries(self) -> None:
        lines = ["movi d0, #0", "fmla s0, s1, s2", "ld4 {v0.4s}, [x0]", "ldp x0, x1, [sp]"]
        by_mnemonic = self._annotate_json(lines)
        for mnemonic, summary in EXPECTED_SUMMARIES.items():
            record = by_mnemonic.get(mnemonic.lower())
            self.assertIsNotNone(record, mnemonic)
            self.assertTrue(record["known"], mnemonic)
            self.assertEqual(record["summary"], summary, mnemonic)

    def test_issue_57a_repro_lines_are_known(self) -> None:
        lines = [
            "b.eq 0x118 <dotn+0x118>",
            "b.ne 0x4c <dotn+0x4c>",
            "b.lt 0x20 <dotn+0x20>",
            "b.hs 0x28 <dotn+0x28>",
            "b.lo 0xe4 <dotn+0xe4>",
            "b 0xd4 <dotn+0xd4>",
        ]
        by_mnemonic = self._annotate_json(lines)
        self.assertEqual(set(by_mnemonic), {"b.eq", "b.ne", "b.lt", "b.hs", "b.lo", "b"})
        for mnemonic, record in by_mnemonic.items():
            self.assertTrue(record["known"], mnemonic)
            if mnemonic == "b":
                self.assertEqual(record["summary"], "B is an Arm instruction.")
            else:
                self.assertEqual(record["summary"], "Branch conditionally.", mnemonic)


if __name__ == "__main__":
    unittest.main()
