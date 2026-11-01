"""Tests for the InstLatX64 dump ingester (node C3)."""

import unittest
from pathlib import Path

from simdref.perf_sources.cores import canonical_core_id
from simdref.perf_sources.instlatx64 import load_vendored, parse_dump
from simdref.perf_sources.merge import SOURCE_INSTLATX64

_FIXTURE = Path(__file__).parent / "fixtures" / "instlatx64" / "zen1_sample.txt"


class TestParseDump(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = _FIXTURE.read_text(encoding="utf-8")
        cls.rows, cls.stats = parse_dump(text, citation_url="https://example.test/zen1")

    def test_row_count(self):
        self.assertEqual(len(self.rows), 3)
        self.assertEqual(self.stats["rows"], 3)
        self.assertEqual(self.stats["fused"], 0)

    def test_layout_a_row(self):
        row = self.rows[0]
        self.assertEqual(row.mnemonic, "MOV")
        self.assertEqual(row.form, "r64, [m64]")
        self.assertEqual(row.latency, "4.0")
        self.assertEqual(row.cpi, "0.50")
        self.assertEqual(row.core, "ZEN1")
        self.assertEqual(row.source, SOURCE_INSTLATX64)
        self.assertEqual(row.source_kind, "measured")
        self.assertEqual(row.source_version, "aida_bench64.dll build: 4.3.758.0 Jun 9 2017")
        self.assertEqual(row.citation_url, "https://example.test/zen1")
        self.assertEqual(row.architecture, "x86")
        self.assertEqual(row.applies_to, "mnemonic")

    def test_layout_b_row(self):
        row = self.rows[1]
        self.assertEqual(row.mnemonic, "VADDPS")
        self.assertEqual(row.form, "ymm, ymm, ymm")
        self.assertEqual(row.latency, "3.0")
        self.assertEqual(row.cpi, "0.50")
        self.assertEqual(row.core, "ZEN1")

    def test_marker_latency_row(self):
        row = self.rows[2]
        self.assertEqual(row.mnemonic, "NOP")
        self.assertEqual(row.form, "")
        self.assertEqual(row.latency, "")
        self.assertEqual(row.cpi, "0.124")

    def test_core_ids_resolve(self):
        for row in self.rows:
            self.assertEqual(canonical_core_id(row.core), row.core)


class TestLoadVendored(unittest.TestCase):
    def test_load_vendored_returns_rows(self):
        rows = load_vendored()
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertEqual(canonical_core_id(row.core), row.core)
            self.assertEqual(row.source, SOURCE_INSTLATX64)
            self.assertEqual(row.source_kind, "measured")


if __name__ == "__main__":
    unittest.main()
