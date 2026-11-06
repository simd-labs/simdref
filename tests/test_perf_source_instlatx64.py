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


class TestParseDumpEdges(unittest.TestCase):
    def test_tags_junk_fused_and_build(self):
        text = (
            'CPU#000 Type: "Intel(R) Core(TM) Ultra 9 285K"\n'
            "Instruction Latency: Used CPUs:  1 ProcMask: 0x0000000000000001\n"
            "aida_bench64.dll build: 4.7.912.8 Dec 20 2024 12:30:07\n"
            "  1 X86 :{EVEX} VPMOVZXBW xmm, xmm L: 1.0ns= 1.0c T: 0.5ns= 0.50c\n"
            "  2 X86 :2x MOV r64, r64 L: [no true dep.] T: 0.5ns= 0.50c\n"
            "  3 X86 :V{EVEX} PMOVZXBW xmm, xmm L: 1.0ns= 1.0c T: 0.5ns= 0.50c\n"
            "  4 X86 :MOV r64, r64+ADD r64, r64 L: 1.0ns= 1.0c T: 0.5ns= 0.50c\n"
        )
        rows, stats = parse_dump(text, citation_url="https://example.test/edge")
        self.assertEqual(stats, {"rows": 4, "fused": 1, "dropped": 0})
        self.assertEqual(
            [(row.mnemonic, row.form) for row in rows],
            [("VPMOVZXBW", "xmm, xmm"), ("MOV", ""), ("VPMOVZXBW", "xmm, xmm")],
        )
        self.assertEqual(rows[0].core, "ARL-P")
        self.assertEqual(rows[0].source_version, "aida_bench64.dll build: 4.7.912.8 Dec 20 2024")
        self.assertEqual((rows[1].latency, rows[1].cpi), ("", "0.50"))


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
