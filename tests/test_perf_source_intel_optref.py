"""Tests for the Intel Optimization Reference Manual perf-source parser."""

from __future__ import annotations

import unittest
from pathlib import Path

from simdref.perf_sources import intel_optref
from simdref.perf_sources.cores import canonical_core_id
from simdref.perf_sources.merge import PerfRow

FIXTURE = Path(__file__).parent / "fixtures" / "intel_optref" / "optref_tables_sample.txt"


def _by_key(rows: list[PerfRow]) -> dict[tuple[str, str], PerfRow]:
    return {(row.mnemonic, row.core): row for row in rows}


class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.stats = intel_optref.parse_text(FIXTURE.read_text(encoding="utf-8"))
        cls.by_key = _by_key(cls.rows)

    def test_exact_values_skl(self):
        row = self.by_key[("ADDPS", "SKL")]
        self.assertEqual(row.latency, "4")
        self.assertEqual(row.cpi, "0.5")

    def test_shared_column_duplicates_to_snb_and_ivb(self):
        for core in ("SNB", "IVB"):
            row = self.by_key[("ADDPS", core)]
            self.assertEqual(row.latency, "3")
            self.assertEqual(row.cpi, "1")

    def test_range_and_parenthesised_values_kept_verbatim(self):
        row = self.by_key[("DIVPS", "IVB")]
        self.assertEqual(row.latency, "10-14")
        self.assertEqual(row.cpi, "14(6)")

    def test_wrapped_row(self):
        row = self.by_key[("MOVMSKPS", "SKL")]
        self.assertEqual(row.latency, "2")
        self.assertEqual(row.cpi, "1")

    def test_footnote_digit_stripped_and_dash_latency_blank(self):
        row = self.by_key[("LFENCE", "SKL")]
        self.assertEqual(row.latency, "")
        self.assertEqual(row.cpi, "6")

    def test_throughput_only_row(self):
        row = self.by_key[("CLC", "SKL")]
        self.assertEqual(row.latency, "")
        self.assertEqual(row.cpi, "0.25")

    def test_atom_shared_column_and_slashed_names(self):
        row = self.by_key[("BSWAP", "GLM")]
        self.assertEqual(row.latency, "1")
        self.assertEqual(row.cpi, "1")
        self.assertIn(("BSWAP", "GLP"), self.by_key)
        self.assertIn(("ADC", "GLP"), self.by_key)
        self.assertIn(("SBB", "GLM"), self.by_key)

    def test_row_fields(self):
        row = self.by_key[("ADDPS", "SKL")]
        self.assertEqual(row.source, "intel-optref")
        self.assertEqual(row.source_kind, "measured")
        self.assertEqual(row.source_version, intel_optref.SOURCE_VERSION)
        self.assertEqual(row.citation_url, intel_optref.CITATION_URL)
        self.assertEqual(row.architecture, "x86")
        self.assertEqual(row.form, "xmm, xmm")

    def test_every_core_id_resolves(self):
        self.assertTrue(self.rows)
        for row in self.rows:
            self.assertEqual(canonical_core_id(row.core), row.core)


class VendoredTests(unittest.TestCase):
    def test_load_vendored_returns_rows(self):
        rows = intel_optref.load_vendored()
        self.assertTrue(rows)
        self.assertTrue(all(isinstance(row, PerfRow) for row in rows))
        for row in rows:
            self.assertEqual(canonical_core_id(row.core), row.core)


if __name__ == "__main__":
    unittest.main()
