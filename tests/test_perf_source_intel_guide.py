"""Tests for the Intel Intrinsics Guide perf source ingester."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from simdref.perf_sources import canonical_core_id
from simdref.perf_sources.intel_guide import (
    CITATION_URL,
    SOURCE_VERSION,
    load_vendored,
    parse_notes_json,
    parse_perf_js,
)
from simdref.perf_sources.merge import SOURCE_INTEL_GUIDE

FIXTURE = Path(__file__).parent / "fixtures" / "intel_guide"
VENDOR = Path(__file__).parent.parent / "vendor" / "intel_guide"


def _rows_by(result, key, cpu):
    for row in result.rows:
        if row.form == key and row.core == cpu:
            return row
    raise AssertionError(f"no row for {key} {cpu}")


class ParseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = parse_perf_js((FIXTURE / "perf2.js").read_text())

    def test_exact_values_for_three_fixture_rows(self):
        row = _rows_by(self.result, "ADCX_GPR32d_GPR32d", "ADL-P")
        self.assertEqual(row.mnemonic, "ADCX")
        self.assertEqual(row.latency, "1")
        self.assertEqual(row.cpi, "0.5")
        row = _rows_by(self.result, "ADDPD_XMMpd_XMMpd", "ICL")
        self.assertEqual(row.latency, "4")
        self.assertEqual(row.cpi, "0.5")
        row = _rows_by(self.result, "XSETBV", "SKL")
        self.assertEqual(row.mnemonic, "XSETBV")
        self.assertEqual(row.latency, "5")
        self.assertEqual(row.cpi, "1.25")

    def test_row_provenance(self):
        row = _rows_by(self.result, "ADCX_GPR32d_GPR32d", "ADL-P")
        self.assertEqual(row.source, SOURCE_INTEL_GUIDE)
        self.assertEqual(row.source_kind, "modeled")
        self.assertEqual(row.source_version, SOURCE_VERSION)
        self.assertEqual(row.citation_url, CITATION_URL)
        self.assertEqual(row.applies_to, "form")
        self.assertEqual(row.architecture, "x86")

    def test_every_core_id_resolves(self):
        self.assertTrue(self.result.rows)
        for row in self.result.rows:
            self.assertEqual(canonical_core_id(row.core), row.core, row.core)

    def test_empty_latency_stays_empty(self):
        row = _rows_by(self.result, "ADDPS_XMMps_XMMps", "ADL-P")
        self.assertEqual(row.latency, "")
        self.assertEqual(row.cpi, "0.5")

    def test_icelake_client_row_wins_over_xeon(self):
        # "Icelake Intel Core" and "Icelake Xeon" both map to ICL. The
        # client label matches the cores.py ``icelake-client`` llvm_cpu.
        rows = [r for r in self.result.rows if r.form == "ADCX_GPR32d_GPR32d" and r.core == "ICL"]
        self.assertEqual(len(rows), 1)

    def test_known_rows_kept_and_unknown_cpu_dropped(self):
        text = '"FOO_XMMp_XMMp":[\n\t\t{"No Such CPU":{l:"1",t:"1"}},\n\t],'
        result = parse_perf_js(text)
        self.assertEqual(result.rows, [])
        self.assertEqual(result.dropped, {"No Such CPU": 1})

    def test_parse_notes_json(self):
        notes = parse_notes_json((FIXTURE / "notes.json").read_text())
        self.assertIn("round_note", notes)
        self.assertIn("sae_note", notes)
        self.assertTrue(notes["sae_note"].startswith("<br />"))


class VendoredTests(unittest.TestCase):
    def test_load_vendored_returns_rows(self):
        rows = load_vendored(VENDOR)
        self.assertGreater(len(rows), 7000)
        for row in rows:
            self.assertEqual(canonical_core_id(row.core), row.core)

    def test_load_vendored_covers_all_expected_cores(self):
        cores = {row.core for row in load_vendored(VENDOR)}
        self.assertEqual(cores, {"ADL-P", "SKL", "EMR", "ICL"})

    def test_load_vendored_default_path(self):
        # Default path is the repo vendor directory next to this checkout.
        self.assertEqual(load_vendored(), load_vendored(VENDOR))

    def test_notes_vendored_side_by_side(self):
        raw = json.loads((VENDOR / "notes.json").read_text())
        self.assertIn("round_note", raw)
        self.assertIn("sae_note", raw)


if __name__ == "__main__":
    unittest.main()
