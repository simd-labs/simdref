"""Tests for the perf-source registration in ingest_catalog._ingest_perf_sources.

Covers the two behaviors added at integration: AMD CPI inversion and the
Intel guide round/SAE note text on intrinsics.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from simdref.ingest_catalog import _attach_intel_guide_notes
from simdref.models import IntrinsicRecord
from simdref.perf_sources.amd_swopt import _cpi, load_vendored, parse_workbook

FIXTURE = Path(__file__).parent / "fixtures" / "amd_swopt"
VENDOR = Path(__file__).parent.parent / "vendor" / "amd_swopt"


class AmdCpiTests(unittest.TestCase):
    def test_inverts_ops_per_cycle(self):
        self.assertEqual(_cpi("2"), "0.5")
        self.assertEqual(_cpi("0.25"), "4")
        self.assertEqual(_cpi("1"), "1")

    def test_non_numeric_stays_empty(self):
        self.assertEqual(_cpi(""), "")
        self.assertEqual(_cpi("variable"), "")
        self.assertEqual(_cpi("0.25-0.5"), "")

    def test_parse_workbook_sets_cpi_and_keeps_raw(self):
        rows, _stats = parse_workbook(
            FIXTURE / "zen5_sample.xlsx",
            core="ZEN5",
            source_version="1.00",
            citation_url="https://docs.amd.com/v/u/en-US/58455_1.00",
        )
        by_mnemonic = {row.mnemonic: row for row in rows}
        self.assertEqual(by_mnemonic["vaddpd"].cpi, "0.5")
        self.assertEqual(by_mnemonic["vdivpd"].cpi, "4")
        for row in rows:
            self.assertIn("amd_ops_per_cycle", row.extra_measurement)

    def test_load_vendored_sets_cpi(self):
        rows = load_vendored(VENDOR)
        self.assertTrue(rows)
        for row in rows:
            raw = row.extra_measurement.get("amd_ops_per_cycle", "")
            if raw:
                self.assertEqual(row.cpi, _cpi(raw), row.mnemonic)


class GuideNotesTests(unittest.TestCase):
    def _intrinsic(self, name: str, description: str) -> IntrinsicRecord:
        return IntrinsicRecord(
            name=name,
            signature=f"void {name}(void)",
            description=description,
            header="immintrin.h",
        )

    def test_note_text_added_for_referencing_intrinsics(self):
        a = self._intrinsic("a", "Round it. [round_note]")
        b = self._intrinsic("b", "Suppress. [sae_note]")
        plain = self._intrinsic("c", "Nothing to reference.")
        _attach_intel_guide_notes([a, b, plain])
        self.assertEqual(len(a.notes), 1)
        self.assertIn("rounding", a.notes[0].lower())
        self.assertEqual(len(b.notes), 1)
        self.assertIn("suppressed", b.notes[0].lower())
        self.assertEqual(plain.notes, [])

    def test_existing_notes_are_kept(self):
        rec = self._intrinsic("a", "[round_note]")
        rec.notes.append("existing note")
        _attach_intel_guide_notes([rec])
        self.assertEqual(rec.notes[0], "existing note")
        self.assertEqual(len(rec.notes), 2)


if __name__ == "__main__":
    unittest.main()
