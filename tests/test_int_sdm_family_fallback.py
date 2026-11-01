"""SDM family keys: concrete records fall back to their family text.

A1's SDM parser emits family keys (CMOVCC, JCC, SETCC, PREFETCHH); concrete
records such as CMOVB or PREFETCHT0 must receive that text.
"""

from __future__ import annotations

import unittest

import simdref.pdfparse.intel  # noqa: F401  # registers the intel-sdm PDF source
from simdref.ingest_pdf import merge_pdf_enrichment
from simdref.models import InstructionRecord
from simdref.pdfparse.types import PdfDescriptionPayload, PdfEnrichmentResult


def _record(mnemonic: str) -> InstructionRecord:
    return InstructionRecord(
        mnemonic=mnemonic,
        form="",
        summary="",
        architecture="x86",
    )


def _result() -> PdfEnrichmentResult:
    def payload(sections: dict[str, str]) -> PdfDescriptionPayload:
        return PdfDescriptionPayload(sections=sections, source_url="https://example/sdm")

    return PdfEnrichmentResult(
        descriptions={
            "CMOVCC": payload({"Description": "conditional move"}),
            "JCC": payload({"Description": "conditional jump"}),
            "SETCC": payload({"Description": "conditional set"}),
            "PREFETCHH": payload({"Description": "prefetch hint"}),
        }
    )


class FamilyFallbackTests(unittest.TestCase):
    def test_concrete_records_get_family_text(self):
        records = [_record(name) for name in ("CMOVB", "JE", "SETNE", "PREFETCHT0")]
        merge_pdf_enrichment(records, "intel-sdm", _result())
        for record in records:
            self.assertIn("Description", record.description, record.mnemonic)

    def test_family_text_content(self):
        records = [_record("CMOVB"), _record("PREFETCHT0")]
        merge_pdf_enrichment(records, "intel-sdm", _result())
        self.assertEqual(records[0].description["Description"], "conditional move")
        self.assertEqual(records[1].description["Description"], "prefetch hint")

    def test_non_family_lookalikes_stay_empty(self):
        records = [_record("JMP"), _record("SETSSBSY"), _record("MOV")]
        merge_pdf_enrichment(records, "intel-sdm", _result())
        for record in records:
            self.assertEqual(record.description, {}, record.mnemonic)


if __name__ == "__main__":
    unittest.main()
