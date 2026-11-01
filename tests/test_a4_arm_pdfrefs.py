"""Arm records must not receive Intel SDM pdf_refs (web audit P10)."""

import simdref.pdfparse.intel  # noqa: F401  registers the intel-sdm source
from simdref.ingest_pdf import merge_pdf_enrichment
from simdref.models import InstructionRecord
from simdref.pdfparse.types import PdfDescriptionPayload, PdfEnrichmentResult


def _adc_records() -> list[InstructionRecord]:
    return [
        InstructionRecord(
            mnemonic="ADC",
            form="ADC R32, R32, IMM32",
            summary="Add with Carry.",
            architecture="x86",
        ),
        InstructionRecord(
            mnemonic="adc",
            form="adc <Wd>, <Wn>, <Wm>",
            summary="Add with Carry.",
            architecture="arm",
        ),
    ]


def _sdm_result() -> PdfEnrichmentResult:
    payload = PdfDescriptionPayload(
        sections={"description": "x86 ADC description."},
        source_url="https://cdrdv2.intel.com/v1/dl/getContent/671200",
        page_start=713,
        page_end=714,
    )
    return PdfEnrichmentResult(descriptions={"ADC": payload})


def test_arm_record_gets_no_intel_pdf_ref() -> None:
    records = _adc_records()
    merge_pdf_enrichment(records, "intel-sdm", _sdm_result())
    arm = records[1]
    assert all(ref["source_id"] != "intel-sdm" for ref in arm.pdf_refs)
    assert "intel-sdm-url" not in arm.metadata


def test_x86_record_keeps_intel_pdf_ref() -> None:
    records = _adc_records()
    merge_pdf_enrichment(records, "intel-sdm", _sdm_result())
    x86 = records[0]
    refs = [ref for ref in x86.pdf_refs if ref["source_id"] == "intel-sdm"]
    assert len(refs) == 1
    assert refs[0]["page_start"] == "713"
