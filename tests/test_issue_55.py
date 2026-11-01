"""Issue 55: uops.info summary text drops the space at CamelCase seams."""

from pathlib import Path

from simdref.ingest_catalog import _instruction_summary, parse_uops_xml

FIXTURE = Path(__file__).parent / "fixtures" / "issue_55" / "uops_summary_typos.xml"


def test_single_precision_typo_fixed() -> None:
    summary = _instruction_summary(
        "VFMADD231PS",
        "Fused Multiply-Add of Packed SinglePrecision Floating-Point Values",
        [],
    )
    assert summary == "Fused Multiply-Add of Packed Single Precision Floating-Point Values."


def test_extract_packed_typo_fixed() -> None:
    summary = _instruction_summary("VEXTRACTI128", "ExtractPacked Integer Values", [])
    assert summary == "Extract Packed Integer Values."


def test_fixture_rows_normalize_summaries() -> None:
    records = parse_uops_xml(FIXTURE.read_text(encoding="utf-8"))
    by_mnemonic = {record.mnemonic: record.summary for record in records}
    assert by_mnemonic["VFMADD231PS"] == (
        "Fused Multiply-Add of Packed Single Precision Floating-Point Values."
    )
    assert by_mnemonic["VEXTRACTI64X4"] == "Extract Packed Integer Values."
    assert by_mnemonic["VEXTRACTI128"] == "Extract Packed Integer Values."


def test_mixed_case_terms_and_acronyms_stay_intact() -> None:
    summary = _instruction_summary("PBLENDMW", "Blend Int Vectors Using an OpMask Control", [])
    assert summary == "Blend Int Vectors Using an OpMask Control."
    summary = _instruction_summary("AESIMC", "Perform the InvMixColumn Transformation", [])
    assert summary == "Perform the InvMixColumn Transformation."
    summary = _instruction_summary("VCVTPH2PS", "Convert FP16 Values to FP32 Values", [])
    assert summary == "Convert FP16 Values to FP32 Values."
