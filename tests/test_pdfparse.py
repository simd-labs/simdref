import dataclasses
from pathlib import Path

import pytest
from simdref.pdfparse.base import (
    chars_to_lines,
    extract_sections_from_chars,
    extract_sections_from_lines,
)


def _make_char(text, fontname, size, x0=0, top=0):
    """Create a minimal char dict matching pdfplumber's char format."""
    return {"text": text, "fontname": fontname, "size": size, "x0": x0, "top": top}


def test_extract_sections_basic():
    """Section headers at size 10, body text at size 9."""
    chars = [
        _make_char("D", "NeoSansMedium", 10.0, x0=0, top=100),
        _make_char("e", "NeoSansMedium", 10.0, x0=8, top=100),
        _make_char("s", "NeoSansMedium", 10.0, x0=16, top=100),
        _make_char("c", "NeoSansMedium", 10.0, x0=24, top=100),
        _make_char("r", "NeoSansMedium", 10.0, x0=32, top=100),
        _make_char("i", "NeoSansMedium", 10.0, x0=40, top=100),
        _make_char("p", "NeoSansMedium", 10.0, x0=48, top=100),
        _make_char("t", "NeoSansMedium", 10.0, x0=56, top=100),
        _make_char("i", "NeoSansMedium", 10.0, x0=64, top=100),
        _make_char("o", "NeoSansMedium", 10.0, x0=72, top=100),
        _make_char("n", "NeoSansMedium", 10.0, x0=80, top=100),
        _make_char("B", "Verdana", 9.0, x0=0, top=120),
        _make_char("o", "Verdana", 9.0, x0=8, top=120),
        _make_char("d", "Verdana", 9.0, x0=16, top=120),
        _make_char("y", "Verdana", 9.0, x0=24, top=120),
        _make_char(".", "Verdana", 9.0, x0=32, top=120),
    ]
    sections = extract_sections_from_chars(chars, heading_min_size=10.0, body_max_size=9.5)
    assert "Description" in sections
    assert any("Body." in text for _, text in sections["Description"])


def test_extract_sections_multiple():
    """Multiple sections detected in sequence."""
    chars = []
    y = 100
    for ch in "Description":
        chars.append(_make_char(ch, "NeoSansMedium", 10.0, top=y))
    y += 20
    for ch in "First body.":
        chars.append(_make_char(ch, "Verdana", 9.0, top=y))
    y += 40
    for ch in "Operation":
        chars.append(_make_char(ch, "NeoSansMedium", 10.0, top=y))
    y += 20
    for ch in "DEST := SRC":
        chars.append(_make_char(ch, "Verdana", 9.0, top=y))

    sections = extract_sections_from_chars(chars, heading_min_size=10.0, body_max_size=9.5)
    assert "Description" in sections
    assert any("First body." in text for _, text in sections["Description"])
    assert "Operation" in sections
    assert any("DEST := SRC" in text for _, text in sections["Operation"])


def test_extract_sections_empty_chars():
    sections = extract_sections_from_chars([], heading_min_size=10.0, body_max_size=9.5)
    assert sections == {}


def test_extract_sections_from_lines_matches_chars():
    chars = []
    y = 100
    for ch in "Description":
        chars.append(_make_char(ch, "NeoSansMedium", 10.0, top=y))
    y += 20
    for ch in "Body line.":
        chars.append(_make_char(ch, "Verdana", 9.0, top=y))

    lines = chars_to_lines(chars)
    assert extract_sections_from_lines(lines, heading_min_size=10.0, body_max_size=9.5) == (
        extract_sections_from_chars(chars, heading_min_size=10.0, body_max_size=9.5)
    )


from simdref.pdfparse.intel import (
    INTEL_SDM_URL,
    KNOWN_SECTIONS,
    INTEL_PDF_SOURCE,
    _instruction_page_ranges,
    _page_might_have_tables,
    _prepare_page_from_pymupdf_dict,
    _prepared_page_needs_fallback,
    _table_bboxes_for_page,
    normalize_section_name,
    parse_instruction_title,
)
from simdref.ingest_pdf import load_or_parse_pdf_source, merge_pdf_enrichment
from simdref.models import InstructionRecord
from simdref.pdfparse.registry import _PDF_SOURCES, get_pdf_source
from simdref.pdfparse.types import PdfDescriptionPayload, PdfEnrichmentResult


def test_parse_instruction_title_basic():
    assert parse_instruction_title(
        "ADDPS\u2014Add Packed Single Precision Floating-Point Values"
    ) == ("ADDPS", "Add Packed Single Precision Floating-Point Values")


def test_parse_instruction_title_with_slash():
    result = parse_instruction_title(
        "MOVDQA/VMOVDQA32/VMOVDQA64\u2014Move Aligned Packed Integer Values"
    )
    assert result == ("MOVDQA/VMOVDQA32/VMOVDQA64", "Move Aligned Packed Integer Values")


def test_parse_instruction_title_no_emdash():
    assert parse_instruction_title("CHAPTER 3") is None


def test_parse_instruction_title_lowercase_rejected():
    assert parse_instruction_title("The Intel\u00ae Pentium\u00ae Processor (1995\u2014") is None


def test_normalize_section_name():
    assert normalize_section_name("Description") == "Description"
    assert (
        normalize_section_name("Intel C/C++ Compiler Intrinsic Equivalent")
        == "Intrinsic Equivalents"
    )
    assert (
        normalize_section_name("Intel C/C++Compiler Intrinsic Equivalent")
        == "Intrinsic Equivalents"
    )
    assert (
        normalize_section_name("SIMD Floating-Point Exceptions") == "SIMD Floating-Point Exceptions"
    )
    assert normalize_section_name("Numeric Exceptions") == "Numeric Exceptions"
    assert normalize_section_name("Other Exceptions") == "Other Exceptions"
    assert normalize_section_name("Flags Affected") == "Flags Affected"


def test_known_sections():
    assert "Description" in KNOWN_SECTIONS
    assert "Operation" in KNOWN_SECTIONS
    assert "Intrinsic Equivalents" in KNOWN_SECTIONS
    assert "Flags Affected" in KNOWN_SECTIONS
    assert "FPU Flags Affected" in KNOWN_SECTIONS
    assert "Floating-Point Exceptions" in KNOWN_SECTIONS


def test_normalize_fpu_sections():
    assert normalize_section_name("FPU Flags Affected") == "FPU Flags Affected"
    assert normalize_section_name("Floating-Point Exceptions") == "Floating-Point Exceptions"


def test_whitelist_rejects_garbage_headings():
    """With known_headings, table captions at heading size are demoted to body."""
    known = frozenset({"description", "operation"})
    chars = []
    y = 100
    # "Description" at heading size — should be detected as heading
    for ch in "Description":
        chars.append(_make_char(ch, "NeoSansMedium", 10.0, top=y))
    y += 20
    for ch in "Real body text.":
        chars.append(_make_char(ch, "Verdana", 9.0, top=y))
    y += 20
    # "Table 5-8. VF..." at heading size — should NOT be a heading
    garbage = "Table 5-8. VFMSUB Notation"
    for ch in garbage:
        chars.append(_make_char(ch, "NeoSansMedium", 10.0, top=y))
    y += 20
    for ch in "More body text.":
        chars.append(_make_char(ch, "Verdana", 9.0, top=y))

    sections = extract_sections_from_chars(
        chars,
        heading_min_size=10.0,
        body_max_size=9.5,
        known_headings=known,
    )
    assert "Description" in sections
    # Garbage heading should NOT appear as a section
    assert "Table 5-8. VFMSUB Notation" not in sections
    # Body text after the garbage line should still be under "Description"
    body_texts = [text for _, text in sections["Description"]]
    assert "Real body text." in body_texts
    assert "More body text." in body_texts


def test_whitelist_passes_through_without_known_headings():
    """Without known_headings, font-size heuristic still works."""
    chars = []
    y = 100
    for ch in "AnyHeading":
        chars.append(_make_char(ch, "NeoSansMedium", 10.0, top=y))
    y += 20
    for ch in "Body.":
        chars.append(_make_char(ch, "Verdana", 9.0, top=y))

    sections = extract_sections_from_chars(
        chars,
        heading_min_size=10.0,
        body_max_size=9.5,
        known_headings=None,
    )
    assert "AnyHeading" in sections


def test_whitelist_drops_heading_size_garbage():
    """Lines at heading font size that aren't known headings are silently dropped."""
    known = frozenset({"description", "operation"})
    chars = []
    y = 100
    for ch in "Description":
        chars.append(_make_char(ch, "NeoSansMedium", 10.0, top=y))
    y += 20
    for ch in "Body line 1.":
        chars.append(_make_char(ch, "Verdana", 9.0, top=y))
    y += 20
    # Pseudocode symbol at heading size — should be dropped (not in body)
    for ch in "IF (COUNT & COUNTMASK) = 1":
        chars.append(_make_char(ch, "NeoSansMedium", 10.0, top=y))
    y += 20
    for ch in "Body line 2.":
        chars.append(_make_char(ch, "Verdana", 9.0, top=y))

    sections = extract_sections_from_chars(
        chars,
        heading_min_size=10.0,
        body_max_size=9.5,
        known_headings=known,
    )
    assert "Description" in sections
    body_texts = [text for _, text in sections["Description"]]
    assert "Body line 1." in body_texts
    assert "Body line 2." in body_texts
    # The garbage line at heading size should NOT appear in any section body
    all_body = " ".join(text for sec in sections.values() for _, text in sec)
    assert "IF (COUNT" not in all_body


def test_merge_pdf_enrichment_adds_sections_and_pdf_reference():
    instructions = [
        InstructionRecord(
            mnemonic="VADDPD",
            form="VADDPD (YMM, YMM, YMM)",
            summary="Add packed double-precision floating-point values.",
            isa=["AVX"],
        )
    ]
    result = PdfEnrichmentResult(
        descriptions={
            "ADDPD": PdfDescriptionPayload(
                sections={
                    "Description": "Add packed double-precision floating-point values.",
                    "Operation": "DEST := SRC1 + SRC2",
                },
                source_url=INTEL_SDM_URL,
                page_start=123,
                page_end=125,
            )
        }
    )

    merge_pdf_enrichment(instructions, "intel-sdm", result)

    item = instructions[0]
    assert item.description["Description"].startswith("Add packed")
    assert item.metadata["intel-sdm-page-start"] == "123"
    assert item.metadata["intel-sdm-page-end"] == "125"
    assert item.metadata["intel-sdm-url"] == f"{INTEL_SDM_URL}#page=123"


def test_load_or_parse_pdf_source_uses_cache(tmp_path, monkeypatch):
    pdf_path = tmp_path / "intel-sdm.pdf"
    cache_path = tmp_path / "intel-sdm-cache.msgpack"
    pdf_path.write_bytes(b"fake-pdf")

    calls: list[Path] = []

    def _fake_parse(path, *, status=None):
        calls.append(path)
        if status is not None:
            status("parsed")
        return PdfEnrichmentResult(
            descriptions={
                "ADDPD": PdfDescriptionPayload(
                    sections={"Description": "Add packed doubles."},
                    source_url=INTEL_SDM_URL,
                    page_start=10,
                    page_end=11,
                )
            }
        )

    spec = get_pdf_source("intel-sdm")
    monkeypatch.setitem(_PDF_SOURCES, "intel-sdm", dataclasses.replace(spec, parser=_fake_parse))

    first = load_or_parse_pdf_source("intel-sdm", pdf_path, cache_path=cache_path)
    second = load_or_parse_pdf_source("intel-sdm", pdf_path, cache_path=cache_path)

    assert first == second
    assert calls == [pdf_path]


def test_intel_pdf_source_registered():
    spec = get_pdf_source("intel-sdm")
    assert spec.source_id == INTEL_PDF_SOURCE.source_id
    assert spec.display_name == "Intel SDM"


class _FakeTable:
    def __init__(self, bbox):
        self.bbox = bbox


class _FakePage:
    def __init__(self, *, rects=(), lines=(), curves=(), width=100.0, height=100.0, tables=()):
        self.rects = list(rects)
        self.lines = list(lines)
        self.curves = list(curves)
        self.width = width
        self.height = height
        self._tables = list(tables)

    def find_tables(self):
        return list(self._tables)


class _FakeOutlineDoc:
    def __init__(self, outlines):
        self._outlines = outlines

    def get_outlines(self):
        return iter(self._outlines)

    def get_dest(self, dest):
        return dest


class _FakePdfPageRef:
    def __init__(self, pageid):
        self.pageid = pageid


class _FakeOutlinePdf:
    def __init__(self, pageids, outlines):
        self.pages = [_FakePage() for _ in pageids]
        for page, pageid in zip(self.pages, pageids, strict=True):
            page.page_obj = _FakePdfPageRef(pageid)
        self.doc = _FakeOutlineDoc(outlines)


def test_page_might_have_tables_uses_graphics_threshold():
    sparse_page = _FakePage(rects=range(2), lines=range(1), curves=range(1))
    dense_page = _FakePage(rects=range(6))

    assert _page_might_have_tables(sparse_page) is False
    assert _page_might_have_tables(dense_page) is True


def test_table_bboxes_for_page_skips_find_tables_when_precheck_fails(monkeypatch):
    page = _FakePage(rects=range(2))

    def _boom():
        raise AssertionError("find_tables should not run")

    monkeypatch.setattr(page, "find_tables", _boom)
    assert _table_bboxes_for_page(page) == []


def test_table_bboxes_for_page_filters_large_false_positives():
    page = _FakePage(
        rects=range(6),
        width=100.0,
        height=100.0,
        tables=[
            _FakeTable((0.0, 0.0, 20.0, 20.0)),
            _FakeTable((0.0, 0.0, 100.0, 100.0)),
        ],
    )

    assert _table_bboxes_for_page(page) == [(0.0, 0.0, 20.0, 20.0)]


def test_prepare_page_from_pymupdf_dict_extracts_title_and_body_lines():
    prepared = _prepare_page_from_pymupdf_dict(
        {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "ADDPS",
                                    "size": 12.0,
                                    "bbox": (72.0, 50.0, 110.0, 62.0),
                                },
                                {
                                    "text": "—",
                                    "size": 12.0,
                                    "bbox": (114.0, 50.0, 118.0, 62.0),
                                },
                                {
                                    "text": "Add Packed Single Precision Floating-Point Values",
                                    "size": 12.0,
                                    "bbox": (122.0, 50.0, 360.0, 62.0),
                                },
                            ]
                        },
                        {
                            "spans": [
                                {
                                    "text": "Description",
                                    "size": 10.0,
                                    "bbox": (72.0, 80.0, 130.0, 90.0),
                                }
                            ]
                        },
                        {
                            "spans": [
                                {
                                    "text": "Adds packed values.",
                                    "size": 9.0,
                                    "bbox": (72.0, 96.0, 170.0, 106.0),
                                }
                            ]
                        },
                    ],
                }
            ]
        }
    )

    assert prepared.backend == "pymupdf"
    assert prepared.title == ("ADDPS", "Add Packed Single Precision Floating-Point Values")
    assert prepared.body_lines == [
        (80.0, 10.0, 72.0, "Description"),
        (96.0, 9.0, 72.0, "Adds packed values."),
    ]


def test_prepare_page_from_pymupdf_dict_filters_tabular_noise():
    prepared = _prepare_page_from_pymupdf_dict(
        {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Opcode",
                                    "size": 9.0,
                                    "bbox": (72.0, 80.0, 100.0, 90.0),
                                }
                            ]
                        },
                        {
                            "spans": [
                                {
                                    "text": "Instruction Operand Encoding",
                                    "size": 9.0,
                                    "bbox": (72.0, 94.0, 210.0, 104.0),
                                }
                            ]
                        },
                        {
                            "spans": [
                                {
                                    "text": "Description",
                                    "size": 10.0,
                                    "bbox": (72.0, 110.0, 130.0, 120.0),
                                }
                            ]
                        },
                    ],
                }
            ]
        }
    )

    assert prepared.body_lines == [(110.0, 10.0, 72.0, "Description")]


def test_prepared_page_needs_fallback_when_title_page_has_no_headings():
    prepared = _prepare_page_from_pymupdf_dict(
        {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "ADDPS",
                                    "size": 12.0,
                                    "bbox": (72.0, 50.0, 110.0, 62.0),
                                },
                                {
                                    "text": "—",
                                    "size": 12.0,
                                    "bbox": (114.0, 50.0, 118.0, 62.0),
                                },
                                {
                                    "text": "Add Packed Single Precision Floating-Point Values",
                                    "size": 12.0,
                                    "bbox": (122.0, 50.0, 360.0, 62.0),
                                },
                            ]
                        },
                        {
                            "spans": [
                                {
                                    "text": "Unclassified body line",
                                    "size": 9.0,
                                    "bbox": (72.0, 90.0, 170.0, 100.0),
                                }
                            ]
                        },
                    ],
                }
            ]
        }
    )

    assert _prepared_page_needs_fallback(prepared) == "missing-heading"


def test_prepared_page_does_not_need_fallback_without_title():
    prepared = _prepare_page_from_pymupdf_dict(
        {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Continuation paragraph.",
                                    "size": 9.0,
                                    "bbox": (72.0, 90.0, 170.0, 100.0),
                                }
                            ]
                        }
                    ],
                }
            ]
        }
    )

    assert _prepared_page_needs_fallback(prepared) is None


def test_instruction_page_ranges_uses_outline_instruction_chapters():
    pdf = _FakeOutlinePdf(
        pageids=range(1, 5001),
        outlines=[
            (
                1,
                "Volume 2 (2A, 2B, 2C, & 2D): Instruction Set Reference, A-Z",
                {"D": [type("Ref", (), {"objid": 587})(), "XYZ"]},
                None,
                None,
            ),
            (
                2,
                "Chapter 3 Instruction Set Reference, A-L",
                {"D": [type("Ref", (), {"objid": 687})(), "XYZ"]},
                None,
                None,
            ),
            (
                2,
                "Chapter 4 Instruction Set Reference, M-U",
                {"D": [type("Ref", (), {"objid": 1284})(), "XYZ"]},
                None,
                None,
            ),
            (
                2,
                "Chapter 33 VMX Instruction Reference",
                {"D": [type("Ref", (), {"objid": 4327})(), "XYZ"]},
                None,
                None,
            ),
            (
                3,
                "35.5 SEAM Instruction Reference",
                {"D": [type("Ref", (), {"objid": 4395})(), "XYZ"]},
                None,
                None,
            ),
            (
                2,
                "Chapter 41 Intel® SGX Instruction References",
                {"D": [type("Ref", (), {"objid": 4527})(), "XYZ"]},
                None,
                None,
            ),
            (
                1,
                "Volume 3 (3A, 3B, 3C, & 3D): System Programming Guide",
                {"D": [type("Ref", (), {"objid": 3143})(), "XYZ"]},
                None,
                None,
            ),
            (
                2,
                "Chapter 34 System Management Mode",
                {"D": [type("Ref", (), {"objid": 4359})(), "XYZ"]},
                None,
                None,
            ),
            (
                2,
                "Chapter 36 Intel® Processor Trace",
                {"D": [type("Ref", (), {"objid": 4401})(), "XYZ"]},
                None,
                None,
            ),
            (
                2,
                "Chapter 42 Intel® SGX Interactions with IA32 and Intel® 64 Architecture",
                {"D": [type("Ref", (), {"objid": 4655})(), "XYZ"]},
                None,
                None,
            ),
        ],
    )

    assert _instruction_page_ranges(pdf) == [
        (686, 3142),
        (4326, 4358),
        (4394, 4400),
        (4526, 4654),
    ]


def test_instruction_page_ranges_falls_back_to_all_pages_without_outline():
    pdf = _FakeOutlinePdf(pageids=range(1, 101), outlines=[])

    assert _instruction_page_ranges(pdf) == [(0, 100)]
