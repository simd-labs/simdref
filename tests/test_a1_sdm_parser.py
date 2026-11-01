"""Regression tests for Intel SDM parser defects A1 (MOVDQU comma title,
mixed-case titles, dropped 64-Bit Mode Exceptions).

Fixture texts come from the vendored Intel SDM PDF pages:
p1354 (MOVDQU), p861 (CMOVcc), p1203 (Jcc), p1911 (SETcc),
p1703 (PREFETCHh), p708 (AAD), p1348 (MOVDQ2Q).
"""

from pathlib import Path

from simdref.pdfparse.base import extract_sections_from_lines
from simdref.pdfparse import intel
from simdref.pdfparse.intel import (
    _ALL_HEADING_NAMES,
    _line_is_tabular_noise,
    _prepare_page_from_pymupdf_dict,
    parse_instruction_title,
)

FIXTURES = Path(__file__).parent / "fixtures" / "a1_sdm"


def _span(text, size, x0, top):
    return {"text": text, "size": size, "bbox": (x0, top, x0 + 100.0, top + 10.0)}


def test_comma_mnemonic_title_parses_and_expands():
    lines = (FIXTURES / "titles.txt").read_text().splitlines()
    parsed = parse_instruction_title(lines[0])
    assert parsed == ("MOVDQU,VMOVDQU8/16/32/64", "Move Unaligned Packed Integer Values")
    assert intel._mnemonic_variants(parsed[0]) == [
        "MOVDQU",
        "VMOVDQU8",
        "VMOVDQU16",
        "VMOVDQU32",
        "VMOVDQU64",
    ]


def test_mixed_case_titles_parse():
    lines = (FIXTURES / "titles.txt").read_text().splitlines()
    expected = {
        "CMOVcc": ("CMOVcc", "Conditional Move"),
        "Jcc": ("Jcc", "Jump if Condition Is Met"),
        "SETcc": ("SETcc", "Set Byte on Condition"),
        "PREFETCHh": ("PREFETCHh", "Prefetch Data Into Caches"),
    }
    for line in lines[1:]:
        mnemonic, _summary = line.split("—", 1)
        expected_pair = expected[mnemonic]
        assert parse_instruction_title(line) == expected_pair
    assert intel._mnemonic_variants("CMOVcc") == ["CMOVCC"]
    assert intel._mnemonic_variants("PREFETCHh") == ["PREFETCHH"]


def test_pymupdf_span_gap_in_mixed_case_title():
    # PyMuPDF splits "CMOVcc" into spans; the gap rule inserts a space.
    prepared = _prepare_page_from_pymupdf_dict(
        {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                _span("CMOV", 12.0, 72.0, 50.0),
                                _span("cc", 12.0, 118.0, 50.0),
                                _span("—", 12.0, 140.0, 50.0),
                                _span("Conditional Move", 12.0, 150.0, 50.0),
                            ]
                        }
                    ],
                }
            ]
        }
    )
    assert prepared.title == ("CMOVcc", "Conditional Move")


def test_tabular_noise_keeps_section_heading():
    assert _line_is_tabular_noise("64-Bit Mode Exceptions") is False
    assert _line_is_tabular_noise("64/32-Bit Mode Support") is True


def test_64bit_mode_exceptions_heading_survives_pymupdf_prep():
    lines = (FIXTURES / "aad_64bit_exceptions.txt").read_text().splitlines()
    text_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"spans": [_span(lines[0], 10.0, 46.0, 66.0)]},
                    {"spans": [_span(lines[1], 9.0, 46.0, 82.0)]},
                    {"spans": [_span(lines[2], 10.0, 46.0, 100.0)]},
                    {"spans": [_span(lines[3], 9.0, 46.0, 120.0)]},
                    {"spans": [_span(lines[4], 9.0, 138.0, 120.0)]},
                ],
            }
        ]
    }
    prepared = _prepare_page_from_pymupdf_dict(text_dict)
    texts = [text for _top, _size, _x0, text in prepared.body_lines]
    assert "64-Bit Mode Exceptions" in texts
    assert "Compatibility Mode Exceptions" in texts

    sections = extract_sections_from_lines(
        prepared.body_lines,
        heading_min_size=9.8,
        body_max_size=9.5,
        known_headings=_ALL_HEADING_NAMES,
    )
    bit64_texts = [text for _x0, text in sections["64-Bit Mode Exceptions"]]
    compat_texts = [text for _x0, text in sections["Compatibility Mode Exceptions"]]
    assert "If in 64-bit mode." in bit64_texts
    assert "If in 64-bit mode." not in compat_texts


def test_repeated_heading_keeps_last_prose_block():
    # The table column header Description repeats the section heading. Keep
    # the last block: it holds the real prose, the first block holds table
    # fragments. The x0 filter in intel.py then keeps the prose lines.
    block_a, _sep, block_b = (
        (FIXTURES / "movdq2q_description_blocks.txt").read_text().partition("\n\n")
    )
    lines = [
        (1.0, 10.0, 46.0, "Description"),
        (2.0, 9.0, 364.0, "F2 0F D6 /r"),
        (3.0, 10.0, 46.0, "Operation"),
        (4.0, 9.0, 46.0, "DEST := SRC[63:0];"),
        (5.0, 10.0, 46.0, "Description"),
        (6.0, 9.0, 46.0, block_a.strip()),
        (7.0, 9.0, 46.0, block_b.strip()),
    ]
    sections = extract_sections_from_lines(
        lines,
        heading_min_size=9.8,
        body_max_size=9.5,
        known_headings=_ALL_HEADING_NAMES,
    )
    description_texts = [text for _x0, text in sections["Description"]]
    assert description_texts == [block_a.strip(), block_b.strip()]
