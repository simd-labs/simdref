"""PTX ISA derivation: parser on the synthetic fixture, keys, picks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdref.catkey import build_key
from simdref.ptx import (
    BUNDLE_FORMAT,
    derive_rows,
    op_chain,
    parse_ptx_instructions_payload,
    pick_form,
    pick_spelling,
)


@pytest.fixture(scope="module")
def fixture_html() -> str:
    return (Path(__file__).parent / "fixtures" / "ptx_isa_sample.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rows(fixture_html: str) -> list[dict]:
    return derive_rows(fixture_html)


@pytest.fixture(scope="module")
def records(rows: list[dict]):
    return parse_ptx_instructions_payload(
        json.dumps({"format": BUNDLE_FORMAT, "instructions": rows})
    )


def test_row_count(rows):
    assert len(rows) == 7


def test_frob_row_fields(rows):
    row = rows[0]
    assert row["mnemonic"] == "frob"
    assert row["form"] == "d, [a]"
    assert row["category"] == "Widget Instructions"
    assert row["syntax"] == "frob{.fast}.vec.type d, [a];"
    assert row["qualifiers"] == {
        "vec": [".v2", ".v4"],
        "type": [".u8", ".f32"],
        "scope": [".cta", ".cluster"],
    }
    assert row["ptx_isa"] == ["2.0", "3.1"]
    assert row["sm"] == ["sm_75", "sm_80"]
    assert row["url"] == "#widget-instructions-frob"


def test_chain_drops_set_name_segments(rows):
    mnemonics = [row["mnemonic"] for row in rows]
    assert "frob.relaxed" in mnemonics
    assert "frob.nc" in mnemonics
    assert "twiddle.sync.aligned.row.col" in mnemonics
    assert "floop.l1" in mnemonics


def test_multi_op_heading(rows):
    flop = next(row for row in rows if row["mnemonic"] == "flop")
    assert flop["form"] == "[a]"
    assert flop["category"] == "Cache Instructions"
    assert flop["url"] == "#cache-instructions-flop-floop"
    assert flop["ptx_isa"] == []
    assert flop["sm"] == []


def test_section_without_syntax_yields_no_rows(rows):
    assert not any(row["mnemonic"].startswith("zzz") for row in rows)


def test_colliding_template_lines_merge_by_key():
    # Two template lines with one (mnemonic, form) share one catalog key:
    # the row lists both lines and unions the qualifier sets.
    fragment = (
        '<section id="widget-instructions-grob">'
        "<h2>Widget Instructions: grob</h2>"
        '<p class="rubric">Syntax</p>'
        '<div class="highlight-text notranslate"><div class="highlight"><pre>'
        "<span></span>grob.type1 d, [a];\n"
        "grob{.sat}.type2 d, [a];\n"
        ".type1 =    { .u8 };\n"
        ".type2 =    { .u8 };\n"
        ".sat =    { .false };\n"
        "</pre></div></div>"
        "</section>"
    )
    rows = derive_rows(fragment)
    assert len(rows) == 1
    assert rows[0]["mnemonic"] == "grob"
    assert rows[0]["syntax"] == "grob.type1 d, [a];\ngrob{.sat}.type2 d, [a];"
    assert rows[0]["qualifiers"]["sat"] == [".false"]


def test_double_colon_set_lines_are_not_rows():
    # `::dimension = { ::x, ::y };` names a set the templates reference as
    # `{::dimension}`: it is a set definition, never an instruction row.
    fragment = (
        '<section id="widget-instructions-grop">'
        "<h2>Widget Instructions: grop</h2>"
        '<p class="rubric">Syntax</p>'
        '<div class="highlight-text notranslate"><div class="highlight"><pre>'
        "<span></span>grop{::dimension}.b32 reg, resp;\n"
        "::dimension = { ::x, ::y };\n"
        "</pre></div></div>"
        "</section>"
    )
    rows = derive_rows(fragment)
    assert len(rows) == 1
    assert rows[0]["mnemonic"] == "grop.b32"
    assert rows[0]["form"] == "reg, resp"
    assert "::dimension" not in rows[0]["mnemonic"]


def test_plain_and_syntaxless_sections_are_skipped():
    page = (
        '<section id="introduction"><h1>1. Introduction</h1><p>Intro text.</p></section>'
        '<section id="broken"><h3>9.0.1. Widget Instructions: broken</h3>'
        '<p class="rubric">Syntax</p><p>No block.</p></section>'
        '<section id="widget-instructions-frob">'
        "<h3>9.1.1. Widget Instructions: frob</h3>"
        '<p class="rubric">Syntax</p>'
        '<div class="highlight-text notranslate"><div class="highlight">'
        "<pre><span></span>frob.type d, [a];\n.type = { .u8 };\n</pre></div></div></section>"
    )
    rows = derive_rows(page)
    assert [row["mnemonic"] for row in rows] == ["frob"]


def test_bundle_round_trip(rows, records):
    assert [record.mnemonic for record in records] == [row["mnemonic"] for row in rows]
    record = records[0]
    assert record.mnemonic == "frob"
    assert record.form == "d, [a]"
    assert record.architecture == "ptx"
    assert record.isa == ["PTX"]
    assert record.source == "ptx-isa"
    assert record.description["Syntax"] == "frob{.fast}.vec.type d, [a];"
    assert record.metadata["url"] == "#widget-instructions-frob"
    assert record.metadata["ptx_isa"] == "2.0, 3.1"
    assert record.metadata["sm"] == "sm_75, sm_80"
    summary = record.summary
    assert "frob instruction in Widget Instructions." in summary
    assert "Qualifier sets: vec, type, scope." in summary
    assert "Introduced in PTX ISA 2.0." in summary
    assert "Requires sm_75 or higher." in summary


def test_raw_html_payload(fixture_html):
    records = parse_ptx_instructions_payload(fixture_html)
    assert len(records) == 7
    assert records[0].mnemonic == "frob"


def test_op_chain():
    assert op_chain("ld.global.v4.f32") == ("ld", ("global", "v4", "f32"))
    assert op_chain("frob") == ("frob", ())
    assert op_chain("FROB.NC.U8") == ("frob", ("nc", "u8"))


def test_build_key_on_ptx_form():
    assert build_key("frob", "d, [a]") == "FROB (D, [A])"


def test_pick_form_ranks_by_coverage(records):
    pick = pick_form(records, "twiddle.sync.aligned.m8n8.row.col.f16", "b, c, d")
    assert pick is not None
    assert pick.form == "b, c, d"


def test_pick_form_ties_on_operand_count(records):
    near = pick_form(records, "twiddle.sync.aligned.m8n8", "b, c, d")
    far = pick_form(records, "twiddle.sync.aligned.m8n8", "b, c, d, e")
    assert near is not None and near.form == "b, c, d"
    assert far is not None and far.form == "b, c, d, e"


def test_pick_form_prefers_chain_prefix(records):
    # frob.nc has the best vocabulary here (relaxed, cta, u8) but is not
    # a chain prefix of the token, so the prefix pool must beat it.
    pick = pick_form(records, "frob.relaxed.cta.u8", "d, [a]")
    assert pick is not None
    assert pick.mnemonic == "frob"


def test_pick_form_unknown_base(records):
    assert pick_form(records, "zork.u8", "d") is None


@pytest.fixture(scope="module")
def match_records() -> list:
    html = (Path(__file__).parent / "fixtures" / "ptx_isa_match_sample.html").read_text(
        encoding="utf-8"
    )
    rows = derive_rows(html)
    return parse_ptx_instructions_payload(
        json.dumps({"format": BUNDLE_FORMAT, "instructions": rows})
    )


def test_merged_rows_name_the_section_of_each_syntax_line(match_records):
    # The merged `add` row keeps the section url of every template line,
    # so a concrete spelling reaches the section that documents it.
    add = next(r for r in match_records if r.mnemonic == "add" and r.form == "d, a, b")
    assert json.loads(add.metadata["urls"]) == [
        "#integer-arithmetic-instructions-add",
        "#integer-arithmetic-instructions-add",
        "#floating-point-instructions-add",
        "#floating-point-instructions-add",
    ]


def test_pick_spelling_resolves_concrete_spellings(match_records):
    # The one matcher the LSP and the `isa` CLI share resolves each
    # concrete spelling to the row and the section that documents it.
    # add.f32 is floating point, add.s32 is integer: one merged `add`
    # row, two sections.
    cases = [
        (
            "ld.global.nc.f32",
            "%f1, [%rd1]",
            "ld.global.nc",
            "#data-movement-and-conversion-instructions-ld-global-nc",
        ),
        (
            "ld.global.cg.nc.f32",
            "%f1, [%rd1]",
            "ld.global.nc",
            "#data-movement-and-conversion-instructions-ld-global-nc",
        ),
        ("add.f32", "%f1, %f2, %f3", "add", "#floating-point-instructions-add"),
        ("add.s32", "%r1, %r2, %r3", "add", "#integer-arithmetic-instructions-add"),
        (
            "vadd.u32.s32.sat",
            "%r1, %r2, %r3",
            "vop.dtype.atype.btype",
            "#scalar-video-instructions-vadd-vsub-vabsdiff-vmin-vmax",
        ),
    ]
    for spelling, operands, mnemonic, url in cases:
        pick = pick_spelling(match_records, spelling, operands)
        assert pick is not None, spelling
        assert pick.mnemonic == mnemonic
        assert pick.metadata["url"] == url
    # A part no template line's vocabulary accepts names no row.
    assert pick_spelling(match_records, "add.z32", "%r1, %r2, %r3") is None
