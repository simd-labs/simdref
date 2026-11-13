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

PTX = "https://docs.nvidia.com/cuda/parallel-thread-execution/"


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
    assert len(rows) == 8


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
    assert row["url"] == PTX + "#widget-instructions-frob"


def test_description_holds_each_sub_section_in_page_order(rows):
    # Each rubric sub-section becomes one entry under its own heading,
    # in page order, and code blocks keep their line breaks. The title
    # line (name: brief) is not a sub-section.
    row = next(r for r in rows if r["mnemonic"] == "floss.sync")
    description = row["description"]
    assert list(description) == [
        "Syntax",
        "Description",
        "Semantics",
        "Notes",
        "PTX ISA Notes",
        "Target ISA Notes",
        "Examples",
    ]
    assert row["brief"] == "Floss one widget against another."
    assert description["Syntax"] == "floss.sync.mode.type d, [a], membermask;"
    assert description["Description"] == (
        "Supports e.g. Rounding and saturation modes. The second sentence is not the summary.\n"
        "A second paragraph stays a separate line."
    )
    assert description["Semantics"] == "Writes d with the flossed widget value."
    assert description["Notes"] == "Only the first sentence of Description sets the summary."
    assert description["PTX ISA Notes"] == "floss introduced in PTX ISA version 7.0."
    assert description["Target ISA Notes"] == "floss requires sm_80 or higher."
    assert description["Examples"] == (
        "floss.sync.cta.u32 %d, [%a], %m;\nfloss.sync.cluster.u32 %d, [%a], %m;"
    )


def test_summary_is_the_title_brief_first_sentence(records):
    record = next(r for r in records if r.mnemonic == "floss.sync")
    assert record.summary == (
        "Floss one widget against another. Introduced in PTX ISA 7.0. Requires sm_80 or higher."
    )


def test_title_brief_is_not_a_description_section():
    # The instruction section's title line (name: brief) is not a
    # description section: its brief leads the summary before the ISA
    # and sm facts, and the Description section keeps its own text.
    fragment = (
        '<section id="widget-instructions-gret">'
        "<h2>Widget Instructions: gret</h2>"
        '<p class="rubric"><code class="docutils literal notranslate">'
        '<span class="pre">gret</span></code></p>'
        "<p>Greets one widget against another.</p>"
        '<p class="rubric">Syntax</p>'
        '<div class="highlight-text notranslate"><div class="highlight"><pre>'
        "<span></span>gret.b32 d, [a];\n</pre></div></div>"
        '<p class="rubric">Description</p>'
        "<p>Greets the widget operand in full.</p>"
        '<p class="rubric">PTX ISA Notes</p>'
        "<p>gret introduced in PTX ISA version 7.0.</p>"
        '<p class="rubric">Target ISA Notes</p>'
        "<p>gret requires sm_80 or higher.</p>"
        "</section>"
    )
    rows = derive_rows(fragment)
    records = parse_ptx_instructions_payload(
        json.dumps({"format": BUNDLE_FORMAT, "instructions": rows})
    )
    record = records[0]
    assert "gret" not in record.description
    assert list(record.description) == [
        "Syntax",
        "Description",
        "PTX ISA Notes",
        "Target ISA Notes",
    ]
    assert record.description["Description"] == "Greets the widget operand in full."
    assert record.summary == (
        "Greets one widget against another. Introduced in PTX ISA 7.0. Requires sm_80 or higher."
    )


def test_summary_stays_generated_without_description():
    # A section with no Description sub-section keeps the generated line.
    fragment = (
        '<section id="widget-instructions-groq">'
        "<h2>Widget Instructions: groq</h2>"
        '<p class="rubric">Syntax</p>'
        '<div class="highlight-text notranslate"><div class="highlight"><pre>'
        "<span></span>groq d;\n</pre></div></div>"
        '<p class="rubric">PTX ISA Notes</p>'
        "<p>groq introduced in PTX ISA version 8.0.</p>"
        "</section>"
    )
    rows = derive_rows(fragment)
    records = parse_ptx_instructions_payload(
        json.dumps({"format": BUNDLE_FORMAT, "instructions": rows})
    )
    record = records[0]
    assert list(record.description) == ["Syntax", "PTX ISA Notes"]
    assert record.summary == ("groq instruction in Widget Instructions. Introduced in PTX ISA 8.0.")


def test_nested_sections_stay_out_of_the_parent_description():
    # A family overview section holds its lead-in prose and its syntax
    # block in a nested instruction section. The parent's description
    # keeps its own text only.
    fragment = (
        '<section id="family-instructions-grok">'
        "<h2>Family Instructions: grok</h2>"
        "<p>Overview of the grok family.</p>"
        '<section id="family-instructions-grok-init">'
        "<h3>Family Instructions: grok.init</h3>"
        '<p class="rubric">Syntax</p>'
        '<div class="highlight-text notranslate"><div class="highlight"><pre>'
        "<span></span>grok.init.b64 [a], count;\n"
        "grok.try.b64 [a];\n"
        "</pre></div></div>"
        '<p class="rubric">Description</p>'
        "<p>Initializes a grok object.</p>"
        "</section>"
        "</section>"
    )
    rows = derive_rows(fragment)
    parent = next(r for r in rows if r["url"].endswith("#family-instructions-grok"))
    assert "grok" not in parent["description"]
    assert parent["brief"] == "Overview of the grok family."
    assert "Initializes" not in parent["brief"]
    assert parent["description"]["Syntax"] == "grok.init.b64 [a], count;"


def test_merged_brief_prefers_the_name_line_over_the_lead():
    # A family section's lead-in steals the first nested section's syntax
    # rows; when the rows merge, the name line's brief leads the summary.
    fragment = (
        '<section id="family-instructions-grok">'
        "<h2>Family Instructions: grok</h2>"
        "<p>Overview of the grok family.</p>"
        '<section id="family-instructions-grok-init">'
        "<h3>Family Instructions: grok.init</h3>"
        '<p class="rubric"><code class="docutils literal notranslate">'
        '<span class="pre">grok.init</span></code></p>'
        "<p>Initializes a grok object.</p>"
        '<p class="rubric">Syntax</p>'
        '<div class="highlight-text notranslate"><div class="highlight"><pre>'
        "<span></span>grok.init.b64 [a], count;\n</pre></div></div>"
        '<p class="rubric">Description</p>'
        "<p>Creates the grok state.</p>"
        "</section>"
        "</section>"
    )
    rows = derive_rows(fragment)
    records = parse_ptx_instructions_payload(
        json.dumps({"format": BUNDLE_FORMAT, "instructions": rows})
    )
    assert records[0].summary == "Initializes a grok object."


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
    assert flop["url"] == PTX + "#cache-instructions-flop-floop"
    assert flop["ptx_isa"] == []
    assert flop["sm"] == []


def test_section_without_syntax_yields_no_rows(rows):
    assert not any(row["mnemonic"].startswith("zzz") for row in rows)


def test_colliding_template_lines_merge_by_key():
    # Two template lines with one (mnemonic, form) share one catalog key:
    # the row lists both lines, unions the qualifier sets and shares one
    # description.
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
        '<p class="rubric">Description</p>'
        "<p>Grobs a widget.</p>"
        "</section>"
    )
    rows = derive_rows(fragment)
    assert len(rows) == 1
    assert rows[0]["mnemonic"] == "grob"
    assert rows[0]["syntax"] == "grob.type1 d, [a];\ngrob{.sat}.type2 d, [a];"
    assert rows[0]["qualifiers"]["sat"] == [".false"]
    assert rows[0]["description"] == {
        "Syntax": "grob.type1 d, [a];\ngrob{.sat}.type2 d, [a];",
        "Description": "Grobs a widget.",
    }


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
    assert record.description["Description"] == (
        "Read the widget operand and write the destination register."
    )
    assert record.metadata["url"] == PTX + "#widget-instructions-frob"
    assert record.metadata["ptx_isa"] == "2.0, 3.1"
    assert record.metadata["sm"] == "sm_75, sm_80"
    summary = record.summary
    assert summary == (
        "Copy a widget from one register file to another."
        " Introduced in PTX ISA 2.0."
        " Requires sm_75 or higher."
    )


def test_raw_html_payload(fixture_html):
    records = parse_ptx_instructions_payload(fixture_html)
    assert len(records) == 8
    assert records[0].mnemonic == "frob"


def test_doc_syntax_block_keeps_every_pre_line():
    # The Syntax code block of the doc entry keeps every line of the
    # pre element: the template lines and the qualifier set lines. A
    # one-line truncation drops the `.mode = { ... };` line of the SHF
    # and SHFL rows.
    fragment = (
        '<section id="data-movement-and-conversion-instructions-shfl-sync">'
        "<h4>Data Movement and Conversion Instructions: shfl.sync</h4>"
        '<p class="rubric">Syntax</p>'
        '<div class="highlight-text notranslate"><div class="highlight"><pre>'
        "<span></span>shfl.sync.mode.b32  d[|p], a, b, c, membermask;\n"
        ".mode = { .up, .down, .bfly, .idx };\n"
        "</pre></div></div>"
        '<p class="rubric">Description</p>'
        "<p>Exchanges data between threads.</p>"
        "</section>"
    )
    rows = derive_rows(fragment)
    assert len(rows) == 1
    row = rows[0]
    syntax = next(
        section
        for entry in row["doc_sources"]
        for section in entry["sections"]
        if section["heading"] == "Syntax"
    )
    code = next(block for block in syntax["blocks"] if block["kind"] == "code")
    assert code["text"] == (
        "shfl.sync.mode.b32  d[|p], a, b, c, membermask;\n.mode = { .up, .down, .bfly, .idx };"
    )


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
        PTX + "#integer-arithmetic-instructions-add",
        PTX + "#integer-arithmetic-instructions-add",
        PTX + "#floating-point-instructions-add",
        PTX + "#floating-point-instructions-add",
    ]


def test_merged_rows_carry_every_sections_description(match_records):
    # Rows of one section share its text; a key merged across sections
    # concatenates the texts in page order. The title briefs merge the
    # same way and lead the summary.
    add = next(r for r in match_records if r.mnemonic == "add" and r.form == "d, a, b")
    assert "add" not in add.description
    assert add.description["Description"] == (
        "Performs integer addition.\nPerforms floating-point addition with rounding."
    )
    assert add.summary == "Add integer values."
    assert len(add.description["Syntax"].split("\n")) == 4


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
            PTX + "#data-movement-and-conversion-instructions-ld-global-nc",
        ),
        (
            "ld.global.cg.nc.f32",
            "%f1, [%rd1]",
            "ld.global.nc",
            PTX + "#data-movement-and-conversion-instructions-ld-global-nc",
        ),
        ("add.f32", "%f1, %f2, %f3", "add", PTX + "#floating-point-instructions-add"),
        ("add.s32", "%r1, %r2, %r3", "add", PTX + "#integer-arithmetic-instructions-add"),
        (
            "vadd.u32.s32.sat",
            "%r1, %r2, %r3",
            "vop.dtype.atype.btype",
            PTX + "#scalar-video-instructions-vadd-vsub-vabsdiff-vmin-vmax",
        ),
    ]
    for spelling, operands, mnemonic, url in cases:
        pick = pick_spelling(match_records, spelling, operands)
        assert pick is not None, spelling
        assert pick.mnemonic == mnemonic
        assert pick.metadata["url"] == url
    # A part no template line's vocabulary accepts names no row.
    assert pick_spelling(match_records, "add.z32", "%r1, %r2, %r3") is None
