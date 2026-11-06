"""catkey.py: catalog-key parsing for the form matcher."""

from simdref.catkey import (
    IMMEDIATE_ACCEPT,
    build_key,
    form_accept,
    key_operand_bits,
    operands_of_key,
    split_isa_parts,
)


def test_operands_of_key_splits_at_top_level_commas():
    # A pre-index key address holds its own commas and reads as one operand.
    assert operands_of_key("LDR (<BT>, [<XNSP>, <HASH>#0]!)") == [
        "<BT>",
        "[<XNSP>, <HASH>#0]!",
    ]
    assert operands_of_key("ADD (M32, I8)") == ["M32", "I8"]
    assert operands_of_key("NOP") == []


def test_key_operand_bits_reads_the_first_digit_run():
    assert key_operand_bits("ADD (M32, I8)") == (32, 8)
    assert key_operand_bits("MOV (AL, Moffs8)") == (None, 8)
    assert key_operand_bits("LDR (<BT>, [<XNSP>, <HASH>#0]!)") == (None, 0)


def test_form_accept_maps_key_operands_to_line_classes():
    assert form_accept("i8") == IMMEDIATE_ACCEPT
    assert form_accept("0") == frozenset({"lit0"})
    assert form_accept("m16_1to8") == frozenset({"mbc"})
    assert form_accept("m32") == frozenset({"m32", "m"})
    assert form_accept("R8h") == frozenset({"r8"})
    assert form_accept("Moffs8") == frozenset({"moffs"})
    assert form_accept("rel32") == frozenset({"rel"})


def test_form_accept_maps_placeholders_to_line_classes():
    # A key repeat is not part of the placeholder name.
    assert form_accept("<HASH>#0#0") == IMMEDIATE_ACCEPT
    assert form_accept("[<XNSP>, <HASH>#0]!") == frozenset({"mpre"})
    assert form_accept("Z0.4S") == frozenset({"av", "avs"})
    # The Z arrangement element class is lowercase: it names the line class.
    assert form_accept("Z1.B") == frozenset({"av", "avb"})
    assert form_accept("D0") == frozenset({"d"})
    assert form_accept("<BT>") == frozenset({"b"})
    assert form_accept("<WORX_EXTEND>") == frozenset({"w", "x", "we", "xe"})


def test_build_key_normalizes_name_and_form():
    assert build_key(" add ", " m32,  i8 ") == "ADD (M32, I8)"
    assert build_key("nop", "  ") == "NOP"
    assert build_key(" ", "M32") == ""


def test_split_isa_parts_splits_on_separators_and_space_runs():
    assert split_isa_parts("SVE2 / SME, NEON||  Base  x ") == [
        "SVE2",
        "SME",
        "NEON",
        "Base",
        "x",
    ]
    assert split_isa_parts("AArch64 Base") == ["AArch64 Base"]
