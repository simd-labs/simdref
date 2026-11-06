"""catkey.py: catalog-key parsing for the form matcher."""

import json

from simdref.arm_instructions import parse_arm_instruction_payload
from simdref.catkey import (
    IMMEDIATE_ACCEPT,
    build_key,
    form_accept,
    key_operand_bits,
    operands_of_key,
    split_isa_parts,
)
from simdref.ingest_catalog import link_records
from simdref.models import IntrinsicRecord


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


def test_ba_placeholder_row_resolves():
    # Pin the one row class the merged placeholder table fixes: a ref key
    # naming the Ba role resolves to the row that renders b3. Dropping "Ba"
    # from RULE_PLACEHOLDER leaves the row as "<BA>" and the ref unresolved.
    payload = {
        "instructions": [
            {
                "_type": "Instruction.InstructionSet",
                "name": "A64",
                "children": [
                    {
                        "_type": "Instruction.Instruction",
                        "operation_id": "test_ba_row",
                        "assembly": {
                            "_type": "Instruction.Assembly",
                            "symbols": [
                                {"_type": "Instruction.Symbols.Literal", "value": "FMLA "},
                                {"_type": "Instruction.Symbols.RuleReference", "rule_id": "Bd"},
                                {"_type": "Instruction.Symbols.Literal", "value": ", "},
                                {"_type": "Instruction.Symbols.RuleReference", "rule_id": "Bn"},
                                {"_type": "Instruction.Symbols.Literal", "value": ", "},
                                {"_type": "Instruction.Symbols.RuleReference", "rule_id": "Ba"},
                            ],
                        },
                    }
                ],
            }
        ]
    }
    rows = parse_arm_instruction_payload(json.dumps(payload))
    assert [row.key for row in rows] == ["FMLA (B0, B1, B3)"]
    intrinsic = IntrinsicRecord(
        name="vtest_ba",
        signature="float8_t vtest_ba(float8_t, float8_t, float8_t)",
        description="Test intrinsic for the Ba placeholder row.",
        header="arm_neon.h",
        architecture="arm",
        instruction_refs=[{"name": "FMLA", "form": "Bd, Bn, Ba"}],
    )
    link_records([intrinsic], rows)
    ref = intrinsic.instruction_refs[0]
    assert ref["resolution"] == "key"
    assert ref["display_key"] == rows[0].key
