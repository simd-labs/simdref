"""Issue #65 m1: rows with a placeholder summary take the A64 XML brief.

The fixture XML files are byte-exact copies from the Arm A64 ISA XML
release (ISA_A64_xml_A_profile-2026-06), except the one final newline the
end-of-file hook adds (the release files end without one). Reverting the
brief lookup in arm_instructions.py must fail every test here.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

import simdref.arm_instructions as arm_instructions
from simdref import ingest_sources
from simdref.arm_instructions import parse_arm_instruction_payload
from simdref.ingest_sources import a64xml_briefs_from_xml, parse_a64xml_briefs

FIXTURES = Path(__file__).parent / "fixtures" / "issue_65" / "a64_xml"
RELEASE = "ISA_A64_xml_A_profile-2026-06"


def _xml(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture()
def xml_archive(tmp_path, monkeypatch):
    """The fixture XMLs as one archive, plus decoys the parser must skip."""
    archive = tmp_path / f"{RELEASE}.tar.gz"
    decoy = tmp_path / "decoy.xml"
    decoy.write_text(
        _xml("abs.xml").replace("Absolute value", "Decoy brief"),
        encoding="utf-8",
    )
    with tarfile.open(archive, "w:gz") as tf:
        for path in sorted(FIXTURES.glob("*.xml")):
            tf.add(path, arcname=f"{RELEASE}/{path.name}")
        tf.add(decoy, arcname="ISA_A64_xml_A_profile-2026-03/decoy.xml")
        tf.add(decoy, arcname=f"{RELEASE}/nested/decoy.xml")
        # A .dtd member like the release's own, not standalone XML.
        dtd = tmp_path / "junk.dtd"
        dtd.write_text("<!ELEMENT toptitle EMPTY>\n", encoding="utf-8")
        tf.add(dtd, arcname=f"{RELEASE}/junk.dtd")
    monkeypatch.setattr(ingest_sources, "LOCAL_ARM_A64_XML_ARCHIVES", [archive])
    arm_instructions._a64xml_summaries.cache_clear()
    yield archive
    arm_instructions._a64xml_summaries.cache_clear()


def _node(
    mnemonic: str,
    name: str,
    *,
    title: str = "",
    aliases: tuple = (),
    alias_children: tuple = (),
    operation_id: str = "",
):
    children = [
        {
            "_type": "Instruction.InstructionAlias",
            "name": alias_name,
            "condition": None,
            "assembly": {
                "_type": "Instruction.Assembly",
                "symbols": [{"_type": "Instruction.Symbols.Literal", "value": alias_name}],
            },
        }
        for alias_name in aliases
    ]
    children += list(alias_children)
    node = {
        "_type": "Instruction.Instruction",
        "name": name,
        "assembly": {
            "_type": "Instruction.Assembly",
            "symbols": [
                {"_type": "Instruction.Symbols.Literal", "value": mnemonic},
                {"_type": "Instruction.Symbols.RuleReference", "rule_id": "SPACE"},
                {"_type": "Instruction.Symbols.RuleReference", "rule_id": "Rd"},
            ],
        },
        "children": children,
    }
    if title:
        node["title"] = title
    if operation_id:
        node["operation_id"] = operation_id
    return node


def _alias_child(name: str, *, operation_id: str = "", condition=None):
    return {
        "_type": "Instruction.InstructionAlias",
        "name": name,
        "condition": condition,
        "operation_id": operation_id,
        "assembly": {
            "_type": "Instruction.Assembly",
            "symbols": [{"_type": "Instruction.Symbols.Literal", "value": name}],
        },
    }


def _tree(nodes, group: str = "dp"):
    return {
        "instructions": [
            {
                "_type": "Instruction.InstructionSet",
                "name": "A64",
                "children": [
                    {"_type": "Instruction.InstructionGroup", "name": group, "children": nodes}
                ],
            }
        ]
    }


def _summary(rows, mnemonic):
    (row,) = [item for item in rows if item.mnemonic == mnemonic]
    return row.summary


def test_brief_and_names_from_one_instruction_file():
    assert a64xml_briefs_from_xml(_xml("abs.xml")) == {"ABS": "Absolute value"}


def test_non_xml_member_is_skipped():
    assert a64xml_briefs_from_xml("<!ELEMENT toptitle EMPTY>\n") == {}


def test_docvars_and_heading_key_the_same_brief():
    briefs = a64xml_briefs_from_xml(_xml("stfmaxnm.xml"))
    assert briefs == {
        "STFMAXNM": "Atomic floating-point maximum number, without return",
        "STFMAXNML": "Atomic floating-point maximum number, without return",
        "STFMAXNM, STFMAXNML": "Atomic floating-point maximum number, without return",
    }


def test_alias_names_share_the_alias_brief():
    briefs = a64xml_briefs_from_xml(_xml("cblo_regs.xml"))
    assert briefs == {
        "CBHI": "Compare unsigned lower than register and branch",
        "CBLO (REGISTER)": "Compare unsigned lower than register and branch",
        "CBLO_REGS": "Compare unsigned lower than register and branch",
    }


def test_entity_spelling_stays_in_keys():
    briefs = a64xml_briefs_from_xml(_xml("cbbcc_regs.xml"))
    assert briefs == {
        "CBB&LT;CC&GT;": "Compare bytes and branch",
        "CBBCC_REGS": "Compare bytes and branch",
        "CBBEQ": "Compare bytes and branch",
        "CBBGE": "Compare bytes and branch",
        "CBBGT": "Compare bytes and branch",
        "CBBHI": "Compare bytes and branch",
        "CBBHS": "Compare bytes and branch",
        "CBBNE": "Compare bytes and branch",
    }


def test_archive_reads_only_the_release_directory(xml_archive):
    by_file, by_name = parse_a64xml_briefs(xml_archive)
    assert by_name["ABS"] == "Absolute value"
    assert by_file["ABS"] == "Absolute value"
    assert "Decoy brief" not in by_name.values()
    assert "Decoy brief" not in by_file.values()


def test_conflicting_names_give_no_brief(xml_archive):
    # cbcc_regs.xml and cblo_regs.xml both name CBHI with different briefs.
    # No file order decides: the name carries no brief and each file keeps
    # its own brief under its file name (#65 m1 round 2).
    by_file, by_name = parse_a64xml_briefs(xml_archive)
    assert "CBHI" not in by_name
    assert by_file["CBCC_REGS"] == "Compare registers and branch"
    assert by_file["CBLO_REGS"] == "Compare unsigned lower than register and branch"


def test_placeholder_row_summary_takes_the_brief(xml_archive):
    rows = parse_arm_instruction_payload(json.dumps(_tree([_node("ABS", "ABS_32")])))
    assert _summary(rows, "ABS") == "Absolute value."


def test_generated_summary_stays_when_no_brief_names_the_row(xml_archive):
    rows = parse_arm_instruction_payload(json.dumps(_tree([_node("QQQQ", "QQQQ_x")])))
    assert _summary(rows, "QQQQ") == "Qqqq is an Arm instruction."


def test_sourced_row_text_is_untouched(xml_archive):
    rows = parse_arm_instruction_payload(
        json.dumps(_tree([_node("STFMAXNM", "stfmaxnm", title="Row title")]))
    )
    assert _summary(rows, "STFMAXNM") == "Row title."


def test_hand_kept_titles_win_over_the_brief(xml_archive):
    rows = parse_arm_instruction_payload(json.dumps(_tree([_node("FMLA", "FMLA_vec")])))
    assert _summary(rows, "FMLA") == "Floating-point fused Multiply-Add."


def test_acle_text_is_untouched(xml_archive, monkeypatch):
    monkeypatch.setattr(
        arm_instructions,
        "_acle_instruction_summaries",
        lambda: {"ABS": "ACLE says absolute value"},
    )
    rows = parse_arm_instruction_payload(
        json.dumps(_tree([_node("ABS", "ABS_32")], group="simd_dp"))
    )
    assert _summary(rows, "ABS") == "ACLE says absolute value."


def test_alias_rows_keep_the_target_title(xml_archive):
    rows = parse_arm_instruction_payload(
        json.dumps(
            _tree(
                [
                    _node(
                        "MSR",
                        "MSR_imm",
                        title="Move to System register",
                        aliases=("ABS", "QQQQ"),
                    )
                ]
            )
        )
    )
    # The target title is sourced text and keeps every alias row
    # byte-identical; the brief fills only placeholder or generated text
    # (#69 behaviour, #65 m1 round 2).
    assert _summary(rows, "ABS") == "Move to System register."
    assert _summary(rows, "QQQQ") == "Move to System register."


def test_bare_mov_takes_its_own_file_brief(xml_archive):
    rows = parse_arm_instruction_payload(
        json.dumps(
            _tree(
                [
                    _node(
                        "ORR",
                        "ORR_log_shift",
                        operation_id="ORR_log_shift",
                        alias_children=(_alias_child("MOV", operation_id="MOV_ORR_log_shift"),),
                    )
                ],
                group="log_shift",
            )
        )
    )
    # The row's operation_id names mov_orr_log_shift.xml ("Move register
    # value"). The bare MOV key names mov_dupm_z_i.xml first in sort order,
    # whose brief is DUPM's; file order must not decide (#65 m1 round 2).
    assert _summary(rows, "MOV") == "Move register value."


def test_acle_alias_text_is_untouched(xml_archive, monkeypatch):
    monkeypatch.setattr(
        arm_instructions,
        "_acle_instruction_summaries",
        lambda: {"MOV": "ACLE move text"},
    )
    rows = parse_arm_instruction_payload(
        json.dumps(
            _tree(
                [
                    _node(
                        "MOVA",
                        "mova_za_p_rz",
                        alias_children=(
                            _alias_child(
                                "MOV",
                                operation_id="MOV_ORR_log_shift",
                                condition={
                                    "_type": "AST.Function",
                                    "name": "IsFeatureImplemented",
                                    "arguments": [{"_type": "AST.Identifier", "value": "FEAT_SME"}],
                                },
                            ),
                        ),
                    )
                ],
                group="mortlach_insert_pred",
            )
        )
    )
    # ACLE text is sourced and ranks above the brief on alias rows too.
    assert _summary(rows, "MOV") == "ACLE move text."


def test_conflicting_brief_without_operation_id_gives_none(xml_archive):
    rows = parse_arm_instruction_payload(json.dumps(_tree([_node("CBHI", "CBcc_regs")])))
    # CBHI has conflicting briefs and the row names no file, so it keeps its
    # generated text (#65 m1 round 2).
    assert _summary(rows, "CBHI") == "Cbhi is an Arm instruction."
