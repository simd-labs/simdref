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


def _node(mnemonic: str, name: str, *, title: str = "", aliases: tuple = ()):
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
    return node


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
    briefs = parse_a64xml_briefs(xml_archive)
    assert briefs["ABS"] == "Absolute value"
    assert "Decoy brief" not in briefs.values()


def test_shared_names_keep_the_first_file_in_sorted_order(xml_archive):
    # cbcc_regs.xml sorts before cblo_regs.xml and both name CBHI.
    assert parse_a64xml_briefs(xml_archive)["CBHI"] == "Compare registers and branch"


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


def test_alias_rows_take_their_own_brief(xml_archive):
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
    # The alias's own brief beats the target title; a name the release does
    # not carry keeps the target title (#69 behaviour).
    assert _summary(rows, "ABS") == "Absolute value."
    assert _summary(rows, "QQQQ") == "Move to System register."
