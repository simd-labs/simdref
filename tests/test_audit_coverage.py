"""Regression tests for tools/audit_coverage.py upstream extractors.

Covers issue #4: the AARCHMRS upstream mnemonic extractor was walking every
``name`` field in the JSON tree, pulling in InstructionSet / group / feature
node names (``A``, ``A1B``, ``ALIGN``, ``ASIMDALL``) alongside real mnemonics.
That made ``arm-a64`` coverage collapse to 0.045 in the 2026-04-23 drift run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import audit_coverage  # type: ignore[import-not-found]  # noqa: E402


def _aarchmrs_payload_with_noise() -> dict:
    """Minimal AARCHMRS-like tree: one real Instruction plus a scaffolding of
    InstructionSet / group nodes whose ``name`` fields must NOT be mistaken
    for mnemonics."""
    return {
        "instructions": [
            {
                "_type": "Instruction.InstructionSet",
                "name": "A64",
                "children": [
                    {
                        "_type": "Instruction.InstructionGroup",
                        "name": "ASIMDALL",  # group name — must be ignored
                        "children": [
                            {
                                "_type": "Instruction.InstructionGroup",
                                "name": "ALIGN",  # group name — must be ignored
                                "children": [
                                    {
                                        "_type": "Instruction.Instruction",
                                        "name": "aarch64/instrs/integer/arithmetic/add/Instruction",  # noqa: E501
                                        "assembly": {
                                            "symbols": [
                                                {
                                                    "_type": "Instruction.Symbols.Literal",
                                                    "value": "ADD",
                                                },
                                                {
                                                    "_type": "Instruction.Symbols.Literal",
                                                    "value": " ",
                                                },
                                                {
                                                    "_type": "Instruction.Symbols.RuleReference",
                                                    "rule_id": "Xd",
                                                },
                                            ],
                                        },
                                    },
                                    {
                                        "_type": "Instruction.InstructionAlias",
                                        "name": "ALIASED",  # alias node name — ignore
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    }


def test_arm_a64_extractor_ignores_group_names():
    """Group/InstructionSet node names must not show up as mnemonics."""
    payload = _aarchmrs_payload_with_noise()
    names = audit_coverage._extract_arm_a64_mnemonics(json.dumps(payload))
    assert names == {"ADD"}, f"expected only real mnemonics, got {names}"
    # Specifically, the contaminating tokens from the 2026-04-23 drift must be absent.
    assert "A64" not in names
    assert "ASIMDALL" not in names
    assert "ALIGN" not in names
    assert "ALIASED" not in names


def test_arm_a64_extractor_handles_fixture_form():
    """Non-AARCHMRS payloads (simple ``mnemonic`` lists) still work."""
    payload = {
        "format": "arm-instructions-fixture-v1",
        "instructions": [
            {"mnemonic": "FADD", "operands": "Vd.4S, Vn.4S, Vm.4S"},
            {"mnemonic": "FMUL", "operands": "Vd.4S, Vn.4S, Vm.4S"},
        ],
    }
    names = audit_coverage._extract_arm_a64_mnemonics(json.dumps(payload))
    assert {"FADD", "FMUL"}.issubset(names)


def test_arm_a64_extractor_handles_wrapped_instructions_json():
    """Payload-wrapping form (``instructions_json`` as embedded string) still works."""
    inner = [{"mnemonic": "LDR"}, {"mnemonic": "STR"}]
    payload = {"format": "arm-aarchmrs-instructions-v1", "instructions_json": json.dumps(inner)}
    names = audit_coverage._extract_arm_a64_mnemonics(json.dumps(payload))
    assert {"LDR", "STR"}.issubset(names)


def test_arm_a64_extractor_scopes_to_a64_instruction_set():
    """FAT AARCHMRS archives ship A64 + A32 + T32. ``arm-a64`` must only
    count the A64 subtree — otherwise A32-only mnemonics (BKPT, BLX, BX,
    CPSID, …) inflate the 'missing' set and push coverage below threshold."""
    payload = {
        "instructions": [
            {
                "_type": "Instruction.InstructionSet",
                "name": "A64",
                "children": [
                    {
                        "_type": "Instruction.Instruction",
                        "assembly": {
                            "symbols": [
                                {"_type": "Instruction.Symbols.Literal", "value": "ADD"},
                            ],
                        },
                    },
                ],
            },
            {
                "_type": "Instruction.InstructionSet",
                "name": "A32",
                "children": [
                    {
                        "_type": "Instruction.Instruction",
                        "assembly": {
                            "symbols": [
                                {"_type": "Instruction.Symbols.Literal", "value": "BKPT"},
                            ],
                        },
                    },
                ],
            },
            {
                "_type": "Instruction.InstructionSet",
                "name": "T32",
                "children": [
                    {
                        "_type": "Instruction.Instruction",
                        "assembly": {
                            "symbols": [
                                {"_type": "Instruction.Symbols.Literal", "value": "IT"},
                            ],
                        },
                    },
                ],
            },
        ],
    }
    names = audit_coverage._extract_arm_a64_mnemonics(json.dumps(payload))
    assert names == {"ADD"}


def test_arm_a64_extractor_returns_empty_on_unparseable():
    assert audit_coverage._extract_arm_a64_mnemonics("not json") == set()


def test_intel_extractor_reads_datajs_wrapper():
    """A data.js wrapper (backslash-continued JS string) yields the names the
    catalog ingest parser reads."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<intrinsics_list version="3.6.4">',
        '  <intrinsic name="_mm_add_ps" rettype="__m128"/>',
        '  <intrinsic name="no_underscore"/>',
        "</intrinsics_list>",
    ]
    text = 'var data_js = "' + "\\n".join(lines) + '";'
    assert audit_coverage._extract_intel_names(text) == {"_mm_add_ps"}


def test_intel_extractor_reads_json_and_ignores_garbage():
    text = json.dumps([{"name": "_mm256_add_pd"}, {"name": "other"}])
    assert audit_coverage._extract_intel_names(text) == {"_mm256_add_pd"}
    assert audit_coverage._extract_intel_names("not a payload") == set()
