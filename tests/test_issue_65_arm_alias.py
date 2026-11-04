"""Issue #65 B8 and m4: AArch64 instruction aliases become named catalog rows.

The fixture holds the SMSTART, SMSTOP, GCSPOPM, GCSSS1 and GCSSS2 alias nodes
and their three target instructions, copied from the vendored AARCHMRS
Instructions.json. Reverting the alias walk in arm_instructions.py must fail
every test here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdref.arm_instructions import parse_arm_instruction_payload
from simdref.asm_identity import _refine_subset, _tool_path, identify_lines
from simdref.models import Catalog, SourceVersion
from simdref.storage import build_sqlite

FIXTURE = Path(__file__).parent / "fixtures" / "issue_65" / "arm_alias_nodes.json"

# name -> (isa tags, llvm-mc-23 -show-encoding word, target mnemonic)
ALIASES = {
    "SMSTART": (["A64", "SME"], 0xD503477F, "MSR"),
    "SMSTOP": (["A64", "SME"], 0xD503467F, "MSR"),
    "GCSPOPM": (["A64", "GCS"], 0xD52B773F, "SYSL"),
    "GCSSS1": (["A64", "GCS"], 0xD50B7740, "SYS"),
    "GCSSS2": (["A64", "GCS"], 0xD52B7760, "SYSL"),
}


@pytest.fixture(scope="module")
def rows():
    return parse_arm_instruction_payload(FIXTURE.read_text())


def _by_mnemonic(rows, mnemonic):
    return [row for row in rows if row.mnemonic == mnemonic]


@pytest.mark.parametrize("name", sorted(ALIASES))
def test_alias_has_named_row(rows, name):
    isa, _word, _target = ALIASES[name]
    (row,) = _by_mnemonic(rows, name)
    assert row.form.startswith(name)
    assert row.isa == isa
    assert row.metadata["operation_id"].startswith(name)
    assert row.source == "arm-a64"


@pytest.mark.parametrize("name", sorted(ALIASES))
def test_alias_encoding_covers_target_and_matches_assembler(rows, name):
    _isa, word, target = ALIASES[name]
    (alias,) = _by_mnemonic(rows, name)
    (target_row,) = _by_mnemonic(rows, target)
    assert alias.enc_bits == 32
    assert word & alias.enc_mask == alias.enc_match
    assert word & target_row.enc_mask == target_row.enc_match
    # Strictly finer than the target, so byte identity prefers the alias.
    assert alias.enc_mask & target_row.enc_mask == target_row.enc_mask
    assert alias.enc_mask != target_row.enc_mask
    picked = _refine_subset([(row.enc_mask, row.enc_match, row) for row in (alias, target_row)])
    assert [row.mnemonic for _m, _x, row in picked] == [name]


def test_siblings_do_not_match_each_other(rows):
    for name, (_isa, word, _target) in ALIASES.items():
        for other in ALIASES:
            if other == name:
                continue
            (row,) = _by_mnemonic(rows, other)
            assert word & row.enc_mask != row.enc_match, (name, other)


def test_target_rows_unchanged(rows):
    payload = json.loads(FIXTURE.read_text())
    for top in payload["instructions"]:
        for group in top["children"]:
            for node in group["children"]:
                node["children"] = []
    stripped = parse_arm_instruction_payload(json.dumps(payload))
    named = {row.mnemonic for row in rows} - {row.mnemonic for row in stripped}
    assert named == set(ALIASES)
    assert [r for r in rows if r.mnemonic not in ALIASES] == stripped


needs_mc = pytest.mark.skipif(
    _tool_path("SIMDREF_LLVM_MC", "llvm-mc") is None, reason="llvm-mc not found"
)


@needs_mc
@pytest.mark.parametrize("line", ["smstart", "smstop"])
def test_smstart_line_names_alias_not_msr(rows, tmp_path, line):
    db = tmp_path / "catalog.db"
    build_sqlite(
        Catalog(
            intrinsics=[],
            instructions=rows,
            sources=[SourceVersion(source="t", version="t", fetched_at="2025", url="t://")],
            generated_at="2025",
        ),
        db,
    )
    (result,) = identify_lines([(line, None)], catalog_path=db)
    assert result.status == "row", result.reason
    assert result.row.mnemonic == line.upper()
