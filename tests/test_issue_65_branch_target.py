"""Issue #65 M1: known must not depend on how objdump prints a branch target.

"je 7f37b" and "beqz a1,7d91c" resolve like "je foo+0xeb" (the <sym> form
after _drop_symbol_target) and "beqz a1,0x46": the probe spells a bare
target as the PC-relative fixup the <sym> form answers with. Each test
builds a small catalog and runs the real llvm-mc and XED, so the tests
skip when the tools are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdref.asm_identity import _reloc_branch_targets, _tool_path, _xed_facts, identify_lines
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.storage import build_sqlite

LLVM_MC = _tool_path("SIMDREF_LLVM_MC", "llvm-mc") or ""
XED = _tool_path("SIMDREF_XED", "xed") or ""
needs_tools = pytest.mark.skipif(
    not (LLVM_MC and XED), reason="llvm-mc or xed absent (set SIMDREF_LLVM_MC, SIMDREF_XED)"
)


def _db(records: list[InstructionRecord], tmp_path: Path) -> Path:
    db = tmp_path / "catalog.db"
    catalog = Catalog(
        intrinsics=[],
        instructions=records,
        sources=[SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")],
        generated_at="2025-01-01",
    )
    build_sqlite(catalog, db)
    return db


def _branch_catalog(tmp_path: Path) -> Path:
    iform = _xed_facts(["7405"], XED)[0]["iform"]
    jz = InstructionRecord(
        mnemonic="JZ",
        form="JZ (REL8)",
        summary="Jump if Condition Is Met.",
        architecture="x86",
        metadata={"iform": iform, "split_attrs": json.dumps({})},
    )
    # opcode and funct3 bits only: the operand and immediate fields are the
    # fixup placeholder on the .+ form and the concrete value on the 0x form.
    beq = InstructionRecord(
        mnemonic="beq",
        form="beq",
        summary="Branch if equal.",
        architecture="riscv",
        enc_bits=32,
        enc_mask=0x0000707F,
        enc_match=0x00000063,
    )
    return _db([jz, beq], tmp_path)


def test_reloc_spelling_leaves_working_text_untouched():
    assert _reloc_branch_targets("je 7f37b") == "je .+0x7f37b"
    assert _reloc_branch_targets("beqz a1,7d91c") == "beqz a1,.+0x7d91c"
    for line in (
        "beqz a1,0x46",
        "je foo+0xeb",
        "addi a0,a0,70",
        "mov $0x7f37b,%eax",
        "lui a1,0x7",
        "mov 0x1a4(%rsp),%eax",
    ):
        assert _reloc_branch_targets(line) == line


@needs_tools
def test_bare_target_rows_match_the_symbol_and_immediate_forms(tmp_path):
    db = _branch_catalog(tmp_path)
    items: list[tuple[str, str | None]] = [
        ("je 7f37b", None),
        ("je foo+0xeb", None),
        ("beqz a1,7d91c", None),
        ("beqz a1,0x46", None),
    ]
    results = identify_lines(items, catalog_path=db)
    rows = [result.row.db_key if result.row is not None else None for result in results]
    assert rows[0] is not None and rows[0] == rows[1]
    assert rows[2] is not None and rows[2] == rows[3]
    assert rows[0].startswith("x86:")
    assert rows[2].startswith("riscv:")
