"""Issue #65 B2/B6: lines of an unsupported ISA never borrow a supported row.

A line every supported assembler rejects (PowerPC ``fadds 1, 2, 3``, MIPS
``lw $t0, 0($t1)``, ARM32 ``add r3, r4, r5``) is foreign: annotate reports
known=false instead of borrowing an Arm, RISC-V or x86 row, and hover agrees.
The acceptance probes need the real llvm-mc, so the tests skip without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdref.annotate import AnnotateOptions, _record_takes_line, annotate_stream
from simdref.asm_identity import _tool_path
from simdref.lsp import _hover_markdown
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.storage import build_sqlite, open_db

LLVM_MC = _tool_path("SIMDREF_LLVM_MC", "llvm-mc") or ""
needs_tools = pytest.mark.skipif(not LLVM_MC, reason="llvm-mc absent (set SIMDREF_LLVM_MC)")

ROWS = [
    # enc_* are the real upstream encoding keys of these forms (issue #65
    # repro rows), so byte identity resolves the supported lines end to end.
    InstructionRecord(
        mnemonic="ADD",
        form="ADD (<XDORXZR__6>, <XNORXZR__12>, <XMORXZR__4>#0)",
        summary="Add optionally-shifted register.",
        architecture="arm",
        isa=["A64"],
        enc_bits=32,
        enc_mask=4280287232,
        enc_match=2332033024,
    ),
    InstructionRecord(
        mnemonic="add",
        form="add",
        summary="Integer add.",
        architecture="riscv",
        isa=["I"],
        enc_bits=32,
        enc_mask=4261441663,
        enc_match=51,
        enc_extra=[["", 32, 4261441663, 51]],
    ),
    InstructionRecord(
        mnemonic="ADD",
        form="ADD (M16, I16)",
        summary="Add 16-bit integer operands.",
        architecture="x86",
        isa=["I86"],
    ),
    InstructionRecord(
        mnemonic="FADD",
        form="FADD (ST, ST(0))",
        summary="Fadd 80-bit operands.",
        architecture="x86",
        isa=["X87"],
    ),
    InstructionRecord(
        mnemonic="FADD",
        form="FADD (D0, D1, D2)",
        summary="Floating-point Add (vector).",
        architecture="arm",
        isa=["NEON"],
        enc_bits=32,
        enc_mask=4292934656,
        enc_match=509618176,
    ),
    InstructionRecord(
        mnemonic="lw",
        form="lw",
        summary="Load word.",
        architecture="riscv",
        isa=["I"],
        enc_bits=32,
        enc_mask=28799,
        enc_match=8195,
        enc_extra=[["", 32, 28799, 8195]],
    ),
]

# (line, known): the foreign lines of B2 and B6 plus the lines that must keep
# today's answer (A64, RISC-V, wasm, the ARM32 branch with no catalog rows).
CASES = [
    ("fadds 1, 2, 3", False),  # PowerPC
    ("lw $t0, 0($t1)", False),  # MIPS
    ("add r3, r4, r5", False),  # ARM32
    ("bx lr", False),  # ARM32, no catalog rows at all
    ("add x0, x1, x2", True),  # AArch64
    ("lw t0, 0(t1)", True),  # RISC-V
    ("i32.add", False),  # wasm
]


def _db(tmp_path: Path) -> Path:
    db = tmp_path / "catalog.db"
    catalog = Catalog(
        intrinsics=[],
        instructions=ROWS,
        sources=[SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")],
        generated_at="2025-01-01",
    )
    build_sqlite(catalog, db)
    return db


def _known(conn, line: str) -> bool:
    opts = AnnotateOptions(fmt="json")
    out = "".join(annotate_stream([line + "\n"], opts=opts, conn=conn))
    return json.loads(out)[0]["known"]


@needs_tools
@pytest.mark.parametrize(("line", "known"), CASES)
def test_foreign_lines_stay_unknown(tmp_path: Path, line: str, known: bool):
    with open_db(_db(tmp_path)) as conn:
        assert _known(conn, line) is known


@needs_tools
def test_foreign_batch_stays_unknown(tmp_path: Path):
    # One stream: the stream family must not answer the foreign lines either.
    text = "".join(f"{line}\n" for line, _known_flag in CASES)
    with open_db(_db(tmp_path)) as conn:
        out = "".join(
            annotate_stream(
                text.splitlines(keepends=True), opts=AnnotateOptions(fmt="json"), conn=conn
            )
        )
    records = json.loads(out)
    assert [r["known"] for r in records] == [known for _line, known in CASES]


@needs_tools
def test_hover_agrees_with_annotate(tmp_path: Path):
    # Hover reads the same catalog through the shared lsp matcher: the ARM32
    # register spellings name no candidate row, so hover shows nothing.
    with open_db(_db(tmp_path)) as conn:
        for line, known in CASES:
            word, _space, operands = line.partition(" ")
            markdown = _hover_markdown(
                conn,
                word,
                operand_count=len(operands.split(",")) if operands else 0,
                operand_text=operands,
                intel=False,
            )
            assert (markdown is not None) is known, line
            assert _known(conn, line) is known, line


# R1 (review of 320dee2): an x86 Intel line with an immediate answers from
# the pool when byte identity misses. The veto must zip the line's operand
# classes against the key in the line's own order: AT&T flips (source first),
# Intel and sigil-less lines do not.
IMM_ROWS = [
    InstructionRecord(
        mnemonic="SUB",
        form="SUB (R64, I8)",
        summary="Subtract register and immediate operands.",
        architecture="x86",
        isa=["I86"],
    ),
    InstructionRecord(
        mnemonic="ADD",
        form="ADD (R64, I8)",
        summary="Add register and immediate operands.",
        architecture="x86",
        isa=["I86"],
    ),
]

# (line, known): the R1 Intel lines and their AT&T forms keep the answer the
# mnemonic pool gives them, so the veto must pass in both operand orders.
IMM_CASES = [
    ("sub rsp, 8", True),
    ("sub rsp, 0x8", True),
    ("add rax, 1", True),
    ("sub $8, %rsp", True),
    ("add $1, %rax", True),
]


def test_veto_zips_classes_in_line_order():
    # The rows carry no encoding keys, so the pick path alone decides.
    sub, add = IMM_ROWS
    assert _record_takes_line(sub, "rsp, 8") is True  # Intel order, no flip
    assert _record_takes_line(sub, "rsp, 0x8") is True
    assert _record_takes_line(add, "rax, 1") is True
    assert _record_takes_line(sub, "$8, %rsp") is True  # AT&T order, flipped
    reg_only = InstructionRecord(
        mnemonic="ADD",
        form="ADD (R64, R64)",
        summary="Add register operands.",
        architecture="x86",
        isa=["I86"],
    )
    assert _record_takes_line(reg_only, "rax, 1") is False  # B2 veto stands
    fadd = InstructionRecord(
        mnemonic="FADD",
        form="FADD (D0, D1, D2)",
        summary="Floating-point Add (vector).",
        architecture="arm",
        isa=["NEON"],
    )
    assert _record_takes_line(fadd, "1, 2, 3") is False  # B2 veto stands


@needs_tools
def test_intel_immediate_batch_keeps_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The R1 batch context is byte identity missing for the line, which is why
    # no IMM_ROWS row carries an encoding key: the name-lookup pick decides.
    monkeypatch.setattr(
        "simdref.annotate.identity_keys", lambda texts, *a, **k: [None] * len(texts)
    )
    db = tmp_path / "catalog.db"
    catalog = Catalog(
        intrinsics=[],
        instructions=ROWS + IMM_ROWS,
        sources=[SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")],
        generated_at="2025-01-01",
    )
    build_sqlite(catalog, db)
    text = "".join(f"{line}\n" for line, _known_flag in IMM_CASES)
    with open_db(db) as conn:
        out = "".join(
            annotate_stream(
                text.splitlines(keepends=True), opts=AnnotateOptions(fmt="json"), conn=conn
            )
        )
    records = json.loads(out)
    assert [r["known"] for r in records] == [known for _line, known in IMM_CASES]
