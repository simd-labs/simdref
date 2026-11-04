"""Issue #65: runtime identity defects in asm_identity.py.

Each test builds a small catalog and runs the real llvm-mc and XED, so the
tests skip when the tools are absent. The probe lines come from the issue.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdref.asm_identity import (
    _probe_lines_split,
    _tool_path,
    _xed_facts,
    identify_lines,
    isa_by_acceptance,
)
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.storage import build_sqlite

LLVM_MC = _tool_path("SIMDREF_LLVM_MC", "llvm-mc") or ""
XED = _tool_path("SIMDREF_XED", "xed") or ""
needs_tools = pytest.mark.skipif(
    not (LLVM_MC and XED), reason="llvm-mc or xed absent (set SIMDREF_LLVM_MC, SIMDREF_XED)"
)

X86_BATCH = [
    "mov %rax,%rbx",
    "add $1,%rax",
    "lea 8(%rsp),%rdi",
    "call foo",
    "ret",
    "nop",
    "push %rbp",
    "xor %eax,%eax",
    "imul %rcx,%rdx",
    "pxor %xmm0,%xmm0",
]


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


def _x86_row(form: str, iform: str, **attrs: str) -> InstructionRecord:
    metadata = {"iform": iform}
    if attrs:
        metadata["split_attrs"] = json.dumps(attrs)
    return InstructionRecord(
        mnemonic=form.split()[0], form=form, summary=form, architecture="x86", metadata=metadata
    )


def _iform(hexbytes: str) -> str:
    return _xed_facts([hexbytes], XED)[0]["iform"]


def _rv_row(name: str, mask: int, match: int) -> InstructionRecord:
    return InstructionRecord(
        mnemonic=name,
        form=name,
        summary=name,
        architecture="riscv",
        enc_bits=16,
        enc_mask=mask,
        enc_match=match,
    )


def _got(results) -> list[str]:
    return [r.row.db_key if r.row is not None else r.reason for r in results]


@needs_tools
@pytest.mark.parametrize(
    "arch, line",
    [
        ("aarch64", "stadd w0, [x1]"),
        ("aarch64", "stclr x0, [x1]"),
        ("aarch64", "steor w2, [x3]"),
        ("aarch64", "stset x4, [x5]"),
        ("aarch64", "staddl w0, [x1]"),
        ("aarch64", "pmull2 v0.1q, v1.2d, v2.2d"),
        ("aarch64", "gcspopm x0"),
        ("aarch64", "gcsss1 x0"),
        ("aarch64", "gcsss2 x0"),
        ("riscv64", "vandn.vv v1, v2, v3"),
        ("riscv64", "vaesef.vv v1, v2"),
        ("riscv64", "vclz.v v1, v2"),
        ("riscv64", "vcpop.v v1, v2"),
        ("riscv64", "vrol.vv v1, v2, v3"),
        ("riscv64", "vrev8.v v1, v2"),
        ("riscv64", "fadd.h fa0, fa1, fa2"),
    ],
)
def test_m2_m9_m13_m4_assembler_accepts_feature_lines(arch, line):
    assert _probe_lines_split([line], arch, "att", LLVM_MC)[0]["status"] == "OK"


@needs_tools
def test_stladd_is_no_instruction():
    # "stladd" is not an AArch64 mnemonic (the real one is staddl): llvm-mc rejects it
    assert _probe_lines_split(["stladd w0, [x1]"], "aarch64", "att", LLVM_MC)[0]["status"] == (
        "REJECTED"
    )


@needs_tools
def test_m2_sext_w_compressed_bytes_skip_rv32_only_c_jal(tmp_path):
    # objdump of rv64gc prints "2501  sext.w a0,a0": c.addiw and RV32-only c.jal share the bits
    db = _db([_rv_row("c.addiw", 0xE003, 0x2001), _rv_row("c.jal", 0xE003, 0x2001)], tmp_path)
    assert _got(identify_lines([("sext.w a0,a0", "2501")], catalog_path=db)) == ["riscv:c.addiw"]


@needs_tools
def test_rvc_xlen_tie_picks_the_listing_decode(tmp_path):
    # A 16-bit word decodes to one instruction per xlen (c.jal versus
    # c.addiw on 0x2001, the Zcf float forms against c.ld/c.sd): the row is
    # the decode whose printed mnemonic is the line's own, and 2501 sext.w
    # keeps its rv64 c.addiw.
    db = _db(
        [
            _rv_row("c.addiw", 0xE003, 0x2001),
            _rv_row("c.jal", 0xE003, 0x2001),
            _rv_row("c.flw", 0xE003, 0x6000),
            _rv_row("c.ld", 0xE003, 0x6000),
            _rv_row("c.fsw", 0xE003, 0xE000),
            _rv_row("c.sd", 0xE003, 0xE000),
            _rv_row("c.flwsp", 0xE003, 0x6002),
            _rv_row("c.ldsp", 0xE003, 0x6002),
            _rv_row("c.fswsp", 0xE003, 0xE002),
            _rv_row("c.sdsp", 0xE003, 0xE002),
        ],
        tmp_path,
    )
    items = [
        ("jal f1", "37fd"),
        ("flw fa5,0(a1)", "619c"),
        ("fsw fa5,0(a0)", "e11c"),
        ("flw ft0,0(sp)", "6002"),
        ("fsw ft0,0(sp)", "e002"),
        ("sext.w a0,a0", "2501"),
    ]
    assert _got(identify_lines(items, catalog_path=db)) == [
        "riscv:c.jal",
        "riscv:c.flw",
        "riscv:c.fsw",
        "riscv:c.flwsp",
        "riscv:c.fswsp",
        "riscv:c.addiw",
    ]


@needs_tools
def test_no_operand_opcode_keeps_its_name():
    # An operand-less MCInst closes its bracket on the opcode line; the
    # capture must not keep it ("FENCE_I>" never names a row).
    from simdref.asm_identity import _probe_lines

    probed = _probe_lines(["fence.i", "c.ebreak"], "riscv64", "att", LLVM_MC)
    assert [entry["opcode"] for entry in probed] == ["FENCE_I", "C_EBREAK"]


@needs_tools
@pytest.mark.parametrize(
    "line",
    [
        "vpcmpltud (%rax),%zmm1,%k1",
        "vpcmpltd (%rax),%xmm1,%k1",
        "vpcmpleq (%rax),%ymm1,%k1",
    ],
)
def test_m3_vpcmp_memory_form_picks_the_plain_row(line, tmp_path):
    from simdref.asm_identity import _probe_lines

    probe = _probe_lines([line], "x86", "att", LLVM_MC)[0]
    iform = _iform(probe["bytes"].replace("0x", "").replace(",", ""))
    db = _db(
        [
            _x86_row("VPCMPUD (K, K, ZMM, M32_1to16, I8)", iform, mask="1"),
            _x86_row("VPCMPUD (K, K, ZMM, M512, I8)", iform, mask="1"),
            _x86_row("VPCMPUD (K, ZMM, M32_1to16, I8)", iform, mask="0"),
            _x86_row("VPCMPUD (K, ZMM, M512, I8)", iform, mask="0"),
        ],
        tmp_path,
    )
    [got] = _got(identify_lines([(line, None)], catalog_path=db))
    assert "m512" in got and "1to" not in got


@needs_tools
def test_m3_vpcmp_broadcast_spelling_picks_the_broadcast_row(tmp_path):
    line = "vpcmpltud (%rax){1to16},%zmm1,%k1"
    from simdref.asm_identity import _probe_lines

    probe = _probe_lines([line], "x86", "att", LLVM_MC)[0]
    iform = _iform(probe["bytes"].replace("0x", "").replace(",", ""))
    db = _db(
        [
            _x86_row("VPCMPUD (K, K, ZMM, M32_1to16, I8)", iform, mask="1"),
            _x86_row("VPCMPUD (K, K, ZMM, M512, I8)", iform, mask="1"),
            _x86_row("VPCMPUD (K, ZMM, M32_1to16, I8)", iform, mask="0"),
            _x86_row("VPCMPUD (K, ZMM, M512, I8)", iform, mask="0"),
        ],
        tmp_path,
    )
    [got] = _got(identify_lines([(line, None)], catalog_path=db))
    assert "1to16" in got


@needs_tools
@pytest.mark.parametrize(
    "line, byte_column",
    [("endbr64", None), ("endbr32", None), ("endbr64", "f3 0f 1e fa"), ("endbr32", "f3 0f 1e fb")],
)
def test_m4_endbr_maps_to_its_row(line, byte_column, tmp_path):
    iform = _iform("f30f1efa")  # XED gives both ENDBR forms this one iform
    db = _db(
        [
            InstructionRecord(
                mnemonic="ENDBR64", form="ENDBR64", summary="e64", metadata={"iform": "ENDBR64"}
            ),
            InstructionRecord(
                mnemonic="ENDBR32", form="ENDBR32", summary="e32", metadata={"iform": "ENDBR32"}
            ),
            _x86_row("NOP (R32, R32)", iform),
            _x86_row("NOP (R16, R16)", iform),
        ],
        tmp_path,
    )
    want = "x86:" + line
    assert _got(identify_lines([(line, byte_column)], catalog_path=db)) == [want]


@needs_tools
def test_b2_b3_foreign_byte_column_is_no_x86_evidence(tmp_path):
    # ARM32 "bx lr" bytes e12fff1e decode as x86 LOOP: a foreign line must not take that row
    iform = _iform("e12fff1e")
    db = _db([_x86_row("LOOP (REL8)", iform)], tmp_path)
    [got] = _got(identify_lines([("bx lr", "e12fff1e")], catalog_path=db))
    assert not got.startswith("x86:")


@needs_tools
def test_b6_batch_majority_does_not_name_a_foreign_line():
    families, majority = isa_by_acceptance(X86_BATCH + ["add x0, x1, x2"])
    assert majority == "x86"
    assert families[-1] is None
    assert set(families[:-1]) == {"x86"}


@needs_tools
def test_b6_identify_lines_gives_the_foreign_line_no_x86_row(tmp_path):
    iform = _iform("2000028b")
    db = _db([_x86_row("ADD (R16, R16, I16)", iform)], tmp_path)
    items = [(text, None) for text in X86_BATCH] + [("add x0, x1, x2", "20 00 02 8b")]
    assert not _got(identify_lines(items, catalog_path=db))[-1].startswith("x86:")
