"""ISA auto-detection by assembler acceptance (lane L1).

Each line's family is the triple whose llvm-mc encodes it. A line several
triples accept (``ret``, ``nop``) takes the document majority of the lines
that only one triple accepts. The register vote stays as the fallback for
the tool-missing case (musl, unsupported platform) and keeps today's
behavior there.

Tool tests skip with a reason when ``_tool_path`` finds no llvm-mc. The
fallback test hides the tool and runs without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdref import annotate as annotate_mod
from simdref.annotate import (
    AnnotateOptions,
    LineKind,
    _SYMBOL_ANNOTATION_RE,
    _parse_objdump_instr,
    annotate_stream,
    parse_asm_line,
)
from simdref.models import InstructionRecord

FIXTURES = Path(__file__).parent / "fixtures"
MIXED_BATCH = (
    "vpcmpd k1, zmm0, zmm1, 1\n"
    "jnle 20 <x>\n"
    "vpcompressd ZMMWORD PTR [rsi+rcx*4]{k1}, zmm0\n"
    "b.eq 0x118 <dotn+0x118>\n"
    "b.ne 0x4c <dotn+0x4c>\n"
    "b 0xd4 <dotn+0xd4>\n"
)


def _have_tool() -> bool:
    from simdref.asm_identity import _tool_path

    return _tool_path("SIMDREF_LLVM_MC", "llvm-mc") is not None


needs_tool = pytest.mark.skipif(
    not _have_tool(), reason="llvm-mc not found (simdref-llvm-mc wheel or SIMDREF_LLVM_MC)"
)


def _cleaned(text: str) -> str | None:
    """The probe text: objdump columns and <symbol> gone, immediates kept."""
    parsed = parse_asm_line(text)
    if parsed.kind != LineKind.INSTRUCTION:
        return None
    obj = _parse_objdump_instr(parsed.raw)
    return _SYMBOL_ANNOTATION_RE.sub(" ", obj[1] if obj is not None else parsed.raw).strip()


def _doc(name: str) -> list[str]:
    text = (FIXTURES / "isa_detect" / name).read_text()
    return [clean for line in text.splitlines() if (clean := _cleaned(line))]


def _batch_texts() -> list[str]:
    return [clean for line in MIXED_BATCH.splitlines() if (clean := _cleaned(line))]


class _FakeConn:
    def close(self):
        pass


def _rows(*specs) -> list[InstructionRecord]:
    return [
        InstructionRecord(mnemonic=m, form=f, summary=s, architecture=a) for m, f, s, a in specs
    ]


def _fake_lookup(rows):
    def fake(mnemonic, conn, isa=None):
        out = [r for r in rows if r.mnemonic.lower() == mnemonic.lower()]
        return [r for r in out if isa is None or r.architecture == isa]

    return fake


def _annotate_json(text: str, rows, monkeypatch, **opts) -> list[dict]:
    monkeypatch.setattr(annotate_mod, "lookup", _fake_lookup(rows))
    out = "".join(
        annotate_stream(
            text.splitlines(keepends=True),
            opts=AnnotateOptions(fmt="json", **opts),
            conn=_FakeConn(),
        )
    )
    return json.loads(out)


@needs_tool
def test_each_line_takes_the_family_whose_assembler_encodes_it():
    from simdref.asm_identity import isa_by_acceptance

    families, majority = isa_by_acceptance(_doc("a64_scalar_regs.s"))
    assert families == ["arm", "arm", "arm"]
    assert majority == "arm"
    families, majority = isa_by_acceptance(_doc("rv_x_regs.s"))
    assert families == ["riscv", "riscv"]
    assert majority == "riscv"


@needs_tool
def test_x86_listings_detect_in_both_syntaxes():
    from simdref.asm_identity import isa_by_acceptance

    for name in ("x86_att.s", "x86_intel.s"):
        families, majority = isa_by_acceptance(_doc(name))
        assert families == ["x86"] * 4
        assert majority == "x86"


@needs_tool
def test_shared_lines_take_the_document_majority():
    from simdref.asm_identity import isa_by_acceptance

    # `ret` alone is accepted by every triple; the two decisive lines vote arm.
    text = (FIXTURES / "issue_51" / "aarch64_stdin.s").read_text()
    families, majority = isa_by_acceptance(
        [clean for line in text.splitlines() if (clean := _cleaned(line))]
    )
    assert families == ["arm", "arm", "arm"]
    assert majority == "arm"


@needs_tool
def test_mixed_batch_gives_each_line_its_own_family():
    from simdref.asm_identity import isa_by_acceptance

    families, majority = isa_by_acceptance(_batch_texts())
    assert families == ["x86", "x86", "x86", "arm", "arm", "arm"]
    # The batch ties between x86 and arm: no majority, neutral lines keep
    # today's vote.
    assert majority is None


@needs_tool
def test_auto_mode_serves_the_rv_x_only_listing_riscv_rows(monkeypatch):
    rows = _rows(
        ("ADD", "ADD (X5, X6, X7)", "Arm add.", "arm"),
        ("ADD", "ADD (X5, X6, X7)", "Riscv add.", "riscv"),
        ("ADDI", "ADDI (X1, X1, 16)", "Arm addi.", "arm"),
        ("ADDI", "ADDI (X1, X1, 16)", "Riscv addi.", "riscv"),
    )
    records = _annotate_json(
        (FIXTURES / "isa_detect" / "rv_x_regs.s").read_text(), rows, monkeypatch
    )
    assert [record["summary"] for record in records] == ["Riscv add.", "Riscv addi."]


@needs_tool
def test_auto_mode_serves_the_a64_scalar_listing_arm_rows(monkeypatch):
    rows = _rows(
        ("FADD", "FADD (S0, S1, S2)", "Arm fadd.", "arm"),
        ("FADD", "FADD (ST0, ST1)", "X86 fadd.", "x86"),
        ("FMUL", "FMUL (D0, D1, D2)", "Arm fmul.", "arm"),
        ("FMUL", "FMUL (ST0, ST1)", "X86 fmul.", "x86"),
        ("LDR", "LDR (Q0, SP)", "Arm ldr.", "arm"),
    )
    records = _annotate_json(
        (FIXTURES / "isa_detect" / "a64_scalar_regs.s").read_text(), rows, monkeypatch
    )
    assert [record["summary"] for record in records] == ["Arm fadd.", "Arm fmul.", "Arm ldr."]


def test_fallback_without_the_tool_keeps_the_register_vote(monkeypatch):
    # The register vote keeps today's behavior when llvm-mc is missing: the
    # xN-only RISC-V listing still reads as arm (the known shortcut).
    monkeypatch.setattr("simdref.asm_identity._tool_path", lambda *args: None)
    rows = _rows(
        ("ADD", "ADD (X5, X6, X7)", "Arm add.", "arm"),
        ("ADD", "ADD (X5, X6, X7)", "Riscv add.", "riscv"),
        ("ADDI", "ADDI (X1, X1, 16)", "Arm addi.", "arm"),
        ("ADDI", "ADDI (X1, X1, 16)", "Riscv addi.", "riscv"),
    )
    monkeypatch.setattr(annotate_mod, "lookup", _fake_lookup(rows))
    out = "".join(
        annotate_stream(
            (FIXTURES / "isa_detect" / "rv_x_regs.s").read_text().splitlines(keepends=True),
            opts=AnnotateOptions(fmt="json"),
            conn=_FakeConn(),
        )
    )
    assert [record["summary"] for record in json.loads(out)] == ["Arm add.", "Arm addi."]


def test_helper_declines_without_the_tool(monkeypatch):
    from simdref.asm_identity import isa_by_acceptance

    monkeypatch.setattr("simdref.asm_identity._tool_path", lambda *args: None)
    families, majority = isa_by_acceptance(_doc("rv_x_regs.s"))
    assert families == [None, None]
    assert majority is None
