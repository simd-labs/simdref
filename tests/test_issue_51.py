"""Issue #51: AArch64 input must not join x86 rows.

The repro lines share mnemonics with x86 rows (LSL is Load Segment Limit
on x86, RET is Return From Procedure). The legacy mnemonic matcher has no
ISA context and joins those x86 rows. The stream tests hide llvm-mc and
xed so byte identity cannot run and the legacy matcher answers, as with a
schema-13 catalog. Each test asserts the picked row family.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from simdref import annotate as annotate_mod
from simdref import cli as cli_mod
from simdref.annotate import AnnotateOptions, annotate_stream, lookup, pick_record
from simdref.lsp import identity_keys
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.storage import build_sqlite, open_db

FIXTURES = Path(__file__).parent / "fixtures" / "issue_51"
STDIN_REPRO = (FIXTURES / "aarch64_stdin.s").read_text()
OBJDUMP_REPRO = (FIXTURES / "aarch64_objdump.s").read_text()

X86_SUMMARIES = [
    "Load Segment Limit.",
    "Return From Procedure.",
    "Move 16-bit operands.",
]


def _rows() -> list[InstructionRecord]:
    return [
        InstructionRecord(
            mnemonic="LSL", form="LSL (R16, R16)", summary="Load Segment Limit.", architecture="x86"
        ),
        InstructionRecord(
            mnemonic="JMP", form="JMP (Rel32)", summary="Jump.", architecture="x86"
        ),
        InstructionRecord(
            mnemonic="ADD", form="ADD (R64, R64)", summary="Add 64-bit operands.", architecture="x86"
        ),
        InstructionRecord(
            mnemonic="ADD", form="ADD (X0, X0, X0)", summary="Arm add.", architecture="arm"
        ),
        InstructionRecord(
            mnemonic="RET",
            form="RET_NEAR",
            summary="Return From Procedure.",
            architecture="x86",
        ),
        InstructionRecord(
            mnemonic="MOV",
            form="MOV (R16, I16)",
            summary="Move 16-bit operands.",
            architecture="x86",
        ),
        InstructionRecord(
            mnemonic="LSL", form="LSL (X0, X0, I12)", summary="Lsl instruction.", architecture="arm"
        ),
        InstructionRecord(
            mnemonic="RET", form="RET (X30)", summary="Ret instruction.", architecture="arm"
        ),
    ]


def _issue_db(tmp_path) -> Path:
    db = tmp_path / "issue-51.db"
    build_sqlite(
        Catalog(
            intrinsics=[],
            instructions=_rows(),
            sources=[
                SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")
            ],
            generated_at="2025-01-01",
        ),
        db,
    )
    return db


def _no_identity(monkeypatch) -> None:
    # The issue plays on the legacy matcher path: byte identity cannot run.
    monkeypatch.setattr("simdref.asm_identity._tool_path", lambda *args: None)


def _run_json(tmp_path, text: str, **opts) -> list[dict]:
    conn = open_db(path=_issue_db(tmp_path))
    try:
        out = "".join(
            annotate_stream(
                text.splitlines(keepends=True),
                opts=AnnotateOptions(fmt="json", **opts),
                conn=conn,
            )
        )
    finally:
        conn.close()
    return json.loads(out)


def test_stdin_repro_joins_arm_rows(tmp_path, monkeypatch):
    _no_identity(monkeypatch)
    records = _run_json(tmp_path, STDIN_REPRO)
    assert records[0]["summary"] == "Lsl instruction."
    assert records[1]["summary"] == "Ret instruction."
    assert records[2]["known"] is False


def test_isa_x86_pin_keeps_x86_rows(tmp_path, monkeypatch):
    _no_identity(monkeypatch)
    records = _run_json(tmp_path, STDIN_REPRO, isa="x86")
    assert [record["summary"] for record in records] == X86_SUMMARIES


def test_objdump_file_format_line_names_the_isa(tmp_path, monkeypatch):
    _no_identity(monkeypatch)
    conn = open_db(path=_issue_db(tmp_path))
    try:
        out = "".join(
            annotate_stream(
                OBJDUMP_REPRO.splitlines(keepends=True), opts=AnnotateOptions(), conn=conn
            )
        )
    finally:
        conn.close()
    assert "Ret instruction." in out
    assert "Return From Procedure." not in out


def test_lookup_keeps_only_pinned_isa_rows(tmp_path):
    conn = open_db(path=_issue_db(tmp_path))
    try:
        arm = lookup("lsl", conn, isa="arm")
        x86 = lookup("lsl", conn, isa="x86")
        riscv = lookup("lsl", conn, isa="riscv")
    finally:
        conn.close()
    assert [record.architecture for record in arm] == ["arm"]
    assert [record.architecture for record in x86] == ["x86"]
    assert riscv == []


def test_pick_record_keeps_only_pinned_isa_rows(tmp_path):
    conn = open_db(path=_issue_db(tmp_path))
    try:
        records = lookup("lsl", conn)
        arm = pick_record(records, operands="x9, x9", isa="arm")
        x86 = pick_record(records, operands="x9, x9", isa="x86")
        riscv = pick_record(records, operands="x9, x9", isa="riscv")
    finally:
        conn.close()
    assert arm.summary == "Lsl instruction."
    assert x86.summary == "Load Segment Limit."
    assert riscv is None


def test_cli_isa_option(tmp_path, monkeypatch):
    _no_identity(monkeypatch)
    monkeypatch.setattr(cli_mod, "ensure_runtime", lambda: None)
    monkeypatch.setattr(cli_mod, "SQLITE_PATH", _issue_db(tmp_path))
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.app, ["annotate", "-", "--format", "json", "--isa", "arm"], input=STDIN_REPRO
    )
    assert result.exit_code == 0
    records = json.loads(result.stdout)
    assert records[0]["summary"] == "Lsl instruction."
    assert records[1]["summary"] == "Ret instruction."
    assert records[2]["known"] is False
    bad = runner.invoke(cli_mod.app, ["annotate", "-", "--isa", "bogus"], input="")
    assert bad.exit_code == 1


def test_identity_keys_exposes_the_doc_arch(tmp_path):
    conn = open_db(path=_issue_db(tmp_path))
    arch_out: list[str] = []
    try:
        identity_keys(["lsl x9, x9", "ret"], None, conn, arch_out=arch_out)
    finally:
        conn.close()
    assert arch_out == ["arm"]


def test_detect_isa_reads_the_input_signals():
    from simdref.annotate import detect_isa

    assert detect_isa(OBJDUMP_REPRO.splitlines()) == "arm"
    assert detect_isa(["lsl x9, x9, #2"]) == "arm"
    assert detect_isa(["b.eq 0x118 <dotn+0x118>"]) == "arm"
    assert detect_isa(["add a0, a1, a2"]) == "riscv"
    assert detect_isa(["drv:     file format elf64-x86-64"]) == "x86"
    assert detect_isa(["addq %rax, %rbx"]) is None


def test_symbol_annotations_do_not_vote_the_isa(tmp_path, monkeypatch):
    # Review finding 1 repro: an objdump <symbol> annotation named like a
    # register must not vote. The two x86 lines take x86 rows, not Arm rows.
    from simdref.annotate import detect_isa

    assert detect_isa(["jmp 50 <x0+0x50>"]) is None
    _no_identity(monkeypatch)
    records = _run_json(tmp_path, "jmp 50 <x0+0x50>\naddq %rax, %rbx\n")
    assert records[0]["known"] is True
    assert records[0]["summary"] == "Jump."
    assert records[1]["known"] is True
    assert records[1]["summary"] == "Add 64-bit operands."
