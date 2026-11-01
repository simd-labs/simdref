"""Back-compat + position tracking in `simdref annotate`."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import ClassVar

from simdref.annotate import AnnotateOptions, annotate_stream, parse_asm_line


def test_objdump_line_address_parsed_with_track_positions():
    line = "  4011a4:\tvfmadd231ps %xmm2,%xmm1,%xmm0"
    parsed = parse_asm_line(line, track_positions=True)
    assert parsed.address == 0x4011A4
    assert parsed.mnemonic == "vfmadd231ps"


def test_objdump_line_parsed_without_track_positions():
    # Issue #21: objdump -d output (the workflow §1d recipe) is a first-class
    # annotate input even without --track-positions — the leading address
    # label is stripped and threaded through as metadata.
    line = "  4011a4:\tvfmadd231ps %xmm2,%xmm1,%xmm0"
    parsed = parse_asm_line(line)
    assert parsed.mnemonic == "vfmadd231ps"
    assert parsed.address == 0x4011A4


def test_annotate_json_with_track_positions_emits_address(monkeypatch):
    # Stub the catalog lookup so we don't depend on the live DB.
    from simdref import annotate as _annotate

    class _FakeRecord:
        summary = "Fused multiply-add of packed single-precision floats."
        arch_details = {
            "skylake-x": {
                "latencies": [{"cycles": 4}],
                "throughput": 0.5,
                "source_kind": "measured",
            }
        }
        operand_details = []
        key = "VFMADD231PS (XMM, XMM, XMM)"

    monkeypatch.setattr(_annotate, "lookup", lambda mn, conn, isa=None: [_FakeRecord()])
    monkeypatch.setattr(
        _annotate,
        "pick_record",
        lambda records, *, arch=None, operands="", isa=None: records[0] if records else None,
    )

    conn = sqlite3.connect(":memory:")
    line = "  4011a4:\tvfmadd231ps %xmm2,%xmm1,%xmm0\n"
    opts = AnnotateOptions(fmt="json", track_positions=True)
    out = "".join(annotate_stream(iter([line]), opts=opts, conn=conn))
    data = json.loads(out)
    assert len(data) == 1
    assert data[0]["address"] == "0x4011a4"


def test_annotate_default_mode_annotates_objdump_input(monkeypatch):
    """Issue #21 regression: default-mode annotate of objdump -d output used
    to echo every line verbatim with no annotation and exit 0."""
    from simdref import annotate as _annotate

    class _FakeRecord:
        summary = "Subtract Packed Double Precision Floating-Point Values."
        arch_details: ClassVar = {
            "MTL-P": {
                "latencies": [{"cycles": 2}],
                "throughput": 0.5,
                "source_kind": "measured",
            }
        }
        operand_details: ClassVar = []
        key = "VSUBPD (YMM, YMM, YMM)"

    monkeypatch.setattr(_annotate, "lookup", lambda mn, conn, isa=None: [_FakeRecord()])
    monkeypatch.setattr(
        _annotate,
        "pick_record",
        lambda records, *, arch=None, operands="", isa=None: records[0] if records else None,
    )

    conn = sqlite3.connect(":memory:")
    lines = [
        "\n",
        "drv:     file format elf64-x86-64\n",
        "\n",
        "Disassembly of section .text:\n",
        "\n",
        "000000000015a950 <main>:\n",
        "  15a95f:\tvsubpd %ymm9,%ymm10,%ymm13\n",
        "  15a964:\tret\n",
    ]
    out = "".join(annotate_stream(iter(lines), opts=AnnotateOptions(fmt="sa"), conn=conn))
    annotated = [ln for ln in out.splitlines() if "vsubpd" in ln]
    assert annotated, f"objdump line was not annotated: {out!r}"
    assert "Subtract Packed" in annotated[0]

    # Same input as JSON: one record, carrying the objdump address.
    stats: dict[str, int] = {}
    out = "".join(
        annotate_stream(iter(lines), opts=AnnotateOptions(fmt="json"), conn=conn, stats=stats)
    )
    data = json.loads(out)
    assert data and data[0]["address"] == "0x15a95f"
    assert stats["parsed"] == 2  # vsubpd + ret
    assert stats["recognized"] >= 1


def test_objdump_parser_cross_arch_and_gas_numeric_labels():
    """objdump emits the encoding differently per arch: x86 spaces every byte
    ("48 8b 45 f8"), AArch64/RISC-V emit one token ("a9bf7bfd"/"00050513").
    GAS numeric local labels at column 0 must not parse as objdump lines."""
    from simdref.annotate import LineKind

    p = parse_asm_line("   4b0:\ta9bf7bfd \tstp\tx29, x30, [sp, #-16]!")
    assert p.kind == LineKind.INSTRUCTION
    assert p.mnemonic == "stp"
    assert p.address == 0x4B0

    p = parse_asm_line("    1013a:\t00050513        \tmv  a0, a0")
    assert p.kind == LineKind.INSTRUCTION
    assert p.mnemonic == "mv"
    assert p.address == 0x1013A

    # Column-0 GAS local label ("1:" referenced by "jne 1b" idioms).
    p = parse_asm_line("1:\tnop")
    assert p.kind == LineKind.COMMENT


def test_annotate_stats_flag_unparseable_input(monkeypatch):
    """The CLI surfaces silent format failures via the stats dict."""
    from simdref import annotate as _annotate

    monkeypatch.setattr(_annotate, "lookup", lambda mn, conn: [])
    conn = sqlite3.connect(":memory:")
    # Raw hexdump: no line parses as an instruction or label.
    stats: dict[str, int] = {}
    src = ["00000000  48 89 e5 c3  |....|\n", "00000004  90 90 90 90  |....|\n"]
    out = "".join(annotate_stream(src, opts=AnnotateOptions(fmt="sa"), conn=conn, stats=stats))
    assert stats.get("parsed", 0) == 0
    assert stats.get("content", 0) == 2
    assert "# ??" not in out  # nothing was (mis)classified as an instruction
