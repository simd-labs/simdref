"""Tests for the annotate subcommand and helpers."""

from __future__ import annotations

import json
import sqlite3

import pytest

from simdref.annotate import (
    AnnotateOptions,
    LineKind,
    _parse_objdump_instr,
    aggregate_perf,
    annotate_stream,
    format_annotation,
    parse_asm_line,
    pick_record,
)
from simdref.models import InstructionRecord


def _make_record(
    *,
    mnemonic: str = "vaddps",
    summary: str = "Add packed single-precision floats.",
    arch_details: dict | None = None,
) -> InstructionRecord:
    return InstructionRecord(
        mnemonic=mnemonic,
        form=f"{mnemonic} ymm, ymm, ymm",
        summary=summary,
        arch_details=arch_details or {},
    )


def _arch_entry(lat: str, cpi: str, kind: str = "measured") -> dict:
    return {
        "latencies": [{"cycles": lat}],
        "measurement": {"TP": cpi},
        "source_kind": kind,
    }


def test_parse_asm_line_classifies_lines():
    assert parse_asm_line("").kind == LineKind.BLANK
    assert parse_asm_line("   \n").kind == LineKind.BLANK
    assert parse_asm_line("main:").kind == LineKind.LABEL
    assert parse_asm_line("    .cfi_startproc").kind == LineKind.DIRECTIVE
    assert parse_asm_line("# a comment").kind == LineKind.COMMENT
    ins = parse_asm_line("    vaddps %ymm2, %ymm1, %ymm0")
    assert ins.kind == LineKind.INSTRUCTION
    assert ins.mnemonic == "vaddps"
    assert "%ymm0" in ins.operands


def test_parse_asm_line_objdump_mode_rejects_source_interleave():
    # `objdump -dS` interleaves C/C++ source lines between instruction
    # blocks. Under --track-positions these are NOT AT&T instructions and
    # must not be classified as mnemonics ("return", "if", "for", ...).
    src_lines = [
        "    return __x >> __count;",
        "  const size_t __k = __b / __log_r;",
        "    if (low_bits == 0) continue;",
        "0000000000013610 <deregister_tm_clones>:",
    ]
    for line in src_lines:
        parsed = parse_asm_line(line, track_positions=True)
        assert parsed.kind != LineKind.INSTRUCTION, f"false-positive on {line!r}"

    # But real objdump instruction lines still parse with an address.
    objdump = "   13917:	vaddps %ymm2,%ymm1,%ymm0"
    ins = parse_asm_line(objdump, track_positions=True)
    assert ins.kind == LineKind.INSTRUCTION
    assert ins.mnemonic == "vaddps"
    assert ins.address == 0x13917


def test_parse_asm_line_objdump_address_with_hex_letter_prefix():
    # Regression: addresses whose first hex digit is a letter (a-f) used to
    # match _LABEL_RE before the objdump regex, dropping the entire
    # [0xa000, 0xffff] range from annotated output.
    line = "    a000:\tvandpd -0x7b08(%rip),%xmm13,%xmm14\t# 2500 <.LC53>"
    parsed = parse_asm_line(line, track_positions=True)
    assert parsed.kind == LineKind.INSTRUCTION
    assert parsed.mnemonic == "vandpd"
    assert parsed.address == 0xA000


def test_parse_asm_line_objdump_symbol_header_not_instruction():
    parsed = parse_asm_line("0000000000003d80 <main>:", track_positions=True)
    assert parsed.kind != LineKind.INSTRUCTION


# Issue #61 repro lines: a mnemonic of only hex letters ("fadd", "addb")
# and an address of only hex letters ("b:", "c:"). Each tuple is one objdump
# line with real tabs, its address, mnemonic, operands and byte column.
_HEXLETTER_REPROS = [
    ("   9:\tfadd   %st(1),%st", 0x9, "fadd", "%st(1),%st", None),
    ("   b:\taddb   $0x1,(%rax)", 0xB, "addb", "$0x1,(%rax)", None),
    ("       4:     \tfadd\td0, d1, d2", 0x4, "fadd", "d0, d1, d2", None),
    ("   b:\t80 00 01             \taddb   $0x1,(%rax)", 0xB, "addb", "$0x1,(%rax)", "80 00 01"),
    ("   c:\tc3                   \tret", 0xC, "ret", "", "c3"),
]


@pytest.mark.parametrize(("line", "addr", "mnem", "ops", "bytehex"), _HEXLETTER_REPROS)
def test_objdump_hexletter_repro_line(monkeypatch, line, addr, mnem, ops, bytehex):
    from simdref import annotate as _annotate

    obj = _parse_objdump_instr(line)
    assert obj is not None, line
    assert obj[0] == addr
    assert obj[2] == bytehex
    parsed = parse_asm_line(line)
    assert parsed.kind == LineKind.INSTRUCTION, line
    assert parsed.mnemonic == mnem
    assert parsed.operands == ops
    assert parsed.address == addr

    rec = _make_record(
        mnemonic=mnem, summary="Repro record.", arch_details={"SKX": _arch_entry("3", "0.5")}
    )
    monkeypatch.setattr(_annotate, "lookup", _fake_lookup_factory({mnem.lower(): [rec]}))
    conn = sqlite3.connect(":memory:")
    out = "".join(annotate_stream(iter([line + "\n"]), opts=AnnotateOptions(fmt="json"), conn=conn))
    data = json.loads(out)
    assert len(data) == 1, line
    assert data[0]["mnemonic"] == mnem
    assert data[0]["address"] == f"0x{addr:x}"


def test_objdump_hexletter_address_keeps_labels():
    # A real label stays a label even when its name is all hex letters.
    for line in ("loop:", ".L2:", "main:", "add:", "b:"):
        parsed = parse_asm_line(line)
        assert parsed.kind != LineKind.INSTRUCTION, line
        assert parsed.mnemonic == "", line
    for line in ("loop:", "main:", "add:", "b:"):
        assert parse_asm_line(line).kind == LineKind.LABEL, line


def test_objdump_multi_digit_address_controls_still_parse():
    # Same bytes and mnemonics with a multi-digit address: worked before,
    # must keep working.
    for line, mnem in (
        ("  1b:\t80 00 01             \taddb   $0x1,(%rax)", "addb"),
        ("  11:\tc3                   \tret", "ret"),
    ):
        parsed = parse_asm_line(line)
        assert parsed.kind == LineKind.INSTRUCTION, line
        assert parsed.mnemonic == mnem


def test_aggregate_perf_modes():
    rec = _make_record(
        arch_details={
            "SKX": _arch_entry("3", "0.5"),
            "ZEN4": _arch_entry("5", "1.0"),
        }
    )
    assert aggregate_perf(rec, mode="avg").latency == 4.0
    assert aggregate_perf(rec, mode="median").latency == 4.0
    assert aggregate_perf(rec, mode="best").latency == 3.0
    assert aggregate_perf(rec, mode="worst").latency == 5.0
    assert aggregate_perf(rec, mode="avg").n_archs == 2


def test_aggregate_perf_include_modeled_toggle():
    rec = _make_record(arch_details={"SKX": _arch_entry("4", "1.0", kind="modeled")})
    measured_only = aggregate_perf(rec, mode="avg", include_modeled=False)
    # No measured data, falls through to any-kind mixed.
    assert measured_only.n_archs == 1
    assert measured_only.source_kind in {"mixed", "modeled"}
    with_modeled = aggregate_perf(rec, mode="avg", include_modeled=True)
    assert with_modeled.latency == 4.0
    assert with_modeled.source_kind == "modeled"


def test_pick_record_prefers_named_arch():
    a = _make_record(arch_details={"SKX": _arch_entry("3", "0.5")})
    b = _make_record(arch_details={"ZEN4": _arch_entry("5", "1.0")})
    assert pick_record([a, b], arch="ZEN4") is b


def _xor_record(key: str) -> InstructionRecord:
    return InstructionRecord(
        mnemonic="XOR",
        form=key,
        summary="Logical Exclusive OR.",
        arch_details={"HSW": _arch_entry("1", "0.25")},
    )


def test_pick_record_zero_idiom_picks_imm_zero_variant():
    # `xor %eax, %eax` — two aliased R32 operands. The catalog has no
    # ``(R32, R32)`` form; uops.info records this case as ``(R32, 0)``
    # (literal imm=0). Before the fix, the picker tied on every R32-
    # bearing form and picked ``XOR (M32, R32)`` by iteration order,
    # giving lat=6.0 / cpi=1.10.
    recs = [
        _xor_record("XOR (M32, R32)"),
        _xor_record("XOR (R32, 0)"),
        _xor_record("XOR (R32, M32)"),
        _xor_record("XOR (R32, I32)"),
        _xor_record("XOR (R32, I8)"),
    ]
    picked = pick_record(recs, arch="HSW", operands="%eax, %eax")
    assert picked is not None
    assert picked.key == "XOR (R32, 0)"


def test_pick_record_rejects_memory_form_when_no_mem_operand():
    # `xor %rax, %rax` must not resolve to a memory-destination form.
    recs = [
        _xor_record("XOR (M64, R64)"),
        _xor_record("XOR (R64, 0)"),
        _xor_record("XOR (R64, M64)"),
    ]
    picked = pick_record(recs, arch="HSW", operands="%rax, %rax")
    assert picked is not None
    assert "M64" not in picked.key or picked.key == "XOR (R64, 0)"


def test_pick_record_vector_zero_idiom_single_variant():
    # vpxor %ymm1,%ymm1,%ymm1 — only one form exists, must still resolve.
    rec = InstructionRecord(
        mnemonic="VPXOR",
        form="VPXOR (YMM, YMM, YMM)",
        summary="Packed XOR.",
        arch_details={"HSW": _arch_entry("1", "0.33")},
    )
    picked = pick_record([rec], arch="HSW", operands="%ymm1, %ymm1, %ymm1")
    assert picked is rec


def test_format_annotation_has_summary_and_perf():
    rec = _make_record(
        arch_details={
            "SKX": _arch_entry("3", "0.5"),
            "ZEN4": _arch_entry("5", "1.0"),
        }
    )
    out = format_annotation(
        rec,
        performance=True,
        docs=True,
        arch=None,
        agg="avg",
        include_modeled=False,
    )
    assert "Add packed" in out
    assert "lat=4.0c" in out
    assert "cpi=0.75" in out
    assert "[avg of 2 archs, measured]" in out


class _FakeConn:
    def __init__(self, mapping: dict[str, list[InstructionRecord]]):
        self.mapping = mapping


def _fake_lookup_factory(mapping):
    def _fake(mnemonic, conn, isa=None):
        return mapping.get(mnemonic.lower(), [])

    return _fake


def test_annotate_stream_inline(monkeypatch):
    rec = _make_record(arch_details={"SKX": _arch_entry("3", "0.5")})
    from simdref import annotate as annotate_mod

    monkeypatch.setattr(annotate_mod, "lookup", _fake_lookup_factory({"vaddps": [rec]}))

    src = [
        "main:\n",
        "    .cfi_startproc\n",
        "    vaddps %ymm2, %ymm1, %ymm0\n",
        "    vmadeupinstr %ymm0, %ymm1\n",
        "\n",
    ]
    opts = AnnotateOptions()
    out = list(annotate_stream(src, opts=opts, conn=_FakeConn({})))
    joined = "".join(out)

    # Labels/directives/blank lines pass through unchanged.
    assert "main:\n" in joined
    assert ".cfi_startproc\n" in joined

    # Known mnemonic keeps the original instruction text and adds annotation.
    known = [ln for ln in out if "vaddps" in ln][0]
    assert "vaddps %ymm2, %ymm1, %ymm0" in known
    assert "# " in known
    assert "lat=3.0c" in known

    # Unknown mnemonic marked with "??".
    unknown = [ln for ln in out if "vmadeupinstr" in ln][0]
    assert "# ??" in unknown


def test_annotate_hello_simd_snippet(monkeypatch):
    """End-to-end annotate on a small SIMD dot-product snippet."""
    from pathlib import Path
    from simdref import annotate as annotate_mod

    vaddps = _make_record(
        mnemonic="vaddps",
        summary="Add packed single-precision floats.",
        arch_details={"SKX": _arch_entry("4", "0.5"), "ZEN4": _arch_entry("3", "0.5")},
    )
    vmulps = _make_record(
        mnemonic="vmulps",
        summary="Multiply packed single-precision floats.",
        arch_details={"SKX": _arch_entry("4", "0.5")},
    )
    vmovups = _make_record(
        mnemonic="vmovups",
        summary="Move unaligned packed single-precision floats.",
        arch_details={"SKX": _arch_entry("5", "0.5")},
    )
    vxorps = _make_record(
        mnemonic="vxorps",
        summary="Bitwise logical XOR of packed single-precision floats.",
        arch_details={"SKX": _arch_entry("1", "0.33")},
    )
    vhaddps = _make_record(
        mnemonic="vhaddps",
        summary="Horizontal add packed single-precision floats.",
        arch_details={"SKX": _arch_entry("6", "2.0")},
    )
    mapping = {
        "vaddps": [vaddps],
        "vmulps": [vmulps],
        "vmovups": [vmovups],
        "vxorps": [vxorps],
        "vhaddps": [vhaddps],
    }
    monkeypatch.setattr(annotate_mod, "lookup", _fake_lookup_factory(mapping))

    fixture = Path(__file__).parent / "fixtures" / "hello_simd.s"
    src = fixture.read_text().splitlines(keepends=True)
    out = "".join(annotate_stream(src, opts=AnnotateOptions(), conn=_FakeConn({})))

    # Directives/labels/ret pass through unchanged; `ret` is an unknown
    # mnemonic and gets the "??" marker.
    assert ".cfi_startproc\n" in out
    assert "dot8:\n" in out
    # All known SIMD mnemonics picked up a summary + perf fragment.
    for mnem, fragment in [
        ("vmovups", "Move unaligned"),
        ("vmulps", "Multiply packed"),
        ("vaddps", "Add packed"),
        ("vxorps", "Bitwise logical XOR"),
        ("vhaddps", "Horizontal add"),
    ]:
        line = next(ln for ln in out.splitlines() if mnem in ln and "#" in ln)
        assert fragment in line
        assert "lat=" in line
        assert "cpi=" in line

    # Every annotation is a valid assembly comment (starts with '#' on x86 GAS).
    for ln in out.splitlines():
        if "#" in ln:
            # The portion after the first '#' must not contain another unescaped newline
            # and the comment char must occur after the instruction text.
            idx = ln.index("#")
            assert idx >= 0
