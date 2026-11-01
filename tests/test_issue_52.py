"""Issue #52: objdump prefix tokens (data16, cs, ...) read as the mnemonic.

objdump prints multi-byte NOP padding and prefixed instructions with the
prefix bytes as tokens before the mnemonic. The parser used the first
token as the mnemonic, so padding lines came out as unknown mnemonics.
"""

from __future__ import annotations

from pathlib import Path

from simdref.annotate import LineKind, _lookup_variants, parse_asm_line

FIXTURE = Path(__file__).parent / "fixtures" / "issue_52" / "prefix_padding.s"


def _fixture_lines() -> list[str]:
    return FIXTURE.read_text().splitlines()


def test_issue_repro_lines_track_positions():
    first, second = _fixture_lines()
    ins = parse_asm_line(first, track_positions=True)
    assert ins.kind == LineKind.INSTRUCTION
    assert ins.address == 0x35
    assert ins.mnemonic == "nopw"
    assert ins.operands == "0x0(%rax,%rax,1)"

    ins = parse_asm_line(second, track_positions=True)
    assert ins.kind == LineKind.INSTRUCTION
    assert ins.address == 0x2B
    assert ins.mnemonic == "nopw"
    assert ins.operands == "0x0(%rax,%rax,1)"


def test_issue_repro_lines_without_track_positions():
    for line in _fixture_lines():
        ins = parse_asm_line(line)
        assert ins.kind == LineKind.INSTRUCTION
        assert ins.mnemonic == "nopw", line


def test_nopw_resolves_to_catalog_nop():
    # The mnemonic "nopw" must reach the catalog row for the 0F 1F /0 NOP.
    assert _lookup_variants("nopw")[:2] == ["nopw", "nop"]


def test_lock_and_rep_prefixes_strip_before_mnemonic():
    ins = parse_asm_line(
        "  40:\tf0 01 07                	\tlock add %eax,(%rdi)\n",
        track_positions=True,
    )
    assert ins.kind == LineKind.INSTRUCTION
    assert ins.mnemonic == "add"
    assert ins.operands == "%eax,(%rdi)"

    ins = parse_asm_line(
        "  48:\tf3 c3                   	\trep ret\n",
        track_positions=True,
    )
    assert ins.kind == LineKind.INSTRUCTION
    assert ins.mnemonic == "ret"


def test_lone_prefix_token_stays_the_mnemonic():
    # objdump prints a lone "data16" for an unpaired prefix byte. There is
    # no token after it, so the old reading stands.
    ins = parse_asm_line("  50:\t66                     \t\tdata16\n", track_positions=True)
    assert ins.mnemonic == "data16"


def test_notrack_and_bnd_prefixes_strip():
    ins = parse_asm_line(
        "  55:\t3e ff e0                	\tnotrack jmp *%rax\n",
        track_positions=True,
    )
    assert ins.kind == LineKind.INSTRUCTION
    assert ins.mnemonic == "jmp"
    assert ins.operands == "*%rax"
