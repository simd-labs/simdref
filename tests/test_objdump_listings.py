"""objdump listings parse to the toolchain truth (issue #65).

tests/data/objdump holds one listing of corpus.c per objdump format and
target, and expected.json with one [address, bytes, mnemonic] record per
instruction. tools/gen_objdump_corpus.sh regenerates both from clang and the
objdumps: addresses and mnemonics come from --no-show-raw-insn listings, bytes
from the raw .text section, never from a listing byte column.
"""

import json
from pathlib import Path

import pytest

from simdref.annotate import LineKind, _asm_line, _listing_records, _objdump_records, parse_asm_line

DATA = Path(__file__).parent / "data" / "objdump"
EXPECTED = json.loads((DATA / "expected.json").read_text())


def _records(lines: list[str]) -> list[list]:
    """[address, memory-order bytes hex, mnemonic] per instruction line, as annotate_stream parses."""
    recs = _listing_records(lines)
    out = []
    for index, line in enumerate(lines):
        parsed = _asm_line(line, recs.get(index), False)
        if parsed.kind == LineKind.INSTRUCTION:
            data = recs[index][1] if index in recs else None
            out.append([parsed.address, data.hex() if data else None, parsed.mnemonic])
    return out


@pytest.mark.parametrize("name", sorted(EXPECTED))
@pytest.mark.parametrize("pasted", [False, True], ids=["tabs", "pasted"])
def test_listing_records_match_toolchain(name, pasted):
    lines = (DATA / name).read_text().splitlines()
    if pasted:  # a terminal or web paste turns each TAB into spaces
        lines = [line.expandtabs(8) for line in lines]
    assert _records(lines) == EXPECTED[name]


def _fragment(text: str) -> list[list]:
    return _records(text.split("\n"))


def test_61_hex_letter_mnemonics_have_no_byte_column():
    assert _fragment("   9:\tfadd   %st(1),%st\n   b:\taddb   $0x1,(%rax)\n   e:\tret") == [
        [0x9, None, "fadd"],
        [0xB, None, "addb"],
        [0xE, None, "ret"],
    ]


def test_61_hex_letter_address_with_bytes():
    assert _fragment(
        "   b:\t80 00 01             \taddb   $0x1,(%rax)\n   e:\tc3                   \tret"
    ) == [[0xB, "800001", "addb"], [0xE, "c3", "ret"]]


def test_63_gnu_wrap_continuation_folds_into_instruction():
    assert _fragment(
        "    4233:\t48 b9 ff ff ff 53 ff \tmovabs $0xff03ffff53ffffff,%rcx\n"
        "    423a:\tff ff 03 \n"
        "    423d:\tc3                   \tret"
    ) == [[0x4233, "48b9ffffff53ffffff03", "movabs"], [0x423D, "c3", "ret"]]


def test_63_wrap_on_last_line_folds_without_next_address():
    assert _fragment(
        "  1ef:\t62 f3 75 48 25 54 98 \tvpternlogd $0x96,0x40(%rax,%rbx,4),%zmm1,%zmm2\n"
        "  1f6:\t01 96 "
    ) == [[0x1EF, "62f375482554980196", "vpternlogd"]]


def test_b4_thumb2_halfword_pairs_are_little_endian():
    assert _fragment(
        "       0: 4770         \tbx\tlr\n"
        "       4: f8d0 c000    \tldr.w\tr12, [r0]\n"
        "       8: e8df f000    \ttbb\t[pc, r0]\n"
        "       c: 4770         \tbx\tlr"
    ) == [
        [0x0, "7047", "bx"],
        [0x4, "d0f800c0", "ldr.w"],
        [0x8, "dfe800f0", "tbb"],
        [0xC, "7047", "bx"],
    ]


def test_b5_llvm_long_x86_columns():
    recs = _fragment(
        "      60: 48 b8 88 77 66 55 44 33 22 11\tmovabsq\t$0x1122334455667788, %rax\n"
        "      6a: 64 f0 81 80 44 33 22 11 44 33 22 11  \tlock\t\taddl\t$0x11223344, %fs:0x11223344(%rax)\n"
        "      76: c7 84 98 44 33 22 11 44 33 22 11 \tmovl\t$0x11223344, 0x11223344(%rax,%rbx,4)\n"
        "      81: c3                           \tretq"
    )
    assert [(a, len(bytes.fromhex(b)), m) for a, b, m in recs] == [
        (0x60, 10, "movabsq"),
        (0x6A, 12, "addl"),
        (0x76, 11, "movl"),
        (0x81, 1, "retq"),
    ]


@pytest.mark.parametrize(
    "line", ["loop:", "dead:", "b:", "\tadd: ret", "\tbeef: nop", "\tdead: fadd %st(1),%st"]
)
def test_labels_are_not_objdump_lines(line):
    assert _objdump_records([line]) == {}
    assert parse_asm_line(line).kind == LineKind.LABEL
