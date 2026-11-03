"""Issue #60 class 6: cross-architecture matcher gaps.

AArch64 B.cond rows must carry their own cond field value, and byte identity
must resolve the RISC-V sp forms, pseudo-branches, ``j`` and the FP
load/store/move lines to one catalog row. The fixtures carry real upstream
data shapes: the AARCHMRS B_cond node (FAT-2025-09) and riscv-unified-db
instruction records with their YAML encoding blocks.

The issue's objdump text (``blez a0,130 <.L48>``) reaches llvm-mc with the
``<sym>`` suffix stripped; the strip lives in the annotate/lsp identity call
sites (lane C237), so these tests feed the stripped text directly to
``identify``. Tool-dependent tests skip when llvm-mc is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdref.arm_instructions import parse_arm_instruction_payload
from simdref.asm_identity import _tool_path, identify
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.riscv import parse_riscv_instruction_payload
from simdref.storage import build_sqlite

LLVM_MC = _tool_path("SIMDREF_LLVM_MC", "llvm-mc") or ""
needs_tools = pytest.mark.skipif(not LLVM_MC, reason="llvm-mc absent (set SIMDREF_LLVM_MC)")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "issue_60_crossarch"

# The Arm ARM cond field value per B.cond spelling. llvm-mc -show-inst prints
# the same value as the Bcc MCOperand immediate (b.le -> Imm:13, b.al ->
# Imm:14), so each row's enc_match must hold it in bits 3:0.
B_COND_VALUES = {
    "B.EQ": 0,
    "B.NE": 1,
    "B.HS": 2,
    "B.CS": 2,
    "B.LO": 3,
    "B.CC": 3,
    "B.MI": 4,
    "B.PL": 5,
    "B.VS": 6,
    "B.VC": 7,
    "B.HI": 8,
    "B.LS": 9,
    "B.GE": 10,
    "B.LT": 11,
    "B.GT": 12,
    "B.LE": 13,
    "B.AL": 14,
    "B.NV": 15,
}

# (enc_bits, enc_mask, enc_match) the fixture's YAML encoding blocks parse to.
RV_PATTERNS = {
    "ld": (32, 0x707F, 0x3003),
    "lbu": (32, 0x707F, 0x4003),
    "addi": (32, 0x707F, 0x13),
    "beq": (32, 0x707F, 0x63),
    "bne": (32, 0x707F, 0x1063),
    "blt": (32, 0x707F, 0x4063),
    "bge": (32, 0x707F, 0x5063),
    "jal": (32, 0x7F, 0x6F),
    "flw": (32, 0x707F, 0x2007),
    "fsw": (32, 0x707F, 0x2027),
    "fld": (32, 0x707F, 0x3007),
    "fsd": (32, 0x707F, 0x3027),
    "fsgnj.s": (32, 0xFE00707F, 0x20000053),
    "fsgnj.d": (32, 0xFE00707F, 0x22000053),
    "c.ldsp": (16, 0xE003, 0x6002),
    "c.sdsp": (16, 0xE003, 0xE002),
    "c.addi": (16, 0xE003, 0x1),
    "c.addi16sp": (16, 0xEF83, 0x6101),
    "c.j": (16, 0xE003, 0xA001),
    "c.jal": (16, 0xE003, 0x2001),
    "c.beqz": (16, 0xE003, 0xC001),
    "c.bnez": (16, 0xE003, 0xE001),
    "c.fldsp": (16, 0xE003, 0x2002),
    "c.fsdsp": (16, 0xE003, 0xA002),
}


def _arm_records() -> list[InstructionRecord]:
    return parse_arm_instruction_payload((FIXTURES / "arm_bcond_instructions.json").read_text())


def _rv_records() -> list[InstructionRecord]:
    return parse_riscv_instruction_payload((FIXTURES / "riscv_instructions.json").read_text())


def _catalog(records: list[InstructionRecord]) -> Catalog:
    return Catalog(
        intrinsics=[],
        instructions=records,
        sources=[SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")],
        generated_at="2025-01-01",
    )


def _db(records: list[InstructionRecord], tmp_path: Path) -> Path:
    db = tmp_path / "catalog.db"
    build_sqlite(_catalog(records), db)
    return db


class TestBCondRows:
    """One B.cond row per condition code, each with its own encoding key."""

    def test_rows_fix_the_condition_field(self):
        rows = [r for r in _arm_records() if r.mnemonic.startswith("B.")]
        assert {r.mnemonic for r in rows} == set(B_COND_VALUES)
        for row in rows:
            assert row.enc_bits == 32
            assert row.enc_mask == 0xFF00001F
            assert row.enc_match == 0x54000000 | B_COND_VALUES[row.mnemonic]

    def test_rows_are_byte_distinct(self):
        encodings = {
            (r.enc_mask, r.enc_match) for r in _arm_records() if r.mnemonic.startswith("B.")
        }
        assert encodings == {(0xFF00001F, 0x54000000 | value) for value in range(16)}


class TestRiscvEncodingKeys:
    """The unified-db YAML encoding.match lands on every base row."""

    def test_patterns_parse_from_the_yaml_block(self):
        by_name = {r.mnemonic: r for r in _rv_records()}
        for mnemonic, (bits, mask, match) in RV_PATTERNS.items():
            row = by_name[mnemonic]
            assert (row.enc_bits, row.enc_mask, row.enc_match) == (bits, mask, match), mnemonic


@needs_tools
class TestAarch64BcondIdentity:
    """Each B.cond line resolves to exactly one row (issue #60 repro class)."""

    # The issue's repro text is "b.le\t114 <my_memcmp+0x40>"; its <sym>
    # annotation becomes the symbol expression after the call-site strip.
    CASES = [
        ("b.al\t401000", "B.AL"),
        ("b.eq\t0x114", "B.EQ"),
        ("b.cc\t0x114", "B.CC"),
        ("b.hs\t0x114", "B.HS"),
        ("b.le my_memcmp+0x40", "B.LE"),
        ("b.eq .Ltmp", "B.EQ"),
        ("b.gt .Ltmp", "B.GT"),
    ]

    def test_each_line_gets_its_own_row(self, tmp_path):
        db = _db(_arm_records(), tmp_path)
        for text, mnemonic in self.CASES:
            result = identify(text + "\n", "aarch64", catalog_path=db)[0]
            assert result.status == "row", (text, result.reason)
            assert result.row.mnemonic == mnemonic, text


@needs_tools
class TestRiscvIdentity:
    """Sp forms, pseudo-branches, j and FP loads/stores/moves resolve."""

    SP_FP_CASES = [
        ("ld\ts0,16(sp)", "c.ldsp"),
        ("ld\ts11,8(sp)", "c.ldsp"),
        ("addi\tsp,sp,-16", "c.addi"),
        ("lbu\ts1,3(sp)", "lbu"),
        ("fsd\tfa1,64(sp)", "c.fsdsp"),
        ("fld\tfa0,8(sp)", "c.fldsp"),
        ("flw\tfa0,0(a0)", "flw"),
        ("fsw\tfa1,64(sp)", "fsw"),
        ("fmv.s\tfa0,fa1", "fsgnj.s"),
        ("fmv.d\tfa0,fa1", "fsgnj.d"),
    ]
    # Symbol targets assemble to fixup placeholders ("A" bits); decimal and
    # hex targets assemble to a concrete encoding. Both spell the same row.
    BRANCH_CASES = [
        ("blez\ta0, .L48", "bge"),
        ("bgez\ta0, .L1", "bge"),
        ("bltz\ta0, .L1", "blt"),
        ("bgtz\ta0, .L1", "blt"),
        ("ble\ta0,a1, .L1", "bge"),
        ("bgt\ta0,a1,0x1c", "blt"),
        ("beqz\ta0, .L1", "c.beqz"),
        ("bnez\ta0, .L1", "c.bnez"),
        ("j\t.L34", "c.j"),
        ("j\t100", "c.j"),
    ]

    def test_sp_and_fp_lines_get_their_rows(self, tmp_path):
        db = _db(_rv_records(), tmp_path)
        for text, mnemonic in self.SP_FP_CASES:
            result = identify(text + "\n", "riscv64", catalog_path=db)[0]
            assert result.status == "row", (text, result.reason)
            assert result.row.mnemonic == mnemonic, text

    def test_pseudo_branches_map_to_the_real_instruction_row(self, tmp_path):
        db = _db(_rv_records(), tmp_path)
        for text, mnemonic in self.BRANCH_CASES:
            result = identify(text + "\n", "riscv64", catalog_path=db)[0]
            assert result.status == "row", (text, result.reason)
            assert result.row.mnemonic == mnemonic, text


@needs_tools
class TestOpcodeDeciderNeedsKnownBitProof:
    """The MCInst name may pick a row only with proof on the known bits.

    A fixup placeholder can cover the whole word (a RISC-V symbolic branch
    prints covered bytes). A row whose key equals the MCInst name but whose
    fixed operand bits contradict the printed bytes then has no byte proof,
    and the name alone must not pick it (review F1)."""

    # Control: the row fixes only op+funct3, the fields the name proves.
    HONEST = InstructionRecord(
        mnemonic="bge",
        form="bge",
        summary="ROW-BGE-HONEST",
        architecture="riscv",
        isa=["rvi"],
        enc_bits=32,
        enc_mask=0x707F,
        enc_match=0x5063,
    )
    # Same key "bge" (matches MCInst BGE), but it fixes rs2 = x1 while the
    # printed bytes carry rs2 = a0. The row contradicts known bits.
    CONTRADICTING = InstructionRecord(
        mnemonic="bge",
        form="bge",
        summary="ROW-BGE-RS2-X1",
        architecture="riscv",
        isa=["rvi"],
        enc_bits=32,
        enc_mask=0x707F | 0x1F00000,
        enc_match=0x5063 | (1 << 20),
    )

    def test_contradicting_row_gets_no_row(self, tmp_path):
        db = _db([self.CONTRADICTING], tmp_path)
        result = identify("blez a0, .L1\n", "riscv64", catalog_path=db)[0]
        assert result.status == "no-row", (result.status, result.reason)
        assert result.row is None

    def test_row_agreeing_with_the_bytes_still_wins(self, tmp_path):
        db = _db([self.HONEST], tmp_path)
        result = identify("blez a0, .L1\n", "riscv64", catalog_path=db)[0]
        assert result.status == "row", result.reason
        assert result.row.mnemonic == "bge"
        assert result.row.enc_match == 0x5063
