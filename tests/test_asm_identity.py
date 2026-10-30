"""Tests for simdref.asm_identity (spec C1 lookup half).

Tool-dependent tests skip cleanly when llvm-mc or the XED decoder is absent;
nothing fakes a pass. The pure-lookup tests need no tools at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdref.asm_identity import (
    ToolUnavailableError,
    _bucketize,
    _bytes_to_words,
    _refine_subset,
    _tool_path,
    identify,
)
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.storage import build_sqlite

LLVM_MC = _tool_path("SIMDREF_LLVM_MC", "llvm-mc") or ""
XED = _tool_path("SIMDREF_XED", "xed") or ""
needs_tools = pytest.mark.skipif(not LLVM_MC, reason="llvm-mc absent (set SIMDREF_LLVM_MC)")
needs_xed = pytest.mark.skipif(not (LLVM_MC and XED), reason="llvm-mc or xed absent")


def _enc_node(name: str, mnemonic: str, entries: list[tuple[int, int, str]]) -> dict:
    return {
        "_type": "Instruction.Instruction",
        "name": name,
        "operation_id": mnemonic,
        "assembly": {
            "symbols": [
                {"_type": "Instruction.Symbols.Literal", "value": mnemonic},
                {"_type": "Instruction.Symbols.RuleReference", "rule_id": "SPACE"},
                {"_type": "Instruction.Symbols.RuleReference", "rule_id": "Xd"},
            ]
        },
        "encoding": {
            "_type": "Instruction.Encodeset",
            "width": 32,
            "values": [
                {
                    "_type": "Instruction.Encodeset.Bits",
                    "range": {"_type": "Range", "start": start, "width": span},
                    "value": {"_type": "Values.Value", "meaning": None, "value": f"'{value}'"},
                }
                for start, span, value in entries
            ],
        },
    }


def _arm_catalog(nodes: list[dict], tmp_path: Path) -> Path:
    from simdref.arm_instructions import parse_arm_instruction_payload

    payload = {
        "instructions": [{"_type": "Instruction.InstructionSet", "name": "A64", "children": nodes}]
    }
    records = parse_arm_instruction_payload(json.dumps(payload))
    db = tmp_path / "catalog.db"
    build_sqlite(_catalog(records), db)
    return db


def _catalog(records: list[InstructionRecord]) -> Catalog:
    return Catalog(
        intrinsics=[],
        instructions=records,
        sources=[SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")],
        generated_at="2025-01-01",
    )


def _rv_row(mnemonic: str, form: str, bits: int, mask: int, match: int) -> InstructionRecord:
    return InstructionRecord(
        mnemonic=mnemonic,
        form=form,
        summary="Row.",
        architecture="riscv",
        enc_bits=bits,
        enc_mask=mask,
        enc_match=match,
        enc_extra=[["", bits, mask, match]],
    )


# ---------------------------------------------------------------------------
# pure lookup logic (no tools)
# ---------------------------------------------------------------------------


class TestSubsetRule:
    def test_covered_candidate_drops(self):
        fine = (0xFF, 0x12, "fine")
        coarse = (0xF0, 0x10, "coarse")
        kept = _refine_subset([fine, coarse])
        assert [entry[2] for entry in kept] == ["fine"]

    def test_unrelated_candidates_all_stay(self):
        a = (0xFF, 0x12, "a")
        b = (0x0F, 0x03, "b")
        assert len(_refine_subset([a, b])) == 2

    def test_identical_patterns_do_not_cover_each_other(self):
        a = (0xF0, 0x10, "a")
        b = (0xF0, 0x10, "b")
        assert len(_refine_subset([a, b])) == 2


class TestBuckets:
    def test_wildcard_window_expands(self):
        row = object()
        buckets = _bucketize([(0xFF00, 0x1200, row)], shift=8)
        assert list(buckets) == [0x12]
        assert buckets[0x12] == [(0xFF00, 0x1200, row)]

    def test_free_window_bits_land_in_every_value(self):
        row = object()
        buckets = _bucketize([(0x00FF, 0x0012, row)], shift=8)
        assert len(buckets) == 256


class TestBytesToWords:
    def test_fixup_shapes_zero_only_unknown_bits(self):
        # Shapes measured from llvm-mc: whole byte, marked byte, binary bits.
        raw = _bytes_to_words("0x63'A',A,0xb5'A',A")
        assert raw == bytes([0x63, 0x00, 0xB5, 0x00])
        raw = _bytes_to_words("0bAAAAAA01,0b101AAAAA")
        assert raw == bytes([0x01, 0xA0])
        raw = _bytes_to_words("0x33,0x85,0xc5,0x00")
        assert raw == bytes([0x33, 0x85, 0xC5, 0x00])


# ---------------------------------------------------------------------------
# end to end through llvm-mc
# ---------------------------------------------------------------------------


@needs_tools
class TestAarch64Document:
    def test_ret_line_hits_the_ret_row(self, tmp_path):
        # encodes RET: 1101011 0 0 10 11111 0000 0 0 xxxxx 00000 (Xd = x30)
        db = _arm_catalog(
            [
                _enc_node(
                    "RET_64R_branch_reg",
                    "RET",
                    [
                        (25, 7, "1101011"),
                        (21, 4, "0010"),
                        (16, 5, "11111"),
                        (10, 6, "000000"),
                    ],
                ),
                _enc_node("HINT_hints", "HINT", [(22, 10, "1101010100")]),
            ],
            tmp_path,
        )
        results = identify("ret\nnop", "aarch64", catalog_path=db)
        assert [r.status for r in results] == ["row", "row"]
        assert results[0].row.mnemonic == "RET"
        assert results[1].row.mnemonic == "HINT"

    def test_line_with_no_row_says_so(self, tmp_path):
        db = _arm_catalog([_enc_node("HINT_hints", "HINT", [(22, 10, "1101010100")])], tmp_path)
        results = identify("ret", "aarch64", catalog_path=db)
        assert results[0].status == "no-row"
        assert results[0].reason == "no catalog row"

    def test_rejected_line_carries_the_error(self, tmp_path):
        db = _arm_catalog([_enc_node("HINT_hints", "HINT", [(22, 10, "1101010100")])], tmp_path)
        results = identify("add x0, x1", "aarch64", catalog_path=db)
        assert results[0].status == "no-row"
        assert results[0].reason and results[0].reason != "no catalog row"

    def test_not_an_instruction(self, tmp_path):
        db = _arm_catalog([_enc_node("HINT_hints", "HINT", [(22, 10, "1101010100")])], tmp_path)
        results = identify("\n# comment\nlabel:\nret", "aarch64", catalog_path=db)
        assert [r.status for r in results] == [
            "not-an-instruction",
            "not-an-instruction",
            "not-an-instruction",
            "no-row",
        ]


@needs_tools
class TestArm32Document:
    def test_arm32_rows_never_come_from_a64_keys(self, tmp_path):
        db = _arm_catalog([_enc_node("HINT_hints", "HINT", [(22, 10, "1101010100")])], tmp_path)
        results = identify("mov r0, r1", "arm", catalog_path=db)
        assert results[0].status == "no-row"
        assert "arm32" in results[0].reason


@needs_tools
class TestRiscvDocument:
    def test_opcode_names_the_row_in_a_tie(self, tmp_path):
        # lr.w and lr.w.aq share one (mask, match); llvm-mc's MCInst opcode
        # names the row (review F3). Disabling the opcode stage makes both
        # lines ambiguous and fails this test.
        db = tmp_path / "catalog.db"
        build_sqlite(
            _catalog(
                [
                    _rv_row("lr.w", "lr.w", 32, 4193284223, 268443695),
                    _rv_row("lr.w.aq", "lr.w.aq", 32, 4193284223, 268443695),
                ]
            ),
            db,
        )
        results = identify("lr.w a0, (a1)\nlr.w.aq a0, (a1)", "riscv64", catalog_path=db)
        assert [r.status for r in results] == ["row", "row"]
        assert [r.row.mnemonic for r in results] == ["lr.w", "lr.w.aq"]

    def test_j_on_rv64_is_c_j_never_c_jal(self, tmp_path):
        # c.j and c.jal differ in the pattern; c.jal is RV32-only and
        # c.addiw (RV64) shares its pattern. The manager requirement: 'j' on
        # RV64 gives c.j; 'c.jal' on RV32 resolves its tie to c.jal.
        db = tmp_path / "catalog.db"
        build_sqlite(
            _catalog(
                [
                    _rv_row("c.j", "c.j", 16, 57347, 40961),
                    _rv_row("c.jal", "c.jal", 16, 57347, 8193),
                    _rv_row("c.addiw", "c.addiw", 16, 57347, 8193),
                ]
            ),
            db,
        )
        results = identify("j 8", "riscv64", catalog_path=db)
        assert results[0].status == "row"
        assert results[0].row.mnemonic == "c.j"
        results = identify("c.jal 8", "riscv32", catalog_path=db)
        assert results[0].status == "row"
        assert results[0].row.mnemonic == "c.jal"

    def test_tie_the_opcode_does_not_name_stays_ambiguous(self, tmp_path):
        db = tmp_path / "catalog.db"
        build_sqlite(
            _catalog([_rv_row("foo", "foo", 16, 0, 0), _rv_row("bar", "bar", 16, 0, 0)]),
            db,
        )
        result = identify("mv a0, a1", "riscv64", catalog_path=db)[0]
        assert result.status == "no-row"
        assert result.reason.startswith("ambiguous")
        assert set(result.candidates) == {"riscv:foo", "riscv:bar"}

    def test_both_spellings_byte_identify_to_c_mv(self, tmp_path):
        # mv a0, a1 and c.mv a0, a1 assemble to the same two bytes; the bytes
        # are the c.mv encoding, so both lines identify the c.mv row.
        db = tmp_path / "catalog.db"
        build_sqlite(
            _catalog(
                [
                    _rv_row("mv", "mv", 16, 0, 0),
                    _rv_row("c.mv", "c.mv", 16, 0, 0),
                    _rv_row("sub", "sub", 32, 0, 0),
                ]
            ),
            db,
        )
        results = identify("mv a0, a1\nc.mv a0, a1\nsub a0, a1, a2", "riscv64", catalog_path=db)
        assert [r.status for r in results] == ["row", "row", "row"]
        assert [r.row.mnemonic for r in results] == ["c.mv", "c.mv", "sub"]

    def test_base_scoped_patterns(self, tmp_path):
        row = InstructionRecord(
            mnemonic="slli",
            form="slli",
            summary="Row.",
            architecture="riscv",
            enc_bits=32,
            enc_mask=0xFFFFFFFF,
            enc_match=0x00C58533,  # add a0, a1, a2
            enc_extra=[
                ["RV32", 32, 0xFFFFFFFF, 0x00C58533],
                ["RV64", 32, 0xFFFFFFFF, 0x40C58533],  # sub a0, a1, a2
            ],
        )
        db = tmp_path / "catalog.db"
        build_sqlite(_catalog([row]), db)
        assert identify("add a0, a1, a2", "riscv64", catalog_path=db)[0].status == "no-row"
        assert identify("sub a0, a1, a2", "riscv32", catalog_path=db)[0].status == "no-row"
        assert identify("add a0, a1, a2", "riscv32", catalog_path=db)[0].status == "row"
        assert identify("sub a0, a1, a2", "riscv64", catalog_path=db)[0].status == "row"

    def test_e_family_rows_serve_their_base(self, tmp_path):
        # RV32E/RV64E keys serve their whole base family (revB probe I).
        row = InstructionRecord(
            mnemonic="add",
            form="add",
            summary="Row.",
            architecture="riscv",
            enc_bits=32,
            enc_mask=0xFFFFFFFF,
            enc_match=0x00C58533,
            enc_extra=[["RV32E", 32, 0xFFFFFFFF, 0x00C58533]],
        )
        db = tmp_path / "catalog.db"
        build_sqlite(_catalog([row]), db)
        assert identify("add a0, a1, a2", "riscv32", catalog_path=db)[0].status == "row"
        assert identify("add a0, a1, a2", "riscv64", catalog_path=db)[0].status == "no-row"


@needs_xed
class TestX86Document:
    def test_split_attrs_pick_the_sized_row(self, tmp_path):
        from simdref.asm_identity import _xed_facts

        facts = _xed_facts(["01d8", "4801c3"], XED)  # addl %eax,%ebx / addq %rax,%rbx
        assert facts[0]["iform"] == facts[1]["iform"], facts
        iform = facts[0]["iform"]
        rows = [
            InstructionRecord(
                mnemonic="ADD",
                form="ADD (R32, R32)",
                summary="Row.",
                architecture="x86",
                metadata={"iform": iform, "split_attrs": json.dumps({"eosz": "2"})},
            ),
            InstructionRecord(
                mnemonic="ADD",
                form="ADD (R64, R64)",
                summary="Row.",
                architecture="x86",
                metadata={"iform": iform, "split_attrs": json.dumps({"eosz": "3"})},
            ),
        ]
        db = tmp_path / "catalog.db"
        build_sqlite(_catalog(rows), db)
        results = identify("addl %eax, %ebx\naddq %rax, %rbx", "x86", "att", catalog_path=db)
        assert [r.status for r in results] == ["row", "row"]
        assert [r.row.form for r in results] == ["ADD (R32, R32)", "ADD (R64, R64)"]

    def test_iform_absent_is_a_reason(self, tmp_path):
        rows = [
            InstructionRecord(
                mnemonic="ADD",
                form="ADD (R64, R64)",
                summary="Row.",
                architecture="x86",
                metadata={"iform": "NOT_A_REAL_IFORM", "split_attrs": "{}"},
            )
        ]
        db = tmp_path / "catalog.db"
        build_sqlite(_catalog(rows), db)
        result = identify("addq %rax, %rbx", "x86", "att", catalog_path=db)[0]
        assert result.status == "no-row"
        assert "absent" in result.reason


@needs_xed
class TestX86SplitStages:
    """Pin each _x86_pick stage (review F2): rows differ only in the stage's
    attribute and the pick must name the right one."""

    CASES = 0

    def _pick(self, tmp_path, hexes, line, attrs, syntax="att"):
        from simdref.asm_identity import _xed_facts

        iform = _xed_facts([hexes], XED)[0]["iform"]
        rows = [
            InstructionRecord(
                mnemonic="T",
                form=label,
                summary="Row.",
                architecture="x86",
                metadata={"iform": iform, "split_attrs": json.dumps(a)},
            )
            for label, a in attrs
        ]
        type(self).CASES += 1
        db = tmp_path / f"catalog-{self.CASES}.db"
        build_sqlite(_catalog(rows), db)
        result = identify(line, "x86", syntax, catalog_path=db)[0]
        assert result.status == "row", result
        return result.row.form

    def test_agen_address_forms(self, tmp_path):
        # baseless, index-only, RIP-relative, disp-only, base forms and
        # segment-prefixed LEA and memory lines (manager F1).
        cases = [
            ("lea (,%r11,4), %rax", "4a8d049d00000000", "IS_D32", "B"),
            ("lea (,%r11), %rax", "4a8d041d00000000", "I_D32", "B"),
            ("lea 0x10(%rip), %rax", "488d0510000000", "R_D32", "B"),
            ("lea 0x1234, %rax", "488d042534120000", "D32", "B"),
            ("lea 32(%r10), %rax", "498d4220", "B_D8", "B"),
            ("lea (%r10,%r11), %rax", "4b8d041a", "B_I", "B_D8"),
            ("lea (%r10,%r11,4), %rax", "4b8d049a", "B_IS", "B_I"),
            ("lea %fs:(,%r11,4), %rax", "644a8d049d00000000", "IS_D32", "B"),
            ("lea %fs:0x10(%rip), %rax", "64488d0510000000", "R_D32", "B"),
            ("mov %fs:(%rax), %rbx", "64488b18", "B", "D32"),
            ("mov 0x1234, %eax", "8b042534120000", "D32", "B"),
        ]
        for line, hexes, target, other in cases:
            picked = self._pick(
                tmp_path,
                hexes,
                line,
                [("TARGET", {"agen": target}), ("OTHER", {"agen": other})],
            )
            assert picked == "TARGET", (line, picked)

    def test_high8_stage(self, tmp_path):
        cases = [
            ("add %ah, %al", "00e0", "H-REG1"),
            ("add %al, %ah", "00c4", "H-REG0"),
            ("add %al, %bl", "00c3", "H-EMPTY"),
        ]
        attrs = [
            ("H-REG0", {"high8": "REG0"}),
            ("H-REG1", {"high8": "REG1"}),
            ("H-EMPTY", {"high8": ""}),
        ]
        for line, hexes, expect in cases:
            assert self._pick(tmp_path, hexes, line, attrs) == expect, line

    def test_immzero_stage(self, tmp_path):
        attrs = [("Z-1", {"immzero": "1"}), ("Z-0", {"immzero": "0"})]
        assert self._pick(tmp_path, "66e500", "in $0, %ax", attrs) == "Z-1"
        assert self._pick(tmp_path, "66e505", "in $5, %ax", attrs) == "Z-0"

    def test_mask_stage(self, tmp_path):
        attrs = [("M-1", {"mask": "1"}), ("M-0", {"mask": "0"})]
        assert (
            self._pick(tmp_path, "62f16c4958c1", "vaddps %zmm1, %zmm2, %zmm0{%k1}", attrs) == "M-1"
        )
        assert self._pick(tmp_path, "62f16c4858c1", "vaddps %zmm1, %zmm2, %zmm0", attrs) == "M-0"

    def test_zeroing_stage(self, tmp_path):
        attrs = [("Q-1", {"zeroing": "1"}), ("Q-0", {"zeroing": "0"})]
        assert (
            self._pick(tmp_path, "62f16cc958c1", "vaddps %zmm1, %zmm2, %zmm0{%k1}{z}", attrs)
            == "Q-1"
        )
        assert (
            self._pick(tmp_path, "62f16c4958c1", "vaddps %zmm1, %zmm2, %zmm0{%k1}", attrs) == "Q-0"
        )

    def test_unknown_agen_token_is_a_reason(self, tmp_path):
        # An unknown or empty agen token reports one no-row reason, never raises.
        from simdref.asm_identity import _xed_facts

        iform = _xed_facts(["498d4220"], XED)[0]["iform"]
        cases = [
            ("B_Z_D32", "no catalog row (unknown agen token 'Z' in 'B_Z_D32')"),
            ("", "no catalog row (unknown agen token '' in '')"),
        ]
        for n, (value, expect) in enumerate(cases):
            rows = [
                InstructionRecord(
                    mnemonic="T",
                    form=label,
                    summary="Row.",
                    architecture="x86",
                    metadata={"iform": iform, "split_attrs": json.dumps({"agen": value})},
                )
                for label in ("BAD-1", "BAD-2")
            ]
            db = tmp_path / f"bad-agen-{n}.db"
            build_sqlite(_catalog(rows), db)
            result = identify("lea 32(%r10), %rax", "x86", "att", catalog_path=db)[0]
            assert result.status == "no-row", result
            assert result.reason == expect


@needs_tools
class TestFixupAndModeDeclines:
    """Byte identity declines when a fixup placeholder or a wrong decoder
    mode would decide the row (Sol's blockers D1/D2); the old matcher then
    answers the line, as with the tools hidden."""

    def test_symbol_immediate_later_set_gets_no_row(self, tmp_path):
        # Sol's D1 probe verbatim: the fixup-covered imm8 feeds the immzero
        # split (zero fill reads imm=0), and its echoed width is placeholder
        # data as well (probe-widths 2).
        from simdref.asm_identity import _xed_facts

        # shll/addl bytes with the imm zeroed, as the placeholder decodes.
        iforms = [fact["iform"] for fact in _xed_facts(["c1e000", "83c000"], XED)]
        rows = [
            InstructionRecord(
                mnemonic="T",
                form=f"{iform}-{zero}",
                summary="Row.",
                architecture="x86",
                metadata={"iform": iform, "split_attrs": json.dumps({"immzero": zero})},
            )
            for iform in iforms
            for zero in ("1", "0")
        ]
        db = tmp_path / "catalog.db"
        build_sqlite(_catalog(rows), db)
        doc = "shll $symbol, %eax\naddl $symbol, %eax\n.set symbol, 2"
        results = identify(doc, "x86", "att", catalog_path=db)
        for result in results[:2]:
            assert result.row is None, result
            assert "fixup" in result.reason, result.reason

    def test_code16_mode_gets_no_row(self, tmp_path):
        # Sol's D2 probe verbatim: .code16 encodes MOV16rr as 89c3, which the
        # 64-bit decode names MOV (R32, R32).
        from simdref.asm_identity import _xed_facts

        iform = _xed_facts(["89c3"], XED)[0]["iform"]
        rows = [
            InstructionRecord(
                mnemonic="MOV",
                form="MOV (R32, R32)",
                summary="Row.",
                architecture="x86",
                metadata={"iform": iform, "split_attrs": "{}"},
            )
        ]
        db = tmp_path / "catalog.db"
        build_sqlite(_catalog(rows), db)
        doc = ".code16\nmovw %ax, %bx\npushw %ax"
        results = identify(doc, "x86", "att", catalog_path=db)
        for result in results[1:]:
            assert result.row is None, result
            assert "16-bit mode" in result.reason, result.reason

    def test_code32_mode_gets_no_row(self, tmp_path):
        # Sol's mode32 probe verbatim: .code32 pushl encodes 50 (PUSH32r),
        # which the 64-bit decode names PUSH (R64).
        from simdref.asm_identity import _xed_facts

        rows = [
            InstructionRecord(
                mnemonic="T",
                form=form,
                summary="Row.",
                architecture="x86",
                metadata={"iform": _xed_facts([hexes], XED)[0]["iform"], "split_attrs": "{}"},
            )
            for form, hexes in [("PUSH (R64)", "50"), ("MOV (R32, R32)", "89c3")]
        ]
        db = tmp_path / "catalog.db"
        build_sqlite(_catalog(rows), db)
        doc = ".code32\npushl %eax\nmovl %eax, %ebx"
        results = identify(doc, "x86", "att", catalog_path=db)
        for result in results[1:]:
            assert result.row is None, result
            assert "32-bit mode" in result.reason, result.reason

    def _mode_catalog(self, tmp_path):
        # MOV16rr assembles to 89c3, which the 64-bit decode names MOV (R32, R32).
        from simdref.asm_identity import _xed_facts

        rows = [
            InstructionRecord(
                mnemonic="MOV",
                form="MOV (R32, R32)" if hexes == "89c3" else "NOP",
                summary="Row.",
                architecture="x86",
                metadata={"iform": iform, "split_attrs": "{}"},
            )
            for hexes in ("89c3", "90")
            for iform in [_xed_facts([hexes], XED)[0]["iform"]]
        ]
        db = tmp_path / "catalog.db"
        build_sqlite(_catalog(rows), db)
        return db

    def _assert_no_row_in_mode(self, tmp_path, doc, mode):
        # The line assembles in a non-64 mode; the 64-bit decoder must not
        # name its row, and the old matcher answers it instead.
        db = self._mode_catalog(tmp_path)
        result = identify(doc, "x86", "att", catalog_path=db)[-1]
        assert result.row is None, result
        assert f"{mode}-bit mode" in result.reason, result.reason

    def test_mode_in_inactive_if_body_gets_no_row(self, tmp_path):
        # Sol's blocker P2 verbatim: the .code64 sits in a skipped .if 0
        # body, so the movw assembles in 16-bit mode (llvm-mc MOV16rr, 89c3).
        self._assert_no_row_in_mode(tmp_path, ".code16\n.if 0\n.code64\n.endif\nmovw %ax, %bx", 16)

    def test_mode_in_macro_body_gets_no_row(self, tmp_path):
        # The .code64 is macro body text only; the movw runs in 16-bit mode.
        self._assert_no_row_in_mode(
            tmp_path, ".code16\n.macro up\n.code64\n.endm\nmovw %ax, %bx", 16
        )

    def test_mode_in_comment_gets_no_row(self, tmp_path):
        # The .code64 is block-comment text; the movw runs in 16-bit mode.
        self._assert_no_row_in_mode(tmp_path, ".code16\n/*\n.code64\n*/\nmovw %ax, %bx", 16)

    def test_mode_after_semicolon_statement_gets_no_row(self, tmp_path):
        # The "; .code16" statement executes after the nop: the nop keeps its
        # row (64-bit mode) and the movw runs in 16-bit mode.
        db = self._mode_catalog(tmp_path)
        results = identify("nop; .code16\nmovw %ax, %bx", "x86", "att", catalog_path=db)
        assert results[0].row is not None, results[0]
        assert results[-1].row is None, results[-1]
        assert "16-bit mode" in results[-1].reason, results[-1].reason

    def test_mode_in_inactive_else_branch_gets_no_row(self, tmp_path):
        # The .code64 sits in the skipped .else branch; the movl runs in
        # 32-bit mode (llvm-mc MOV32rr).
        self._assert_no_row_in_mode(
            tmp_path, ".code16\n.if 1\n.code32\n.else\n.code64\n.endif\nmovl %eax, %ebx", 32
        )

    def test_fixup_bits_never_pick_a_word_row(self, tmp_path):
        # A64 adrp (probe-words D2): the fixup covers the immlo bits that
        # separate these two rows, so the zero fill must not pick one.
        db = _arm_catalog(
            [
                _enc_node("ADRP_op1", "ADRP", [(31, 1, "1")]),
                _enc_node("ADRP_op1_immlo0", "ADRP", [(31, 1, "1"), (30, 1, "0")]),
            ],
            tmp_path,
        )
        result = identify("adrp x0, sym", "aarch64", catalog_path=db)[0]
        assert result.row is None, result
        assert "fixup" in result.reason, result.reason


@needs_tools
class TestUserTextNeverMovesIdentity:
    """User text never decides a mode, a byte string or an opcode (Sol r3
    P2 and the fix5 audit sites).

    .print writes to stdout, and llvm-mc's output file reprints user names
    and strings (".set .code16, 1" echoes ".code16 = 1"; .ascii echoes its
    string). Only tool-generated shapes may be read."""

    def _x86_db(self, tmp_path, specs):
        from simdref.asm_identity import _xed_facts

        rows = [
            InstructionRecord(
                mnemonic=form.split()[0],
                form=form,
                summary="Row.",
                architecture="x86",
                metadata={"iform": _xed_facts([hexes], XED)[0]["iform"], "split_attrs": "{}"},
            )
            for form, hexes in specs
        ]
        db = tmp_path / "catalog.db"
        build_sqlite(_catalog(rows), db)
        return db

    def test_sol_print_reproducer_gets_no_row(self, tmp_path):
        # Sol r3's reproducer verbatim: the printed ".code64" must not move
        # the tracked mode, so the movw stays a 16-bit decline. On c1c4c9b
        # the printed flag gives it the wrong MOV (R32, R32) row.
        db = self._x86_db(tmp_path, [("MOV (R32, R32)", "89c3")])
        doc = ".code16\n" + "nop\n" * 1500 + '.print ""\n.print ".code64"\nmovw %ax, %bx'
        result = identify(doc, "x86", "att", catalog_path=db)[-1]
        assert result.row is None, result
        assert "16-bit mode" in result.reason, result.reason

    def test_printed_flag_never_declines_a_64bit_line(self, tmp_path):
        # The false-decline counterpart: a printed ".code16" in 64-bit mode
        # must not hide the movq row (Sol r3: tip gave no row here).
        db = self._x86_db(tmp_path, [("MOV (R64, R64)", "4889d8")])
        result = identify('.print ".code16"\nmovq %rax, %rbx', "x86", "att", catalog_path=db)[-1]
        assert result.status == "row", result
        assert result.row.form == "MOV (R64, R64)", result

    def test_assignment_print_never_sets_the_mode(self, tmp_path):
        # Audit site 1: '.set .code64, 1' echoes '.code64 = 1', and on
        # c1c4c9b that assignment print reads as a mode flag in both
        # directions.
        db = self._x86_db(tmp_path, [("MOV (R32, R32)", "89c3"), ("ADD (R64, R64)", "4801c3")])
        wrong_row = identify(
            ".code16\n.set .code64, 1\nmovw %ax, %bx", "x86", "att", catalog_path=db
        )[-1]
        assert wrong_row.row is None, wrong_row
        assert "16-bit mode" in wrong_row.reason, wrong_row.reason
        false_decline = identify(".set .code16, 1\naddq %rax, %rbx", "x86", "att", catalog_path=db)[
            -1
        ]
        assert false_decline.status == "row", false_decline

    def test_string_echo_never_fakes_an_encoding_comment(self, tmp_path):
        # Audit site 2: a macro call expanding to .ascii reprints the string
        # where the call line's block is read, and on c1c4c9b the string's
        # 'encoding: [0xc3]' gives the call line the ret row.
        db = self._x86_db(tmp_path, [("RET_NEAR", "c3")])
        doc = '.macro m\n.ascii " # encoding: [0xc3]"\n.endm\nm\nretq'
        results = identify(doc, "x86", "att", catalog_path=db)
        assert results[3].row is None, results[3]
        assert results[4].row is not None, results[4]  # control: real retq

    def test_string_echo_never_fakes_an_mcinst_line(self, tmp_path):
        # Audit site 3: one string carries fake bytes and a fake MCInst that
        # names a different tie row; the opcode tie-break must read neither.
        db = tmp_path / "catalog.db"
        build_sqlite(
            _catalog(
                [
                    _rv_row("lr.w", "lr.w", 32, 4193284223, 268443695),
                    _rv_row("lr.w.aq", "lr.w.aq", 32, 4193284223, 268443695),
                ]
            ),
            db,
        )
        doc = (
            ".macro m\n"
            '.ascii " # encoding: [0xaf,0x22,0x00,0x10] # <MCInst #0 LR_W_AQ x"\n'
            ".endm\nm\nlr.w a0, (a1)"
        )
        results = identify(doc, "riscv64", catalog_path=db)
        assert results[3].row is None, results[3]
        assert results[4].row.mnemonic == "lr.w", results[4]  # control: real lr.w


def _keyless_db(tmp_path) -> Path:
    """A schema-13 shaped catalog: 9 instruction columns, no encoding keys."""
    import sqlite3

    db = tmp_path / "keyless.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('schema_version', '13')")
        conn.execute(
            "CREATE TABLE instructions_data (db_key TEXT, key TEXT, architecture TEXT,"
            " mnemonic TEXT, form TEXT, summary TEXT, isa TEXT, category TEXT, payload BLOB)"
        )
        conn.execute("CREATE INDEX idx_instruction_category ON instructions_data (category)")
        conn.execute(
            "CREATE TABLE intrinsics_data (id TEXT, name TEXT, architecture TEXT,"
            " signature TEXT, description TEXT, header TEXT, isa TEXT, category TEXT,"
            " subcategory TEXT, arm_arch TEXT, payload BLOB)"
        )
        conn.execute("CREATE INDEX idx_intrinsic_arm_arch ON intrinsics_data (arm_arch)")
        conn.commit()
    finally:
        conn.close()
    return db


def test_keyless_catalog_gives_one_reason(tmp_path):
    # C4: a schema-13 catalog has no encoding keys; every instruction line
    # reports that one reason. Needs no tools (the keyless path runs first).
    results = identify(
        "addq %rax, %rbx\n\n# comment\n", "x86", "att", catalog_path=_keyless_db(tmp_path)
    )
    assert [r.status for r in results] == ["no-row", "not-an-instruction", "not-an-instruction"]
    assert results[0].reason == "catalog lacks encoding keys (schema 13)"


@needs_tools
class TestToolFailureModes:
    def test_echo_mismatch_reports_every_line(self, tmp_path):
        # A fake llvm-mc emits one echo block for two source lines: the
        # line-to-bytes mapping is unverified, so no line gets a row (F4).
        fake = tmp_path / "fake-mc"
        fake.write_text(
            "#!/bin/sh\n"
            "out=\n"
            "while [ $# -gt 0 ]; do\n"
            '  if [ "$1" = "-o" ]; then out="$2"; shift 2; else shift; fi\n'
            "done\n"
            '[ -n "$out" ] || exit 1\n'
            "cat > \"$out\" <<'EOF'\n"
            "    ret                             # encoding: [0xd6,0x5f,0x03,0xd5]\n"
            "                                    # <MCInst #1 RET>\n"
            "EOF\n"
        )
        fake.chmod(0o755)
        db = _arm_catalog([_enc_node("RET_64R_branch_reg", "RET", [(25, 7, "1101011")])], tmp_path)
        results = identify("ret\nret", "aarch64", catalog_path=db, llvm_mc=str(fake))
        assert [r.status for r in results] == ["no-row", "no-row"]
        for result in results:
            assert result.reason.startswith("line mapping unverified"), result.reason

    def test_xed_empty_output_line_is_a_reason(self, tmp_path):
        # A fake xed decoder printing an empty line must not crash (F4).
        fake = tmp_path / "fake-xed"
        fake.write_text("#!/bin/sh\necho\n")
        fake.chmod(0o755)
        rows = [
            InstructionRecord(
                mnemonic="ADD",
                form="ADD (R64, R64)",
                summary="Row.",
                architecture="x86",
                metadata={"iform": "ADD_GPRv_GPRv_01", "split_attrs": "{}"},
            )
        ]
        db = tmp_path / "catalog.db"
        build_sqlite(_catalog(rows), db)
        result = identify("addq %rax, %rbx", "x86", "att", catalog_path=db, xed=str(fake))[0]
        assert result.status == "no-row"
        assert "EMPTY_OUTPUT" in result.reason

    def test_catalog_cache_invalidates_on_rebuild(self, tmp_path):
        # F4: the cache keys on the file mtime, so a rebuilt catalog at the
        # same path is re-read.
        db = _arm_catalog([_enc_node("HINT_hints", "HINT", [(22, 10, "1101010100")])], tmp_path)
        first = identify("nop", "aarch64", catalog_path=db)[0]
        assert first.status == "row" and first.row.mnemonic == "HINT"
        _arm_catalog([_enc_node("RET_64R_branch_reg", "RET", [(25, 7, "1101011")])], tmp_path)
        second = identify("ret", "aarch64", catalog_path=db)[0]
        assert second.status == "row" and second.row.mnemonic == "RET"


def test_missing_tool_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMDREF_LLVM_MC", "")
    monkeypatch.setattr("simdref.asm_identity._tool_path", lambda *_a, **_k: None)
    db = _arm_catalog([_enc_node("HINT_hints", "HINT", [(22, 10, "1101010100")])], tmp_path)
    with pytest.raises(ToolUnavailableError):
        identify("ret", "aarch64", catalog_path=db)


def test_bad_arch_and_syntax_are_rejected():
    with pytest.raises(ValueError):
        identify("ret", "mips")
    with pytest.raises(ValueError):
        identify("mov", "x86", "gas")


@needs_xed
class TestLineMapFromToolOutput:
    """One test per source-line/echo divergence class (review E1/E2).

    Probe lines are verbatim from the website-port review artifacts
    (review-web/probes/probe_all.py, review-web/reviewer-a/att-clean.s,
    logs/review-web.log). The review quotes the probe lines
    'GOT x86:cmp (i32, i32, i32) WANT x86:test (i32, i32)' and
    'line mapping unverified (echo 3 vs source ...)'.
    """

    def _db(self, tmp_path, items, name):
        from simdref.asm_identity import _xed_facts

        rows = []
        for form, hexes, attrs in items:
            iform = _xed_facts([hexes], XED)[0]["iform"]
            rows.append(
                InstructionRecord(
                    mnemonic=form.split(" ")[0],
                    form=form,
                    summary="Row.",
                    architecture="x86",
                    metadata={"iform": iform, "split_attrs": json.dumps(attrs)},
                )
            )
        db = tmp_path / name
        build_sqlite(_catalog(rows), db)
        return db

    def test_semicolon_statement_separator_never_shifts_the_map(self, tmp_path):
        # ';' is a statement separator for llvm-mc x86 AT&T ('; comment line'
        # from att-clean.s is two statements there), so the test line must
        # keep its own row and never the cmp row of the line above.
        db = self._db(
            tmp_path,
            [("CMP (R32, R32)", "39c1", {}), ("TEST (R32, R32)", "85c1", {})],
            "semi.db",
        )
        results = identify(
            "\tcmpl %eax, %ecx\n; comment line\n\ttestl %eax, %ecx\n",
            "x86",
            "att",
            catalog_path=db,
        )
        assert results[0].row.db_key == "x86:cmp (r32, r32)"
        assert results[1].status == "no-row"  # llvm-mc's own error
        assert results[2].row.db_key == "x86:test (r32, r32)", (
            "GOT x86:cmp (i32, i32, i32) WANT x86:test (i32, i32)"
        )

    def test_comment_marker_line_with_code_still_identifies(self, tmp_path):
        # '# comment line' and '// slash comment' (probe_all.py edge docs) drop
        # on both sides, but a comment marker in front of code is code to
        # llvm-mc and must not hide the instruction from the map.
        db = self._db(
            tmp_path,
            [
                ("ADD (R64, R64)", "4801c3", {"eosz": "3"}),
                ("ADD (R32, R32)", "01c1", {"eosz": "2"}),
            ],
            "comments.db",
        )
        results = identify(
            "/* c */ addq %rax, %rbx\n# comment line\n// slash comment\n\taddl %eax, %ecx\n",
            "x86",
            "att",
            catalog_path=db,
        )
        assert results[0].row.form == "ADD (R64, R64)"
        assert results[1].reason == "comment"
        assert results[2].reason == "comment"
        assert results[3].row.form == "ADD (R32, R32)"

    def test_trailing_blank_lines_keep_their_rows(self, tmp_path):
        # E2 probe line: 'line mapping unverified (echo 3 vs source ...)'.
        # The document join dropped the trailing blank lines (find_trigger.py
        # "trailing-blanks": instr + ["", ""]) and no line got a row.
        db = self._db(
            tmp_path,
            [
                ("ADD (R64, R64)", "4801c3", {"eosz": "3"}),
                ("ADD (R32, R32)", "01c1", {"eosz": "2"}),
            ],
            "trailing.db",
        )
        results = identify(
            "\taddq %rax, %rbx\n\taddl %eax, %ecx\n\n\n",
            "x86",
            "att",
            catalog_path=db,
        )
        assert results[0].row.form == "ADD (R64, R64)", (
            "line mapping unverified (echo 3 vs source ...)"
        )
        assert results[1].row.form == "ADD (R32, R32)"
        assert [results[2].reason, results[3].reason] == ["blank", "blank"]

    def test_crlf_trailing_lines_keep_their_rows(self, tmp_path):
        # The CRLF goal docs reported 'line mapping unverified (echo 39 vs
        # source 38)' (logs/review-web.log); the same join drops a trailing
        # CRLF line.
        db = self._db(
            tmp_path,
            [
                ("ADD (R64, R64)", "4801c3", {"eosz": "3"}),
                ("ADD (R32, R32)", "01c1", {"eosz": "2"}),
            ],
            "crlf.db",
        )
        results = identify(
            "\taddq %rax, %rbx\r\n\taddl %eax, %ecx\r\n\r\n",
            "x86",
            "att",
            catalog_path=db,
        )
        assert results[0].row.form == "ADD (R64, R64)", (
            "line mapping unverified (echo 39 vs source 38)"
        )
        assert results[1].row.form == "ADD (R32, R32)"
        assert results[2].reason == "blank"

    def test_form_feed_terminator_keeps_the_last_line(self, tmp_path):
        # A form feed is a line terminator for splitlines; the join dropped
        # the line it ends unless a newline follows.
        db = self._db(
            tmp_path,
            [
                ("ADD (R64, R64)", "4801c3", {"eosz": "3"}),
                ("ADD (R32, R32)", "01c1", {"eosz": "2"}),
            ],
            "formfeed.db",
        )
        results = identify(
            "\taddq %rax, %rbx\n\taddl %eax, %ecx\n\x0c",
            "x86",
            "att",
            catalog_path=db,
        )
        assert results[0].row.form == "ADD (R64, R64)"
        assert results[1].row.form == "ADD (R32, R32)"
        assert results[2].reason == "blank"

    def test_line_continuation_lines_report_no_row(self, tmp_path):
        # A backslash continuation (the '.irp x, 1, \' probe line) creates
        # lines llvm-mc reports only as <instantiation> errors, invisible to
        # the <stdin> error map: the continuation class must not hide the
        # rows of the rest of the document.
        db = self._db(tmp_path, [("NOP", "90", {})], "cont.db")
        results = identify(
            ".irp x, 1, \\\n2\n.byte \\x\n.endr\n\tnop\n", "x86", "att", catalog_path=db
        )
        assert results[0].reason == "directive"
        assert results[1].status == "no-row"  # twice echoed marker: unproven
        assert "line mapping unverified" in results[1].reason
        assert [results[2].reason, results[3].reason] == ["directive", "directive"]
        assert results[4].row.db_key == "x86:nop"

    def test_directive_that_emits_no_instruction_keeps_neighbours(self, tmp_path):
        # '.endm' and a macro body emit no echo block; the instruction after
        # them must keep its row.
        db = self._db(
            tmp_path,
            [("NOP", "90", {}), ("ADD (R64, R64)", "4801c3", {"eosz": "3"})],
            "directive.db",
        )
        results = identify(
            ".macro mym\n\tnop\n.endm\n\taddq %rax, %rbx\n",
            "x86",
            "att",
            catalog_path=db,
        )
        assert [results[0].reason, results[2].reason] == ["directive", "directive"]
        assert results[1].status == "no-row"  # body emits nothing at definition
        assert results[3].row.form == "ADD (R64, R64)"

    def test_macro_expansion_maps_to_the_call_line(self, tmp_path):
        # A macro call emits the expansion's blocks on the call line: one
        # instruction means the call line carries its bytes, and the body
        # line reports no row.
        db = self._db(
            tmp_path,
            [("NOP", "90", {}), ("ADD (R32, R32)", "01c1", {"eosz": "2"})],
            "macro.db",
        )
        results = identify(
            ".macro mym\n\tnop\n.endm\nmym\n\taddl %eax, %ecx\n",
            "x86",
            "att",
            catalog_path=db,
        )
        assert results[1].status == "no-row"
        assert results[3].row.form == "NOP"
        assert results[4].row.form == "ADD (R32, R32)"
