"""Tests for the AMD Software Optimization Guide workbook parser."""

from __future__ import annotations

import unittest
from pathlib import Path

from simdref.perf_sources.amd_swopt import (
    WORKBOOKS,
    _alt_lines,
    _alt_width,
    _operand_alts,
    _op_column,
    _vendor_alt_classes,
    concrete_lines,
    load_vendored,
    parse_workbook,
)
from simdref.perf_sources.cores import canonical_core_id
from simdref.perf_sources.merge import SOURCE_AMD_SWOPT, PerfRow

FIXTURES = Path(__file__).parent / "fixtures" / "amd_swopt"


class ParseWorkbookTests(unittest.TestCase):
    def test_zen5_fixture_rows_exact(self):
        rows, stats = parse_workbook(
            FIXTURES / "zen5_sample.xlsx",
            core="ZEN5",
            source_version="1.00",
            citation_url="https://docs.amd.com/v/u/en-US/58455_1.00",
        )
        self.assertEqual(stats.dropped_unknown_core, 0)
        self.assertEqual([r.mnemonic for r in rows], ["vaddpd", "vdivpd", "vsqrtpd"])

        addpd, divpd, sqrtpd = rows
        self.assertEqual(addpd.form, "xyz,xyz,xyz/m")
        self.assertEqual(addpd.latency, "2")
        self.assertEqual(addpd.extra_measurement["amd_ops_per_cycle"], "2")
        self.assertEqual(addpd.extra_measurement["cpuid"], "AVX512VL AVX512F")

        self.assertEqual(divpd.form, "xy,xy,xy/m")
        self.assertEqual(divpd.latency, "13")
        self.assertEqual(divpd.extra_measurement["amd_ops_per_cycle"], "0.25")
        self.assertEqual(divpd.extra_measurement["cpuid"], "avx")

        self.assertEqual(sqrtpd.form, "xyz,xyz/m")
        self.assertEqual(sqrtpd.latency, "20")
        self.assertEqual(sqrtpd.extra_measurement["amd_ops_per_cycle"], "0.13300000000000001")
        self.assertEqual(sqrtpd.extra_measurement["cpuid"], "AVX512VL AVX512F")

        for row in rows:
            self.assertEqual(row.core, "ZEN5")
            self.assertEqual(row.source, SOURCE_AMD_SWOPT)
            self.assertEqual(row.source_kind, "modeled")
            self.assertEqual(row.source_version, "1.00")
            self.assertEqual(row.architecture, "x86")
            self.assertEqual(row.citation_url, "https://docs.amd.com/v/u/en-US/58455_1.00")

    def test_family17h_fixture_row_exact(self):
        rows, stats = parse_workbook(
            FIXTURES / "family17h_sample.xlsx",
            core="zen1",
            source_version="1.00",
            citation_url="https://docs.amd.com/v/u/en-US/55723_3.01",
        )
        self.assertEqual(stats.dropped_unknown_core, 0)
        self.assertEqual(len(rows), 1)
        (row,) = rows
        # "zen1" resolves to the canonical spelling; the mnemonic stays verbatim.
        self.assertEqual(row.core, "ZEN1")
        self.assertEqual(row.mnemonic, "AAA")
        self.assertEqual(row.form, "")
        self.assertEqual(row.latency, "3")
        self.assertEqual(row.extra_measurement["amd_ops_per_cycle"], "0.3")
        self.assertEqual(row.extra_measurement["cpuid"], "general")

    def test_every_registry_core_resolves(self):
        for spec in WORKBOOKS:
            with self.subTest(workbook=spec.match):
                self.assertEqual(canonical_core_id(spec.core), spec.core)


class ConcreteLinesTests(unittest.TestCase):
    """The rewrite table turns workbook shapes into concrete instructions."""

    def _lines(self, mnemonic: str, form: str) -> list[str]:
        row = PerfRow(
            mnemonic=mnemonic,
            core="ZEN4",
            source=SOURCE_AMD_SWOPT,
            source_kind="modeled",
            source_version="test",
            architecture="x86",
            form=form,
        )
        return concrete_lines(row)

    def test_implicit_register_writes_as_its_register(self):
        self.assertEqual(self._lines("BLENDVPD", "xmm1,xmm2,<xmm0>"), ["BLENDVPD xmm1, xmm2, xmm0"])

    def test_shape_expands_to_one_instruction_per_width(self):
        self.assertEqual(self._lines("vaddps", "xy"), ["vaddps xmm1", "vaddps ymm1"])

    def test_slash_alternatives_name_each_form(self):
        self.assertEqual(
            self._lines("addpd", "x,x/m")[:2],
            ["addpd xmm1, xmm2", "addpd xmm1, byte ptr [rax]"],
        )

    def test_width_alternatives_take_the_family_of_the_first_token(self):
        self.assertEqual(
            self._lines("fld", "mem32/64"),
            ["fld dword ptr [rax]", "fld qword ptr [rax]"],
        )

    def test_bare_mnemonic_row_writes_the_mnemonic(self):
        self.assertEqual(self._lines("aaa", ""), ["aaa"])

    def test_x87_memory_form_drops_the_implicit_st0(self):
        lines = self._lines("fadd", "s/m,s_")
        self.assertIn("fadd st(1), st(0)", lines)
        self.assertIn("fadd dword ptr [rax]", lines)

    def test_unknown_shape_writes_no_instruction(self):
        self.assertEqual(self._lines("vaddps", "quux,quux"), [])


class OperandDecoderTests(unittest.TestCase):
    """Pin the structural operand decoder to the workbook's token grammar."""

    def test_width_alternatives_widen_with_the_first_token_family(self):
        self.assertEqual(_operand_alts("reg8/16/32"), ["reg8", "reg16", "reg32"])
        self.assertEqual(_operand_alts("[8/16"), ["[8", "16"])
        self.assertEqual(_operand_alts("r32/r64"), ["r32", "r64"])

    def test_operand_columns_are_op_plus_one_digit(self):
        self.assertTrue(_op_column("op1"))
        self.assertFalse(_op_column("op12"))
        self.assertFalse(_op_column("operand1"))

    def test_alt_classes_name_the_class_with_its_width(self):
        self.assertEqual(_vendor_alt_classes("mem32/64"), frozenset({"mem32", "mem64"}))
        self.assertEqual(_vendor_alt_classes("memem32bcst"), frozenset({"mem32"}))
        self.assertEqual(_vendor_alt_classes("[r32]"), frozenset({"gpr32"}))
        self.assertEqual(_vendor_alt_classes("[r]"), frozenset({"other"}))
        self.assertEqual(_vendor_alt_classes("imm8/16"), frozenset({"imm0", "imm8", "imm16"}))
        self.assertEqual(_vendor_alt_classes("k2{k1}"), frozenset({"k"}))
        self.assertEqual(_vendor_alt_classes("s"), frozenset({"xmm"}))

    def test_alt_width_and_lines_decode_register_and_memory_forms(self):
        self.assertEqual(_alt_width("memem32bcst"), 32)
        self.assertEqual(_alt_width("XMM2"), 128)
        self.assertEqual(_alt_lines("reg32", 0, 512), ["eax"])
        self.assertEqual(_alt_lines("mmx2", 1, 512), ["mm2"])
        self.assertEqual(_alt_lines("mem32bcst", 0, 512), ["dword ptr [rax]{1to16}"])
        self.assertEqual(_alt_lines("imm8", 0, 512), ["2"])
        self.assertEqual(_alt_lines("imm32", 0, 512), ["0x12345678"])
        self.assertEqual(_alt_lines("K", 0, 512), ["k1"])


class LoadVendoredTests(unittest.TestCase):
    def test_load_vendored_returns_perf_rows(self):
        rows = load_vendored()
        self.assertGreater(len(rows), 0)
        for row in rows:
            with self.subTest(mnemonic=row.mnemonic, core=row.core):
                self.assertIsInstance(row, PerfRow)
                self.assertEqual(row.source, SOURCE_AMD_SWOPT)
                self.assertIsNotNone(canonical_core_id(row.core))

    def test_load_vendored_missing_dir_is_empty(self):
        rows = load_vendored(FIXTURES / "no-such-dir")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
