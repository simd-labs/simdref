"""Tests for AMD workbook rows matched to catalog forms by encoded bytes.

Lane L2: the bytes decide which catalog forms a guide row names. The
operand-class matcher stays only as the fallback when the tool is missing.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from simdref.asm_identity import _tool_path
from simdref.models import InstructionRecord
from simdref.perf_sources.amd_swopt import PerfRow, form_classes
from simdref.perf_sources.merge import merge_perf_rows

FIXTURES = Path(__file__).parent / "fixtures" / "amd_match"
HAVE_TOOLS = bool(_tool_path("SIMDREF_LLVM_MC", "llvm-mc")) and bool(
    _tool_path("SIMDREF_XED", "xed")
)


def _records() -> list[InstructionRecord]:
    raw = json.loads((FIXTURES / "records.json").read_text(encoding="utf-8"))
    return [
        InstructionRecord(
            mnemonic=item["mnemonic"],
            form=item["form"],
            summary=item["summary"],
            architecture="x86",
            isa=["SSE4.1"],
            metadata={"iform": item["iform"]},
        )
        for item in raw
    ]


def _amd_row(form: str, *, mnemonic: str = "BLENDVPD") -> PerfRow:
    return PerfRow(
        mnemonic=mnemonic,
        core="ZEN4",
        source="amd-swopt",
        source_kind="modeled",
        source_version="test",
        architecture="x86",
        form=form,
        latency="1",
        form_classes=form_classes(form),
    )


class ByteMatchTests(unittest.TestCase):
    """The encoded bytes pick the catalog rows at ingest."""

    @unittest.skipUnless(
        HAVE_TOOLS, "llvm-mc and the XED decoder not found (simdref_llvm_mc wheel missing)"
    )
    def test_register_row_lands_only_on_the_register_form(self):
        # The reviewer case: the workbook row names registers only, so the
        # M128 form must not carry it.
        reg, mem = _records()
        merge_perf_rows([reg, mem], [_amd_row("xmm1,xmm2,<xmm0>")])
        self.assertIn("ZEN4", reg.arch_details)
        self.assertNotIn("ZEN4", mem.arch_details)

    @unittest.skipUnless(
        HAVE_TOOLS, "llvm-mc and the XED decoder not found (simdref_llvm_mc wheel missing)"
    )
    def test_register_or_memory_row_lands_on_both_named_forms(self):
        reg, mem = _records()
        merge_perf_rows([reg, mem], [_amd_row("x,x/m,x")])
        self.assertIn("ZEN4", reg.arch_details)
        self.assertIn("ZEN4", mem.arch_details)

    @unittest.skipUnless(
        HAVE_TOOLS, "llvm-mc and the XED decoder not found (simdref_llvm_mc wheel missing)"
    )
    def test_row_that_does_not_assemble_is_dropped_not_spread(self):
        reg, mem = _records()
        written = merge_perf_rows([reg, mem], [_amd_row("xmm1,xmm2,imm8,imm16")])
        self.assertEqual(written, 0)
        self.assertNotIn("ZEN4", reg.arch_details)
        self.assertNotIn("ZEN4", mem.arch_details)


class FallbackTests(unittest.TestCase):
    """Without the tool the operand-class matcher keeps today's behavior."""

    def test_tool_missing_keeps_the_class_matcher(self):
        with mock.patch("simdref.asm_identity._tool_path", return_value=None):
            reg, mem = _records()
            merge_perf_rows([reg, mem], [_amd_row("xmm1,xmm2,<xmm0>")])
        # Three row operands match no two-operand form, so the old
        # fallback attaches the row to every variant.
        self.assertIn("ZEN4", reg.arch_details)
        self.assertIn("ZEN4", mem.arch_details)


if __name__ == "__main__":
    unittest.main()
