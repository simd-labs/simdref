"""Tests for perf-source priority in merge_perf_rows and the new core ids.

Covers node C0: one instruction form on one core keeps exactly one perf
value, chosen by the fixed source priority:
uops.info > instlatx64 > intel-guide > intel-optref/amd-swopt/arm-swopt
> LLVM modeled (llvm-mca / llvm_scheduling).
"""

import unittest

from simdref.models import InstructionRecord
from simdref.perf_sources.amd_swopt import form_classes
from simdref.perf_sources.cores import canonical_core_id, core_architecture
from simdref.perf_sources.merge import PerfRow, merge_perf_rows


def _record(arch_details=None):
    return InstructionRecord(
        mnemonic="VADDPD",
        form="VADDPD zmm1, zmm2, zmm3",
        summary="Add packed double-precision floating-point values.",
        architecture="x86",
        arch_details=dict(arch_details or {}),
    )


def _row(source, latency, *, kind="measured"):
    return PerfRow(
        mnemonic="VADDPD",
        core="SKX",
        source=source,
        source_kind=kind,
        source_version="test",
        architecture="x86",
        latency=latency,
    )


class TestSourcePriority(unittest.TestCase):
    def test_uops_info_wins_over_three_sources(self):
        record = _record()
        rows = [
            _row("intel-guide", "6"),
            _row("instlatx64", "5"),
            _row("uops.info", "4"),
        ]
        merge_perf_rows([record], rows)
        entry = record.arch_details["SKX"]
        self.assertEqual(entry["source"], "uops.info")
        self.assertEqual(entry["latencies"], [{"cycles": "4"}])

    def test_instlatx64_beats_vendor_guide(self):
        record = _record()
        rows = [
            _row("intel-optref", "7"),
            _row("instlatx64", "5"),
        ]
        merge_perf_rows([record], rows)
        entry = record.arch_details["SKX"]
        self.assertEqual(entry["source"], "instlatx64")
        self.assertEqual(entry["latencies"], [{"cycles": "5"}])

    def test_intel_guide_beats_vendor_guides(self):
        for vendor in ("intel-optref", "amd-swopt", "arm-swopt"):
            record = _record()
            merge_perf_rows([record], [_row(vendor, "7"), _row("intel-guide", "6")])
            self.assertEqual(record.arch_details["SKX"]["source"], "intel-guide")

    def test_modeled_only_row_is_kept(self):
        record = _record()
        merge_perf_rows([record], [_row("llvm-mca", "3", kind="modeled")])
        entry = record.arch_details["SKX"]
        self.assertEqual(entry["source"], "llvm-mca")
        self.assertEqual(entry["source_kind"], "modeled")
        self.assertEqual(entry["latencies"], [{"cycles": "3"}])

    def test_measured_sources_beat_modeled(self):
        for measured in ("uops.info", "instlatx64", "intel-guide", "amd-swopt"):
            record = _record()
            rows = [_row(measured, "5"), _row("llvm-mca", "3", kind="modeled")]
            merge_perf_rows([record], rows)
            self.assertEqual(record.arch_details["SKX"]["source"], measured)

    def test_legacy_uops_row_without_label_wins(self):
        # uops.info rows enter arch_details via parse_uops_xml, not PerfRow,
        # and carry no ``source`` key. The priority must hold for them too.
        legacy = {"measurement": {"TP_loop": "1.0"}, "latencies": [{"cycles": "4"}]}
        record = _record({"SKX": dict(legacy)})
        merge_perf_rows([record], [_row("instlatx64", "5")])
        self.assertEqual(record.arch_details["SKX"], legacy)


class TestNewCoreIds(unittest.TestCase):
    def test_new_x86_ids_resolve(self):
        expected = {
            "ZEN1": "ZEN1",
            "ZEN+": "ZEN+",
            "ZEN2": "ZEN2",
            "ZEN3": "ZEN3",
            "ZEN4": "ZEN4",
            "ZEN5": "ZEN5",
            "ADL-P": "ADL-P",
            "ADL-E": "ADL-E",
            "RPL": "RPL",
            "MTL-P": "MTL-P",
            "MTL-E": "MTL-E",
            "LNL-P": "LNL-P",
            "LNL-E": "LNL-E",
            "ARL-P": "ARL-P",
            "ARL-E": "ARL-E",
            "SPR": "EMR",
            "EMR": "EMR",
            "GNR": "GNR",
            "SRF": "SRF",
        }
        for name, canonical in expected.items():
            with self.subTest(name=name):
                self.assertEqual(canonical_core_id(name), canonical)
                self.assertEqual(core_architecture(canonical), "x86")

    def test_new_aarch64_ids_resolve(self):
        for name in ("N1", "N2", "N3", "V1", "V2", "V3"):
            with self.subTest(name=name):
                canonical = canonical_core_id(name)
                self.assertEqual(canonical, f"neoverse-{name.lower()}")
                self.assertEqual(core_architecture(canonical), "aarch64")


class TestAmdFormClassMatch(unittest.TestCase):
    """FIXUI defect 5: AMD workbook rows name operand classes (xmm1,xmm2;
    reg,reg/mem), not catalog forms. A class row lands on the catalog
    forms whose operands match the classes, not on every variant."""

    def _records(self):
        reg = InstructionRecord(
            mnemonic="ADDPD",
            form="ADDPD (XMM, XMM)",
            summary="Add packed double-precision floating-point values.",
            architecture="x86",
            isa=["SSE2"],
        )
        mem = InstructionRecord(
            mnemonic="ADDPD",
            form="ADDPD (XMM, M128)",
            summary="Add packed double-precision floating-point values.",
            architecture="x86",
            isa=["SSE2"],
        )
        return reg, mem

    def _amd_row(self, form: str, mnemonic: str = "ADDPD") -> PerfRow:
        return PerfRow(
            mnemonic=mnemonic,
            core="ZEN4",
            source="amd-swopt",
            source_kind="modeled",
            source_version="test",
            architecture="x86",
            form=form,
            latency="3",
            form_classes=form_classes(form),
        )

    def test_register_row_hits_only_the_register_form(self):
        reg, mem = self._records()
        rows = [self._amd_row("xmm1,xmm2")]
        merge_perf_rows([reg, mem], rows)
        self.assertIn("ZEN4", reg.arch_details)
        self.assertNotIn("ZEN4", mem.arch_details)

    def test_reg_or_mem_row_hits_both_forms(self):
        reg, mem = self._records()
        rows = [self._amd_row("xmm1,xmm2/mem128")]
        merge_perf_rows([reg, mem], rows)
        self.assertIn("ZEN4", reg.arch_details)
        self.assertIn("ZEN4", mem.arch_details)

    def test_width_specific_row_skips_other_widths(self):
        r32 = InstructionRecord(
            mnemonic="ADCX",
            form="ADCX (R32, R32)",
            summary="Unsigned integer addition with carry.",
            architecture="x86",
            isa=["ADX"],
        )
        r64 = InstructionRecord(
            mnemonic="ADCX",
            form="ADCX (R64, R64)",
            summary="Unsigned integer addition with carry.",
            architecture="x86",
            isa=["ADX"],
        )
        rows = [self._amd_row("reg32,reg32", mnemonic="ADCX")]
        merge_perf_rows([r32, r64], rows)
        self.assertIn("ZEN4", r32.arch_details)
        self.assertNotIn("ZEN4", r64.arch_details)

    def test_unmatched_form_attaches_nowhere(self):
        reg, mem = self._records()
        rows = [self._amd_row("xmm1,xmm2,imm8,imm16")]
        merge_perf_rows([reg, mem], rows)
        self.assertNotIn("ZEN4", reg.arch_details)
        self.assertNotIn("ZEN4", mem.arch_details)


if __name__ == "__main__":
    unittest.main()
