"""Tests for the structural scans in perf_sources.merge and export."""

import unittest

from simdref.export import _intrinsic_chunk_prefix
from simdref.perf_sources.merge import (
    _form_class_match,
    _key_operand_class,
    _record_form_classes,
    borrowed_field_sources,
    donor_source_kinds,
)


class KeyOperandClassTests(unittest.TestCase):
    def test_mem_operand_digit_run(self):
        self.assertEqual(_key_operand_class("m128"), "mem128")
        self.assertEqual(_key_operand_class("M256"), "mem256")
        self.assertEqual(_key_operand_class("m8"), "mem8")

    def test_mem_operand_edges(self):
        # A bare "m" is mem0; leading zeros collapse through int(); the
        # digit run stops at the first non-digit and the tail is ignored.
        self.assertEqual(_key_operand_class("m"), "mem0")
        self.assertEqual(_key_operand_class("m08"), "mem8")
        self.assertEqual(_key_operand_class("m12x"), "mem12")
        self.assertEqual(_key_operand_class("mmx"), "other")


class RecordFormClassesTests(unittest.TestCase):
    def test_paren_form_splits_operands(self):
        self.assertEqual(_record_form_classes("ADD (R64, M64)"), ("gpr64", "mem64"))

    def test_paren_form_edges(self):
        self.assertEqual(_record_form_classes("ADD"), ())
        self.assertEqual(_record_form_classes("(R64)"), ())
        self.assertEqual(_record_form_classes("ADD (R64"), ())
        self.assertEqual(_record_form_classes("ADD ((R64))"), ("other",))
        self.assertEqual(_record_form_classes("ADD (R64, M64) extra"), ("gpr64", "mem64"))

    def test_class_match_prefix_star(self):
        self.assertTrue(_form_class_match((frozenset({"mem*"}),), ("mem128",)))
        self.assertTrue(_form_class_match((frozenset({"gpr*"}),), ("gpr64",)))
        self.assertTrue(_form_class_match((frozenset({"x*"}),), ("x87",)))
        self.assertTrue(_form_class_match((frozenset({"mem128"}),), ("mem128",)))
        self.assertFalse(_form_class_match((frozenset({"mem*"}),), ("gpr64",)))


class IntrinsicChunkPrefixTests(unittest.TestCase):
    def test_width_prefix_stripped(self):
        self.assertEqual(_intrinsic_chunk_prefix("_mm256_add_ps"), "add")
        self.assertEqual(_intrinsic_chunk_prefix("__riscv_v_add_vv"), "add")
        self.assertEqual(_intrinsic_chunk_prefix("svadd_f32_m"), "add")
        self.assertEqual(_intrinsic_chunk_prefix("vqadd_s32"), "add")

    def test_prefix_kept_when_it_consumes_the_name(self):
        self.assertEqual(_intrinsic_chunk_prefix("mm256_"), "mm2")
        self.assertEqual(_intrinsic_chunk_prefix("sv"), "sv")

    def test_short_names_fall_back_to_misc(self):
        self.assertEqual(_intrinsic_chunk_prefix("mm256x"), "misc")
        self.assertEqual(_intrinsic_chunk_prefix("__a"), "misc")


class DonorSourceTests(unittest.TestCase):
    def test_donor_source_kinds_counts_borrowed_fields(self):
        details = {
            "source_kind": "measured",
            "latency_source": "intel-guide",
            "measurement": {"TP_loop": 0.5},
        }
        self.assertEqual(donor_source_kinds(details), {"measured", "modeled"})

    def test_borrowed_field_sources_only_fills_present_fields(self):
        details = {
            "latency_source": "instlatx64",
            "latencies": [1, 2],
            "cpi_source": "intel-guide",
            "measurement": {"TP_loop": 0.5},
        }
        self.assertEqual(
            borrowed_field_sources(details),
            {"latency_source": "instlatx64", "cpi_source": "intel-guide"},
        )


if __name__ == "__main__":
    unittest.main()
