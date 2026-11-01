"""Tests for the Arm Neoverse software optimization guide parser."""

from __future__ import annotations

import unittest
from pathlib import Path

from simdref.perf_sources.arm_swopt import (
    CPUS,
    load_vendored,
    parse_groups,
    parse_text,
)
from simdref.perf_sources.cores import canonical_core_id
from simdref.perf_sources.merge import SOURCE_ARM_SWOPT

_FIXTURES = Path(__file__).parent / "fixtures" / "arm_swopt"


def _parse_groups(name: str):
    return parse_groups((_FIXTURES / name).read_text())


class ParseGroupsTests(unittest.TestCase):
    def test_v2_alu_basic_row(self):
        rows = _parse_groups("v2-sample.txt")
        alu = next(r for r in rows if r["group"] == "ALU, basic")
        self.assertEqual(alu["latency"], "1")
        self.assertEqual(alu["throughput"], "6")
        self.assertEqual(alu["pipelines"], "I")
        self.assertEqual(
            alu["mnemonics"],
            ["ADD", "ADC", "AND", "BIC", "EON", "EOR", "ORN", "ORR", "SUB", "SBC"],
        )

    def test_v2_asimd_wrapped_row(self):
        rows = _parse_groups("v2-sample.txt")
        asimd = next(r for r in rows if r["group"] == "ASIMD arith, basic")
        self.assertEqual(asimd["latency"], "2")
        self.assertEqual(asimd["throughput"], "4")
        self.assertEqual(asimd["pipelines"], "V")
        self.assertEqual(
            asimd["mnemonics"],
            [
                "ABS",
                "ADD",
                "NEG",
                "SADDL(2)",
                "SADDW(2)",
                "SHADD",
                "SHSUB",
                "SSUBL(2)",
                "SSUBW(2)",
                "SUB",
                "UADDL(2)",
                "UADDW(2)",
                "UHADD",
                "UHSUB",
                "USUBL(2)",
                "USUBW(2)",
            ],
        )

    def test_n1_group_name_wraps_after_mnemonics(self):
        rows = _parse_groups("n1-sample.txt")
        groups = {r["group"]: r for r in rows}
        row = groups["ASIMD arith, reduce, 4H/4S"]
        self.assertEqual(row["latency"], "3")
        self.assertEqual(row["throughput"], "1")
        self.assertEqual(row["pipelines"], "V1")
        self.assertEqual(row["mnemonics"], ["ADDV", "SADDLV", "UADDLV"])
        # Fractional throughput: 1/2 instruction per cycle.
        self.assertEqual(groups["ASIMD arith, reduce, 16B"]["throughput"], "1/2")

    def test_perf_row_fields(self):
        rows = parse_text((_FIXTURES / "n1-sample.txt").read_text(), "neoverse-n1")
        addv = next(
            r
            for r in rows
            if r.extra_measurement["instruction_group"] == "ASIMD arith, reduce, 4H/4S"
            and r.mnemonic == "ADDV"
        )
        self.assertEqual(addv.core, "neoverse-n1")
        self.assertEqual(addv.source, SOURCE_ARM_SWOPT)
        self.assertEqual(addv.source_kind, "modeled")
        self.assertEqual(addv.source_version, "Issue 4.0")
        self.assertEqual(addv.architecture, "arm")
        self.assertEqual(addv.latency, "3")
        self.assertEqual(addv.cpi, "1")
        self.assertEqual(addv.applies_to, "mnemonic")
        self.assertEqual(addv.extra_measurement["pipelines"], "V1")
        frac = next(
            r
            for r in rows
            if r.extra_measurement["instruction_group"] == "ASIMD arith, reduce, 16B"
            and r.mnemonic == "ADDV"
        )
        self.assertEqual(frac.cpi, "2")


class VendoredTests(unittest.TestCase):
    def test_every_vendored_cpu_resolves_to_a_core(self):
        known = {c.cpu for c in CPUS if canonical_core_id(c.cpu) is not None}
        self.assertIn("neoverse-v2", known)
        # CPUs without a core id are dropped by load_vendored.
        self.assertEqual(
            [c.cpu for c in CPUS if canonical_core_id(c.cpu) is None],
            ["neoverse-e1"],
        )

    def test_load_vendored_returns_rows(self):
        rows = load_vendored()
        self.assertGreater(len(rows), 1000)
        for row in rows:
            self.assertEqual(row.source, SOURCE_ARM_SWOPT)
            self.assertEqual(row.architecture, "arm")
            self.assertIsNotNone(canonical_core_id(row.core))
            self.assertIn("pipelines", row.extra_measurement)
        cores = {row.core for row in rows}
        self.assertEqual(
            cores,
            {
                "neoverse-v1",
                "neoverse-v2",
                "neoverse-v3",
                "neoverse-n1",
                "neoverse-n2",
                "neoverse-n3",
            },
        )


if __name__ == "__main__":
    unittest.main()
