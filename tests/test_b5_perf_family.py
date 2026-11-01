"""B5/3c: perf group labels come from the core's real vendor and family.

The SHOT1 card labelled Arm and RISC-V perf rows "Intel Client". The
label must follow cores.py core_architecture: aarch64 cores group under
"Arm", riscv cores under "RISC-V"; x86 keeps its current vendor labels.
Rows exported for the web carry the label so the web UI needs no
guess.

Fixture record: an Arm instruction whose measurements sit on
neoverse-v1 (aarch64) and sifive-x280 (riscv).
"""

from __future__ import annotations

import unittest

from simdref.models import InstructionRecord


def _instruction() -> InstructionRecord:
    return InstructionRecord(
        mnemonic="FADD",
        form="FADD (V0.4S, V1.4S, V2.4S)",
        summary="Floating-point add.",
        architecture="arm",
        isa=["NEON"],
        arch_details={
            "neoverse-v1": {
                "source_kind": "modeled",
                "measurement": {"TP_loop": "1"},
                "latencies": [{"cycles": "2"}],
            },
            "sifive-x280": {
                "source_kind": "measured",
                "measurement": {"TP_loop": "4", "uops": "1"},
                "latencies": [{"cycles": "16"}],
            },
        },
    )


class PerfFamilyLabelTests(unittest.TestCase):
    def test_aarch64_cores_label_as_arm(self):
        from simdref.perf import perf_family_label

        self.assertEqual(perf_family_label("neoverse-v1"), "Arm")
        self.assertEqual(perf_family_label("cortex-a72"), "Arm")

    def test_riscv_cores_label_as_riscv(self):
        from simdref.perf import perf_family_label

        self.assertEqual(perf_family_label("sifive-x280"), "RISC-V")

    def test_x86_vendor_labels_are_unchanged(self):
        from simdref.perf import perf_family_label

        self.assertEqual(perf_family_label("SKL"), "Intel Client")
        self.assertEqual(perf_family_label("SKX"), "Intel Server")
        self.assertEqual(perf_family_label("ZEN3"), "AMD")

    def test_unknown_core_has_no_label(self):
        from simdref.perf import perf_family_label

        self.assertEqual(perf_family_label(""), "")
        self.assertEqual(perf_family_label("not-a-core"), "")


class WebExportTests(unittest.TestCase):
    def test_measurement_rows_carry_the_family_label(self):
        from simdref.export import _web_measurements

        rows = _web_measurements(_instruction())
        by_uarch = {row["uarch"]: row for row in rows}
        self.assertEqual(by_uarch["neoverse-v1"]["family"], "Arm")
        self.assertEqual(by_uarch["sifive-x280"]["family"], "RISC-V")


if __name__ == "__main__":
    unittest.main()
