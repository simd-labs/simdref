"""Core ids and source mappings added at integration.

Silvermont (SLM), Ice Lake server (ICX), Neoverse E1 and Neoverse V3AE.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from simdref.perf_sources import arm_swopt, intel_optref
from simdref.perf_sources.cores import canonical_core_id
from simdref.perf_sources.intel_guide import parse_perf_js

FIXTURES = Path(__file__).parent / "fixtures"


class CoreIdTests(unittest.TestCase):
    def test_new_x86_ids_resolve(self):
        self.assertEqual(canonical_core_id("SLM"), "SLM")
        self.assertEqual(canonical_core_id("Silvermont"), "SLM")
        self.assertEqual(canonical_core_id("ICX"), "ICX")
        self.assertEqual(canonical_core_id("icelake-server"), "ICX")
        self.assertEqual(canonical_core_id("icelake-client"), "ICL")

    def test_new_arm_ids_resolve(self):
        self.assertEqual(canonical_core_id("neoverse-e1"), "neoverse-e1")
        self.assertEqual(canonical_core_id("Neoverse-E1"), "neoverse-e1")
        self.assertEqual(canonical_core_id("neoverse-v3ae"), "neoverse-v3ae")
        self.assertEqual(canonical_core_id("V3AE"), "neoverse-v3ae")


class IntelGuideMappingTests(unittest.TestCase):
    def test_icelake_xeon_rows_map_to_icx(self):
        result = parse_perf_js((FIXTURES / "intel_guide" / "perf2.js").read_text())
        xeon = [r for r in result.rows if r.form == "ADDPD_XMMpd_XMMpd" and r.core == "ICX"]
        self.assertEqual(len(xeon), 1)
        self.assertEqual(xeon[0].latency, "4")
        client = [r for r in result.rows if r.form == "ADCX_GPR32d_GPR32d" and r.core == "ICL"]
        self.assertEqual(len(client), 1)
        self.assertEqual(client[0].cpi, "")


class IntelOptrefMappingTests(unittest.TestCase):
    def test_silvermont_column_recovers_slm_rows(self):
        rows, stats = intel_optref.parse_text(
            (FIXTURES / "intel_optref" / "optref_tables_sample.txt").read_text(encoding="utf-8")
        )
        self.assertEqual(stats.dropped_no_core, 0)
        slm = {(r.mnemonic, r.form): r for r in rows if r.core == "SLM"}
        self.assertIn(("BSWAP", "r32"), slm)
        self.assertEqual(slm[("BSWAP", "r32")].latency, "1")
        self.assertEqual(slm[("BSWAP", "r32")].cpi, "1")


class ArmSwoptMappingTests(unittest.TestCase):
    def test_new_cores_have_cpu_specs(self):
        cpus = {spec.cpu for spec in arm_swopt.CPUS}
        self.assertIn("neoverse-e1", cpus)
        self.assertIn("neoverse-v3ae", cpus)

    def test_e1_groups_no_longer_dropped(self):
        groups = [
            {
                "group": "g",
                "latency": "1",
                "throughput": "1",
                "pipelines": "I",
                "notes": "",
                "mnemonics": ["ADD"],
            }
        ]
        rows, dropped = arm_swopt.perf_rows(groups, "neoverse-e1")
        self.assertEqual(dropped, 0)
        self.assertEqual([row.core for row in rows], ["neoverse-e1"])


if __name__ == "__main__":
    unittest.main()
