"""Tests for simdref.perf helpers."""

import unittest

from simdref.perf import (
    MISSING_PERF,
    PerfValue,
    _is_numeric,
    best_cpi,
    best_cpi_labeled,
    best_cpi_measured,
    best_cpi_modeled,
    best_latency,
    best_latency_labeled,
    best_latency_measured,
    best_latency_modeled,
    best_numeric,
    latency_cycle_values,
    variant_perf_summary,
    variant_perf_summary_labeled,
)


class PerfTests(unittest.TestCase):
    def test_is_numeric_integers(self):
        self.assertTrue(_is_numeric("5"))
        self.assertTrue(_is_numeric("123"))

    def test_is_numeric_decimals(self):
        self.assertTrue(_is_numeric("1.5"))
        self.assertTrue(_is_numeric("0.25"))

    def test_is_numeric_rejects_non_numeric(self):
        self.assertFalse(_is_numeric(""))
        self.assertFalse(_is_numeric("abc"))
        self.assertFalse(_is_numeric("1.2.3"))
        self.assertFalse(_is_numeric("variable"))

    def test_latency_cycle_values_extracts_unique(self):
        latencies = [
            {"cycles": "3", "cycles_mem": "7"},
            {"cycles": "3", "cycles_addr": "5"},
        ]
        values = latency_cycle_values(latencies)
        self.assertEqual(values, ["3", "7", "5"])

    def test_latency_cycle_values_empty(self):
        self.assertEqual(latency_cycle_values([]), [])

    def test_latency_cycle_values_drops_upper_bound_zero_mem(self):
        # uops.info convention: cycles_mem="0" with the upper-bound flag
        # is a not-measured marker for memory operands like MOV r32,m32.
        latencies = [
            {
                "start_op": "2",
                "target_op": "1",
                "cycles_addr": "5",
                "cycles_addr_index": "5",
                "cycles_mem": "0",
                "cycles_mem_is_upper_bound": "1",
            }
        ]
        values = latency_cycle_values(latencies)
        self.assertEqual(values, ["5"])
        self.assertEqual(best_numeric(values), "5")

    def test_latency_cycle_values_keeps_real_zero_when_not_upper_bound(self):
        latencies = [{"cycles_mem": "0"}]
        self.assertEqual(latency_cycle_values(latencies), ["0"])

    def test_latency_cycle_values_reads_min_max_pair(self):
        # simdref#50: uops.info divide/sqrt rows carry only a min/max range.
        latencies = [
            {"min_cycles": "13", "max_cycles": "15", "min_cycles_is_upper_bound": "1"},
            {"min_cycles": "13", "max_cycles": "15", "min_cycles_is_upper_bound": "1"},
        ]
        self.assertEqual(latency_cycle_values(latencies), ["13-15"])

    def test_latency_cycle_values_min_max_equal(self):
        self.assertEqual(latency_cycle_values([{"min_cycles": "7", "max_cycles": "7"}]), ["7"])

    def test_latency_cycle_values_min_only(self):
        self.assertEqual(latency_cycle_values([{"min_cycles": "7"}]), ["7"])

    def test_latency_cycle_values_drops_upper_bound_zero_mem_range(self):
        latencies = [{"min_cycles_mem": "0", "min_cycles_mem_is_upper_bound": "1"}]
        self.assertEqual(latency_cycle_values(latencies), [])

    def test_min_max_pair_flows_to_every_output(self):
        # simdref#50: the perf-root fix must reach every caller output: the
        # string picks, the variant summary and the float annotate view.
        from simdref.annotate import _latency_for, _per_arch_value

        details = {"latencies": [{"min_cycles": "13", "max_cycles": "15"}]}
        arch_details = {"MTL-P": details}
        self.assertEqual(best_latency(arch_details), "13-15")
        self.assertEqual(best_latency_labeled(arch_details).value, "13-15")
        lat, _ = variant_perf_summary(arch_details)
        self.assertEqual(lat, "13-15")
        self.assertEqual(_per_arch_value(details, _latency_for), 13.0)

    def test_best_numeric_picks_smallest(self):
        self.assertEqual(best_numeric(["5", "3", "7"]), "3")
        self.assertEqual(best_numeric(["1.5", "2.0"]), "1.5")

    def test_best_numeric_falls_back_to_first(self):
        self.assertEqual(best_numeric(["variable"]), "variable")

    def test_best_numeric_empty(self):
        self.assertEqual(best_numeric([]), "-")

    def test_best_latency(self):
        arch_details = {
            "SKL": {"latencies": [{"cycles": "4"}]},
            "HSW": {"latencies": [{"cycles": "5"}]},
        }
        self.assertEqual(best_latency(arch_details), "4")

    def test_best_latency_empty(self):
        self.assertEqual(best_latency({}), "-")

    def test_best_cpi(self):
        arch_details = {
            "SKL": {"measurement": {"TP_loop": "1.0", "TP_unrolled": "0.5"}},
        }
        self.assertEqual(best_cpi(arch_details), "0.5")

    def test_best_cpi_drops_harness_contaminated_measurement(self):
        # Upstream uops.info records RET inside a call/ret loop harness;
        # the raw measurement block reports harness-wide ``uops=20`` with
        # ``TP_unrolled=33.35``. That figure is meaningless for the lone
        # ret, so ``_cpi_values`` must drop the measurement and let the
        # caller surface "-" rather than a nonsense CPI.
        arch_details = {
            "HSW": {
                "measurement": {
                    "TP_loop": "28.91",
                    "TP_ports": "3.00",
                    "TP_unrolled": "33.35",
                    "uops": "20",
                },
            },
        }
        self.assertEqual(best_cpi(arch_details), "-")

    def test_variant_perf_summary(self):
        arch_details = {
            "SKL": {
                "latencies": [{"cycles": "4"}],
                "measurement": {"TP_loop": "1.0"},
            },
        }
        lat, cpi = variant_perf_summary(arch_details)
        self.assertEqual(lat, "4")
        self.assertEqual(cpi, "1.0")


class LabeledPerfTests(unittest.TestCase):
    def test_missing_perf_sentinel(self):
        self.assertEqual(MISSING_PERF, PerfValue("-", "", ""))
        self.assertEqual(str(MISSING_PERF), "-")

    def test_measured_preferred_over_modeled(self):
        arch_details = {
            "neoverse-n1": {
                "latencies": [{"cycles": "3"}],
                "source_kind": "modeled",
            },
            "A64FX": {
                "latencies": [{"cycles": "5"}],
                "source_kind": "measured",
            },
        }
        # Prefer measured even though its numeric value is larger.
        result = best_latency_labeled(arch_details)
        self.assertEqual(result, PerfValue("5", "measured", "A64FX"))

    def test_legacy_entries_default_to_measured(self):
        arch_details = {"SKL": {"latencies": [{"cycles": "4"}]}}
        result = best_latency_labeled(arch_details)
        self.assertEqual(result.source_kind, "measured")
        self.assertEqual(result.core, "SKL")
        self.assertEqual(result.value, "4")

    def test_modeled_only_falls_back_to_modeled(self):
        arch_details = {
            "neoverse-n1": {
                "latencies": [{"cycles": "3"}],
                "source_kind": "modeled",
            },
        }
        result = best_latency_labeled(arch_details)
        self.assertEqual(result, PerfValue("3", "modeled", "neoverse-n1"))

    def test_empty_returns_sentinel(self):
        self.assertEqual(best_latency_labeled({}), MISSING_PERF)
        self.assertEqual(best_cpi_labeled({}), MISSING_PERF)

    def test_measured_modeled_filters(self):
        arch_details = {
            "SKL": {"latencies": [{"cycles": "4"}], "source_kind": "measured"},
            "N1": {"latencies": [{"cycles": "2"}], "source_kind": "modeled"},
        }
        self.assertEqual(best_latency_measured(arch_details).value, "4")
        self.assertEqual(best_latency_modeled(arch_details).value, "2")

    def test_cpi_labeled_prefers_measured(self):
        arch_details = {
            "N1": {"measurement": {"TP": "0.25"}, "source_kind": "modeled"},
            "SKL": {"measurement": {"TP_unrolled": "0.5"}, "source_kind": "measured"},
        }
        result = best_cpi_labeled(arch_details)
        self.assertEqual(result, PerfValue("0.5", "measured", "SKL"))

    def test_cpi_measured_modeled_split(self):
        arch_details = {
            "SKL": {"measurement": {"TP_unrolled": "0.5"}, "source_kind": "measured"},
            "N1": {"measurement": {"TP": "0.25"}, "source_kind": "modeled"},
        }
        self.assertEqual(best_cpi_measured(arch_details).value, "0.5")
        self.assertEqual(best_cpi_modeled(arch_details).value, "0.25")

    def test_best_latency_prefers_measured_string_api(self):
        arch_details = {
            "SKL": {"latencies": [{"cycles": "4"}], "source_kind": "measured"},
            "N1": {"latencies": [{"cycles": "2"}], "source_kind": "modeled"},
        }
        self.assertEqual(best_latency(arch_details), "4")
        self.assertEqual(best_cpi(arch_details), "-")

    def test_variant_perf_summary_labeled(self):
        arch_details = {
            "SKL": {
                "latencies": [{"cycles": "4"}],
                "measurement": {"TP_loop": "1.0"},
                "source_kind": "measured",
            },
        }
        lat, cpi = variant_perf_summary_labeled(arch_details)
        self.assertEqual(lat, PerfValue("4", "measured", "SKL"))
        self.assertEqual(cpi, PerfValue("1.0", "measured", "SKL"))

    def test_non_numeric_fallback_labeled(self):
        arch_details = {
            "SKL": {
                "latencies": [{"cycles": "variable"}],
                "source_kind": "measured",
            },
        }
        result = best_latency_labeled(arch_details)
        self.assertEqual(result.value, "variable")
        self.assertEqual(result.source_kind, "measured")


if __name__ == "__main__":
    unittest.main()
