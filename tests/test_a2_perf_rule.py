"""A2: one rule picks the displayed LAT and CPI of a perf row.

The rule lives in :func:`simdref.perf.perf_row_values`:

- LAT is the largest latency over the read-operand pairs of the row
  (simdref#65), "MIN-MAX" when a pair stores a range.
- CPI is what the CLI shows today: TP_loop, falling back to TP, else "-".

Every Python surface (CLI table, TUI table, LSP aggregation, web export)
must agree on the picked values for the same fixture row.
"""

from __future__ import annotations

import unittest

from simdref.export import _web_measurements
from simdref.models import InstructionRecord
from simdref.perf import best_cpi, best_latency, perf_row_values

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover
    HAS_TEXTUAL = False


# One arch row with two operand pairs whose latencies differ; the first
# listed pair is not the maximum.
def _arch_details() -> dict:
    return {
        "SKL": {
            "source_kind": "measured",
            "measurement": {"TP_loop": "0.50", "uops": "2"},
            "latencies": [
                {"cycles": "3", "start_op": "2", "target_op": "1"},
                {"cycles": "7", "start_op": "1", "target_op": "1"},
            ],
        }
    }


def _instruction() -> InstructionRecord:
    return InstructionRecord(
        mnemonic="VADDPD",
        form="VADDPD (YMM, YMM, YMM)",
        summary="Add packed double.",
        architecture="x86",
        isa=["AVX"],
        arch_details=_arch_details(),
    )


class PerfRowRuleTests(unittest.TestCase):
    def test_lat_is_max_over_operand_pairs_not_first_listed(self):
        lat, cpi = perf_row_values(_arch_details()["SKL"])
        self.assertEqual(lat, "7")
        self.assertEqual(cpi, "0.50")

    def test_cpi_prefers_tp_loop_over_tp_unrolled(self):
        details = {
            "measurement": {"TP_unrolled": "0.25", "TP_loop": "0.50"},
            "latencies": [{"cycles": "3"}],
        }
        self.assertEqual(perf_row_values(details), ("3", "0.50"))

    def test_cpi_falls_back_to_tp_when_no_tp_loop(self):
        details = {"measurement": {"TP": "2.00"}, "latencies": []}
        self.assertEqual(perf_row_values(details), ("-", "2.00"))

    def test_missing_data_gives_dashes(self):
        self.assertEqual(perf_row_values({}), ("-", "-"))


class SurfaceParityTests(unittest.TestCase):
    def test_cli_rows_match_rule(self):
        from simdref.display import measurement_rows

        row = measurement_rows(_instruction())[0]
        self.assertEqual(row["latency"], "7")
        self.assertEqual(row["TP_loop"], "0.50")

    @unittest.skipUnless(HAS_TEXTUAL, "textual required")
    def test_tui_perf_table_matches_rule(self):
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        payload = app._build_instruction_payload(_instruction())
        panels = [entry[1].renderable for entry in payload if entry[0] == "collapsible"]
        self.assertTrue(panels, "no perf panel in TUI payload")
        table = panels[0]
        headers = [c.header for c in table.columns]
        row = {h: str(list(c._cells)[0]) for h, c in zip(headers, table.columns)}
        self.assertEqual(row["LAT"], "7")
        self.assertEqual(row["CPI"], "0.50")

    def test_lsp_aggregation_matches_rule(self):
        # lsp.py displays best_latency / best_cpi over linked variants: the
        # smallest value over every pair and core, not the per-row pick.
        self.assertEqual(best_latency(_arch_details()), "3")
        self.assertEqual(best_cpi(_arch_details()), "0.50")

    def test_web_export_rows_match_rule(self):
        row = _web_measurements(_instruction())[0]
        self.assertEqual(row["latency"], "7")
        self.assertEqual(row["tpLoop"], "0.50")


if __name__ == "__main__":
    unittest.main()
