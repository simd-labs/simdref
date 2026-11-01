"""B1: latency per operand pair, with min/max ranges, upper-bound marks,
and same-register latency, on every surface.

Data source: uops.info latency rows carry start_op/target_op plus
min_cycles/max_cycles, *_is_upper_bound flags, and cycles_same_reg. The
LAT cell keeps the A2 rule (perf_row_values); the pair detail is shown
under the perf table on each surface.

Surfaces under test, each with the fixture record from
tests/fixtures/b1/latency_pairs.json:

- helper formatting (simdref.perf.latency_pairs / latency_pair_text)
- CLI show detail view (simdref.display.print_latency_pair_tables)
- TUI detail page (SimdrefApp._build_instruction_payload)
- LSP/instruction markdown hover line
- web export detail chunk (_web_measurements latencyPairs)
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from simdref.export import _web_measurements
from simdref.models import InstructionRecord
from simdref.perf import latency_pair_text, latency_pairs

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover
    HAS_TEXTUAL = False

FIXTURE = Path(__file__).parent / "fixtures" / "b1" / "latency_pairs.json"
EXPECTED_TEXT = "10-<=22 (xmm1->xmm0), <=8 (mem->xmm0), 13 (xmm0->xmm0, same reg 10)"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _instruction() -> InstructionRecord:
    data = _fixture()
    return InstructionRecord(
        mnemonic=data["mnemonic"],
        form=data["form"],
        summary=data["summary"],
        architecture=data["architecture"],
        isa=data["isa"],
        operand_details=data["operand_details"],
        arch_details=data["arch_details"],
    )


class LatencyPairHelperTests(unittest.TestCase):
    def test_pairs_carry_range_upper_bound_and_same_reg(self):
        pairs = latency_pairs(_fixture()["arch_details"]["SNB"], _fixture()["operand_details"])
        self.assertEqual(
            pairs,
            [
                {"start": "xmm1", "target": "xmm0", "value": "10-<=22"},
                {"start": "mem", "target": "xmm0", "value": "<=8"},
                {"start": "xmm0", "target": "xmm0", "value": "13", "same_reg": "10"},
            ],
        )

    def test_pair_text_is_one_compact_line(self):
        text = latency_pair_text(_fixture()["arch_details"]["SNB"], _fixture()["operand_details"])
        self.assertEqual(text, EXPECTED_TEXT)

    def test_missing_pairs_return_empty(self):
        self.assertEqual(latency_pairs({}), [])
        self.assertEqual(latency_pair_text({}, []), "")


class CliSurfaceTests(unittest.TestCase):
    def test_cli_detail_view_shows_pair_line(self):
        from simdref import display

        with display.console.capture() as capture:
            display.print_latency_pair_tables(_instruction())
        # Rich folds long pair strings over table rules; unwrap for matching.
        out = " ".join(capture.get().replace("│", " ").split())
        self.assertIn("latency per operand pair", out)
        self.assertIn("10-<=22 (xmm1->xmm0)", out)
        self.assertIn("<=8 (mem->xmm0)", out)
        self.assertIn("same reg 10", out)

    def test_cli_detail_without_pairs_shows_nothing(self):
        from simdref import display

        record = _instruction()
        record.arch_details = {"SNB": {"measurement": {"TP_loop": "1.00"}}}
        with display.console.capture() as capture:
            display.print_latency_pair_tables(record)
        self.assertEqual(capture.get(), "")

    def test_cli_show_full_view_includes_pairs(self):
        from simdref import display

        with display.console.capture() as capture:
            display.render_instruction(catalog=None, item=_instruction())
        self.assertIn(
            "<=8 (mem->xmm0)",
            " ".join(capture.get().replace("│", " ").split()),
        )


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class TuiSurfaceTests(unittest.TestCase):
    def _pair_panel_text(self, item) -> str:
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        payload = app._build_instruction_payload(item)
        for entry in payload:
            if entry[0] == "collapsible" and "operand pair" in entry[2]:
                renderable = entry[1].renderable
                rows = []
                for col_idx in range(len(renderable.columns)):
                    rows.append(
                        " | ".join(str(cell) for cell in renderable.columns[col_idx]._cells)
                    )
                return entry[2] + "\n" + "\n".join(rows)
        return ""

    def test_tui_detail_page_shows_pair_panel(self):
        out = self._pair_panel_text(_instruction())
        self.assertIn("latency per operand pair", out)
        self.assertIn("10-<=22 (xmm1->xmm0)", out)
        self.assertIn("<=8 (mem->xmm0)", out)

    def test_tui_detail_without_pairs_has_no_pair_panel(self):
        record = _instruction()
        record.arch_details = {"SNB": {"measurement": {"TP_loop": "1.00"}}}
        self.assertEqual(self._pair_panel_text(record), "")


class LspSurfaceTests(unittest.TestCase):
    def test_hover_line_lists_pairs_with_core(self):
        from simdref.display import instruction_markdown

        markdown = instruction_markdown(_instruction())
        self.assertIn(f"(SNB).** {EXPECTED_TEXT}", markdown)

    def test_hover_without_pairs_has_no_pair_line(self):
        from simdref.display import instruction_markdown

        record = _instruction()
        record.arch_details = {"SNB": {"measurement": {"TP_loop": "1.00"}}}
        self.assertNotIn("(SNB).**", instruction_markdown(record))


class ExportSurfaceTests(unittest.TestCase):
    def test_detail_chunk_row_carries_pairs(self):
        rows = _web_measurements(_instruction())
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["latencyPairs"],
            [
                {"start": "xmm1", "target": "xmm0", "value": "10-<=22"},
                {"start": "mem", "target": "xmm0", "value": "<=8"},
                {"start": "xmm0", "target": "xmm0", "value": "13", "sameReg": "10"},
            ],
        )

    def test_detail_chunk_row_without_pairs_omits_key(self):
        record = _instruction()
        record.arch_details = {"SNB": {"measurement": {"TP_loop": "1.00"}}}
        rows = _web_measurements(record)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("latencyPairs", rows[0])


if __name__ == "__main__":
    unittest.main()
