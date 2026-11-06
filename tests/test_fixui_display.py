"""FIXUI: display-layer template guard and per-row perf source labels.

Defect 4: upstream AARCHMRS rows keep raw <RULE> tokens in their record
keys. Keys stay as they are; display strings must show only the mnemonic
when the operand text holds an unexpandable token.

Defect 2: every perf row carries a compact source label; the group
footer citations stay.
"""

from __future__ import annotations

import unittest

from simdref.display import (
    console,
    display_instruction_form,
    instruction_markdown,
    measurement_rows,
    print_mapping,
    render_instruction,
)
from simdref.export import _detail_chunks, _search_instructions, _web_measurements
from simdref.models import Catalog, InstructionRecord, IntrinsicRecord

TOKEN_FORM = "FMUL (V0.<T_FMUL_HSD>, V1.<T_FMUL_HSD>, V2.<T_FMUL_HSD>)"


def _token_record() -> InstructionRecord:
    return InstructionRecord(
        mnemonic="FMUL",
        form=TOKEN_FORM,
        summary="Floating-point multiply.",
        architecture="arm",
        isa=["SVE"],
    )


def _catalog(*instructions: InstructionRecord) -> Catalog:
    return Catalog(
        intrinsics=[],
        instructions=list(instructions),
        sources=[],
        generated_at="2026-01-01T00:00:00Z",
    )


class TemplateTokenGuardTests(unittest.TestCase):
    def test_display_form_shows_only_the_mnemonic(self):
        self.assertEqual(display_instruction_form(TOKEN_FORM), "FMUL")
        self.assertEqual(
            display_instruction_form("FADD (V0.4S, V1.4S, V2.4S)"),
            "FADD (V0.4S, V1.4S, V2.4S)",
        )

    def test_markdown_hides_the_token(self):
        self.assertNotIn("<T_FMUL_HSD>", instruction_markdown(_token_record()))

    def test_cli_render_hides_the_token(self):
        with console.capture() as capture:
            render_instruction(_catalog(_token_record()), _token_record(), short=True)
        self.assertNotIn("<T_FMUL_HSD>", capture.get())

    def test_cli_mapping_hides_the_token(self):
        record = _token_record()
        record.linked_intrinsics.append("vmulq_f32")
        intrinsic = IntrinsicRecord(
            name="vmulq_f32",
            signature="float32x4_t vmulq_f32(float32x4_t a, float32x4_t b)",
            description="Floating-point multiply.",
            header="arm_neon.h",
            architecture="arm",
            isa=["NEON"],
        )
        with console.capture() as capture:
            print_mapping("instruction", _catalog(record), intrinsic)
        output = capture.get()
        self.assertNotIn("<T_FMUL_HSD>", output)
        self.assertIn("FMUL", output)

    def test_export_display_strings_hide_the_token(self):
        catalog = _catalog(_token_record())
        detail = _detail_chunks(catalog)["FMU"][_token_record().db_key]
        self.assertNotIn("<", detail["display_form"])
        search = _search_instructions(
            catalog, {item.db_key: ("-", "-") for item in catalog.instructions}
        )
        self.assertNotIn("<", next(iter(search))["display_key"])


class PerfSourceLabelTests(unittest.TestCase):
    def _record(self) -> InstructionRecord:
        return InstructionRecord(
            mnemonic="FADDP",
            form="FADDP (XMM1, XMM2, XMM3)",
            summary="Packed add.",
            architecture="x86",
            isa=["AVX"],
            arch_details={
                "SKX": {
                    "source": "uops.info",
                    "source_kind": "measured",
                    "source_version": "2025-10-30",
                    "citation_url": "https://uops.info/table.html",
                    "measurement": {"TP_loop": "0.55", "uops": "2"},
                    "latencies": [{"cycles": "4"}],
                },
                "neoverse-v1": {
                    "source": "llvm-mca",
                    "source_kind": "modeled",
                    "source_version": "24.0.0",
                    "citation_url": "https://llvm.org/docs/CommandGuide/llvm-mca.html",
                    "measurement": {"TP_loop": "0.5", "uops": "1"},
                    "latencies": [{"cycles": "2"}],
                },
            },
        )

    def test_cli_perf_table_labels_each_row_and_keeps_the_footer(self):
        record = self._record()
        with console.capture() as capture:
            render_instruction(_catalog(record), record, short=True)
        output = capture.get()
        self.assertIn("uops.info", output)
        self.assertIn("llvm-mca", output)
        self.assertIn("src", output)
        # The group footer citation lines stay.
        self.assertIn("https://llvm.org/docs/CommandGuide/llvm-mca.html", output)

    def test_web_measurement_rows_label_each_row(self):
        rows = _web_measurements(self._record())
        by_uarch = {row["uarch"]: row for row in rows}
        self.assertEqual(by_uarch["SKX"]["source"], "uops.info")
        self.assertEqual(by_uarch["neoverse-v1"]["source"], "llvm-mca")

    def test_uops_rows_default_to_the_uops_label(self):
        record = InstructionRecord(
            mnemonic="VADDPS",
            form="VADDPS (YMM0, YMM1, YMM2)",
            summary="Packed add.",
            architecture="x86",
            isa=["AVX"],
            arch_details={
                "SKX": {
                    "source_kind": "measured",
                    "measurement": {"TP_loop": "0.55", "uops": "2"},
                    "latencies": [{"cycles": "4"}],
                },
            },
        )
        rows = measurement_rows(record)
        self.assertEqual(rows[0]["source_tool"], "uops.info")
        web = _web_measurements(record)
        self.assertEqual(web[0]["source"], "uops.info")


if __name__ == "__main__":
    unittest.main()
