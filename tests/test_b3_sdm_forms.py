"""B3: SDM opcode-table modes, CPUID feature flag column and the
Instruction Operand Encoding table, parsed and shown per surface.

Fixture texts come from the vendored Intel SDM pages for ADDPS (p723),
VPERMB and ADCX (p716), extracted with pdftotext -layout.

Surfaces under test, each with a fixture record that holds the fields:

- parser tables (simdref.pdfparse.intel.parse_opcode_table /
  parse_operand_encoding_table / form_table_sections)
- CLI show detail view (simdref.display.render_instruction)
- TUI detail page (SimdrefApp._build_instruction_payload)
- web app.js detail view (simdref-web tests/test_b3_sdm_fields_web.py)
"""

from __future__ import annotations

import unittest
from pathlib import Path

from simdref.models import InstructionRecord
from simdref.pdfparse import intel

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover
    HAS_TEXTUAL = False

FIXTURES = Path(__file__).parent / "fixtures" / "b3"

EXCEPTIONS_SECTION = {"Exceptions": "Exceptions Type 14; see Table 2-31."}


def _span(text, size, x0, top):
    return {"text": text, "size": size, "bbox": (x0, top, x0 + 100.0, top + 10.0)}


def _fixture_lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text().splitlines()


def _instruction(name: str = "addps_page.txt") -> InstructionRecord:
    description = intel.form_table_sections(_fixture_lines(name))
    description.update(EXCEPTIONS_SECTION)
    return InstructionRecord(
        mnemonic="ADDPS",
        form="ADDPS xmm1, xmm2/m128",
        summary="Add packed single precision floating-point values",
        description=description,
    )


class OpcodeTableTests(unittest.TestCase):
    def test_addps_rows_carry_modes_and_cpuid(self):
        rows = intel.parse_opcode_table(_fixture_lines("addps_page.txt"))
        self.assertEqual([r["op_en"] for r in rows], ["A", "B", "B", "C", "C", "C"])
        self.assertTrue(all(r["mode_64"] == "V" and r["mode_32"] == "V" for r in rows))
        self.assertEqual(rows[0]["opcode"], "NP 0F 58 /r")
        self.assertEqual(rows[0]["cpuid"], "SSE")
        self.assertEqual(rows[1]["cpuid"], "AVX")
        self.assertEqual(rows[3]["cpuid"], "(AVX512VL AND AVX512F) OR AVX10.1")
        self.assertEqual(rows[5]["op_en"], "C")

    def test_adcx_row_carries_not_encodable_compat_mode(self):
        rows = intel.parse_opcode_table(_fixture_lines("adcx_page.txt"))
        self.assertEqual([r["op_en"] for r in rows], ["RM", "RM"])
        self.assertEqual(rows[1]["opcode"], "66 REX.w 0F 38 F6 /r")
        self.assertEqual((rows[1]["mode_64"], rows[1]["mode_32"]), ("V", "N.E."))
        self.assertEqual(rows[1]["cpuid"], "ADX")

    def test_vpermb_rows_share_one_op_en(self):
        rows = intel.parse_opcode_table(_fixture_lines("vpermb_page.txt"))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["op_en"] == "A" for r in rows))

    def test_no_table_lines_give_no_rows(self):
        self.assertEqual(intel.parse_opcode_table(["Description", "plain prose."]), [])


class OperandEncodingTableTests(unittest.TestCase):
    def test_addps_operand_roles_and_tuple_type(self):
        rows = intel.parse_operand_encoding_table(_fixture_lines("addps_page.txt"))
        self.assertEqual([r["op_en"] for r in rows], ["A", "B", "C"])
        self.assertEqual(rows[0]["operands"], ["ModRM:reg (r, w)", "ModRM:r/m (r)"])
        self.assertEqual(rows[1]["tuple_type"], "")
        self.assertEqual(rows[2]["tuple_type"], "Full")

    def test_vpermb_tuple_type_full_mem(self):
        rows = intel.parse_operand_encoding_table(_fixture_lines("vpermb_page.txt"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tuple_type"], "Full Mem")
        self.assertEqual(
            rows[0]["operands"],
            ["ModRM:reg (w)", "EVEX.vvvv (r)", "ModRM:r/m (r)"],
        )

    def test_adcx_operand_encoding_without_tuple_column(self):
        rows = intel.parse_operand_encoding_table(_fixture_lines("adcx_page.txt"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["op_en"], "RM")
        self.assertEqual(rows[0]["tuple_type"], "")
        self.assertEqual(rows[0]["operands"], ["ModRM:reg (r, w)", "ModRM:r/m (r)"])

    def test_no_table_lines_give_no_rows(self):
        self.assertEqual(intel.parse_operand_encoding_table(["Description", "prose."]), [])


class PymupdfCellRowsTests(unittest.TestCase):
    """PyMuPDF gives one line per table cell, sorted per visual row. The
    production path (parse_intel_sdm) joins same-row cells, then parses."""

    def test_cell_rows_form_sections(self):
        from simdref.pdfparse.intel import (
            _cell_joined_lines,
            _prepare_page_from_pymupdf_dict,
        )

        def row(cells, top):
            # PyMuPDF emits each table cell as its own line on one row.
            return [{"spans": [_span(text, 9.0, x0, top)]} for text, x0 in cells]

        text_dict = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        *row(
                            [
                                ("NP 0F 58 /r", 49.0),
                                ("A", 205.0),
                                ("V/V", 236.0),
                                ("SSE", 285.0),
                                ("Add packed values", 355.0),
                            ],
                            116.0,
                        ),
                        *row(
                            [
                                ("Tuple Type", 97.0),
                                ("Operand 1", 150.0),
                                ("Operand 2", 250.0),
                            ],
                            180.0,
                        ),
                        *row(
                            [
                                ("A", 72.0),
                                ("Full", 96.0),
                                ("ModRM:reg (w)", 150.0),
                                ("ModRM:r/m (r)", 250.0),
                            ],
                            196.0,
                        ),
                        *row([("Description", 46.0)], 220.0),
                        *row([("Adds four values.", 46.0)], 236.0),
                    ],
                }
            ]
        }
        prepared = _prepare_page_from_pymupdf_dict(text_dict)
        sections = intel.form_table_sections(_cell_joined_lines(prepared.body_lines))
        self.assertIn(
            "NP 0F 58 /r (A): 64-bit V, compat/legacy V; CPUID SSE",
            sections["Opcode & Modes"],
        )
        self.assertIn(
            "A: ModRM:reg (w); ModRM:r/m (r); tuple: Full",
            sections["Operand Encoding"],
        )


class FormTableSectionsTests(unittest.TestCase):
    def test_addps_sections_text(self):
        sections = intel.form_table_sections(_fixture_lines("addps_page.txt"))
        modes = sections["Opcode & Modes"]
        self.assertIn("NP 0F 58 /r (A): 64-bit V, compat/legacy V; CPUID SSE", modes)
        self.assertIn("CPUID (AVX512VL AND AVX512F) OR AVX10.1", modes)
        encoding = sections["Operand Encoding"]
        self.assertIn("C: ModRM:reg (w); EVEX.vvvv (r); ModRM:r/m (r); tuple: Full", encoding)

    def test_adcx_section_keeps_not_encodable_mode(self):
        sections = intel.form_table_sections(_fixture_lines("adcx_page.txt"))
        self.assertIn("compat/legacy N.E.", sections["Opcode & Modes"])
        self.assertIn("RM: ModRM:reg (r, w); ModRM:r/m (r)", sections["Operand Encoding"])

    def test_no_table_gives_no_sections(self):
        self.assertEqual(intel.form_table_sections(["Description", "prose."]), {})


class CliSurfaceTests(unittest.TestCase):
    def test_cli_detail_view_shows_modes_and_encoding(self):
        from simdref import display

        with display.console.capture() as capture:
            display.render_instruction(catalog=None, item=_instruction())
        out = " ".join(capture.get().replace("│", " ").split())
        self.assertIn("Opcode & Modes", out)
        self.assertIn("NP 0F 58 /r (A): 64-bit V, compat/legacy V; CPUID SSE", out)
        self.assertIn("CPUID (AVX512VL AND AVX512F) OR AVX10.1", out)
        self.assertIn("Operand Encoding", out)
        self.assertIn("C: ModRM:reg (w); EVEX.vvvv (r); ModRM:r/m (r); tuple: Full", out)

    def test_cli_full_view_shows_exception_reference(self):
        from simdref import display

        with display.console.capture() as capture:
            display.render_instruction(catalog=None, item=_instruction(), full=True)
        out = " ".join(capture.get().replace("│", " ").split())
        self.assertIn("Exceptions Type 14; see Table 2-31.", out)

    def test_cli_detail_without_table_sections_shows_nothing(self):
        from simdref import display

        record = InstructionRecord(mnemonic="ADDPS", form="ADDPS xmm1, xmm2/m128", summary="s")
        with display.console.capture() as capture:
            display.render_instruction(catalog=None, item=record)
        out = capture.get()
        self.assertNotIn("Opcode & Modes", out)
        self.assertNotIn("Operand Encoding", out)


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class TuiSurfaceTests(unittest.TestCase):
    def _sections(self, item) -> dict[str, tuple]:
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        payload = app._build_instruction_payload(item)
        return {
            entry[2]: (entry[1], entry[3])
            for entry in payload
            if entry[0] == "collapsible" and isinstance(entry[1], str)
        }

    def test_tui_detail_page_shows_table_sections(self):
        sections = self._sections(_instruction())
        modes_body, modes_collapsed = sections["Opcode & Modes"]
        self.assertFalse(modes_collapsed)
        self.assertIn("CPUID (AVX512VL AND AVX512F) OR AVX10.1", modes_body)
        encoding_body, encoding_collapsed = sections["Operand Encoding"]
        self.assertFalse(encoding_collapsed)
        self.assertIn("tuple: Full", encoding_body)
        exceptions_body, _ = sections["Exceptions"]
        self.assertIn("Exceptions Type 14; see Table 2-31.", exceptions_body)

    def test_tui_detail_without_table_sections_shows_nothing(self):
        record = InstructionRecord(mnemonic="ADDPS", form="ADDPS xmm1, xmm2/m128", summary="s")
        sections = self._sections(record)
        self.assertNotIn("Opcode & Modes", sections)
        self.assertNotIn("Operand Encoding", sections)


if __name__ == "__main__":
    unittest.main()
