"""B2: stored-but-hidden metadata in the detail views.

Fields under test, each from the fixture record in
tests/fixtures/b2/metadata_records.json:

- RVV policy keys (masking, policy, tail_policy, mask_policy) on the
  intrinsic and instruction metadata sections of every detail surface.
- Intel category and subcategory of x86 intrinsics on CLI, TUI, web and
  the search export.
- uops.info operand flag reads/writes (flag_CF and friends) plus the
  suppressed mark, from ingest through the operand tables.

A record that lacks a field renders nothing for it: the tests below
also cover the fixture fields stripped from the record.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover
    HAS_TEXTUAL = False

FIXTURE = Path(__file__).parent / "fixtures" / "b2" / "metadata_records.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _intrinsic(key: str):
    from simdref.models import IntrinsicRecord

    data = _fixture()[key]
    return IntrinsicRecord(
        name=data["name"],
        signature=data["signature"],
        description=data["description"],
        header=data["header"],
        architecture=data["architecture"],
        isa=data["isa"],
        category=data.get("category", ""),
        subcategory=data.get("subcategory", ""),
        metadata=data.get("metadata", {}),
    )


def _instruction(key: str):
    from simdref.models import InstructionRecord

    data = _fixture()[key]
    return InstructionRecord(
        mnemonic=data["mnemonic"],
        form=data["form"],
        summary=data["summary"],
        architecture=data["architecture"],
        isa=data["isa"],
        operand_details=data.get("operand_details", []),
        metadata=data.get("metadata", {}),
    )


def _panel_text(rows: list[tuple[str, str]]) -> str:
    return " ".join(f"{key} {value}" for key, value in rows)


class OperandFlagsHelperTests(unittest.TestCase):
    def test_flags_and_suppressed_are_compact_text(self):
        from simdref.display import operand_flags_text

        operand = _fixture()["uops_instruction"]["operand_details"][2]
        self.assertEqual(operand_flags_text(operand), "CF r/w; suppressed")

    def test_operand_without_flags_has_empty_text(self):
        from simdref.display import operand_flags_text

        operand = _fixture()["uops_instruction"]["operand_details"][0]
        self.assertEqual(operand_flags_text(operand), "")


class CliSurfaceTests(unittest.TestCase):
    def _capture(self, fn, *args) -> str:
        from simdref import display

        with display.console.capture() as capture:
            fn(*args)
        # Rich folds long strings over table rules; unwrap for matching.
        return " ".join(capture.get().replace("│", " ").split())

    def test_rvv_intrinsic_show_lists_policy_keys(self):
        from simdref.display import render_intrinsic

        out = self._capture(render_intrinsic, None, _intrinsic("rvv_intrinsic"))
        for expected in ("tail_policy undisturbed", "mask_policy agnostic", "policy undisturbed"):
            self.assertIn(expected, out)

    def test_x86_intrinsic_show_lists_category_and_subcategory(self):
        from simdref.display import render_intrinsic

        out = self._capture(render_intrinsic, None, _intrinsic("x86_intrinsic"))
        self.assertIn("AVX / Arithmetic", out)

    def test_rvv_instruction_metadata_rows_list_policy_keys(self):
        from simdref.display import instruction_metadata_rows

        text = _panel_text(instruction_metadata_rows(_instruction("rvv_instruction")))
        self.assertIn("tail_policy agnostic", text)
        self.assertIn("mask_policy agnostic", text)

    def test_operand_table_lists_flags_and_suppressed(self):
        from simdref.display import print_operand_block

        out = self._capture(print_operand_block, _instruction("uops_instruction"))
        self.assertIn("CF r/w", out)
        self.assertIn("suppressed", out)

    def test_record_without_the_fields_shows_no_empty_labels(self):
        from simdref.display import instruction_metadata_rows, operand_flags_text

        text = _panel_text(instruction_metadata_rows(_instruction("uops_instruction")))
        self.assertNotIn("tail_policy", text)
        operand = _fixture()["uops_instruction"]["operand_details"][0]
        self.assertEqual(operand_flags_text(operand), "")


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class TuiSurfaceTests(unittest.TestCase):
    def _meta_text(self, payload: list[tuple]) -> str:
        rows = []
        for entry in payload:
            if entry[0] != "static":
                continue
            renderable = entry[1].renderable
            if not hasattr(renderable, "columns"):
                continue
            cells = [list(col._cells) for col in renderable.columns]
            for pair in zip(*cells):
                rows.append(" ".join(str(cell) for cell in pair))
        return " ".join(rows)

    def _catalog_conn(self, records):
        from simdref.models import Catalog
        from simdref.storage import build_sqlite, open_db

        self._tmpdir = TemporaryDirectory()  # noqa: F821
        db_path = Path(self._tmpdir.name) / "catalog.db"
        catalog = Catalog(
            intrinsics=[],
            instructions=records,
            sources=[],
            generated_at="2026-10-31T00:00:00",
        )
        build_sqlite(catalog, path=db_path)
        return open_db(db_path)

    def test_rvv_intrinsic_panel_lists_policy_keys(self):
        from simdref.tui import SimdrefApp

        conn = self._catalog_conn([])
        payload = SimdrefApp(initial_view="annotate")._build_intrinsic_payload(
            _intrinsic("rvv_intrinsic"), conn=conn
        )
        conn.close()
        out = self._meta_text(payload)
        self.assertIn("tail_policy undisturbed", out)
        self.assertIn("mask_policy agnostic", out)

    def test_rvv_instruction_panel_lists_policy_keys(self):
        from simdref.tui import SimdrefApp

        payload = SimdrefApp(initial_view="annotate")._build_instruction_payload(
            _instruction("rvv_instruction")
        )
        self.assertIn("tail_policy agnostic", self._meta_text(payload))

    def test_operand_panel_lists_flags_and_suppressed(self):
        from simdref.tui import SimdrefApp

        payload = SimdrefApp(initial_view="annotate")._build_instruction_payload(
            _instruction("uops_instruction")
        )
        out = self._meta_text(payload)
        self.assertIn("CF r/w", out)
        self.assertIn("suppressed", out)


class LspSurfaceTests(unittest.TestCase):
    def test_hover_shows_rvv_policy_line(self):
        from simdref.lsp import _hover_markdown
        from simdref.models import Catalog
        from simdref.storage import build_sqlite, open_db

        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "catalog.db"
            catalog = Catalog(
                intrinsics=[_intrinsic("rvv_intrinsic")],
                instructions=[],
                sources=[],
                generated_at="2026-10-31T00:00:00",
            )
            build_sqlite(catalog, path=db_path)
            conn = open_db(db_path)
            try:
                markdown = _hover_markdown(conn, "__riscv_vadd_vv_i32m1_tu")
            finally:
                conn.close()
        self.assertIsNotNone(markdown)
        self.assertIn("tail_policy undisturbed", markdown)
        self.assertIn("mask_policy agnostic", markdown)


class IngestTests(unittest.TestCase):
    def test_uops_operand_keeps_flag_names_and_suppressed(self):
        from simdref.ingest_catalog import parse_uops_xml

        records = parse_uops_xml(_fixture()["uops_xml"])
        self.assertEqual(len(records), 1)
        flags_operand = records[0].operand_details[3]
        self.assertEqual(flags_operand["flag_CF"], "r/w")
        self.assertEqual(flags_operand["suppressed"], "1")


class ExportSurfaceTests(unittest.TestCase):
    def _catalog(self):
        from simdref.models import Catalog

        return Catalog(
            intrinsics=[_intrinsic("x86_intrinsic"), _intrinsic("rvv_intrinsic")],
            instructions=[_instruction("uops_instruction"), _instruction("rvv_instruction")],
            sources=[],
            generated_at="2026-10-31T00:00:00",
        )

    def test_search_intrinsics_export_category_and_subcategory(self):
        from simdref.export import _search_intrinsics

        entries = {e["name"]: e for e in _search_intrinsics(self._catalog(), {})}
        self.assertEqual(entries["_mm256_add_ps"]["category"], "Arithmetic")
        self.assertEqual(entries["_mm256_add_ps"]["subcategory"], "AVX")
        # A record without the field omits the key entirely.
        self.assertNotIn("category", entries["__riscv_vadd_vv_i32m1_tu"])

    def test_columnar_intrinsics_carry_category_and_subcategory(self):
        from simdref.export import _columnar_intrinsics, _search_intrinsics

        data = _columnar_intrinsics(_search_intrinsics(self._catalog(), {}))
        self.assertIn("cat", data)
        self.assertIn("sub", data)
        self.assertIn("Arithmetic", data["cat"])
        self.assertIn("AVX", data["sub"])

    def test_instruction_detail_chunk_lists_policy_keys_and_operand_flags(self):
        from simdref.export import _detail_chunks

        chunks = _detail_chunks(self._catalog())
        vadd = chunks["VAD"][f"riscv:vadd.vv vd, vs2, vs1, vm"]
        self.assertEqual(vadd["metadata"]["tail_policy"], "agnostic")
        self.assertEqual(vadd["metadata"]["mask_policy"], "agnostic")
        adc = chunks["ADC"][f"x86:adc r64, r64"]
        flags_operand = adc["operand_details"][2]
        self.assertEqual(flags_operand["flag_CF"], "r/w")
        self.assertEqual(flags_operand["suppressed"], "1")
        # A record without the policy keys omits them entirely.
        self.assertNotIn("tail_policy", adc["metadata"])


if __name__ == "__main__":
    unittest.main()
