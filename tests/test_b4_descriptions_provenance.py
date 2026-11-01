"""B4: intrinsic description in the CLI show view and perf provenance
footnote (source name, source_version and citation_url) per surface.

Fixture record: an x86 intrinsic with a description, and an ARM modeled
instruction whose arch_details carry llvm-mca provenance
(tests/fixtures/b4/fmla_modeled.json).

Surfaces under test, each with the fixture record that holds the fields:

- CLI show (simdref.display.render_intrinsic / render_instruction)
- TUI detail page (SimdrefApp._build_intrinsic_payload /
  _build_instruction_payload)
- web export rows (simdref.export._web_measurements; the web table
  itself renders the link, checked by simdref-web tests/test_b4_*)
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simdref.models import InstructionRecord, IntrinsicRecord

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover
    HAS_TEXTUAL = False

FIXTURE = Path(__file__).parent / "fixtures" / "b4" / "fmla_modeled.json"
CITATION_URL = "https://llvm.org/docs/CommandGuide/llvm-mca.html"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _intrinsic(description: str | None = None) -> IntrinsicRecord:
    data = _fixture()["intrinsic"]
    return IntrinsicRecord(
        name=data["name"],
        signature=data["signature"],
        description=data["description"] if description is None else description,
        header=data["header"],
        architecture=data["architecture"],
        isa=data["isa"],
        category=data["category"],
    )


def _instruction(arch_details: dict | None = None) -> InstructionRecord:
    data = _fixture()["instruction"]
    return InstructionRecord(
        mnemonic=data["mnemonic"],
        form=data["form"],
        summary=data["summary"],
        architecture=data["architecture"],
        isa=data["isa"],
        arch_details=_fixture()["instruction"]["arch_details"]
        if arch_details is None
        else arch_details,
    )


def _flat(text: str) -> str:
    return " ".join(text.replace("│", " ").split())


class CliDescriptionTests(unittest.TestCase):
    def test_cli_show_prints_intrinsic_description(self):
        from simdref import display

        with display.console.capture() as capture:
            display.render_intrinsic(catalog=None, item=_intrinsic())
        out = _flat(capture.get())
        self.assertIn("description", out)
        self.assertIn("Add packed single-precision", out)

    def test_cli_show_omits_description_label_when_absent(self):
        from simdref import display

        with display.console.capture() as capture:
            display.render_intrinsic(catalog=None, item=_intrinsic(description=""))
        self.assertNotIn("description", _flat(capture.get()))


class CliProvenanceTests(unittest.TestCase):
    def test_cli_perf_panel_shows_source_version_and_citation(self):
        from simdref import display

        with display.console.capture() as capture:
            display.render_instruction(catalog=None, item=_instruction(), short=True)
        out = _flat(capture.get())
        self.assertIn("llvm-mca", out)
        self.assertIn("22.1.8", out)
        self.assertIn(CITATION_URL, out)
        # One footnote per distinct source, not per core row.
        self.assertEqual(out.count("llvm-mca 22.1.8"), 1)

    def test_cli_perf_panel_omits_footnote_without_source(self):
        from simdref import display

        with display.console.capture() as capture:
            display.render_instruction(
                catalog=None, item=_instruction(_fixture()["measured_details"]), short=True
            )
        self.assertNotIn("llvm-mca", _flat(capture.get()))


@unittest.skipUnless(HAS_TEXTUAL, "textual not installed")
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

    def _catalog_conn(self):
        from simdref.models import Catalog
        from simdref.storage import build_sqlite, open_db

        self._tmpdir = TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "catalog.db"
        catalog = Catalog(
            intrinsics=[_intrinsic()],
            instructions=[],
            sources=[],
            generated_at="2026-10-31T00:00:00",
        )
        build_sqlite(catalog, path=db_path)
        return open_db(db_path)

    def test_tui_intrinsic_page_shows_description(self):
        from simdref.tui import SimdrefApp

        conn = self._catalog_conn()
        payload = SimdrefApp(initial_view="annotate")._build_intrinsic_payload(
            _intrinsic(), conn=conn
        )
        conn.close()
        rendered = self._meta_text(payload)
        self.assertIn("description", rendered)
        self.assertIn("Add packed single-precision", rendered)

    def test_tui_perf_panel_shows_provenance_footnote(self):
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        payload = app._build_instruction_payload(_instruction())
        texts = [entry[1] for entry in payload if entry[0] == "static_markup"]
        self.assertTrue(
            any("llvm-mca" in text and "22.1.8" in text and CITATION_URL in text for text in texts),
            f"no provenance footnote in static_markup entries: {texts!r}",
        )

    def test_tui_perf_panel_omits_footnote_without_source(self):
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        payload = app._build_instruction_payload(_instruction(_fixture()["measured_details"]))
        texts = [entry[1] for entry in payload if entry[0] == "static_markup"]
        self.assertFalse(any("llvm-mca" in text for text in texts))


class WebExportTests(unittest.TestCase):
    def test_measurement_rows_carry_provenance_keys(self):
        from simdref.export import _web_measurements

        rows = _web_measurements(_instruction())
        row = rows[0]
        self.assertEqual(row["source"], "llvm-mca")
        self.assertEqual(row["sourceVersion"], "22.1.8")
        self.assertEqual(row["citationUrl"], CITATION_URL)

    def test_measurement_rows_label_uops_source_when_absent(self):
        from simdref.export import _web_measurements

        # FIXUI: every row carries a source label; a row without a stored
        # source is a uops.info row, the same default the merge rank uses.
        rows = _web_measurements(_instruction(_fixture()["measured_details"]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "uops.info")
        self.assertNotIn("sourceVersion", rows[0])
        self.assertNotIn("citationUrl", rows[0])


if __name__ == "__main__":
    unittest.main()
