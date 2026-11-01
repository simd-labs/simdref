"""B5: keep the 1,072 SME-only intrinsics and tag their streaming mode.

Fixture record (tests/fixtures/b5/arm_sme_bundle.json): one SME-only
intrinsic (svld1_hor_vnum_za128, SIMD_ISA "SME and SME2") and one NEON
intrinsic (vaddq_f32). Before the fix the SIMD filter silently dropped
every SME-only entry.

Surfaces under test, each with the fixture record that holds the fields:

- parse keeps the SME entry and tags the streaming mode from sme_modes
- CLI show (simdref.display.render_intrinsic) prints the tag
- TUI detail page (SimdrefApp._build_intrinsic_payload) shows the tag
- CLI/TUI/web Arm filters name SME as a sub-ISA of the Arm family
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover
    HAS_TEXTUAL = False

BUNDLE = Path(__file__).parent / "fixtures" / "b5" / "arm_sme_bundle.json"


def _records():
    from simdref.ingest_catalog import parse_arm_intrinsics_json_bundle

    payload = {
        "intrinsics_json": BUNDLE.read_text(),
        "operations_json": "[]",
    }
    return {record.name: record for record in parse_arm_intrinsics_json_bundle(payload)}


def _flat(text: str) -> str:
    return " ".join(text.replace("│", " ").split())


class SmeParseTests(unittest.TestCase):
    def test_sme_only_entry_survives_the_simd_filter(self):
        records = _records()
        self.assertIn("svld1_hor_vnum_za128", records)
        self.assertIn("vaddq_f32", records)

    def test_sme_entry_gets_sme_isa_and_header(self):
        sme = _records()["svld1_hor_vnum_za128"]
        self.assertEqual(sme.isa, ["SME"])
        self.assertEqual(sme.header, "arm_sme.h")

    def test_neon_entry_is_unchanged(self):
        neon = _records()["vaddq_f32"]
        self.assertEqual(neon.isa, ["NEON"])
        self.assertEqual(neon.header, "arm_neon.h")

    def test_sme_entry_tagged_with_streaming_mode(self):
        sme = _records()["svld1_hor_vnum_za128"]
        attrs = sme.doc_sections.get("Required keyword attributes", "")
        self.assertIn("__arm_streaming", attrs)
        streaming = sme.doc_sections.get("Required streaming features", "")
        self.assertIn("FEAT_SME", streaming)
        self.assertIn("Requires streaming features", sme.notes)


class SmeFilterTests(unittest.TestCase):
    def test_sme_maps_to_arm_family(self):
        from simdref.display import display_isa, isa_family

        self.assertEqual(display_isa(["SME"]), "SME")
        self.assertEqual(isa_family("SME"), "Arm")

    def test_sme_is_an_arm_sub_isa(self):
        from simdref.display import isa_to_sub_isa

        self.assertEqual(isa_to_sub_isa("SME"), "SME")

    def test_arm_filter_config_lists_sme(self):
        from simdref.filters import DEFAULT_SUBS, FAMILY_SUB_ORDER

        self.assertIn("SME", FAMILY_SUB_ORDER["Arm"])
        self.assertIn("SME", DEFAULT_SUBS["Arm"])


class CliSurfaceTests(unittest.TestCase):
    def test_cli_show_prints_the_streaming_mode_tag(self):
        from simdref import display

        sme = _records()["svld1_hor_vnum_za128"]
        with display.console.capture() as capture:
            display.render_intrinsic(catalog=None, item=sme, full=True)
        out = _flat(capture.get())
        self.assertIn("Required keyword attributes", out)
        self.assertIn("__arm_streaming", out)


@unittest.skipUnless(HAS_TEXTUAL, "textual not installed")
class TuiSurfaceTests(unittest.TestCase):
    def test_tui_detail_page_shows_the_streaming_mode_tag(self):
        from tempfile import TemporaryDirectory

        from simdref.models import Catalog
        from simdref.storage import build_sqlite, open_db
        from simdref.tui import SimdrefApp

        sme = _records()["svld1_hor_vnum_za128"]
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.db"
            build_sqlite(
                Catalog(
                    intrinsics=[sme],
                    instructions=[],
                    sources=[],
                    generated_at="2026-10-31T00:00:00",
                ),
                path=db_path,
            )
            conn = open_db(db_path)
            payload = SimdrefApp(initial_view="annotate")._build_intrinsic_payload(sme, conn=conn)
            conn.close()
        sections = [(entry[2], entry[1]) for entry in payload if entry[0] == "collapsible"]
        titles = [title for title, _ in sections]
        self.assertIn("Required keyword attributes", titles)
        body = next(body for title, body in sections if title == "Required keyword attributes")
        self.assertIn("__arm_streaming", str(body))


class WebExportTests(unittest.TestCase):
    def test_web_search_entry_marks_sme_as_arm_sub_isa(self):
        from simdref.export import _search_intrinsics
        from simdref.models import Catalog

        catalog = Catalog(
            intrinsics=[_records()["svld1_hor_vnum_za128"]],
            instructions=[],
            sources=[],
            generated_at="2026-10-31T00:00:00",
        )
        entry = _search_intrinsics(catalog, {})[0]
        self.assertEqual(entry["isa_families"], ["Arm"])
        self.assertEqual(entry["isa_subs"], ["SME"])


if __name__ == "__main__":
    unittest.main()
