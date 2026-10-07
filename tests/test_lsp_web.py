import json
import tempfile
import unittest
from pathlib import Path

from simdref.ingest import build_catalog
from simdref.lsp import _completion_candidates, _hover_markdown
from simdref.storage import _unpack_payload, build_sqlite, save_catalog, open_db
from simdref.export import export_site_data
from conftest import build_fixture_catalog
from test_export import _decode_instructions, _decode_intrinsics


class LspWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(cls._tmpdir.name)
        cls._catalog = build_fixture_catalog()
        save_catalog(cls._catalog, path=tmp_path / "catalog.msgpack")
        build_sqlite(cls._catalog, path=tmp_path / "catalog.db")
        cls._conn = open_db(path=tmp_path / "catalog.db")

    @classmethod
    def tearDownClass(cls):
        cls._conn.close()
        cls._tmpdir.cleanup()

    def test_intrinsic_hover_contains_signature(self):
        markdown = _hover_markdown(self._conn, "_mm256_add_ps")
        self.assertIsNotNone(markdown)
        self.assertIn("_mm256_add_ps", markdown)
        self.assertIn("Instructions:", markdown)

    def test_riscv_instruction_hover_supports_dotted_mnemonic(self):
        markdown = _hover_markdown(self._conn, "vadd.vv")
        self.assertIsNotNone(markdown)
        self.assertIn("vadd.vv", markdown)
        self.assertIn("**isa.** V", markdown)

    def test_sqlite_runtime_preserves_riscv_counts_sections_and_policy_metadata(self):
        intrinsic_count = self._conn.execute(
            "SELECT COUNT(*) FROM intrinsics_data WHERE architecture = 'riscv'"
        ).fetchone()[0]
        instruction_count = self._conn.execute(
            "SELECT COUNT(*) FROM instructions_data WHERE architecture = 'riscv'"
        ).fetchone()[0]
        self.assertEqual(intrinsic_count, 20)
        self.assertEqual(instruction_count, 20)

        instruction_payload = _unpack_payload(
            self._conn.execute(
                "SELECT payload FROM instructions_data WHERE key = ?",
                ("vsub.vv [masked]",),
            ).fetchone()[0],
        )
        self.assertIn("Description", instruction_payload["description"])
        self.assertIn("Operation", instruction_payload["description"])
        self.assertEqual(instruction_payload["metadata"]["policy"], "agnostic")
        self.assertEqual(instruction_payload["metadata"]["masking"], "masked")
        self.assertEqual(instruction_payload["metadata"]["tail_policy"], "agnostic")
        self.assertEqual(instruction_payload["metadata"]["mask_policy"], "agnostic")

        intrinsic_payload = _unpack_payload(
            self._conn.execute(
                "SELECT payload FROM intrinsics_data WHERE name = ?",
                ("__riscv_vsub_vv_i32m1_m",),
            ).fetchone()[0],
        )
        self.assertEqual(intrinsic_payload["instructions"], ["vsub.vv [masked]"])
        self.assertEqual(intrinsic_payload["metadata"]["policy"], "agnostic")
        self.assertEqual(intrinsic_payload["metadata"]["masking"], "masked")
        self.assertEqual(intrinsic_payload["metadata"]["tail_policy"], "agnostic")
        self.assertEqual(intrinsic_payload["metadata"]["mask_policy"], "agnostic")

    def test_completion_returns_intrinsics(self):
        items = _completion_candidates(self._conn, "_mm256_a", limit=5)
        labels = [item["label"] for item in items]
        self.assertIn("_mm256_add_ps", labels)

    def test_export_web_produces_expected_files(self):
        catalog = build_fixture_catalog()
        catalog.instructions[0].pdf_refs = [
            {
                "source_id": "intel-sdm",
                "label": "Intel SDM",
                "url": "https://example.com/intel-sdm.pdf#page=42",
                "page_start": "42",
                "page_end": "43",
            }
        ]
        catalog.instructions[0].metadata["intel-sdm-url"] = (
            "https://example.com/intel-sdm.pdf#page=42"
        )
        catalog.instructions[0].metadata["intel-sdm-page-start"] = "42"
        catalog.instructions[0].metadata["intel-sdm-page-end"] = "43"
        with tempfile.TemporaryDirectory() as tmpdir:
            export_site_data(catalog, Path(tmpdir))

            # Search index shards: meta carries the ISA config + available
            # ISAs union; the two pool shards are columnar-encoded.
            meta = json.loads((Path(tmpdir) / "search-index-meta.json").read_text())
            self.assertIn("isa_config", meta)
            self.assertIn("available_isas", meta)
            self.assertTrue(meta["available_isas"], "meta.available_isas is empty")
            intrinsics = _decode_intrinsics(
                json.loads((Path(tmpdir) / "search-index-intrinsics.json").read_text())
            )
            instructions = _decode_instructions(
                json.loads((Path(tmpdir) / "search-index-instructions.json").read_text())
            )
            self.assertTrue(len(intrinsics) > 0)
            self.assertTrue(len(instructions) > 0)

            # Sanity: the exported intrinsic shard carries a known probe so
            # we can catch silent hydration failures that would otherwise
            # surface as "Loading…" on the live web page.
            intrinsic_names = {item["name"] for item in intrinsics}
            self.assertIn("_mm256_add_ps", intrinsic_names)

            # Intrinsic details are sharded into per-prefix chunks to avoid
            # shipping the 144 MB monolithic file to the client. The
            # client derives the bucket from the intrinsic name with the
            # same rule as ``simdref.export._intrinsic_chunk_prefix``.
            from simdref.export import _intrinsic_chunk_prefix

            intrinsic_chunks_dir = Path(tmpdir) / "intrinsic-chunks"
            self.assertTrue(intrinsic_chunks_dir.is_dir())

            def _load_intrinsic_detail(name: str) -> dict:
                bucket = _intrinsic_chunk_prefix(name)
                chunk = json.loads((intrinsic_chunks_dir / f"{bucket}.json").read_text())
                return chunk[name]

            arm_detail = _load_intrinsic_detail("vaddq_u8")
            self.assertEqual(
                arm_detail["url"],
                "https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddq_u8",
            )
            self.assertIn("argument_preparation", arm_detail["metadata"])
            riscv_intr = next(
                item for item in intrinsics if item["name"] == "__riscv_vadd_vv_i32m1"
            )
            self.assertEqual(riscv_intr["display_architecture"], "RISC-V")
            riscv_detail = _load_intrinsic_detail("__riscv_vadd_vv_i32m1")
            self.assertEqual(
                riscv_detail["url"], "https://github.com/riscv-non-isa/riscv-rvv-intrinsic-doc"
            )
            self.assertIn("riscv:vsub.vv", [item["key"] for item in instructions])
            # Search index instructions have key but no measurements
            instr = instructions[0]
            self.assertIn("key", instr)
            self.assertIn("display_key", instr)
            self.assertIn("architecture", instr)
            self.assertIn("search_fields", instr)
            self.assertNotIn("measurements", instr)

            # Detail chunks directory
            chunks_dir = Path(tmpdir) / "detail-chunks"
            self.assertTrue(chunks_dir.is_dir())
            chunk_files = list(chunks_dir.glob("*.json"))
            self.assertTrue(len(chunk_files) > 0)

            # Spot-check chunks have measurements/operand details and preserve SDM metadata
            saw_sdm = False
            for chunk_file in chunk_files:
                chunk = json.loads(chunk_file.read_text())
                self.assertIsInstance(chunk, dict)
                for detail in chunk.values():
                    self.assertIn("measurements", detail)
                    self.assertIn("operand_details", detail)
                    self.assertIn("architecture", detail)
                    metadata = detail.get("metadata", {})
                    pdf_refs = detail.get("pdf_refs", [])
                    if pdf_refs:
                        self.assertEqual(pdf_refs[0]["source_id"], "intel-sdm")
                        self.assertEqual(pdf_refs[0]["page_start"], "42")
                    if metadata.get("intel-sdm-url"):
                        saw_sdm = True
                        self.assertEqual(
                            metadata["intel-sdm-url"], "https://example.com/intel-sdm.pdf#page=42"
                        )
                        self.assertEqual(metadata["intel-sdm-page-start"], "42")
                        self.assertEqual(metadata["intel-sdm-page-end"], "43")
                    if detail["architecture"] == "riscv" and detail["mnemonic"] == "vsub.vv":
                        self.assertIn("Description", detail["description"])
                        self.assertIn("Operation", detail["description"])
            self.assertTrue(saw_sdm)

            # Filter spec: shared source of truth for ISA + category facets
            filter_spec = json.loads((Path(tmpdir) / "filter_spec.json").read_text())
            self.assertIn("family_order", filter_spec)
            self.assertIn("family_sub_order", filter_spec)
            self.assertIn("default_enabled", filter_spec)
            self.assertIn("categories", filter_spec)
            # Every category references a family known to the family_order map.
            known_families = set(filter_spec["family_order"].keys())
            for cat in filter_spec["categories"]:
                self.assertIn(cat["family"], known_families)

            # Build stamp: lets the SPA warn when static bundle ages out of sync
            stamp = json.loads((Path(tmpdir) / "build_stamp.json").read_text())
            self.assertIn("version", stamp)
            self.assertIn("catalog_generated_at", stamp)
            self.assertEqual(stamp["intrinsics"], len(catalog.intrinsics))

            # Intrinsic details live in per-prefix chunks; hydrate the
            # bucket that carries ``vaddq_u8`` and spot-check.
            from simdref.export import _intrinsic_chunk_prefix

            vaddq_bucket = _intrinsic_chunk_prefix("vaddq_u8")
            intr_chunk = json.loads(
                (Path(tmpdir) / "intrinsic-chunks" / f"{vaddq_bucket}.json").read_text()
            )
            self.assertIsInstance(intr_chunk, dict)
            self.assertTrue(len(intr_chunk) > 0)
            self.assertIn("doc_sections", intr_chunk["vaddq_u8"])
            self.assertIn("ACLE Documentation", intr_chunk["vaddq_u8"]["doc_sections"])

    def test_export_web_preserves_x86_detail_sections_for_rendering(self):
        catalog = build_fixture_catalog()
        x86_instruction = next(
            item
            for item in catalog.instructions
            if item.architecture == "x86" and item.mnemonic == "VPEXPANDD"
        )
        x86_instruction.description = {
            "Description": "Expand packed integers under writemask control.",
            "Operation": "FOR j := 0 TO KL-1",
            "Exceptions": "Type 11 class exceptions.",
            "Intrinsic Equivalents": "_mm512_maskz_expandloadu_epi32",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            export_site_data(catalog, Path(tmpdir))
            chunk = json.loads((Path(tmpdir) / "detail-chunks" / "VPE.json").read_text())
            detail = chunk[x86_instruction.db_key]
            self.assertIn("description", detail)
            self.assertIn("Description", detail["description"])
            self.assertIn("Operation", detail["description"])
            self.assertIn("Exceptions", detail["description"])
            self.assertIn("Intrinsic Equivalents", detail["description"])
            self.assertTrue(detail["measurements"])


if __name__ == "__main__":
    unittest.main()
