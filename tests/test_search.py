import json
import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
import tempfile

from conftest import build_fixture_catalog
from simdref.cli import _resolve_query_payload
from simdref.display import (
    display_instruction_form,
    instruction_query_text,
    instruction_variant_items,
    strip_instruction_decorators,
)
from simdref.queries import instruction_rows_for_intrinsic
from simdref.ingest import (
    _instruction_summary,
    _normalize_operand_xtype,
    build_catalog,
    parse_uops_xml,
)
from simdref.arm_instructions import parse_arm_instruction_payload
from simdref.ingest_catalog import parse_arm_intrinsics_payload
from simdref.models import InstructionRecord, IntrinsicRecord
from simdref.riscv import (
    _infer_intrinsic_policy,
    parse_riscv_instruction_payload,
    parse_riscv_intrinsics_payload,
)
from simdref.search import find_instruction, find_intrinsic, search_catalog, search_records
from simdref.storage import build_sqlite, open_db, save_catalog
from simdref.tui import _fts_search, _normalize_sub_isa_selection


class SearchTests(unittest.TestCase):
    def test_intrinsic_lookup(self):
        catalog = build_fixture_catalog()
        record = find_intrinsic(catalog, "_mm256_add_epi32")
        self.assertIsNotNone(record)
        self.assertIn("VPADDD", " ".join(record.instructions))
        self.assertIsNotNone(find_intrinsic(catalog, "_mm256_add_ps"))
        self.assertIsNotNone(find_intrinsic(catalog, "_mm256_add_pd"))
        arm_record = find_intrinsic(catalog, "vaddq_u8")
        self.assertIsNotNone(arm_record)
        self.assertEqual(arm_record.architecture, "arm")
        self.assertIn("ADD (Vd.16B, Vn.16B, Vm.16B)", arm_record.instructions)
        self.assertNotIn("ADD (Zd.S, Pg/M, Zn.S, Zm.S)", arm_record.instructions)
        self.assertEqual(
            arm_record.url,
            "https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddq_u8",
        )
        self.assertEqual(arm_record.metadata["argument_preparation"], "a -> Vn.16B;b -> Vm.16B")
        self.assertEqual(
            arm_record.metadata["reference_url"],
            "https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#addition",
        )
        self.assertIn("ACLE Documentation", arm_record.doc_sections)
        self.assertIn("AArch64 instruction:", arm_record.doc_sections["ACLE Documentation"])
        riscv_record = find_intrinsic(catalog, "__riscv_vadd_vv_i32m1")
        self.assertIsNotNone(riscv_record)
        self.assertEqual(riscv_record.architecture, "riscv")
        self.assertEqual(riscv_record.instructions, ["vadd.vv"])
        self.assertEqual(
            riscv_record.url, "https://github.com/riscv-non-isa/riscv-rvv-intrinsic-doc"
        )

    def test_instruction_lookup(self):
        catalog = build_fixture_catalog()
        record = find_instruction(catalog, "ADDPS")
        self.assertIsNotNone(record)
        self.assertIn("_mm_add_ps", record.linked_intrinsics)
        self.assertEqual(record.metadata["iclass"], "ADDPS")
        self.assertTrue(record.arch_details["SKL"]["measurement"])
        self.assertTrue(record.arch_details["SKL"]["doc"])
        arm_record = find_instruction(catalog, "ADD (Zd.S, Pg/M, Zn.S, Zm.S)")
        self.assertIsNotNone(arm_record)
        self.assertEqual(arm_record.architecture, "arm")
        self.assertIn("svadd_s32_z", arm_record.linked_intrinsics)
        riscv_record = find_instruction(catalog, "vadd.vv")
        self.assertIsNotNone(riscv_record)
        self.assertEqual(riscv_record.architecture, "riscv")
        self.assertIn("__riscv_vadd_vv_i32m1", riscv_record.linked_intrinsics)

    def test_instruction_lookup_accepts_tokenized_form(self):
        catalog = build_fixture_catalog()
        record = find_instruction(catalog, "VADDPS YMM YMM YMM")
        self.assertIsNotNone(record)
        self.assertEqual(record.key, "VADDPS (YMM, YMM, YMM)")

    def test_search(self):
        catalog = build_fixture_catalog()
        results = search_catalog(catalog, "expandload")
        self.assertTrue(any(result.kind == "intrinsic" for result in results))
        arm_results = search_catalog(catalog, "svadd")
        self.assertTrue(
            any(
                result.kind == "intrinsic" and result.title == "svadd_s32_z"
                for result in arm_results
            )
        )
        self.assertTrue(
            any(result.kind == "intrinsic" and result.title == "svadd" for result in arm_results)
        )
        riscv_mnemonic = search_catalog(catalog, "vadd.vv")
        self.assertEqual(riscv_mnemonic[0].kind, "instruction")
        self.assertEqual(riscv_mnemonic[0].title, "vadd.vv")
        riscv_intrinsic = search_catalog(catalog, "__riscv_vadd_vv_i32m1")
        self.assertEqual(riscv_intrinsic[0].kind, "intrinsic")
        self.assertEqual(riscv_intrinsic[0].title, "__riscv_vadd_vv_i32m1")
        riscv_extension = search_catalog(catalog, "zvkned")
        self.assertEqual(riscv_extension[0].kind, "instruction")
        self.assertEqual(riscv_extension[0].title, "vaesdf.vv")

    def test_llm_exact_intrinsic_payload(self):
        catalog = build_fixture_catalog()
        payload = _resolve_query_payload(catalog, "_mm_add_ps")
        self.assertEqual(payload["mode"], "exact")
        self.assertEqual(payload["match_kind"], "intrinsic")
        self.assertEqual(payload["intrinsic"]["name"], "_mm_add_ps")
        self.assertTrue(payload["performance"])

    def test_llm_search_payload(self):
        catalog = build_fixture_catalog()
        payload = _resolve_query_payload(catalog, "_mm_add")
        self.assertEqual(payload["mode"], "search")
        self.assertTrue(payload["results"])

    def test_intrinsic_prefers_intrinsic_results(self):
        catalog = build_fixture_catalog()
        results = search_catalog(catalog, "_mm_add")
        self.assertEqual(results[0].kind, "intrinsic")

    def test_mm_add_prefers_intrinsic_results(self):
        catalog = build_fixture_catalog()
        results = search_catalog(catalog, "mm add")
        self.assertEqual(results[0].kind, "intrinsic")
        self.assertNotIn("_mm512_maskz_expandloadu_epi32", [result.title for result in results])

    def test_add_prefers_instruction_results(self):
        catalog = build_fixture_catalog()
        results = search_catalog(catalog, "ADD")
        self.assertEqual(results[0].kind, "instruction")

    def test_mm256_add_prefers_mm256_family(self):
        catalog = build_fixture_catalog()
        results = search_catalog(catalog, "_mm256_add")
        titles = [result.title for result in results[:3]]
        self.assertIn("_mm256_add_ps", titles)
        self.assertIn("_mm256_add_epi32", titles)
        self.assertNotIn("_mm512_maskz_expandloadu_epi32", [result.title for result in results])

    def test_intrinsic_performance_rows(self):
        catalog = build_fixture_catalog()
        intrinsic = find_intrinsic(catalog, "_mm_add_ps")
        rows = instruction_rows_for_intrinsic(catalog, intrinsic)
        self.assertTrue(any(row["uarch"] == "SKL" for row in rows))

    def test_normalize_operand_xtype(self):
        self.assertEqual(_normalize_operand_xtype("4i8"), "i8")
        self.assertEqual(_normalize_operand_xtype("2u16"), "u16")
        self.assertEqual(_normalize_operand_xtype("int"), "i32")
        self.assertEqual(_normalize_operand_xtype("int", "16"), "i16")
        self.assertEqual(_normalize_operand_xtype("int", "64"), "i64")
        self.assertEqual(_normalize_operand_xtype("f32"), "f32")

    def test_generated_summary_for_terse_instruction(self):
        summary = _instruction_summary(
            "ADD",
            "Add",
            [
                {"type": "reg", "width": "32", "xtype": "i32", "w": "1"},
                {"type": "imm", "width": "32", "xtype": "i32", "r": "1"},
            ],
        )
        self.assertEqual(summary, "Add 32-bit integer operands.")

    def test_generated_summary_keeps_operand_width(self):
        records = parse_uops_xml(
            '<root><instruction asm="MOV" string="MOV (R64, M64)" summary="Move">'
            '<operand idx="0" type="reg" width="64" xtype="int" name="R64" w="1"/>'
            '<operand idx="1" type="mem" width="64" xtype="int" name="M64" r="1"/>'
            "</instruction></root>"
        )
        self.assertEqual(records[0].summary, "Move 64-bit integer operands.")

    def test_build_catalog_emits_status_messages(self):
        messages: list[str] = []
        catalog = build_fixture_catalog(status=messages.append)
        self.assertTrue(catalog.intrinsics)
        self.assertTrue(catalog.instructions)
        self.assertTrue(any("Loading Intel fixture" in msg for msg in messages))
        self.assertTrue(any("Linking intrinsics to instructions" in msg for msg in messages))
        self.assertTrue(any("Loading Arm ACLE fixture" in msg for msg in messages))
        self.assertTrue(any("Loading Arm A64" in msg for msg in messages))

    def test_masked_summary_prefix(self):
        summary = _instruction_summary(
            "VADDPS",
            "Add Packed Single Precision Floating-Point Values",
            [
                {"type": "reg", "width": "128", "xtype": "f32", "w": "1"},
                {"type": "reg", "width": "64", "xtype": "i1", "r": "1"},
                {"type": "reg", "width": "128", "xtype": "f32", "r": "1"},
                {"type": "reg", "width": "128", "xtype": "f32", "r": "1"},
            ],
        )
        self.assertEqual(summary, "Masked Add Packed Single Precision Floating-Point Values.")

    def test_riscv_policy_suffixes_mark_masked_variants(self):
        self.assertEqual(_infer_intrinsic_policy("__riscv_vadd_vv_i32m1_mu"), ("mu", "masked"))
        self.assertEqual(_infer_intrinsic_policy("__riscv_vadd_vv_i32m1_tum"), ("tum", "masked"))
        self.assertEqual(_infer_intrinsic_policy("__riscv_vadd_vv_i32m1_tumu"), ("tumu", "masked"))

    def test_maskz_query_prefers_maskz_intrinsic(self):
        catalog = build_fixture_catalog()
        results = search_catalog(catalog, "_mm512_maskz_expandloadu_epi32")
        self.assertEqual(results[0].kind, "intrinsic")
        self.assertEqual(results[0].title, "_mm512_maskz_expandloadu_epi32")

    def test_masked_query_prefers_masked_family_variant(self):
        intrinsics = [
            IntrinsicRecord(
                name="_mm512_mask_add_ps",
                signature="__m512 _mm512_mask_add_ps(__m512 src, __mmask16 k, __m512 a, __m512 b)",
                description="Masked add.",
                header="immintrin.h",
                isa=["AVX512F"],
                instructions=["VADDPS (ZMM{k}, ZMM, ZMM)"],
            ),
            IntrinsicRecord(
                name="_mm512_maskz_add_ps",
                signature="__m512 _mm512_maskz_add_ps(__mmask16 k, __m512 a, __m512 b)",
                description="Mask-zero add.",
                header="immintrin.h",
                isa=["AVX512F"],
                instructions=["VADDPS (ZMM{k}{z}, ZMM, ZMM)"],
            ),
        ]
        instructions = [
            InstructionRecord(
                mnemonic="VADDPS",
                form="VADDPS (ZMM{k}, ZMM, ZMM)",
                summary="Masked add packed single precision floating-point values.",
                isa=["AVX512F"],
            ),
            InstructionRecord(
                mnemonic="VADDPS",
                form="VADDPS (ZMM{k}{z}, ZMM, ZMM)",
                summary="Mask-zero add packed single precision floating-point values.",
                isa=["AVX512F"],
            ),
        ]
        results = search_records(intrinsics, instructions, "_mm512_maskz_add_ps", limit=3)
        self.assertEqual(results[0].kind, "intrinsic")
        self.assertEqual(results[0].title, "_mm512_maskz_add_ps")

    def test_mnemonic_query_prefers_instruction_over_intrinsic_collisions(self):
        intrinsics = [
            IntrinsicRecord(
                name="_addcarryx_u64",
                signature="unsigned char _addcarryx_u64(...)",
                description="Carry add helper.",
                header="immintrin.h",
                isa=["ADX"],
                instructions=["ADCX (R64, R64)"],
            ),
        ]
        instructions = [
            InstructionRecord(
                mnemonic="ADCX",
                form="ADCX (R64, R64)",
                summary="Unsigned integer add with carry.",
                isa=["ADX"],
            ),
        ]
        results = search_records(intrinsics, instructions, "adcx", limit=2)
        self.assertEqual(results[0].kind, "instruction")

    def test_scalar_vector_collision_prefers_requested_width(self):
        intrinsics = [
            IntrinsicRecord(
                name="_mm_add_ps",
                signature="__m128 _mm_add_ps(__m128 a, __m128 b)",
                description="128-bit add.",
                header="xmmintrin.h",
                isa=["SSE"],
                instructions=["ADDPS (XMM, XMM)"],
            ),
            IntrinsicRecord(
                name="_mm256_add_ps",
                signature="__m256 _mm256_add_ps(__m256 a, __m256 b)",
                description="256-bit add.",
                header="immintrin.h",
                isa=["AVX"],
                instructions=["VADDPS (YMM, YMM, YMM)"],
            ),
        ]
        results = search_records(intrinsics, [], "_mm256_add_ps", limit=2)
        self.assertEqual(results[0].title, "_mm256_add_ps")

    def test_instruction_variants_sort_naturally_within_isa(self):
        items = [
            SimpleNamespace(mnemonic="DIV", form="DIV (M16)", isa=["I86"], key="DIV (M16)"),
            SimpleNamespace(mnemonic="DIV", form="DIV (M8)", isa=["I86"], key="DIV (M8)"),
            SimpleNamespace(mnemonic="DIV", form="DIV (R16)", isa=["I86"], key="DIV (R16)"),
            SimpleNamespace(mnemonic="DIV", form="DIV (R8l)", isa=["I86"], key="DIV (R8l)"),
        ]
        queries = [instruction_query_text(item) for item in instruction_variant_items(items)]
        self.assertLess(queries.index("DIV M8"), queries.index("DIV M16"))
        self.assertLess(queries.index("DIV R8l"), queries.index("DIV R16"))

    def test_strip_instruction_decorators_for_display(self):
        self.assertEqual(strip_instruction_decorators("{load} ADD"), "ADD")
        self.assertEqual(strip_instruction_decorators("{load} {disp8} ADD"), "ADD")
        self.assertEqual(display_instruction_form("{load} ADD (R64, R64)"), "ADD (R64, R64)")

    def test_instruction_query_text_strips_leading_decorators(self):
        item = SimpleNamespace(
            mnemonic="{load} ADD",
            form="{load} ADD (R64, R64)",
            isa=["I86"],
            key="{load} ADD (R64, R64)",
        )
        self.assertEqual(instruction_query_text(item), "ADD R64 R64")

    def test_mixed_catalog_keeps_arm_and_x86_instruction_keys_separate(self):
        catalog = build_fixture_catalog()
        arm_records = [
            item
            for item in catalog.instructions
            if item.architecture == "arm" and item.mnemonic == "ADD"
        ]
        self.assertEqual(len(arm_records), 2)
        self.assertEqual(len({item.db_key for item in arm_records}), 2)
        self.assertTrue(all("_mm_add_ps" not in item.linked_intrinsics for item in arm_records))
        neon_add = next(item for item in arm_records if item.form == "ADD (Vd.16B, Vn.16B, Vm.16B)")
        sve_add = next(item for item in arm_records if item.form == "ADD (Zd.S, Pg/M, Zn.S, Zm.S)")
        self.assertIn("vaddq_u8", neon_add.linked_intrinsics)
        self.assertNotIn("vaddq_u8", sve_add.linked_intrinsics)

    def test_mixed_catalog_keeps_riscv_keys_separate(self):
        catalog = build_fixture_catalog()
        riscv_records = [
            item
            for item in catalog.instructions
            if item.architecture == "riscv" and item.mnemonic == "vadd.vv"
        ]
        self.assertGreaterEqual(len(riscv_records), 4)
        self.assertEqual(len({item.db_key for item in riscv_records}), len(riscv_records))
        plain = next(item for item in riscv_records if item.form == "vadd.vv")
        masked = next(item for item in riscv_records if item.form == "vadd.vv [masked]")
        self.assertIn("__riscv_vadd_vv_i32m1", plain.linked_intrinsics)
        self.assertNotIn("__riscv_vadd_vv_i32m1_m", plain.linked_intrinsics)
        self.assertIn("__riscv_vadd_vv_i32m1_m", masked.linked_intrinsics)

    def test_parse_riscv_fixture_payloads(self):
        intrinsics = parse_riscv_intrinsics_payload(
            Path("tests/fixtures/riscv_rvv_intrinsics_sample.json").read_text()
        )
        instructions = parse_riscv_instruction_payload(
            Path("tests/fixtures/riscv_unified_db_sample.json").read_text()
        )
        self.assertTrue(any(item.name == "__riscv_vadd_vv_i32m1" for item in intrinsics))
        self.assertTrue(any(item.mnemonic == "vadd.vv" for item in instructions))
        self.assertTrue(any(item.mnemonic == "vsub.vv" for item in instructions))
        self.assertTrue(any(item.name == "__riscv_vse32_v_i32m1" for item in intrinsics))
        crypto = next(item for item in instructions if item.mnemonic == "vaesdf.vv")
        self.assertEqual(crypto.isa, ["Zvkned"])

    def test_riscv_policy_variants_do_not_cross_link(self):
        catalog = build_fixture_catalog()
        plain = find_intrinsic(catalog, "__riscv_vadd_vv_i32m1")
        masked = find_intrinsic(catalog, "__riscv_vadd_vv_i32m1_m")
        tu = find_intrinsic(catalog, "__riscv_vadd_vv_i32m1_tu")
        tumu = find_intrinsic(catalog, "__riscv_vadd_vv_i32m1_tumu")
        self.assertEqual(plain.instructions, ["vadd.vv"])
        self.assertEqual(masked.instructions, ["vadd.vv [masked]"])
        self.assertEqual(tu.instructions, ["vadd.vv [tu]"])
        self.assertEqual(tumu.instructions, ["vadd.vv [tumu]"])

    def test_riscv_vsub_variants_link_without_crossing_masking(self):
        catalog = build_fixture_catalog()
        plain = find_intrinsic(catalog, "__riscv_vsub_vv_i32m1")
        masked = find_intrinsic(catalog, "__riscv_vsub_vv_i32m1_m")
        self.assertEqual(plain.instructions, ["vsub.vv"])
        self.assertEqual(masked.instructions, ["vsub.vv [masked]"])

    def test_riscv_docs_html_enrichment_fills_missing_sections(self):
        payload = {
            "format": "riscv-unified-db-v1",
            "instructions": [
                {
                    "mnemonic": "vslideup.vx",
                    "form": "vslideup.vx",
                    "summary": "Slide vector elements upward",
                    "isa": ["V"],
                    "url": "https://docs.riscv.org/reference/isa/unpriv/v-st-ext.html#vslideup-vx",
                    "description": {},
                }
            ],
            "docs_pages": {
                "https://docs.riscv.org/reference/isa/unpriv/v-st-ext.html": """
                    <html><body>
                    <h2>Description</h2><p>Slides elements in <code>vs2</code> upward by the scalar offset.</p>
                    <h2>Operation</h2><pre>vd[i + offset] = vs2[i]</pre>
                    </body></html>
                """
            },
        }
        records = parse_riscv_instruction_payload(json.dumps(payload))
        self.assertEqual(
            records[0].description["Description"],
            "Slides elements in vs2 upward by the scalar offset.",
        )
        self.assertIn("vd[i + offset] = vs2[i]", records[0].description["Operation"])

    def test_generic_sve_mapping_links_to_sve_form_only(self):
        catalog = build_fixture_catalog()
        intrinsic = find_intrinsic(catalog, "svadd")
        self.assertIsNotNone(intrinsic)
        self.assertIn("ADD (Zd.S, Pg/M, Zn.S, Zm.S)", intrinsic.instructions)
        self.assertNotIn("ADD (Vd.16B, Vn.16B, Vm.16B)", intrinsic.instructions)
        self.assertIn("ACLE Prototypes", intrinsic.doc_sections)
        self.assertIn("svadd[_s32]_z", intrinsic.doc_sections["ACLE Prototypes"])

    def test_parse_arm_live_intrinsics_bundle(self):
        payload = {
            "format": "arm-intrinsics-json-v1",
            "intrinsics_json": """[
              {
                "SIMD_ISA": ["Neon"],
                "name": "vaddq_u8",
                "return_type": {"value": "uint8x16_t"},
                "arguments": ["uint8x16_t a", "uint8x16_t b"],
                "description": "Add (vector).",
                "instruction_group": "Vector arithmetic|Add|Addition",
                "results": [{"Vd.16B": "result"}],
                "instructions": [{
                  "preamble": "This intrinsic compiles to the following instructions:",
                  "list": [{
                    "base_instruction": "ADD",
                    "url": "https://developer.arm.com/documentation/ddi0602/2025-06/SIMD-FP-Instructions/ADD--vector---Add--vector--",
                    "operands": "Vd.16B,Vn.16B,Vm.16B"
                  }]
                }],
                "Arguments_Preparation": {"a": {"register": "Vn.16B"}, "b": {"register": "Vm.16B"}},
                "Architectures": ["A64"],
                "Operation": "NeonOperationId_00001"
              },
              {
                "SIMD_ISA": ["SVE"],
                "name": "svadd[_s32]_z",
                "return_type": {"value": "svint32_t"},
                "arguments": ["svbool_t pg", "svint32_t op1", "svint32_t op2"],
                "description": "Add",
                "instruction_group": "Vector arithmetic|Add|Addition",
                "results": [{"Zresult.S": "result"}],
                "instructions": [{
                  "preamble": "When result is in a different register from op2, this intrinsic can use:",
                  "list": [{
                    "base_instruction": "ADD",
                    "url": "https://developer.arm.com/documentation/ddi0602/2025-06/SVE-Instructions/ADD--vectors--predicated---Add-vectors--predicated--",
                    "operands": "Zresult.S, Pg/M, Zresult.S, Zop2.S"
                  }]
                }],
                "Arguments_Preparation": {"pg": {"register": "Pg.S"}, "op1": {"register": "Zop1.S"}, "op2": {"register": "Zop2.S"}},
                "Architectures": ["A64"],
                "Operation": "SveOperation_svadd_s32_z",
                "required_streaming_features": {"title": "Required streaming features", "intro": "Streaming intro", "features": "FEAT_SME"}
              }
            ]""",
            "operations_json": """[
              {"item": {"id": "NeonOperationId_00001", "content": "<h4>Operation</h4><pre>V[d] = result;</pre>"}},
              {"item": {"id": "SveOperation_svadd_s32_z", "content": "<h4>Operation</h4><pre>Zresult = op1 + op2;</pre>"}}
            ]""",
            "examples_json": "[]",
        }
        records = parse_arm_intrinsics_payload(json.dumps(payload))
        neon = next(item for item in records if item.name == "vaddq_u8")
        sve = next(item for item in records if item.name == "svadd_s32_z")
        self.assertEqual(neon.source, "arm-intrinsics-site")
        self.assertIn("ACLE Operation", neon.doc_sections)
        self.assertIn("V[d] = result", neon.doc_sections["ACLE Operation"])
        self.assertEqual(sve.header, "arm_sve.h")
        self.assertIn("Required streaming features", sve.doc_sections)
        self.assertEqual(sve.metadata["supported_architectures"], "A64")

    def test_x86_linked_instruction_refs_include_resolution_metadata(self):
        catalog = build_fixture_catalog()
        intrinsic = find_intrinsic(catalog, "_mm_add_ps")
        self.assertIsNotNone(intrinsic)
        ref = intrinsic.instruction_refs[0]
        self.assertEqual(ref["resolution"], "mnemonic")
        self.assertEqual(ref["match_count"], "1")
        self.assertTrue(ref["key"].startswith("x86:"))

    def test_parse_arm_aarchmrs_instruction_bundle(self):
        payload = {
            "format": "arm-aarchmrs-instructions-v1",
            "instructions_json": """{
              "instructions": [
                {
                  "name": "ADD",
                  "operands": "Zd.S, Pg/M, Zn.S, Zm.S",
                  "brief": "Add predicated vectors",
                  "category": "SVE arithmetic",
                  "section": "SVE Instructions",
                  "url": "https://developer.arm.com/documentation/ddi0602/latest/SVE-Instructions/ADD--vectors--predicated---Add-vectors--predicated--",
                  "descriptions": {
                    "Description": "Adds predicated SVE lanes.",
                    "Operation": "Zd = predicated_add(Zn, Zm)"
                  },
                  "aliases": ["ADD (vectors, predicated)"]
                },
                {
                  "base_instruction": "LDP",
                  "operands": "Xt1, Xt2, [Xn|SP{, #imm}]",
                  "summary": "Load pair of registers",
                  "section": "Base Instructions",
                  "url": "https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/LDP--Load-pair-of-registers-"
                }
              ]
            }""",
        }

        records = parse_arm_instruction_payload(json.dumps(payload))
        sve_add = next(item for item in records if item.mnemonic == "ADD")
        ldp = next(item for item in records if item.mnemonic == "LDP")

        self.assertEqual(sve_add.form, "ADD (ZD.S, PG/M, ZN.S, ZM.S)")
        self.assertEqual(sve_add.isa, ["SVE"])
        self.assertIn("Description", sve_add.description)
        self.assertIn("predicated SVE lanes", sve_add.description["Description"])
        self.assertEqual(sve_add.metadata["section"], "SVE Instructions")
        self.assertIn("ADD (vectors, predicated)", sve_add.aliases)

        self.assertEqual(ldp.isa, ["A64"])
        self.assertEqual(ldp.metadata["section"], "Base Instructions")
        self.assertEqual(ldp.summary, "Load pair of registers.")


class TuiSearchFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = TemporaryDirectory()
        tmp_path = Path(cls._tmpdir.name)
        cls._catalog = build_fixture_catalog()
        cls._catalog_path = tmp_path / "catalog.msgpack"
        cls._db_path = tmp_path / "catalog.db"
        save_catalog(cls._catalog, path=cls._catalog_path)
        build_sqlite(cls._catalog, path=cls._db_path)
        cls._conn = open_db(path=cls._db_path)

    @classmethod
    def tearDownClass(cls):
        cls._conn.close()
        cls._tmpdir.cleanup()

    def test_fts_search_respects_sub_isa_filter(self):
        results = _fts_search(
            self._conn,
            "_mm256_add_epi32",
            {"AVX"},
            {"AVX2"},
            limit=10,
        )
        self.assertIn("_mm256_add_epi32", [result.key for result in results])

        filtered = _fts_search(
            self._conn,
            "_mm256_add_epi32",
            {"AVX"},
            {"F16C"},
            limit=10,
        )
        self.assertNotIn("_mm256_add_epi32", [result.key for result in filtered])

    def test_fts_search_normalizes_avx512_family_and_sub_isa(self):
        results = _fts_search(
            self._conn,
            "_mm512_maskz_expandloadu_epi32",
            {"AVX-512"},
            {"AVX512F"},
            limit=10,
        )
        self.assertIn("_mm512_maskz_expandloadu_epi32", [result.key for result in results])

        filtered = _fts_search(
            self._conn,
            "_mm512_maskz_expandloadu_epi32",
            {"AVX-512"},
            {"AVX512VL"},
            limit=10,
        )
        self.assertNotIn("_mm512_maskz_expandloadu_epi32", [result.key for result in filtered])

    def test_fts_search_supports_arm_family_and_sub_isa(self):
        results = _fts_search(
            self._conn,
            "svadd",
            {"Arm"},
            {"SVE"},
            limit=10,
        )
        self.assertIn("svadd_s32_z", [result.key for result in results])
        self.assertIn("svadd", [result.key for result in results])

        filtered = _fts_search(
            self._conn,
            "svadd",
            {"Arm"},
            {"NEON"},
            limit=10,
        )
        self.assertNotIn("svadd_s32_z", [result.key for result in filtered])

    def test_fts_search_supports_riscv_family_and_sub_isa(self):
        results = _fts_search(
            self._conn,
            "vadd.vv",
            {"RISC-V"},
            {"V"},
            limit=10,
        )
        keys = [result.key for result in results]
        self.assertIn("__riscv_vadd_vv_i32m1", keys)
        self.assertIn("riscv:vadd.vv", keys)

        filtered = _fts_search(
            self._conn,
            "vaesdf",
            {"RISC-V"},
            {"V"},
            limit=10,
        )
        self.assertNotIn("__riscv_vaesdf_vv_u32m1", [result.key for result in filtered])

    def test_fts_search_sub_returns_riscv_results(self):
        results = _fts_search(
            self._conn,
            "sub",
            {"RISC-V"},
            {"V"},
            limit=10,
        )
        keys = [result.key for result in results]
        self.assertIn("__riscv_vsub_vv_i32m1", keys)
        self.assertIn("riscv:vsub.vv", keys)

    def test_fts_search_load_returns_riscv_results(self):
        results = _fts_search(
            self._conn,
            "load",
            {"RISC-V"},
            {"V"},
            limit=10,
        )
        keys = [result.key for result in results]
        self.assertIn("__riscv_vle32_v_i32m1", keys)
        self.assertIn("riscv:vle32.v", keys)

    def test_fts_search_add_returns_arm_results(self):
        results = _fts_search(
            self._conn,
            "add",
            {"Arm"},
            {"NEON", "SVE", "SVE2"},
            limit=10,
        )
        keys = [result.key for result in results]
        self.assertIn("vaddq_u8", keys)
        self.assertIn("arm:add (vd.16b, vn.16b, vm.16b)", keys)

    def test_normalize_sub_isa_selection_drops_stale_family_entries(self):
        normalized = _normalize_sub_isa_selection(
            {"RISC-V"},
            {"SSE2", "AVX2", "V"},
            {"RISC-V": ["V", "ZV"], "SSE": ["SSE2"], "AVX": ["AVX2"]},
        )
        self.assertEqual(normalized, {"V"})

    def test_fts_search_add_can_reach_riscv_after_many_x86_matches(self):
        noisy_catalog = build_fixture_catalog()
        for index in range(250):
            noisy_catalog.instructions.append(
                InstructionRecord(
                    mnemonic=f"ADDNOISE{index}",
                    form=f"ADDNOISE{index} (R32, R32)",
                    summary="Add noisy scalar operands.",
                    architecture="x86",
                    isa=["I86"],
                )
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.db"
            build_sqlite(noisy_catalog, path=db_path)
            conn = open_db(path=db_path)
            try:
                results = _fts_search(
                    conn,
                    "add",
                    {"RISC-V"},
                    {"V", "ZV"},
                    limit=10,
                )
            finally:
                conn.close()
        self.assertIn("riscv:vadd.vv", [result.key for result in results])


if __name__ == "__main__":
    unittest.main()
