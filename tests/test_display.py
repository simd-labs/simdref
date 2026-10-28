"""Tests for simdref.display helpers."""

import unittest
from types import SimpleNamespace

from simdref.display import (
    _CODE_SECTION_LANG,
    canonical_url,
    console,
    display_architecture,
    display_isa,
    isa_family,
    isa_to_sub_isa,
    display_instruction_form,
    instruction_query_text,
    instruction_variant_items,
    isa_sort_key,
    isa_visible,
    measurement_rows,
    normalize_instruction_query,
    natural_query_sort_key,
    perf_panel_title,
    render_instruction,
    split_perf_rows,
    uarch_sort_key,
)


class IsaToSubIsaTests(unittest.TestCase):
    def test_sse_variants_do_not_collapse_to_sse(self):
        # Longest-match rule: SSE2/SSE3/SSE4.1/SSE4.2 must resolve to
        # themselves, not to the SSE prefix that appears first in the list.
        self.assertEqual(isa_to_sub_isa("SSE"), "SSE")
        self.assertEqual(isa_to_sub_isa("SSE2"), "SSE2")
        self.assertEqual(isa_to_sub_isa("SSE3"), "SSE3")
        self.assertEqual(isa_to_sub_isa("SSSE3"), "SSSE3")
        self.assertEqual(isa_to_sub_isa("SSE4.1"), "SSE4.1")
        self.assertEqual(isa_to_sub_isa("SSE4.2"), "SSE4.2")
        self.assertEqual(isa_to_sub_isa("SSE4_1"), "SSE4.1")
        self.assertEqual(isa_to_sub_isa("SSE4_2"), "SSE4.2")

    def test_avx512_variants_prefer_exact_match(self):
        self.assertEqual(isa_to_sub_isa("AVX512F"), "AVX512F")
        self.assertEqual(isa_to_sub_isa("AVX512VL"), "AVX512VL")
        self.assertEqual(isa_to_sub_isa("AVX_VNNI"), "AVX_VNNI")


from simdref.models import Catalog, InstructionRecord, IntrinsicRecord


class DisplayISATests(unittest.TestCase):
    def test_display_isa_avx512(self):
        self.assertIn("AVX512", display_isa(["AVX512F"]))

    def test_display_isa_avx10(self):
        result = display_isa(["AVX10_2"])
        self.assertIn("AVX10", result)

    def test_display_isa_amx(self):
        result = display_isa(["AMX_TILE"])
        self.assertEqual(result, "AMX-TILE")

    def test_display_isa_strips_width_suffix(self):
        result = display_isa(["AVX512F_128"])
        self.assertIn("AVX512", result)
        self.assertNotIn("128", result)

    def test_display_isa_empty(self):
        self.assertEqual(display_isa([]), "-")

    def test_display_isa_arm_tokens(self):
        self.assertEqual(display_isa(["advsimd", "SVE2p1"]), "NEON, SVE2")

    def test_display_isa_riscv_tokens(self):
        self.assertEqual(display_isa(["V", "Zve32x", "Zvkned"]), "V, Zve32x, Zvkned")

    def test_display_isa_deduplicates(self):
        result = display_isa(["SSE", "SSE"])
        self.assertEqual(result, "SSE")

    def test_isa_sort_key_chronological(self):
        sse = isa_sort_key(["SSE"])
        avx = isa_sort_key(["AVX"])
        avx512 = isa_sort_key(["AVX512F"])
        self.assertLess(sse[:2], avx[:2])
        self.assertLess(avx[:2], avx512[:2])

    def test_isa_visible_hides_apx(self):
        self.assertFalse(isa_visible(["APX_F"]))

    def test_isa_visible_hides_fp16_by_default(self):
        self.assertFalse(isa_visible(["AVX512FP16"]))
        self.assertTrue(isa_visible(["AVX512FP16"], show_fp16=True))

    def test_isa_family_arm(self):
        self.assertEqual(isa_family("SVE2"), "Arm")
        self.assertEqual(isa_family("NEON"), "Arm")
        self.assertEqual(isa_family("Zvkned"), "RISC-V")

    def test_display_architecture(self):
        self.assertEqual(display_architecture("arm"), "Arm")
        self.assertEqual(display_architecture("x86"), "x86")
        self.assertEqual(display_architecture("riscv"), "RISC-V")

    def test_acle_operation_uses_code_highlighting(self):
        self.assertEqual(_CODE_SECTION_LANG["ACLE Operation"], "asm")


class DisplayInstructionTests(unittest.TestCase):
    def test_instruction_query_text(self):
        item = SimpleNamespace(mnemonic="VADDPS", form="VADDPS (YMM, YMM, YMM)")
        result = instruction_query_text(item)
        self.assertEqual(result, "VADDPS YMM YMM YMM")

    def test_display_instruction_form_strips_evex(self):
        result = display_instruction_form("{evex} VADDPS (YMM, YMM, YMM)")
        self.assertNotIn("evex", result.lower())
        self.assertIn("VADDPS", result)

    def test_normalize_instruction_query(self):
        result = normalize_instruction_query("VADDPS (YMM, YMM, YMM)")
        self.assertEqual(result, "vaddps ymm ymm ymm")

    def test_natural_query_sort_key_orders_numbers(self):
        k8 = natural_query_sort_key("M8")
        k16 = natural_query_sort_key("M16")
        self.assertLess(k8, k16)

    def test_instruction_variant_items_sorted(self):
        items = [
            SimpleNamespace(mnemonic="ADD", form="ADD (R16)", isa=["I86"], key="ADD (R16)"),
            SimpleNamespace(mnemonic="ADD", form="ADD (R8l)", isa=["I86"], key="ADD (R8l)"),
        ]
        sorted_items = instruction_variant_items(items)
        keys = [item.key for item in sorted_items]
        self.assertLess(keys.index("ADD (R8l)"), keys.index("ADD (R16)"))


class DisplayMiscTests(unittest.TestCase):
    def test_canonical_url_adds_https(self):
        self.assertEqual(canonical_url("uops.info/table.html"), "https://www.uops.info/table.html")

    def test_canonical_url_preserves_https(self):
        self.assertEqual(canonical_url("https://example.com"), "https://example.com")

    def test_canonical_url_empty(self):
        self.assertEqual(canonical_url(""), "")

    def test_uarch_sort_key_known(self):
        skl = uarch_sort_key("SKL")
        hsw = uarch_sort_key("HSW")
        self.assertLess(skl, hsw)  # SKL (2015) before HSW (2013) in order list

    def test_uarch_sort_key_unknown(self):
        _, name = uarch_sort_key("UNKNOWN")
        self.assertEqual(name, "UNKNOWN")

    def test_render_instruction_includes_x86_description_sections(self):
        instruction = InstructionRecord(
            mnemonic="VPEXPANDD",
            form="VPEXPANDD (ZMM{k}{z}, M512)",
            summary="Expand packed 32-bit integers from memory.",
            isa=["AVX512F"],
            linked_intrinsics=["_mm512_maskz_expandloadu_epi32"],
            description={
                "Description": "Expand packed integers under writemask control.",
                "Operation": "FOR j := 0 TO KL-1",
                "SIMD Floating-Point Exceptions": "None.",
            },
            arch_details={
                "SKL": {
                    "measurement": {"TP_loop": "1.0", "uops": "2"},
                    "latencies": [{"cycles": "6"}],
                    "doc": {},
                    "iaca": [],
                }
            },
        )
        catalog = Catalog(
            intrinsics=[
                IntrinsicRecord(
                    name="_mm512_maskz_expandloadu_epi32",
                    signature="__m512i _mm512_maskz_expandloadu_epi32(__mmask16 k, void const* mem_addr)",
                    description="Mask-zero expand load.",
                    header="immintrin.h",
                    isa=["AVX512F"],
                    instructions=[instruction.key],
                    instruction_refs=[
                        {
                            "key": instruction.db_key,
                            "display_key": instruction.key,
                            "architecture": "x86",
                        }
                    ],
                )
            ],
            instructions=[instruction],
            sources=[],
            generated_at="2026-01-01T00:00:00Z",
        )
        with console.capture() as capture:
            render_instruction(catalog, instruction, short=False, full=True)
        output = capture.get()
        self.assertIn("Description", output)
        self.assertIn("Operation", output)
        self.assertIn("SIMD Floating-Point Exceptions", output)
        self.assertIn("instruction to intrinsic mapping", output)
        self.assertIn("Mask-zero expand load.", output)


class PerfPanelTests(unittest.TestCase):
    def test_perf_panel_title_named_kinds(self):
        self.assertEqual(perf_panel_title("measured"), "perf (measured)")
        self.assertEqual(perf_panel_title("modeled"), "perf (modeled)")

    def test_split_perf_rows_orders_measured_before_modeled(self):
        rows = [
            {"uarch": "neoverse-n1", "source": "modeled"},
            {"uarch": "SKL", "source": "measured"},
            {"uarch": "apple-m1", "source": "modeled"},
        ]
        groups = split_perf_rows(rows)
        self.assertEqual([kind for kind, _ in groups], ["measured", "modeled"])
        self.assertEqual(len(groups[1][1]), 2)

    def test_split_perf_rows_treats_missing_source_as_measured(self):
        rows = [{"uarch": "SKL"}]
        groups = split_perf_rows(rows)
        self.assertEqual(groups, [("measured", rows)])

    def test_measurement_rows_carries_source_kind(self):
        item = InstructionRecord(
            mnemonic="FMLA",
            form="FMLA V0.4S, V1.4S, V2.4S",
            summary="",
            architecture="arm",
            isa=[],
            arch_details={
                "neoverse-n1": {
                    "source_kind": "modeled",
                    "measurement": {"TP_loop": "0.5", "uops": "1", "ports": "0.50*V0"},
                    "latencies": [{"cycles": "3"}],
                },
                "SKL": {
                    "source_kind": "measured",
                    "measurement": {"TP_loop": "1.0"},
                    "latencies": [{"cycles": "4"}],
                },
            },
        )
        rows = measurement_rows(item)
        by_uarch = {r["uarch"]: r for r in rows}
        self.assertEqual(by_uarch["neoverse-n1"]["source"], "modeled")
        self.assertEqual(by_uarch["SKL"]["source"], "measured")

    def test_render_instruction_emits_separate_measured_and_modeled_panels(self):
        instruction = InstructionRecord(
            mnemonic="FMLA",
            form="FMLA V0.4S, V1.4S, V2.4S",
            summary="Fused multiply-add",
            architecture="arm",
            isa=["NEON"],
            arch_details={
                "neoverse-n1": {
                    "source_kind": "modeled",
                    "measurement": {
                        "TP_loop": "0.5",
                        "uops": "1",
                        "ports": "0.50*V0 0.50*V1",
                    },
                    "latencies": [{"cycles": "4"}],
                },
                "SKL": {
                    "source_kind": "measured",
                    "measurement": {"TP_loop": "1.0", "uops": "1"},
                    "latencies": [{"cycles": "4"}],
                },
            },
        )
        catalog = Catalog(
            intrinsics=[],
            instructions=[instruction],
            sources=[],
            generated_at="2026-01-01T00:00:00Z",
        )
        with console.capture() as capture:
            render_instruction(catalog, instruction, short=True)
        output = capture.get()
        # Two separate panels, one per source kind.
        self.assertIn("perf (measured)", output)
        self.assertIn("perf (modeled)", output)
        # No ``src`` column inside the tables — the panel title carries
        # the provenance label now.
        self.assertNotIn(" src ", output)

    def test_render_instruction_modeled_only_has_no_measured_panel(self):
        instruction = InstructionRecord(
            mnemonic="FADD",
            form="FADD (D0, D1, D2)",
            summary="",
            architecture="arm",
            isa=["NEON"],
            arch_details={
                "neoverse-n1": {
                    "source_kind": "modeled",
                    "measurement": {"TP_loop": "0.5", "uops": "1"},
                    "latencies": [{"cycles": "2"}],
                }
            },
        )
        catalog = Catalog(
            intrinsics=[],
            instructions=[instruction],
            sources=[],
            generated_at="2026-01-01T00:00:00Z",
        )
        with console.capture() as capture:
            render_instruction(catalog, instruction, short=True)
        output = capture.get()
        self.assertIn("perf (modeled)", output)
        self.assertNotIn("perf (measured)", output)


class InstructionMarkdownTests(unittest.TestCase):
    """Parse the hover Markdown with CommonMark + tables, as editors render it."""

    def setUp(self):
        from markdown_it import MarkdownIt

        from simdref.display import instruction_markdown

        item = InstructionRecord(
            mnemonic="VADDPS",
            form="VADDPS (YMM, YMM, YMM)",
            summary="Add packed single *precision* values",
            architecture="x86",
            isa=["AVX"],
            metadata={"extension": "AVX", "url": "https://example.com/vaddps"},
            description={
                "Description": "First line.\nSecond line with a_b and [x].",
                "Operation": "IF k1[j] OR *no writemask*\n    THEN DEST := 0",
                "Intrinsic Equivalents": "VADDPS __m256 _mm256_add_ps (__m256 a, __m256 b);",
                "Protected Mode Exceptions": "#UD If the LOCK prefix is used.",
            },
            arch_details={
                "SKL": {
                    "measurement": {"TP_loop": "0.5", "ports": "1*p01+1*p23"},
                    "latencies": [{"cycles": "4"}],
                }
            },
        )
        self.text = instruction_markdown(item)
        self.tokens = MarkdownIt("commonmark").enable("table").parse(self.text)

    def inline_text(self, kind):
        return [t.content for t in self.tokens if t.type == "inline" and kind in t.content]

    def test_code_sections_are_fenced(self):
        fences = {t.info: t.content for t in self.tokens if t.type == "fence"}
        self.assertIn("*no writemask*", fences["asm"])
        self.assertIn("_mm256_add_ps", fences["c"])

    def test_description_lines_keep_line_breaks(self):
        para = next(t for t in self.tokens if t.type == "inline" and "First line" in t.content)
        self.assertIn("hardbreak", [c.type for c in para.children])
        self.assertIn("a_b and [x].", "".join(c.content for c in para.children))

    def test_text_is_not_emphasis_or_heading(self):
        types = {c.type for t in self.tokens if t.type == "inline" for c in t.children}
        self.assertNotIn("em_open", types)
        headings = [
            self.tokens[i + 1].content
            for i, t in enumerate(self.tokens)
            if t.type == "heading_open"
        ]
        self.assertNotIn("UD If the LOCK prefix is used.", headings)

    def test_perf_ports_cell_is_literal(self):
        cells = [
            "".join(c.content for c in t.children)
            for t in self.tokens
            if t.type == "inline" and "p01" in t.content
        ]
        self.assertEqual(cells, ["1*p01+1*p23"])

    def test_isa_row_shown_once(self):
        self.assertEqual(self.text.count("**isa.**"), 1)


if __name__ == "__main__":
    unittest.main()
