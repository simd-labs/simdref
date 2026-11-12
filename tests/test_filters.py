"""Tests for simdref.filters — the shared filter spec."""

import json
import sqlite3
import unittest
from types import SimpleNamespace

from simdref.filters import (
    ARCH_PRESETS,
    ARM32_SUBS,
    ARM64_SUBS,
    CategorySpec,
    FilterSpec,
    PRESET_ONLY_FAMILIES,
    PresetSpec,
    X86_64_V4_SUBS,
    build_filter_spec,
    load_categories_from_db,
)


class FilterSpecJsonTests(unittest.TestCase):
    def test_to_json_roundtrips_through_json_module(self):
        spec = FilterSpec()
        spec.categories = [CategorySpec("SSE", "Arithmetic", "", 42)]
        payload = spec.to_json()
        # Must be JSON-serialisable for the web export.
        text = json.dumps(payload)
        recovered = json.loads(text)
        self.assertEqual(recovered["default_enabled"], list(spec.default_enabled))
        self.assertEqual(recovered["categories"][0]["family"], "SSE")
        self.assertEqual(recovered["categories"][0]["count"], 42)

    def test_default_subs_are_sorted_for_stable_output(self):
        spec = FilterSpec()
        payload = spec.to_json()
        # Stable ordering matters for diff-friendly static bundles.
        self.assertEqual(payload["default_subs"]["SSE"], sorted(payload["default_subs"]["SSE"]))


class FilterSpecMatchesTests(unittest.TestCase):
    def test_matches_by_family_uses_isa_family_mapping(self):
        spec = FilterSpec()
        rec_arm = SimpleNamespace(isa=["SVE2"], category="")
        rec_x86 = SimpleNamespace(isa=["AVX512F"], category="")
        self.assertTrue(spec.matches(rec_arm, enabled_families={"Arm"}))
        self.assertFalse(spec.matches(rec_arm, enabled_families={"SSE"}))
        self.assertTrue(spec.matches(rec_x86, enabled_families={"AVX-512"}))

    def test_matches_by_category_uses_record_category(self):
        spec = FilterSpec()
        rec = SimpleNamespace(isa=["SSE"], category="Arithmetic")
        self.assertTrue(spec.matches(rec, enabled_categories={"Arithmetic"}))
        self.assertFalse(spec.matches(rec, enabled_categories={"Swizzle"}))

    def test_matches_falls_back_to_metadata_category(self):
        spec = FilterSpec()
        rec = SimpleNamespace(isa=["AVX"], metadata={"category": "shift"})
        self.assertTrue(spec.matches(rec, enabled_categories={"shift"}))

    def test_none_filters_always_match(self):
        spec = FilterSpec()
        rec = SimpleNamespace(isa=[], category="")
        self.assertTrue(spec.matches(rec))


class FilterSpecSqlPredicateTests(unittest.TestCase):
    def test_sql_predicate_empty_filters_returns_empty_clause(self):
        spec = FilterSpec()
        clause, binds = spec.sql_predicate("intrinsics_data")
        self.assertEqual(clause, "")
        self.assertEqual(binds, [])

    def test_sql_predicate_family_expands_to_sub_tokens(self):
        spec = FilterSpec()
        clause, binds = spec.sql_predicate("intrinsics_data", enabled_families=["Arm"])
        self.assertIn("intrinsics_data.isa LIKE ?", clause)
        # Family "Arm" should expand to its sub tokens plus the family label.
        self.assertTrue(any("NEON" in b or "SVE" in b or "Arm" in b for b in binds))

    def test_sql_predicate_category_applies_to_both_tables(self):
        # Schema v10 adds an indexed `category` column to instructions_data;
        # the predicate must push category filters into SQL for both tables.
        spec = FilterSpec()
        _, binds_intr = spec.sql_predicate("intrinsics_data", enabled_categories=["Arithmetic"])
        _, binds_instr = spec.sql_predicate("instructions_data", enabled_categories=["Arithmetic"])
        self.assertIn("Arithmetic", binds_intr)
        self.assertIn("Arithmetic", binds_instr)

    def test_sql_predicate_category_skipped_for_unrelated_tables(self):
        spec = FilterSpec()
        sql, binds = spec.sql_predicate("some_other_table", enabled_categories=["Arithmetic"])
        self.assertNotIn("Arithmetic", binds)
        self.assertNotIn("category", sql)

    def test_sql_predicate_runs_against_real_sqlite(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE intrinsics_data (isa TEXT, category TEXT)")
        conn.executemany(
            "INSERT INTO intrinsics_data VALUES (?, ?)",
            [("AVX512F", "Arithmetic"), ("NEON", "Logical"), ("SSE2", "Arithmetic")],
        )
        spec = FilterSpec()
        clause, binds = spec.sql_predicate(
            "intrinsics_data",
            enabled_families=["Arm"],
            enabled_categories=["Logical"],
        )
        sql = "SELECT COUNT(*) FROM intrinsics_data"
        if clause:
            sql += f" WHERE {clause}"
        count = conn.execute(sql, binds).fetchone()[0]
        self.assertEqual(count, 1)
        conn.close()


class CategoryAggregationTests(unittest.TestCase):
    def _make_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE intrinsics_data (category TEXT, subcategory TEXT, isa TEXT)")
        conn.executemany(
            "INSERT INTO intrinsics_data VALUES (?, ?, ?)",
            [
                ("Arithmetic", "Add", "AVX512F"),
                ("Arithmetic", "Add", "AVX512F"),
                ("Logical", "", "NEON"),
                ("", "", "SSE"),  # no category: excluded
            ],
        )
        return conn

    def test_load_categories_aggregates_and_assigns_family(self):
        conn = self._make_conn()
        cats = load_categories_from_db(conn)
        families = {c.family for c in cats}
        self.assertIn("AVX-512", families)
        self.assertIn("Arm", families)
        arith = next(c for c in cats if c.category == "Arithmetic")
        self.assertEqual(arith.count, 2)
        conn.close()

    def test_build_filter_spec_without_conn_has_empty_categories(self):
        spec = build_filter_spec(None)
        self.assertEqual(spec.categories, [])


class ArchPresetsTests(unittest.TestCase):
    def test_all_expected_presets_present(self):
        expected = {"default", "intel", "arm32", "arm64", "riscv", "cuda", "none", "all"}
        self.assertEqual(expected, set(ARCH_PRESETS.keys()))

    def test_cuda_preset_enables_cuda_and_ptx(self):
        preset = ARCH_PRESETS["cuda"]
        self.assertEqual(preset.families, frozenset({"CUDA", "PTX"}))
        self.assertEqual(preset.subs, frozenset())
        self.assertIsNone(preset.arm_arch)
        self.assertEqual(preset.kind, frozenset({"intrinsic", "instruction"}))

    def test_intel_preset_is_strict_x86_64_v4(self):
        preset = ARCH_PRESETS["intel"]
        self.assertEqual(preset.families, frozenset({"x86", "SSE", "AVX", "AVX-512"}))
        self.assertEqual(set(preset.subs), X86_64_V4_SUBS)
        # v2/v3/v4 baseline must all be present.
        for sub in (
            "SSE",
            "SSE2",
            "SSE3",
            "SSSE3",
            "SSE4.1",
            "SSE4.2",
            "AVX",
            "AVX2",
            "F16C",
            "FMA",
            "BMI1",
            "BMI2",
            "POPCNT",
            "LZCNT",
            "AVX512F",
            "AVX512VL",
            "AVX512BW",
            "AVX512DQ",
            "AVX512CD",
        ):
            self.assertIn(sub, preset.subs, f"Intel preset missing {sub}")
        # Intel preset must NOT enable AMX/APX/AVX10/SVML/VAES/GFNI/AVX-512 _VNNI etc.
        forbidden = {
            "AMX-TILE",
            "AVX512_VNNI",
            "AVX512_FP16",
            "AVX512_BF16",
            "AVX512_VBMI",
            "VAES",
            "GFNI",
            "AVX512_VP2INTERSECT",
        }
        self.assertFalse(preset.subs & frozenset(forbidden))
        self.assertEqual(preset.kind, frozenset({"intrinsic", "instruction"}))
        self.assertIsNone(preset.arm_arch)

    def test_arm32_preset_restricts_to_a32_and_both(self):
        preset = ARCH_PRESETS["arm32"]
        self.assertEqual(preset.families, frozenset({"Arm"}))
        self.assertEqual(set(preset.subs), ARM32_SUBS)
        self.assertEqual(preset.arm_arch, frozenset({"A32", "BOTH"}))
        self.assertEqual(preset.kind, frozenset({"intrinsic", "instruction"}))

    def test_arm64_preset_restricts_to_a64_and_both(self):
        preset = ARCH_PRESETS["arm64"]
        self.assertEqual(preset.families, frozenset({"Arm"}))
        self.assertEqual(set(preset.subs), ARM64_SUBS)
        self.assertEqual(preset.arm_arch, frozenset({"A64", "BOTH"}))
        self.assertEqual(preset.kind, frozenset({"intrinsic", "instruction"}))

    def test_riscv_preset_includes_instructions(self):
        preset = ARCH_PRESETS["riscv"]
        self.assertEqual(preset.families, frozenset({"RISC-V"}))
        self.assertEqual(preset.kind, frozenset({"intrinsic", "instruction"}))

    def test_default_preset_is_x86_only_and_intrinsic_only(self):
        preset = ARCH_PRESETS["default"]
        self.assertEqual(preset.kind, frozenset({"intrinsic"}))
        self.assertEqual(preset.families, frozenset({"SSE", "AVX", "AVX-512"}))
        self.assertFalse({"Arm", "RISC-V"} & preset.families)

    def test_chip_families_exclude_preset_only_families(self):
        spec = FilterSpec()
        expected = tuple(n for n in spec.family_order if n not in PRESET_ONLY_FAMILIES)
        self.assertEqual(spec.chip_families, expected)
        self.assertNotIn("CUDA", spec.chip_families)
        self.assertNotIn("PTX", spec.chip_families)
        self.assertIn("CUDA", spec.family_order)
        self.assertIn("PTX", spec.family_order)

    def test_all_preset_enables_instruction_kind(self):
        preset = ARCH_PRESETS["all"]
        self.assertIn("instruction", preset.kind)
        self.assertIn("intrinsic", preset.kind)
        self.assertIsNone(preset.arm_arch)

    def test_none_preset_empty(self):
        preset = ARCH_PRESETS["none"]
        self.assertEqual(preset.families, frozenset())
        self.assertEqual(preset.subs, frozenset())
        self.assertEqual(preset.kind, frozenset())

    def test_presets_roundtrip_through_filter_spec_json(self):
        spec = FilterSpec()
        payload = spec.to_json()
        self.assertIn("presets", payload)
        self.assertEqual(set(payload["presets"].keys()), set(ARCH_PRESETS.keys()))
        intel = payload["presets"]["intel"]
        self.assertEqual(set(intel["subs"]), X86_64_V4_SUBS)
        self.assertIn("arm_arch_values", payload)
        self.assertIn("A64", payload["arm_arch_values"])


class ArmArchSqlPredicateTests(unittest.TestCase):
    def test_arm_arch_pushed_into_sql_where(self):
        spec = FilterSpec()
        clause, binds = spec.sql_predicate("intrinsics_data", enabled_arm_arch=["A64", "BOTH"])
        self.assertIn("arm_arch IN", clause)
        self.assertEqual(set(binds), {"A64", "BOTH"})

    def test_arm_arch_ignored_for_instructions_table(self):
        spec = FilterSpec()
        clause, binds = spec.sql_predicate("instructions_data", enabled_arm_arch=["A64"])
        self.assertNotIn("arm_arch", clause)
        self.assertEqual(binds, [])


if __name__ == "__main__":
    unittest.main()
