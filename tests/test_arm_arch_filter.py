"""Tests for the arm_arch facet: the shared rule, FilterState and the TUI preset.

Every test here fails on e47f97e, which has no arm_arch filter: the
Arm32 preset lists A64 rows.
"""

from __future__ import annotations

import asyncio
import unittest

from simdref.filters import (
    FilterState,
    arm_arch_visible,
    arm_archs_from_db,
    record_arm_arch,
)
from simdref.models import IntrinsicRecord

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - Textual is a runtime dep
    HAS_TEXTUAL = False

from simdref.storage import CATALOG_PATH, SQLITE_PATH, sqlite_schema_is_current


CATALOG_READY = CATALOG_PATH.exists() and SQLITE_PATH.exists() and sqlite_schema_is_current()


def _arm(name: str = "vadd_f32", supported: str = "A64") -> IntrinsicRecord:
    metadata: dict[str, str] = {"classification": "unclassified"}
    if supported:
        metadata["supported_architectures"] = supported
    return IntrinsicRecord(
        name=name,
        signature=f"float32x2_t {name}(float32x2_t, float32x2_t)",
        description="An Arm intrinsic.",
        header="arm_neon.h",
        url="",
        architecture="arm",
        isa=["NEON"],
        metadata=metadata,
    )


def _x86() -> IntrinsicRecord:
    return IntrinsicRecord(
        name="_mm_zyx_ps",
        signature="__m128 _mm_zyx_ps(__m128, __m128)",
        description="An x86 intrinsic.",
        header="immintrin.h",
        url="",
        architecture="x86",
        isa=["AVX"],
        metadata={},
    )


class ArmArchVisibleTests(unittest.TestCase):
    def test_no_choice_passes_every_row(self):
        self.assertTrue(arm_arch_visible(["NEON"], None, None))
        self.assertTrue(arm_arch_visible(["NEON"], "A64", frozenset()))

    def test_non_arm_rows_are_never_filtered(self):
        self.assertTrue(arm_arch_visible(["AVX"], None, frozenset({"A32", "BOTH"})))

    def test_arm_row_without_bucket_hides(self):
        self.assertFalse(arm_arch_visible(["NEON"], None, frozenset({"A32", "BOTH"})))

    def test_bucket_must_be_in_the_chosen_set(self):
        self.assertFalse(arm_arch_visible(["NEON"], "A64", frozenset({"A32", "BOTH"})))
        self.assertTrue(arm_arch_visible(["NEON"], "BOTH", frozenset({"A32", "BOTH"})))
        self.assertTrue(arm_arch_visible(["NEON"], "A64", frozenset({"A64", "BOTH"})))


class RecordArmArchTests(unittest.TestCase):
    def test_supported_architectures_drives_the_bucket(self):
        self.assertEqual(record_arm_arch(_arm(supported="A64")), "A64")
        self.assertEqual(record_arm_arch(_arm(supported="A32, A64")), "BOTH")
        self.assertEqual(record_arm_arch(_arm(supported="v7")), "A32")

    def test_mve_only_rows_infer_a32(self):
        record = IntrinsicRecord(
            name="vzyxq",
            signature="int vzyxq(int)",
            description="An MVE intrinsic.",
            header="arm_mve.h",
            url="",
            architecture="arm",
            isa=["MVE"],
            metadata={},
        )
        self.assertEqual(record_arm_arch(record), "A32")

    def test_no_metadata_gives_no_bucket(self):
        self.assertIsNone(record_arm_arch(_arm(supported="")))
        self.assertIsNone(record_arm_arch(_x86()))

    def test_dict_payloads_work_too(self):
        self.assertEqual(
            record_arm_arch({"isa": ["NEON"], "metadata": {"supported_architectures": "A64"}}),
            "A64",
        )


class FilterStateArmArchTests(unittest.TestCase):
    def test_arm32_state_hides_a64_rows(self):
        state = FilterState(frozenset({"Arm"}), "", frozenset({"A32", "BOTH"}))
        self.assertFalse(state.matches(_arm(supported="A64")))
        self.assertTrue(state.matches(_arm(supported="A32, A64")))

    def test_arm64_state_hides_a32_rows(self):
        state = FilterState(frozenset({"Arm"}), "", frozenset({"A64", "BOTH"}))
        self.assertFalse(state.matches(_arm(supported="v7")))
        self.assertTrue(state.matches(_arm(supported="A64")))

    def test_rows_without_a_bucket_hide_while_the_facet_is_on(self):
        state = FilterState(frozenset({"Arm"}), "", frozenset({"A64", "BOTH"}))
        self.assertFalse(state.matches(_arm(supported="")))

    def test_no_facet_keeps_every_row(self):
        self.assertTrue(FilterState().matches(_arm(supported="A64")))

    def test_non_arm_rows_pass_the_facet(self):
        state = FilterState(frozenset({"AVX"}), "", frozenset({"A32", "BOTH"}))
        self.assertTrue(state.matches(_x86()))


@unittest.skipUnless(HAS_TEXTUAL and CATALOG_READY, "textual + catalog required")
class TuiArmArchPresetTests(unittest.TestCase):
    def test_arm32_preset_hides_a64_only_rows(self):
        from simdref.tui import SimdrefApp

        async def scenario():
            app = SimdrefApp(initial_preset="arm32", initial_query="vadd")
            async with app.run_test(size=(160, 45)) as pilot:
                for _ in range(6):
                    await pilot.pause()
                self.assertEqual(app._arm_arch, frozenset({"A32", "BOTH"}))
                results = list(app._current_results)
                if not results:
                    self.skipTest("no Arm rows for 'vadd' in this catalog")
                for result in results:
                    self.assertIn(app._arm_archs.get(result.key), {"A32", "BOTH"})

        asyncio.run(scenario())

    def test_facet_map_covers_the_catalog_buckets(self):
        import contextlib

        from simdref.storage import open_db

        with contextlib.closing(open_db()) as conn:
            buckets = arm_archs_from_db(conn)
        self.assertTrue(buckets)
        self.assertTrue(set(buckets.values()) <= {"A32", "A64", "BOTH"})
