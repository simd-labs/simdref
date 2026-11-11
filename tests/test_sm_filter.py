"""Tests for the sm level filter: FilterState and the sm predicate.

The FilterState tests fail on a5095c6, which has no FilterState, no
suffixed floor parse and no --sm option.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from simdref.filters import (
    FilterSpec,
    FilterState,
    record_sm_floor,
    runs_on_sm,
    sm_floors_from_db,
    sm_levels,
    sm_number,
)
from simdref.models import Catalog, InstructionRecord, IntrinsicRecord


def test_empty_sides_pass() -> None:
    assert runs_on_sm("", "sm_80")
    assert runs_on_sm("sm_70", "")
    assert runs_on_sm("", "")
    assert runs_on_sm(None, None)


def test_floor_compare() -> None:
    assert runs_on_sm("sm_70", "sm_80")
    assert runs_on_sm("sm_70", "sm_70")
    assert not runs_on_sm("sm_90", "sm_80")


def test_plain_number_choice() -> None:
    assert runs_on_sm("sm_30", "80")
    assert not runs_on_sm("sm_90", "20")


def _cuda(name: str = "__zyx_sync_v1", min_sm: str = "") -> IntrinsicRecord:
    metadata: dict[str, str] = {"classification": "unclassified"}
    if min_sm:
        metadata["min_sm"] = min_sm
    return IntrinsicRecord(
        name=name,
        signature=f"int {name}(int)",
        description="A zyx intrinsic.",
        header="__clang_cuda_intrinsics.h",
        url="",
        architecture="cuda",
        isa=["CUDA"],
        metadata=metadata,
    )


def _ptx(sm: str = "") -> InstructionRecord:
    return InstructionRecord(
        mnemonic="zyx.sync.b32",
        form="d, p, a, b, membermask",
        summary="zyx.sync.b32 instruction.",
        architecture="ptx",
        isa=["PTX"],
        metadata={"sm": sm, "ptx_isa": "6.0"},
    )


class SmNumberTests(unittest.TestCase):
    def test_suffix_letter_stays_out_of_the_number(self):
        self.assertEqual(sm_number("sm_90a"), 90)
        self.assertEqual(sm_number("sm_100f"), 100)
        self.assertEqual(sm_number("sm_90"), 90)

    def test_plain_number_works_too(self):
        self.assertEqual(sm_number("80"), 80)

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            sm_number("banana")


class RunsOnSmSuffixTests(unittest.TestCase):
    def test_suffixed_floor_compares_by_level(self):
        self.assertTrue(runs_on_sm("sm_90a", "sm_100"))
        self.assertTrue(runs_on_sm("sm_90a", "sm_90"))
        self.assertFalse(runs_on_sm("sm_90a", "sm_80"))


class RecordSmFloorTests(unittest.TestCase):
    def test_cuda_floor_key_wins(self):
        self.assertEqual(record_sm_floor(_cuda(min_sm="sm_70")), "sm_70")
        self.assertEqual(record_sm_floor(_cuda()), "")

    def test_ptx_floor_is_the_smallest_token(self):
        self.assertEqual(record_sm_floor(_ptx("sm_53, sm_80")), "sm_53")
        self.assertEqual(record_sm_floor(_ptx("sm_20, sm_100, sm_53, sm_90")), "sm_20")

    def test_suffixed_floor_keeps_its_label(self):
        self.assertEqual(record_sm_floor(_ptx("sm_100a, sm_90a")), "sm_90a")

    def test_non_cuda_ptx_records_carry_no_floor(self):
        x86 = InstructionRecord(
            mnemonic="zyx",
            form="y, x",
            summary="s",
            architecture="x86",
            isa=["SSE2"],
            metadata={"iform": "ZYX"},
        )
        self.assertEqual(record_sm_floor(x86), "")

    def test_dict_payloads_work_too(self):
        self.assertEqual(record_sm_floor({"metadata": {"min_sm": "sm_70"}}), "sm_70")


class SmLevelsTests(unittest.TestCase):
    def test_sorted_numerically_and_suffix_kept(self):
        self.assertEqual(
            sm_levels(["sm_90a", "sm_50", "sm_50", "sm_70", ""]),
            ["sm_50", "sm_70", "sm_90a"],
        )


class FilterStateTests(unittest.TestCase):
    def test_min_sm_hides_and_shows(self):
        row = _cuda(min_sm="sm_70")
        self.assertFalse(FilterState(min_sm="sm_50").matches(row))
        self.assertTrue(FilterState(min_sm="sm_90").matches(row))

    def test_default_state_keeps_every_row(self):
        self.assertTrue(FilterState().matches(_cuda(min_sm="sm_90")))

    def test_family_filter_still_applies(self):
        row = _cuda(min_sm="sm_70")
        self.assertFalse(FilterState(frozenset({"SSE"}), "sm_90").matches(row))
        self.assertTrue(FilterState(frozenset({"CUDA"}), "sm_90").matches(row))

    def test_empty_floor_passes_every_level(self):
        self.assertTrue(FilterState(min_sm="sm_20").matches(_cuda()))


class FilterSpecLevelsTests(unittest.TestCase):
    def test_to_json_carries_sm_levels(self):
        spec = FilterSpec(sm_levels=("sm_70", "sm_90"))
        self.assertEqual(spec.to_json()["sm_levels"], ["sm_70", "sm_90"])


class SmFloorsFromDbTests(unittest.TestCase):
    def test_map_and_levels_from_a_real_sqlite_catalog(self):
        from conftest import build_fixture_catalog
        from simdref.storage import build_sqlite, open_db

        base = build_fixture_catalog()
        catalog = Catalog(
            intrinsics=sorted(
                list(base.intrinsics) + [_cuda("__zyx_sync_v1", "sm_70")],
                key=lambda item: item.name,
            ),
            instructions=list(base.instructions) + [_ptx("sm_90, sm_90a")],
            sources=base.sources,
            generated_at=base.generated_at,
        )
        database = Path(tempfile.mkdtemp(prefix="simdref-sm-")) / "catalog.db"
        build_sqlite(catalog, database)
        with open_db(database) as conn:
            floors = sm_floors_from_db(conn)
        self.assertEqual(floors["__zyx_sync_v1"], "sm_70")
        self.assertEqual(floors["ptx:zyx.sync.b32 d, p, a, b, membermask"], "sm_90")
        self.assertEqual(sm_levels(floors.values()), sorted(sm_levels(floors.values())))


class BareQuerySmFlagTests(unittest.TestCase):
    NAME = "__zyxq_floored_v1"

    def _run(self, argv: list[str], database: Path) -> tuple[int, str]:
        import contextlib
        import io

        from simdref import cli
        from simdref.storage import open_db as storage_open_db

        captured = io.StringIO()
        with (
            mock.patch.object(cli, "ensure_runtime", lambda: None),
            mock.patch.object(
                cli, "open_db", lambda *args, **kwargs: storage_open_db(path=database)
            ),
            mock.patch.object(cli.sys.stdin, "isatty", return_value=False),
            mock.patch.object(cli.sys.stdout, "isatty", return_value=False),
            mock.patch.object(cli.sys, "argv", ["isa", *argv]),
            contextlib.redirect_stdout(captured),
        ):
            code = cli.main()
        return code, captured.getvalue()

    def _database(self) -> Path:
        from conftest import build_fixture_catalog
        from simdref.storage import build_sqlite

        base = build_fixture_catalog()
        catalog = Catalog(
            intrinsics=sorted(
                list(base.intrinsics) + [_cuda(self.NAME, "sm_70")],
                key=lambda item: item.name,
            ),
            instructions=list(base.instructions),
            sources=base.sources,
            generated_at=base.generated_at,
        )
        database = Path(tempfile.mkdtemp(prefix="simdref-sm-cli-")) / "catalog.db"
        build_sqlite(catalog, database)
        return database

    def test_sm_50_hides_a_floor_70_name(self):
        database = self._database()
        code, out = self._run(["--preset", "cuda", "--sm", "sm_50", self.NAME], database)
        self.assertEqual(code, 2)
        self.assertNotIn(self.NAME, out)

    def test_sm_90_shows_the_same_name(self):
        database = self._database()
        code, out = self._run(["--preset", "cuda", "--sm", "sm_90", self.NAME], database)
        self.assertEqual(code, 0)
        self.assertIn(self.NAME, out)

    def test_no_sm_keeps_the_old_behaviour(self):
        database = self._database()
        code, out = self._run([self.NAME], database)
        self.assertEqual(code, 0)
        self.assertIn(self.NAME, out)

    def test_bad_sm_value_is_a_usage_error(self):
        database = self._database()
        code, _out = self._run(["--sm", "banana", self.NAME], database)
        self.assertEqual(code, 1)


class SmChipCycleTests(unittest.TestCase):
    def test_click_cycles_off_then_each_level(self):
        try:
            from simdref.tui import SmChip
        except ImportError:  # pragma: no cover - textual is a runtime dep
            self.skipTest("textual not installed")
        chip = SmChip(["sm_50", "sm_90"])
        chip.on_click()
        self.assertEqual(chip.min_sm, "sm_50")
        chip.on_click()
        self.assertEqual(chip.min_sm, "sm_90")
        chip.on_click()
        self.assertEqual(chip.min_sm, "")

    def test_render_shows_the_picked_level(self):
        try:
            from simdref.tui import SmChip
        except ImportError:  # pragma: no cover - textual is a runtime dep
            self.skipTest("textual not installed")
        chip = SmChip(["sm_80"])
        self.assertEqual(chip.render().plain, "SM: off")
        chip.on_click()
        self.assertEqual(chip.render().plain, "SM: sm_80")


class CudaPayloadFloorTests(unittest.TestCase):
    def test_parse_keeps_min_sm_for_the_filter(self):
        from simdref.cuda import parse_cuda_intrinsics_payload

        bundle = json.dumps(
            {
                "format": "clang-cuda-headers-v2",
                "intrinsics": [
                    {
                        "name": "__zyx_sync_v1",
                        "signature": "int __zyx_sync_v1(int)",
                        "header": "__clang_cuda_intrinsics.h",
                        "calls": [],
                        "arch_guard": "",
                        "classification": "unclassified",
                        "min_sm": "sm_70",
                        "min_ptx": "6.0",
                        "instruction_refs": [],
                        "metadata": {"classification": "unclassified"},
                    }
                ],
            }
        )
        row = parse_cuda_intrinsics_payload(bundle)[0]
        self.assertEqual(record_sm_floor(row), "sm_70")
        self.assertFalse(FilterState(min_sm="sm_50").matches(row))
        self.assertTrue(FilterState(min_sm="sm_90").matches(row))


if __name__ == "__main__":
    unittest.main()
