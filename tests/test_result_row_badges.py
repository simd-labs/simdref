"""Tests for result row badges: ISA and LAT/CPI on every row.

Every test here fails on e47f97e, which renders rows without badges.
"""

from __future__ import annotations

import contextlib
import unittest

from simdref.search import SearchResult
from simdref.storage import CATALOG_PATH, SQLITE_PATH, sqlite_schema_is_current


CATALOG_READY = CATALOG_PATH.exists() and SQLITE_PATH.exists() and sqlite_schema_is_current()


class RowSubtitleTests(unittest.TestCase):
    def test_badges_never_clip(self):
        from simdref.tui import _ROW_SUBTITLE_MAX, _row_subtitle

        subtitle = _row_subtitle("x" * 400, "x86 AVX-512", "LAT 4 CPI 1")
        self.assertTrue(subtitle.startswith("x86 AVX-512 | LAT 4 CPI 1 | "))
        self.assertLessEqual(len(subtitle), _ROW_SUBTITLE_MAX)

    def test_no_badges_keeps_the_summary(self):
        from simdref.tui import _row_subtitle

        self.assertEqual(_row_subtitle("a summary", "", ""), "a summary")


@unittest.skipUnless(CATALOG_READY, "catalog required")
class SearchRowBadgeTests(unittest.TestCase):
    def test_rows_carry_the_web_badges(self):
        from simdref.display import display_architecture, display_isa
        from simdref.perf import record_iclass, variant_perf_summary
        from simdref.storage import load_intrinsic_from_db, load_instruction_from_db, open_db
        from simdref.tui import _ROW_SUBTITLE_MAX, _attach_row_badges, _fts_search

        with contextlib.closing(open_db()) as conn:
            rows = _fts_search(conn, "add", set(), None, limit=10)
            if not rows:
                self.skipTest("no rows for 'add' in this catalog")
            _attach_row_badges(conn, rows)
            for row in rows:
                self.assertLessEqual(len(row.subtitle), _ROW_SUBTITLE_MAX)
                if row.kind == "intrinsic":
                    item = load_intrinsic_from_db(conn, row.key)
                else:
                    item = load_instruction_from_db(conn, row.key)
                self.assertIsNotNone(item)
                arch = display_architecture(item.architecture)
                self.assertTrue(row.subtitle.startswith(arch))
                isa = display_isa(item.isa)
                if isa not in {"", "-"} and isa != arch:
                    self.assertIn(isa, row.subtitle)
                if row.kind == "instruction":
                    lat, cpi = variant_perf_summary(item.arch_details, record_iclass(item))
                    if lat not in {"-", ""}:
                        self.assertIn(f"LAT {lat}", row.subtitle)
                    if cpi not in {"-", ""}:
                        self.assertIn(f"CPI {cpi}", row.subtitle)

    def test_search_result_subtitle_is_rewritten_in_place(self):
        result = SearchResult(kind="instruction", key="k", title="t", subtitle="s", score=1)
        result.subtitle = "x86 | s"
        self.assertEqual(result.subtitle, "x86 | s")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
