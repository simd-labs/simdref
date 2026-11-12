"""Tests for the TUI category filter.

Every test here fails on e47f97e, which has no category select and no
category facet in the search.
"""

from __future__ import annotations

import asyncio
import contextlib
import unittest

from simdref.storage import CATALOG_PATH, SQLITE_PATH, sqlite_schema_is_current

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - Textual is a runtime dep
    HAS_TEXTUAL = False


CATALOG_READY = CATALOG_PATH.exists() and SQLITE_PATH.exists() and sqlite_schema_is_current()


@unittest.skipUnless(HAS_TEXTUAL and CATALOG_READY, "textual + catalog required")
class TuiCategoryFilterTests(unittest.TestCase):
    def test_select_offers_the_filter_spec_categories(self):
        from simdref.filters import load_categories_from_db
        from simdref.storage import open_db
        from simdref.tui import Select, SimdrefApp

        with contextlib.closing(open_db()) as conn:
            expected = sorted({c.category for c in load_categories_from_db(conn) if c.category})
        if not expected:
            self.skipTest("no categories in this catalog")

        async def scenario():
            app = SimdrefApp()
            async with app.run_test(size=(160, 45)) as pilot:
                for _ in range(4):
                    await pilot.pause()
                select = app.query_one("#cat-select", Select)
                values = [value for _label, value in select._options]
                self.assertEqual(values[0], "")
                self.assertEqual(sorted(values[1:]), expected)

        asyncio.run(scenario())

    def test_category_narrows_rows_that_carry_a_category(self):
        from simdref.filters import load_categories_from_db
        from simdref.storage import open_db
        from simdref.tui import SimdrefApp, _fts_search

        with contextlib.closing(open_db()) as conn:
            counts = sorted(
                ((c.count, c.category) for c in load_categories_from_db(conn) if c.category),
                reverse=True,
            )
        if not counts:
            self.skipTest("no categories in this catalog")
        category = counts[0][1]

        async def scenario():
            app = SimdrefApp(initial_query="add")
            async with app.run_test(size=(160, 45)) as pilot:
                for _ in range(6):
                    await pilot.pause()
                conn = app._conn
                plain = _fts_search(conn, "add", set(), None, limit=50)
                self.assertTrue(plain)
                rows = _fts_search(conn, "add", set(), None, limit=50, enabled_category=category)
                self.assertTrue(rows)
                # Rows without a category stay visible; category-carrying
                # intrinsics narrow to the chosen category.
                for result in rows:
                    if result.kind != "intrinsic":
                        continue
                    row = conn.execute(
                        "SELECT category FROM intrinsics_data WHERE name = ?", (result.key,)
                    ).fetchone()
                    self.assertIn(row["category"], {"", category})

        asyncio.run(scenario())
