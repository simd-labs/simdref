"""Tests for the empty-query state.

Every test here fails on e47f97e, which shows a blank list and a blank
status line for an empty query.
"""

from __future__ import annotations

import asyncio
import types
import unittest

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - Textual is a runtime dep
    HAS_TEXTUAL = False

from simdref.storage import CATALOG_PATH, SQLITE_PATH, sqlite_schema_is_current


CATALOG_READY = CATALOG_PATH.exists() and SQLITE_PATH.exists() and sqlite_schema_is_current()


@unittest.skipUnless(HAS_TEXTUAL and CATALOG_READY, "textual + catalog required")
class EmptyQueryStateTests(unittest.TestCase):
    def test_empty_query_shows_the_web_examples_and_catalog_counts(self):
        from simdref.tui import ExampleItem, SimdrefApp, _EMPTY_QUERY_EXAMPLES

        async def scenario():
            app = SimdrefApp()
            async with app.run_test(size=(160, 45)) as pilot:
                for _ in range(4):
                    await pilot.pause()
                items = [item for item in app.query_one("#results-list").children]
                examples = [item.example for item in items if isinstance(item, ExampleItem)]
                self.assertEqual(tuple(examples), _EMPTY_QUERY_EXAMPLES)
                counts = app._catalog_counts()
                self.assertRegex(counts, r"\d+ intrinsics · \d+ instructions")
                note = items[0].children[0]
                self.assertTrue(note.has_class("empty-note"))
                self.assertIn(counts, str(note.render()))

        asyncio.run(scenario())

    def test_example_row_runs_its_query(self):
        from simdref.tui import ExampleItem, SimdrefApp, _EMPTY_QUERY_EXAMPLES

        async def scenario():
            app = SimdrefApp()
            async with app.run_test(size=(160, 45)) as pilot:
                for _ in range(4):
                    await pilot.pause()
                item = next(
                    child
                    for child in app.query_one("#results-list").children
                    if isinstance(child, ExampleItem)
                )
                app.on_result_selected(types.SimpleNamespace(item=item))
                await pilot.pause()
                self.assertEqual(app._current_query, _EMPTY_QUERY_EXAMPLES[0])
                self.assertTrue(app._current_results)

        asyncio.run(scenario())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
