"""Tests for the annotate microarch choice.

Every test here fails on e47f97e, which pins no arch: the annotate call
passes a hard-coded arch=None.
"""

from __future__ import annotations

import asyncio
import unittest

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - Textual is a runtime dep
    HAS_TEXTUAL = False


class AnnotateUarchOptionsTests(unittest.TestCase):
    def test_options_are_the_canonical_microarch_ids(self):
        from simdref.perf_sources.cores import CANONICAL_CORES
        from simdref.tui import _annotate_uarch_pairs

        expected = sorted({core.canonical_id for core in CANONICAL_CORES})
        pairs = _annotate_uarch_pairs()
        self.assertEqual([value for _label, value in pairs], expected)
        self.assertTrue(expected)


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class AnnotateArchSelectTests(unittest.TestCase):
    def test_select_pins_the_microarch_in_the_options(self):
        from simdref.tui import Select, SimdrefApp, _annotate_uarch_pairs

        async def scenario():
            app = SimdrefApp()
            async with app.run_test(size=(160, 45)) as pilot:
                await pilot.pause()
                select = app.query_one("#ann-arch", Select)
                values = [value for _label, value in select._options]
                self.assertEqual(values[0], "")
                self.assertEqual(values[1:], [v for _l, v in _annotate_uarch_pairs()])
                options, _isa = app._annotate_options()
                self.assertIsNone(options.arch)
                select.value = values[1]
                await pilot.pause()
                options, _isa = app._annotate_options()
                self.assertEqual(options.arch, values[1])

        asyncio.run(scenario())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
