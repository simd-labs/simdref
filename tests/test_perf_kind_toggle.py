"""Tests for the perf kind filter: measured vs modeled panels.

Every test here fails on e47f97e, which has no perf kind toggles and
always shows every perf panel.
"""

from __future__ import annotations

import asyncio
import unittest

from simdref.models import InstructionRecord

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - Textual is a runtime dep
    HAS_TEXTUAL = False


def _record() -> InstructionRecord:
    return InstructionRecord(
        mnemonic="VZYX",
        form="VZYX zmm, zmm",
        summary="A synthetic op with measured and modeled slots.",
        architecture="x86",
        isa=["AVX-512"],
        arch_details={
            "skl": {"source_kind": "measured", "measurement": {"TP_loop": "1", "uops": "2"}},
            "n1": {"source_kind": "modeled", "measurement": {"TP_loop": "2", "uops": "4"}},
        },
    )


def _panel_titles(payload: list[tuple]) -> list[str]:
    return [
        entry[2] for entry in payload if entry[0] == "collapsible" and isinstance(entry[2], str)
    ]


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class PerfKindFilterTests(unittest.TestCase):
    def test_perf_panels_hide_by_source_kind(self):
        from simdref.tui import SimdrefApp

        app = SimdrefApp()
        payload = app._build_instruction_payload(_record())
        titles = _panel_titles(payload)
        self.assertIn("perf (measured)", titles)
        self.assertIn("perf (modeled)", titles)

        app._enabled_perf_kinds = {"measured"}
        payload = app._build_instruction_payload(_record())
        titles = _panel_titles(payload)
        self.assertIn("perf (measured)", titles)
        self.assertNotIn("perf (modeled)", titles)

        app._enabled_perf_kinds = {"modeled"}
        payload = app._build_instruction_payload(_record())
        titles = _panel_titles(payload)
        self.assertNotIn("perf (measured)", titles)
        self.assertIn("perf (modeled)", titles)

    def test_kind_bar_offers_the_perf_kind_toggles(self):
        from simdref.tui import PerfKindToggle, SimdrefApp

        async def scenario():
            app = SimdrefApp(initial_query="add")
            async with app.run_test(size=(160, 45)) as pilot:
                for _ in range(6):
                    await pilot.pause()
                toggles = {t.kind: t for t in app.query(PerfKindToggle)}
                self.assertEqual(set(toggles), {"measured", "modeled"})
                self.assertTrue(all(t.enabled for t in toggles.values()))
                toggles["measured"].on_click()
                await pilot.pause()
                self.assertEqual(app._enabled_perf_kinds, {"modeled"})

        asyncio.run(scenario())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
