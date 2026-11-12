"""Tests for the preset chip sync.

Every test here fails on e47f97e: a preset applied at startup leaves the
ISA and kind chips on their bootstrap highlights.
"""

from __future__ import annotations

import asyncio
import unittest

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - Textual is a runtime dep
    HAS_TEXTUAL = False

from simdref.storage import CATALOG_PATH, SQLITE_PATH, sqlite_schema_is_current


CATALOG_READY = CATALOG_PATH.exists() and SQLITE_PATH.exists() and sqlite_schema_is_current()


@unittest.skipUnless(HAS_TEXTUAL and CATALOG_READY, "textual + catalog required")
class PresetChipSyncTests(unittest.TestCase):
    def test_startup_preset_syncs_the_chip_visuals(self):
        from simdref.tui import IsaToggle, KindToggle, SimdrefApp

        async def scenario():
            app = SimdrefApp(initial_preset="arm32")
            async with app.run_test(size=(160, 45)) as pilot:
                for _ in range(4):
                    await pilot.pause()
                self.assertEqual(app._enabled_families, {"Arm"})
                for toggle in app.query(IsaToggle):
                    self.assertEqual(toggle.enabled, toggle.family in app._enabled_families)
                for toggle in app.query(KindToggle):
                    self.assertEqual(toggle.enabled, toggle.kind in app._enabled_kinds)

        asyncio.run(scenario())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
