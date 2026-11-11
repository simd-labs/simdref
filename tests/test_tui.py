"""Smoke tests for TUI keybindings.

These use Textual's ``App.run_test()`` harness, so they require an
interactive catalog (SQLite DB built in ``data/derived``). Tests that
cannot find a catalog skip rather than fail.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - Textual is a runtime dep
    HAS_TEXTUAL = False

from simdref.storage import CATALOG_PATH, SQLITE_PATH, sqlite_schema_is_current


CATALOG_READY = CATALOG_PATH.exists() and SQLITE_PATH.exists() and sqlite_schema_is_current()


@unittest.skipUnless(HAS_TEXTUAL and CATALOG_READY, "textual + catalog required")
class TuiKeybindingSmokeTests(unittest.TestCase):
    """Boot the TUI headless and hammer the bindings plan (Phase 6) advertises."""

    def _run(self, coro):
        asyncio.run(coro)

    def test_slash_focuses_search_input(self):
        from simdref.tui import SimdrefApp

        async def scenario():
            app = SimdrefApp()
            async with app.run_test() as pilot:
                await pilot.press("/")
                focused = app.focused
                self.assertIsNotNone(focused)
                self.assertEqual(getattr(focused, "id", None), "search-input")

        self._run(scenario())

    def test_ctrl_k_aliases_focus_search(self):
        from simdref.tui import SimdrefApp

        async def scenario():
            app = SimdrefApp()
            async with app.run_test() as pilot:
                await pilot.press("escape")
                await pilot.press("ctrl+k")
                focused = app.focused
                self.assertEqual(getattr(focused, "id", None), "search-input")

        self._run(scenario())

    def test_question_mark_opens_help_modal(self):
        from simdref.tui import HelpScreen, SimdrefApp

        async def scenario():
            app = SimdrefApp()
            async with app.run_test() as pilot:
                await pilot.press("escape")
                # '?' on most keymaps is shift+slash; press the literal '?' glyph.
                await pilot.press("question_mark")
                self.assertIsInstance(app.screen, HelpScreen)
                await pilot.press("escape")
                self.assertNotIsInstance(app.screen, HelpScreen)

        self._run(scenario())

    def test_j_and_k_move_list_cursor(self):
        from simdref.tui import SimdrefApp
        from textual.widgets import ListView

        async def scenario():
            app = SimdrefApp(initial_query="_mm_add")
            async with app.run_test() as pilot:
                # Let the initial search populate.
                await pilot.pause()
                await pilot.pause()
                results = app.query_one("#results-list", ListView)
                if len(results) < 2:
                    self.skipTest("not enough results in the dev catalog to exercise j/k")
                results.focus()
                initial_index = results.index or 0
                await pilot.press("j")
                await pilot.pause()
                self.assertEqual(results.index, initial_index + 1)
                await pilot.press("k")
                await pilot.pause()
                self.assertEqual(results.index, initial_index)

        self._run(scenario())

    def test_escape_clears_detail(self):
        from simdref.tui import SimdrefApp
        from textual.containers import VerticalScroll

        async def scenario():
            app = SimdrefApp(initial_query="_mm_add")
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.pause()
                await pilot.press("1")
                await pilot.pause()
                detail = app.query_one("#detail-scroll", VerticalScroll)
                # Detail pane populates on pick.
                if len(detail.children) == 0:
                    self.skipTest("pick did not populate detail pane")
                await pilot.press("escape")
                await pilot.pause()
                detail = app.query_one("#detail-scroll", VerticalScroll)
                self.assertEqual(len(detail.children), 0)

        self._run(scenario())


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class TuiAnnotateClipboardTests(unittest.TestCase):
    def _run(self, coro):
        asyncio.run(coro)

    def test_annotate_input_does_not_define_custom_clipboard_aliases(self):
        from simdref.tui import _AnnInput

        keys = {getattr(binding, "key", None) for binding in _AnnInput.__dict__.get("BINDINGS", [])}
        self.assertNotIn("ctrl+shift+c", keys)
        self.assertNotIn("ctrl+shift+x", keys)
        self.assertNotIn("ctrl+shift+v", keys)
        self.assertNotIn("shift+insert", keys)

    def test_ctrl_v_pastes_via_native_textarea_binding(self):
        from textual.app import App, ComposeResult
        from simdref.tui import _AnnInput

        class ClipboardHarness(App):
            def compose(self) -> ComposeResult:
                yield _AnnInput.code_editor("", language=None, soft_wrap=False, id="ann-input")

            def _prepare_annotate_clipboard_for_paste(self) -> str | None:
                self._clipboard = "vaddps %ymm2, %ymm1, %ymm0\n"
                return self._clipboard

        async def scenario():
            app = ClipboardHarness()
            async with app.run_test() as pilot:
                ta = app.query_one("#ann-input", _AnnInput)
                ta.focus()
                await pilot.press("ctrl+v")
                await pilot.pause()
                self.assertEqual(ta.text, "vaddps %ymm2, %ymm1, %ymm0\n")

        self._run(scenario())

    def test_prepare_paste_prefers_system_clipboard_over_local(self):
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        app._clipboard = "stale-local"
        app._ann_local_clipboard_valid = True
        app.notify = Mock()
        with patch.object(app, "_read_system_clipboard", return_value="external-clipboard"):
            text = app._prepare_annotate_clipboard_for_paste()
        self.assertEqual(text, "external-clipboard")
        self.assertEqual(app.clipboard, "external-clipboard")
        self.assertFalse(app._ann_local_clipboard_valid)

    def test_prepare_paste_uses_local_clipboard_only_for_app_copy(self):
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        app._clipboard = "internal-copy"
        app._ann_local_clipboard_valid = True
        app.notify = Mock()
        with patch.object(app, "_read_system_clipboard", return_value=None):
            text = app._prepare_annotate_clipboard_for_paste()
        self.assertEqual(text, "internal-copy")

    def test_prepare_paste_does_not_use_stale_local_clipboard(self):
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        app._clipboard = "stale-local"
        app._ann_local_clipboard_valid = False
        app.notify = Mock()
        with patch.object(app, "_read_system_clipboard", return_value=None):
            text = app._prepare_annotate_clipboard_for_paste()
        self.assertIsNone(text)
        app.notify.assert_called()


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class TuiMissingSectionTests(unittest.TestCase):
    """A detail with no operands or no perf data says so, instead of omitting the section."""

    def _markup(self, payload):
        return [s[1] for s in payload if s[0] == "static_markup"]

    def test_intrinsic_without_instruction_marks_missing_sections(self):
        from simdref.models import IntrinsicRecord
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        intrinsic = IntrinsicRecord(
            name="_mm512_mask_reduce_min_epi32",
            signature="int _mm512_mask_reduce_min_epi32(__mmask16 k, __m512i a)",
            description="Reduce by minimum using mask k.",
            header="immintrin.h",
            isa=["AVX512F"],
        )
        with patch.object(app, "_find_linked_instruction", return_value=None):
            lines = self._markup(app._build_intrinsic_payload(intrinsic))
        self.assertTrue(any("no linked instruction" in line for line in lines), lines)

    def test_instruction_without_data_marks_operands_and_perf(self):
        from simdref.models import InstructionRecord
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        bare = InstructionRecord(mnemonic="NEG", form="NEG (R32)", summary="Negate.", isa=["I86"])
        lines = self._markup(app._build_instruction_payload(bare))
        self.assertTrue(any("operands: no data" in line for line in lines), lines)
        self.assertTrue(any("performance: no data" in line for line in lines), lines)

        full = InstructionRecord(
            mnemonic="NEG",
            form="NEG (R32)",
            summary="Negate.",
            isa=["I86"],
            operand_details=[{"idx": "1", "r": "1", "w": "1", "type": "reg", "width": "32"}],
            arch_details={
                "SKL": {"measurement": {"TP_loop": "0.25"}, "latencies": [{"cycles": "1"}]}
            },
        )
        lines = self._markup(app._build_instruction_payload(full))
        self.assertFalse(any("no data" in line for line in lines), lines)


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class TuiCudaPresetTests(unittest.TestCase):
    """The CUDA preset button sits by None and scopes results to CUDA rows."""

    def _run(self, coro):
        asyncio.run(coro)

    def test_cuda_button_orders_bar_and_scopes_results(self):
        from simdref import tui
        from simdref.display import isa_family
        from simdref.storage import open_db
        from test_cuda_wiring import _build_cuda_catalog

        async def scenario():
            with tempfile.TemporaryDirectory() as tmp:
                database = _build_cuda_catalog(Path(tmp))
                with (
                    patch.object(tui, "open_db", lambda *a, **k: open_db(database)),
                    patch.dict(os.environ, {"XDG_STATE_HOME": tmp}),
                ):
                    app = tui.SimdrefApp(initial_query="__shfl_sync")
                    async with app.run_test(size=(160, 45)) as pilot:
                        # Let queued mounts land, or the preset click races
                        # the initial sub-ISA row build.
                        for _ in range(3):
                            await pilot.pause()
                        modes = [b.mode for b in app.query(tui.PresetButton)]
                        self.assertEqual(
                            modes,
                            [
                                "default",
                                "intel",
                                "arm32",
                                "arm64",
                                "riscv",
                                "cuda",
                                "none",
                                "all",
                            ],
                        )
                        cuda_btn = next(b for b in app.query(tui.PresetButton) if b.mode == "cuda")
                        await pilot.click(cuda_btn)
                        await pilot.pause()
                        self.assertEqual(app._enabled_families, {"CUDA"})
                        # CUDA has no sub-ISAs: the bar must drop the stale
                        # rows of the previously enabled families.
                        self.assertEqual(list(app.query(tui.SubIsaToggle)), [])
                        results = app._current_results
                        self.assertTrue(results, "CUDA preset shows no __shfl_sync results")
                        with open_db(database) as conn:
                            for res in results:
                                if res.kind == "intrinsic":
                                    row = conn.execute(
                                        "SELECT isa FROM intrinsics_data WHERE name = ?", (res.key,)
                                    ).fetchone()
                                else:
                                    row = conn.execute(
                                        "SELECT isa FROM instructions_data WHERE db_key = ?",
                                        (res.key,),
                                    ).fetchone()
                                self.assertIsNotNone(row, res.key)
                                families = {
                                    isa_family(v) for v in (row[0] or "").replace(",", " ").split()
                                }
                                self.assertEqual(families, {"CUDA"}, res.key)

        self._run(scenario())


if __name__ == "__main__":
    unittest.main()
