"""Tests for the related entries lists in the detail.

Every test here fails on e47f97e, which shows one linked instruction at
most and no linked intrinsic list.
"""

from __future__ import annotations

import asyncio
import unittest

from simdref.models import IntrinsicRecord, InstructionRecord

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - Textual is a runtime dep
    HAS_TEXTUAL = False

from simdref.storage import CATALOG_PATH, SQLITE_PATH, sqlite_schema_is_current


CATALOG_READY = CATALOG_PATH.exists() and SQLITE_PATH.exists() and sqlite_schema_is_current()


def _intrinsic() -> IntrinsicRecord:
    return IntrinsicRecord(
        name="_mm256_zyx_ps",
        signature="__m256 _mm256_zyx_ps(__m256, __m256)",
        description="A synthetic intrinsic.",
        header="immintrin.h",
        url="",
        architecture="x86",
        isa=["AVX"],
        metadata={},
        instruction_refs=[
            {"key": "x86:zyxps (ymm, ymm, ymm)", "display_key": "ZYXPS (YMM, YMM, YMM)"},
            {"key": "x86:zyxps (xmm, xmm, xmm)", "display_key": "ZYXPS (XMM, XMM, XMM)"},
        ],
    )


def _instruction() -> InstructionRecord:
    return InstructionRecord(
        mnemonic="ZYXPS",
        form="ZYXPS ymm, ymm, ymm",
        summary="A synthetic op.",
        architecture="x86",
        isa=["AVX"],
        linked_intrinsics=["_mm256_zyx_ps", "_mm_zyx_ps"],
    )


def _related(payload: list[tuple]) -> list[tuple[str, list[tuple[str, str, str]]]]:
    return [(section[1], section[2]) for section in payload if section[0] == "related"]


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class RelatedEntryPayloadTests(unittest.TestCase):
    def test_intrinsic_detail_lists_every_linked_instruction(self):
        from simdref.tui import SimdrefApp

        app = SimdrefApp()
        app._find_linked_instruction = lambda intrinsic, conn=None: None
        payload = app._build_intrinsic_payload(_intrinsic())
        sections = _related(payload)
        self.assertEqual([title for title, _rows in sections], ["Instructions"])
        rows = sections[0][1]
        self.assertEqual(
            [(kind, key) for kind, key, _label in rows],
            [
                ("instruction", "x86:zyxps (ymm, ymm, ymm)"),
                ("instruction", "x86:zyxps (xmm, xmm, xmm)"),
            ],
        )

    def test_instruction_detail_lists_every_linked_intrinsic(self):
        from simdref.tui import SimdrefApp

        app = SimdrefApp()
        payload = app._build_instruction_payload(_instruction())
        sections = _related(payload)
        self.assertEqual([title for title, _rows in sections], ["Intrinsics"])
        rows = sections[0][1]
        self.assertEqual(
            [(kind, key) for kind, key, _label in rows],
            [("intrinsic", "_mm256_zyx_ps"), ("intrinsic", "_mm_zyx_ps")],
        )


@unittest.skipUnless(HAS_TEXTUAL and CATALOG_READY, "textual + catalog required")
class RelatedEntryOpenTests(unittest.TestCase):
    def test_enter_on_a_related_entry_opens_it(self):
        from simdref.tui import RelatedList, SimdrefApp

        async def scenario():
            app = SimdrefApp(initial_query="add")
            async with app.run_test(size=(160, 45)) as pilot:
                for _ in range(8):
                    await pilot.pause()
                lists = list(app.query(RelatedList))
                if not lists:
                    self.skipTest("no related list on the first result detail")
                related = lists[0]
                if not related.entries:
                    self.skipTest("empty related list")
                opened: list = []
                app._show_detail = lambda result: opened.append(result)
                related.highlighted = 0
                related.action_select()
                await pilot.pause()
                self.assertTrue(opened)
                self.assertEqual(opened[0].kind, related.entries[0][0])
                self.assertEqual(opened[0].key, related.entries[0][1])

        asyncio.run(scenario())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
