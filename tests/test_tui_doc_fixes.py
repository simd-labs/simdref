"""TUI and CLI rendering fixes from the real-CUDA-data check.

Every test here fails on e62bfbe: hint notes print the label alone,
the TUI instruction panel title shows the operand form only, the
instruction metadata table prints one URL on two rows and wrapped
list items hang to the left margin. The records are synthetic.
"""

from __future__ import annotations

import unittest

from simdref.display import doc_block_renderable

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - Textual is a runtime dep
    HAS_TEXTUAL = False


def render_plain(renderable, width: int = 120) -> str:
    """One renderable as console text, styles stripped."""
    from rich.console import Console

    console = Console(record=True, width=width)
    console.print(renderable)
    return console.export_text()


def text_lines(renderable, width: int = 120) -> list[str]:
    """Rendered lines of one renderable, right spaces cut."""
    return [line.rstrip() for line in render_plain(renderable, width).splitlines()]


_LONG_HINT_RUNS = [
    ["text", "\n\nUse the "],
    ["code", "lane"],
    ["text", " variant on the hot path.\n"],
]


class NoteLabelRunsTests(unittest.TestCase):
    def test_note_text_follows_the_label_for_every_kind(self):
        # Blank note runs ride along from the parse of the indented NVIDIA
        # note blocks. The text keeps one line behind the label for every
        # note kind, under one code path.
        labels = {"note": "Note", "hint": "Hint", "warning": "Warning", "seealso": "See also"}
        for kind, label in labels.items():
            block = {
                "kind": "note",
                "note": kind,
                "text": "Use the lane variant on the hot path.",
                "runs": _LONG_HINT_RUNS,
            }
            renderable = doc_block_renderable(block)
            self.assertEqual(
                renderable.plain,
                f"{label}: Use the lane variant on the hot path.",
                kind,
            )
            self.assertNotIn("\n", renderable.plain, kind)


def payload_text(payload) -> str:
    """Full rendered text of one TUI payload, collapsible content included."""
    from rich.console import Console

    console = Console(record=True, width=200)
    for entry in payload:
        if entry[0] in {"static", "static_markup"}:
            console.print(entry[1])
        elif entry[0] == "collapsible":
            console.print(entry[1])
            console.print(entry[2])
        elif entry[0] == "related":
            console.print(entry[1])
    return console.export_text()


def _ptx_instruction(metadata=None):
    from simdref.models import InstructionRecord

    return InstructionRecord(
        mnemonic="shfl.sync.b32",
        form="d[|p], a, b, c, membermask",
        summary="Register data shuffle within threads of a warp.",
        architecture="ptx",
        isa=["PTX"],
        metadata=metadata or {},
    )


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class InstructionPanelTitleTests(unittest.TestCase):
    def test_title_names_the_mnemonic_and_the_form(self):
        # The web key reads "shfl.sync.b32 d[|p], a, b, c, membermask"; the
        # TUI panel title must name the same key, not the operand form alone.
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        text = payload_text(app._build_instruction_payload(_ptx_instruction()))
        self.assertIn("instruction: shfl.sync.b32 d[|p], a, b, c, membermask", text)


_PTX_URL = "https://example.org/ptx/#shfl-sync"


def _ptx_sources(url=_PTX_URL):
    return [
        {
            "source": "ptx-isa",
            "title": "PTX ISA",
            "url": url,
            "heading": "Introduction",
            "sections": [{"heading": "", "blocks": [{"kind": "paragraph", "text": "Body."}]}],
        }
    ]


def _ptx_instruction_with_header_url():
    import json

    return _ptx_instruction(metadata={"url": _PTX_URL, "doc_sources": json.dumps(_ptx_sources())})


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class TuiHeaderUrlTests(unittest.TestCase):
    def test_panel_shows_each_url_once_with_the_source_title(self):
        # The row's own url and the doc-source entry carry one URL; the web
        # header shows it once, under the source title label.
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        text = payload_text(app._build_instruction_payload(_ptx_instruction_with_header_url()))
        self.assertEqual(text.count(_PTX_URL), 1)
        self.assertIn("PTX ISA", text)


class CliHeaderUrlTests(unittest.TestCase):
    def test_cli_table_shows_each_url_once_with_the_source_title(self):
        import json
        from unittest.mock import patch

        from rich.console import Console

        from simdref import display

        console = Console(record=True, width=160)
        record = _ptx_instruction_with_header_url()
        with patch.object(display, "console", console):
            display.render_instruction(None, record)
        text = console.export_text()
        self.assertEqual(text.count(_PTX_URL), 1)
        self.assertIn("PTX ISA", text)


class ListHangingIndentTests(unittest.TestCase):
    def test_wrapped_bullet_keeps_the_marker_indent(self):
        block = {
            "kind": "list",
            "items": ["one two three four five six seven eight nine ten eleven"],
        }
        lines = text_lines(doc_block_renderable(block), width=30)
        self.assertEqual(lines[0], "  - one two three four five")
        self.assertTrue(lines[1].startswith("    six"), lines)
        self.assertTrue(lines[-1].startswith("    "), lines)

    def test_wrapped_ordered_item_keeps_the_marker_indent(self):
        block = {
            "kind": "list",
            "ordered": True,
            "items": ["one two three four five six seven eight nine ten eleven"],
        }
        lines = text_lines(doc_block_renderable(block), width=30)
        self.assertEqual(lines[0], "  1. one two three four five")
        self.assertTrue(lines[1].startswith("     six"), lines)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
