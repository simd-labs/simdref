"""Rich formatting of the C.1 doc blocks in the TUI and the CLI.

Every test here fails on 1c10f16. That tree renders runs as plain text,
ignores the block ``lang`` and ``caption`` fields, bullets every list
item at one indent and leaves the note label unstyled. The synthetic
records carry the extended block shape: lang, caption, runs, ordered
lists and nested list items.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from simdref.display import doc_block_renderable

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - Textual is a runtime dep
    HAS_TEXTUAL = False

_WARP_URL = "https://example.org/cuda/warp"

# One guide cut with every extended block shape, and one extra-key
# entry the renderer must ignore (page, signature, signature_runs).
FMT_DOC_SOURCES = [
    {
        "source": "cuda-guide",
        "title": "CUDA C++ Programming Guide",
        "url": _WARP_URL,
        "heading": "Warp Lane Functions",
        "page": "guide.html",
        "signature": "T __fxlane_sync(unsigned mask, T value, int srcLane);",
        "signature_runs": [["code", "T __fxlane_sync"]],
        "sections": [
            {
                "heading": "",
                "blocks": [
                    {
                        "kind": "note",
                        "note": "hint",
                        "text": "The lane instruction reads a value from another thread.",
                        "runs": [
                            ["text", "The "],
                            ["code", "lane"],
                            ["text", " instruction reads a value from "],
                            ["em", "another"],
                            ["text", " thread."],
                        ],
                    },
                    {
                        "kind": "code",
                        "lang": "cuda",
                        "caption": "Signature",
                        "text": (
                            "T __fxlane_sync(unsigned mask, T value,"
                            " int srcLane, int width=warpSize);"
                        ),
                    },
                    {
                        "kind": "paragraph",
                        "text": "The mask selects the lanes that take part. See warp.",
                        "runs": [
                            ["text", "The "],
                            ["code", "mask"],
                            ["text", " selects the lanes that take part. See "],
                            ["link", "warp", _WARP_URL],
                            ["text", "."],
                        ],
                    },
                    {
                        "kind": "deflist",
                        "items": [
                            {
                                "name": "Returns",
                                "text": "the value read from the source lane.",
                                "runs": [
                                    ["text", "the "],
                                    ["strong", "value"],
                                    ["text", " read from the source lane."],
                                ],
                            }
                        ],
                    },
                    {
                        "kind": "list",
                        "ordered": True,
                        "items": [
                            {"text": "Read the old value."},
                            {
                                "text": "Compute the sum.",
                                "blocks": [
                                    {
                                        "kind": "list",
                                        "items": [
                                            {"text": "One add per lane."},
                                            {"text": "One store per lane."},
                                        ],
                                    }
                                ],
                            },
                            {"text": "Store the result."},
                        ],
                    },
                    {
                        "kind": "math",
                        "text": "sin(x) fast approximation",
                        "tex": "\\sin(x)",
                    },
                ],
            }
        ],
    },
    {
        "source": "ptx-isa",
        "title": "PTX ISA",
        "url": "https://example.org/ptx/#lane",
        "heading": "",
        "sections": [
            {
                "heading": "Syntax",
                "blocks": [
                    {
                        "kind": "code",
                        "lang": "ptx",
                        "text": "lane.sync.b32 d[|p], a, b, c, membermask;",
                    }
                ],
            },
            {
                "heading": "Description",
                "blocks": [
                    {
                        "kind": "note",
                        "note": "warning",
                        "text": "This instruction does not imply a memory barrier.",
                    }
                ],
            },
        ],
    },
]


def _record(doc_sources):
    from simdref.models import IntrinsicRecord

    return IntrinsicRecord(
        name="__fxlane_sync",
        signature="T __fxlane_sync(unsigned mask, T value, int srcLane)",
        description="Copy from an indexed lane.",
        header="cuda_wrappers.h",
        architecture="cuda",
        isa=["CUDA"],
        metadata={
            "classification": "unclassified",
            "doc_sources": json.dumps(doc_sources),
        },
    )


def render_plain(renderable, width: int = 120) -> str:
    """One renderable as console text, styles stripped."""
    from rich.console import Console

    console = Console(record=True, width=width)
    console.print(renderable)
    return console.export_text()


def text_lines(renderable) -> list[str]:
    """Rendered lines of one renderable, right spaces cut."""
    return [line.rstrip() for line in render_plain(renderable).splitlines()]


class CodeLangTests(unittest.TestCase):
    def test_code_lang_cuda_renders_with_the_cpp_lexer(self):
        from rich.syntax import Syntax

        for lang, lexer in (("cuda", "C++"), ("c++", "C++"), ("bash", "Bash")):
            block = {"kind": "code", "lang": lang, "text": "T f();"}
            renderable = doc_block_renderable(block)
            self.assertIsInstance(renderable, Syntax, lang)
            self.assertEqual(getattr(renderable.lexer, "name", ""), lexer, lang)
        # text has no pinned lexer and renders as plain text.
        from rich.text import Text

        renderable = doc_block_renderable({"kind": "code", "lang": "text", "text": "x = 1;"})
        self.assertIsInstance(renderable, Text)
        self.assertEqual(renderable.plain, "x = 1;")

    def test_ptx_code_block_uses_the_ptx_lexer(self):
        # ptx maps to pygments' PTX lexer; it must fail when the renderer
        # still renders ptx as plain text (pre-2c9bc55 behavior).
        from rich.syntax import Syntax

        ptx_entry = next(e for e in FMT_DOC_SOURCES if e["source"] == "ptx-isa")
        block = ptx_entry["sections"][0]["blocks"][0]
        self.assertEqual(block.get("lang"), "ptx")
        renderable = doc_block_renderable(block)
        self.assertIsInstance(renderable, Syntax)
        self.assertEqual(getattr(renderable.lexer, "name", ""), "PTX")

    def test_code_caption_line_above_the_code(self):
        block = {
            "kind": "code",
            "lang": "cuda",
            "caption": "Example 1: lane copy",
            "text": "T f();",
        }
        lines = [line for line in text_lines(doc_block_renderable(block)) if line.strip()]
        self.assertEqual(lines[0].strip(), "Example 1: lane copy")
        self.assertIn("T f();", lines[1])


class RunsTests(unittest.TestCase):
    def test_runs_render_as_styled_spans(self):
        block = {
            "kind": "paragraph",
            "text": "Use __shfl_sync on the fast path: see warp for sin(x)",
            "runs": [
                ["text", "Use "],
                ["code", "__shfl_sync"],
                ["text", " on the "],
                ["em", "fast"],
                ["text", " "],
                ["strong", "path"],
                ["text", ": see "],
                ["link", "warp", _WARP_URL],
                ["text", " for "],
                ["math", "sin(x)", "\\sin(x)"],
            ],
        }
        renderable = doc_block_renderable(block)
        self.assertEqual(
            renderable.plain,
            f"Use __shfl_sync on the fast path: see warp ({_WARP_URL}) for sin(x)",
        )
        spans = {(renderable.plain[sp.start : sp.end], str(sp.style)) for sp in renderable.spans}
        self.assertIn(("__shfl_sync", "bold cyan"), spans)
        self.assertIn(("fast", "italic"), spans)
        self.assertIn(("path", "bold"), spans)
        link = next(sp for sp in renderable.spans if renderable.plain[sp.start : sp.end] == "warp")
        self.assertEqual(getattr(link.style, "link", ""), _WARP_URL)
        self.assertIn("underline", str(link.style))


class ListTests(unittest.TestCase):
    def test_ordered_list_numbers_its_items(self):
        block = {
            "kind": "list",
            "ordered": True,
            "items": ["Read the old value.", "Compute the sum.", "Store the result."],
        }
        self.assertEqual(
            text_lines(doc_block_renderable(block)),
            ["  1. Read the old value.", "  2. Compute the sum.", "  3. Store the result."],
        )

    def test_nested_list_indents_one_level_deeper(self):
        block = {
            "kind": "list",
            "items": [
                {
                    "text": "Outer item.",
                    "blocks": [
                        {
                            "kind": "list",
                            "ordered": True,
                            "items": ["Inner one.", "Inner two."],
                        }
                    ],
                }
            ],
        }
        self.assertEqual(
            text_lines(doc_block_renderable(block)),
            ["  - Outer item.", "    1. Inner one.", "    2. Inner two."],
        )


class NoteTests(unittest.TestCase):
    def test_note_label_is_bold_and_worded(self):
        labels = {"note": "Note", "hint": "Hint", "warning": "Warning", "seealso": "See also"}
        for kind, label in labels.items():
            renderable = doc_block_renderable({"kind": "note", "note": kind, "text": "body"})
            self.assertEqual(renderable.plain, f"{label}: body", kind)
            spans = [
                (renderable.plain[sp.start : sp.end], str(sp.style)) for sp in renderable.spans
            ]
            self.assertIn((label, "bold"), spans, kind)


class TableTests(unittest.TestCase):
    def test_table_header_rows_of_cells_render_columns(self):
        block = {
            "kind": "table",
            "caption": "Accuracy",
            "header": [["Input", "Result"]],
            "rows": [["x", "sin(x)"]],
        }
        text = render_plain(doc_block_renderable(block))
        for cell in ("Input", "Result", "sin(x)", "Accuracy"):
            self.assertIn(cell, text)
        self.assertNotIn("['Input'", text)


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


class DocRecordTests(unittest.TestCase):
    def _cli_text(self):
        from rich.console import Console

        from simdref import display

        console = Console(record=True, width=200)
        with (
            patch.object(display, "console", console),
            patch.object(display, "linked_instruction_records", lambda *a, **k: []),
        ):
            display.render_intrinsic(None, _record(FMT_DOC_SOURCES))
        return console.export_text()

    def test_entry_extra_keys_are_ignored(self):
        # page, signature and signature_runs ride along and never render.
        text = self._cli_text()
        self.assertIn("Signature", text)
        self.assertIn("lane.sync.b32", text)
        self.assertNotIn("guide.html", text)
        self.assertNotIn("signature_runs", text)

    @unittest.skipUnless(HAS_TEXTUAL, "textual required")
    def test_tui_payload_shows_the_new_formatting(self):
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        app._find_linked_instruction = lambda intrinsic, conn=None: None
        payload = app._build_intrinsic_payload(_record(FMT_DOC_SOURCES))
        text = payload_text(payload)
        self.assertIn("  1. Read the old value.", text)
        self.assertIn("    - One add per lane.", text)
        self.assertLess(text.index("Signature"), text.index("width=warpSize"))
        self.assertIn("Hint:", text)
        self.assertIn(_WARP_URL, text)

    def test_cli_shows_the_new_formatting(self):
        text = self._cli_text()
        self.assertIn("  1. Read the old value.", text)
        self.assertIn("    - One add per lane.", text)
        self.assertLess(text.index("Signature"), text.index("width=warpSize"))
        self.assertIn("Warning:", text)
        self.assertIn(_WARP_URL, text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
