"""TUI and CLI rendering of the ordered C.1 doc_sources block shape.

Every test here fails on a568160. That tree ignores doc_sources, shows
the METADATA Result row, and marks missing operands and perf with
placeholder lines. The synthetic records carry doc_sources as JSON text
in the metadata, the same location a catalog payload can hold it.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simdref.display import record_doc_sources

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - Textual is a runtime dep
    HAS_TEXTUAL = False

# The shuffle cut: one shared family group plus four owned function
# groups, and the linked PTX entry last. This list is the shared
# expected-headings fixture (design node 4).
SHUFFLE_HEADINGS = [
    "Warp Shuffle Functions (CUDA C++ Programming Guide) [family]",
    "__shfl_sync",
    "__shfl_up_sync",
    "__shfl_down_sync",
    "__shfl_xor_sync",
    "PTX ISA",
    "Syntax",
    "Description",
    "Semantics",
    "PTX ISA Notes",
    "Target ISA Notes",
    "Examples",
]

_GUIDE_URL = (
    "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions"
)

_SHUFFLE_DOC_SOURCES = [
    {
        "source": "cuda-guide",
        "title": "CUDA C++ Programming Guide",
        "url": _GUIDE_URL,
        "heading": "Warp Shuffle Functions",
        "shared": True,
        "sections": [
            {
                "heading": "",
                "shared": True,
                "blocks": [
                    {
                        "kind": "note",
                        "note": "hint",
                        "text": "The shuffle instruction permits threads of a warp to read a value from another thread.",
                    },
                    {
                        "kind": "code",
                        "text": "T __shfl_sync(unsigned mask, T value, int srcLane, int width=warpSize);\nT __shfl_up_sync(unsigned mask, T value, unsigned int delta, int width=warpSize);",
                    },
                    {
                        "kind": "paragraph",
                        "text": "Each subsection of a warp behaves as a separate entity when width is smaller.",
                    },
                ],
            },
            {
                "heading": "__shfl_sync",
                "shared": False,
                "blocks": [
                    {
                        "kind": "paragraph",
                        "text": "Direct copy from indexed lane.",
                        "owner": "__shfl_sync",
                    },
                    {
                        "kind": "list",
                        "items": [
                            "If width is less than warpSize, each subsection behaves as a separate entity.",
                            "The source lane value is read from the indexed lane.",
                        ],
                    },
                ],
            },
            {
                "heading": "__shfl_up_sync",
                "shared": False,
                "blocks": [
                    {
                        "kind": "paragraph",
                        "text": "Copy from a lane with a lower ID relative to the caller.",
                        "owner": "__shfl_up_sync",
                    }
                ],
            },
            {
                "heading": "__shfl_down_sync",
                "shared": False,
                "blocks": [
                    {
                        "kind": "paragraph",
                        "text": "Copy from a lane with a higher ID relative to the caller.",
                        "owner": "__shfl_down_sync",
                    }
                ],
            },
            {
                "heading": "__shfl_xor_sync",
                "shared": False,
                "blocks": [
                    {
                        "kind": "paragraph",
                        "text": "Copy from a lane based on bitwise XOR of the caller lane ID.",
                        "owner": "__shfl_xor_sync",
                    }
                ],
            },
        ],
    },
    {
        "source": "ptx-isa",
        "title": "PTX ISA",
        "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync",
        "heading": "",
        "shared": False,
        "sections": [
            {
                "heading": "Syntax",
                "blocks": [
                    {"kind": "code", "text": "shfl.sync.mode.b32 d[|p], a, b, c, membermask;"}
                ],
            },
            {
                "heading": "Description",
                "blocks": [
                    {
                        "kind": "paragraph",
                        "text": "The shuffle and broadcast instruction moves a value between lanes.",
                    },
                    {
                        "kind": "deflist",
                        "items": [
                            {"name": ".all", "text": "copy from the lane indicated by c."},
                            {"name": ".any", "text": "copy from the first lane indicated by c."},
                        ],
                    },
                ],
            },
            {"heading": "Semantics", "blocks": [{"kind": "code", "text": "d = value at lane c;"}]},
            {
                "heading": "PTX ISA Notes",
                "blocks": [{"kind": "paragraph", "text": "Introduced in PTX ISA version 6.0."}],
            },
            {
                "heading": "Target ISA Notes",
                "blocks": [{"kind": "paragraph", "text": "Supported on sm_30 and higher."}],
            },
            {
                "heading": "Examples",
                "blocks": [
                    {"kind": "code", "text": "shfl.sync.mode.b32 %r1, %r2, %r3, %r4, 0xffffffff;"}
                ],
            },
        ],
    },
]

_MATH_DOC_SOURCES = [
    {
        "source": "cuda-math-api",
        "title": "CUDA Math API",
        "url": "https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html#gafa0ea84c0876f187341b9d9a218752b3",
        "heading": "Single Precision Intrinsics",
        "shared": False,
        "sections": [
            {
                "heading": "",
                "blocks": [
                    {
                        "kind": "paragraph",
                        "text": "Calculate the fast approximation to the sine of the input argument.",
                    },
                    {"kind": "note", "note": "seealso", "text": "sinf"},
                    {
                        "kind": "note",
                        "note": "note",
                        "text": "For accuracy information, see the Mathematical Functions appendix.",
                    },
                    {
                        "kind": "note",
                        "note": "note",
                        "text": "Output in the denormal range is flushed to zero.",
                    },
                    {"kind": "math", "text": "sin(x) fast approximation", "tex": "\\sin(x)"},
                    {
                        "kind": "deflist",
                        "items": [{"name": "Returns", "text": "the approximate sine of x."}],
                    },
                ],
            }
        ],
    }
]


def _intrinsic(
    doc_sources=None,
    name="__shfl_sync",
    metadata=None,
    signature=None,
    description="Direct copy from indexed lane.",
):
    from simdref.models import IntrinsicRecord

    meta = {"classification": "unclassified"}
    if metadata:
        meta.update(metadata)
    if doc_sources is not None:
        meta["doc_sources"] = json.dumps(doc_sources)
    return IntrinsicRecord(
        name=name,
        signature=signature
        or "T __shfl_sync(unsigned mask, T value, int srcLane, int width=warpSize)",
        description=description,
        header="cuda_wrappers.h",
        architecture="cuda",
        isa=["CUDA"],
        metadata=meta,
    )


def _ptx_instruction():
    from simdref.models import InstructionRecord

    return InstructionRecord(
        mnemonic="shfl.sync.mode.b32",
        form="d[|p], a, b, c, membermask",
        summary="Shuffle and broadcast.",
        architecture="ptx",
        isa=["PTX"],
        metadata={"doc_sources": json.dumps(_SHUFFLE_DOC_SOURCES[1:])},
    )


def payload_headings(payload: list[tuple]) -> list[str]:
    """Headings of one payload: group heading lines and section titles."""
    from rich.text import Text

    out: list[str] = []
    for entry in payload:
        if entry[0] == "collapsible":
            out.append(entry[2])
        elif entry[0] == "static" and isinstance(entry[1], Text):
            out.append(entry[1].plain)
    return out


def payload_text(payload: list[tuple]) -> str:
    """Full rendered text of one payload, collapsible content included."""
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


class RecordDocSourcesTests(unittest.TestCase):
    def test_reads_field_or_metadata(self):
        field = SimpleNamespace(doc_sources=[{"source": "ptx-isa"}], metadata={})
        self.assertEqual(record_doc_sources(field), [{"source": "ptx-isa"}])
        meta = SimpleNamespace(metadata={"doc_sources": json.dumps([{"source": "ulp"}])})
        self.assertEqual(record_doc_sources(meta), [{"source": "ulp"}])
        self.assertEqual(record_doc_sources(SimpleNamespace(metadata={})), [])
        broken = SimpleNamespace(metadata={"doc_sources": "{not json"})
        self.assertEqual(record_doc_sources(broken), [])


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class TuiDocBlockTests(unittest.TestCase):
    def _payload(self, intrinsic, linked=None):
        from simdref.tui import SimdrefApp

        app = SimdrefApp(initial_view="annotate")
        app._find_linked_instruction = lambda intrinsic, conn=None: linked
        return app._build_intrinsic_payload(intrinsic)

    def test_shuffle_headings_render_in_list_order(self):
        payload = self._payload(_intrinsic(_SHUFFLE_DOC_SOURCES))
        self.assertEqual(payload_headings(payload), SHUFFLE_HEADINGS)

    def test_two_entries_with_one_url_give_one_header_link(self):
        # Family entry plus the row's own cut of the same source page.
        entry = _SHUFFLE_DOC_SOURCES[0]
        own = dict(entry, heading="__shfl_sync", shared=False)
        payload = self._payload(_intrinsic([entry, own]))
        text = payload_text(payload)
        self.assertEqual(text.count(_GUIDE_URL), 1)
        headings = payload_headings(payload)
        self.assertIn("Warp Shuffle Functions (CUDA C++ Programming Guide) [family]", headings)
        self.assertIn("__shfl_sync (CUDA C++ Programming Guide)", headings)

    def test_math_record_shows_returns_once(self):
        intrinsic = _intrinsic(
            _MATH_DOC_SOURCES,
            name="__sinf",
            metadata={"result": "the approximate sine of x."},
        )
        text = payload_text(self._payload(intrinsic))
        self.assertEqual(text.count("Returns"), 1)
        self.assertNotIn("Result", text)
        self.assertIn("See also: sinf", text)
        self.assertIn("sin(x) fast approximation", text)
        self.assertNotIn("\\sin(x)", text)

    def test_cuda_payload_has_no_operand_or_perf_entries(self):
        from simdref.models import InstructionRecord

        linked = InstructionRecord(
            mnemonic="shfl.sync.mode.b32",
            form="d[|p], a, b, c, membermask",
            summary="Shuffle and broadcast.",
            architecture="ptx",
            isa=["PTX"],
        )
        intrinsic = _intrinsic(_MATH_DOC_SOURCES, name="__sinf")
        text = payload_text(self._payload(intrinsic, linked=linked)).lower()
        self.assertNotIn("operands", text)
        self.assertNotIn("perf (", text)
        self.assertNotIn("no data", text)
        self.assertNotIn("no linked instruction", text)

    def test_x86_row_shows_perf_without_result_row(self):
        from simdref.models import InstructionRecord

        linked = InstructionRecord(
            mnemonic="VADDPS",
            form="VADDPS (XMM, XMM, XMM)",
            summary="Add packed single-precision.",
            architecture="x86",
            isa=["AVX"],
            operand_details=[
                {"idx": "1", "r": "1", "w": "1", "type": "reg", "width": "128", "name": "xmm0"}
            ],
            arch_details={"SKL": {"measurement": {"TP_loop": "0.25"}}},
        )
        intrinsic = _intrinsic(
            name="vaddps",
            metadata={"result": "xmm0 <- a + b"},
        )
        payload = self._payload(intrinsic, linked=linked)
        text = payload_text(payload)
        self.assertIn("operands", text.lower())
        self.assertTrue(
            any(title.startswith("perf (") for title in payload_headings(payload)),
            payload_headings(payload),
        )
        self.assertNotIn("Result", text)

    def test_ptx_instruction_headings_render_in_list_order(self):
        from simdref.tui import SimdrefApp

        payload = SimdrefApp(initial_view="annotate")._build_instruction_payload(_ptx_instruction())
        self.assertEqual(
            payload_headings(payload),
            ["PTX ISA"] + SHUFFLE_HEADINGS[6:],
        )


class CliDocBlockTests(unittest.TestCase):
    def _render(self, record):
        from rich.console import Console
        from simdref import display

        console = Console(record=True, width=200)
        with (
            patch.object(display, "console", console),
            patch.object(display, "linked_instruction_records", lambda *a, **k: []),
        ):
            display.render_intrinsic(None, record)
        return console.export_text()

    def test_cli_shows_the_same_groups_in_order(self):
        text = self._render(_intrinsic(_SHUFFLE_DOC_SOURCES))
        position = -1
        for heading in SHUFFLE_HEADINGS:
            found = text.index(heading, position + 1)
            self.assertGreater(found, position, heading)
            position = found

    def test_cli_math_record_shows_returns_once(self):
        intrinsic = _intrinsic(
            _MATH_DOC_SOURCES,
            name="__sinf",
            metadata={"result": "the approximate sine of x."},
        )
        text = self._render(intrinsic)
        self.assertEqual(text.count("Returns"), 1)
        self.assertNotIn("Result", text)
        self.assertIn("Single Precision Intrinsics (CUDA Math API)", text)

    def test_cli_two_entries_with_one_url_give_one_header_link(self):
        entry = _SHUFFLE_DOC_SOURCES[0]
        own = dict(entry, heading="__shfl_sync", shared=False)
        text = self._render(_intrinsic([entry, own]))
        self.assertEqual(text.count(_GUIDE_URL), 1)
        self.assertIn("__shfl_sync (CUDA C++ Programming Guide)", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
