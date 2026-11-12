"""Doc section order survives the export and drives the TUI.

The record's description and doc_sections carry the doc page order.
``_write_json`` sorts every JSON key for byte determinism, which made the
web detail pane show the sections alphabetically. The export writes the
sections as ``[heading, text]`` pairs, and the TUI renders the record
order, the same order the web renders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdref.export import export_site_data
from simdref.models import Catalog, InstructionRecord, IntrinsicRecord, SourceVersion

# NVIDIA PTX ISA page order of the activemask section, not alphabetical.
PAGE_ORDER = ["Syntax", "Description", "PTX ISA Notes", "Target ISA Notes", "Examples"]
# CUDA guide doc-section order of a device intrinsic, not alphabetical either.
DOC_ORDER = ["Description", "Return", "Parameters"]


def _catalog() -> Catalog:
    return Catalog(
        intrinsics=[
            IntrinsicRecord(
                name="__gret",
                signature="int __gret(int x)",
                description="Greets.",
                header="cuda_runtime.h",
                architecture="cuda",
                isa=["CUDA"],
                doc_sections={heading: f"{heading} text." for heading in DOC_ORDER},
            )
        ],
        instructions=[
            InstructionRecord(
                mnemonic="gret.b32",
                form="d",
                summary="Greets.",
                architecture="ptx",
                isa=["PTX"],
                description={heading: f"{heading} text." for heading in PAGE_ORDER},
            )
        ],
        sources=[
            SourceVersion(
                source="test",
                version="t",
                fetched_at="2025-01-01T00:00:00+00:00",
                url="test://",
            )
        ],
        generated_at="2025-01-01T00:00:00+00:00",
    )


def _all_entries(directory: Path) -> dict:
    entries: dict = {}
    for path in sorted(directory.glob("*.json")):
        entries.update(json.loads(path.read_text()))
    return entries


def test_exported_section_order_matches_the_catalog_order(tmp_path: Path):
    catalog = _catalog()
    export_site_data(catalog, tmp_path)
    (instruction,) = _all_entries(tmp_path / "detail-chunks").values()
    exported = [heading for heading, _text in instruction["description"]]
    assert exported == list(catalog.instructions[0].description)
    assert exported == PAGE_ORDER  # positive control: not the sorted keys
    (intrinsic,) = _all_entries(tmp_path / "intrinsic-chunks").values()
    exported_docs = [heading for heading, _text in intrinsic["doc_sections"]]
    assert exported_docs == list(catalog.intrinsics[0].doc_sections)
    assert exported_docs == DOC_ORDER


def test_exported_pairs_hold_the_section_text(tmp_path: Path):
    catalog = _catalog()
    export_site_data(catalog, tmp_path)
    (instruction,) = _all_entries(tmp_path / "detail-chunks").values()
    assert instruction["description"] == [[heading, f"{heading} text."] for heading in PAGE_ORDER]


def test_tui_sections_render_in_record_order():
    # The TUI hoisted the headings in a hand order, so it disagreed with
    # the web on a PTX row. Both now render the record order.
    pytest.importorskip("textual")
    from simdref.tui import SimdrefApp

    app = SimdrefApp()
    payload = app._build_instruction_payload(_catalog().instructions[0])
    headings = [entry[2] for entry in payload if entry[0] == "collapsible"]
    assert headings == PAGE_ORDER
