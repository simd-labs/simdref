"""Merging doc entries never mutates the source entry of another row.

A CUDA row linked to two PTX instructions of one page folds the links
into one ptx-isa entry; the instruction records keep their own
sections (the activemask.b32 leak). merge_doc_entries folds a copy:
the caller's input dict is not aliased into the row.
"""

from __future__ import annotations

import json

from simdref.cuda import append_linked_ptx_docs
from simdref.docblocks import merge_doc_entries
from simdref.models import InstructionRecord, IntrinsicRecord

PAGE = "https://example.invalid/ptx.html"


def _instruction(mnemonic: str, section_id: str, lines: list[str]) -> InstructionRecord:
    return InstructionRecord(
        mnemonic=mnemonic,
        form="d",
        summary="",
        architecture="ptx",
        isa=["PTX"],
        metadata={"url": f"{PAGE}#{section_id}"},
        doc_sources=[
            {
                "source": "ptx-isa",
                "url": f"{PAGE}#{section_id}",
                "page": PAGE,
                "title": "PTX",
                "heading": f"Widget Instructions: {mnemonic}",
                "shared": False,
                "sections": [
                    {
                        "heading": "Syntax",
                        "shared": False,
                        "blocks": [{"kind": "code", "text": line, "lang": "ptx"} for line in lines],
                    }
                ],
            }
        ],
        source="ptx-isa",
    )


def _row(refs: list[dict[str, str]]) -> IntrinsicRecord:
    return IntrinsicRecord(
        name="__zyx_ballot_v1",
        signature="int __zyx_ballot_v1(int mask)",
        description="CUDA device intrinsic from __clang_cuda_intrinsics.h.",
        header="__clang_cuda_intrinsics.h",
        url="",
        architecture="cuda",
        isa=["CUDA"],
        instruction_refs=refs,
        source="clang-cuda-headers",
    )


def test_linked_instructions_keep_their_own_sections() -> None:
    active = _instruction("activemask", "activemask-section", ["activemask.b32 d;"])
    vote = _instruction("vote.ballot", "vote-section", ["vote.ballot.b32 d, {!}a;"])
    before = json.dumps([r.doc_sources for r in (active, vote)], sort_keys=True)
    row = _row([{"name": "activemask"}, {"name": "vote.ballot"}])
    append_linked_ptx_docs([row], [active, vote])
    after = json.dumps([r.doc_sources for r in (active, vote)], sort_keys=True)
    assert after == before
    # The row folds the two links of the same page into one entry.
    ptx_entries = [e for e in row.doc_sources if e.get("source") == "ptx-isa"]
    assert len(ptx_entries) == 1
    texts = [block["text"] for section in ptx_entries[0]["sections"] for block in section["blocks"]]
    assert texts == ["activemask.b32 d;", "vote.ballot.b32 d, {!}a;"]


def test_merge_never_aliases_the_incoming_entry() -> None:
    entries: list[dict] = []
    incoming = {
        "source": "ptx-isa",
        "url": f"{PAGE}#one",
        "page": PAGE,
        "sections": [
            {
                "heading": "Syntax",
                "blocks": [{"kind": "code", "text": "one.b32 d;", "lang": "ptx"}],
            }
        ],
    }
    snapshot = json.dumps(incoming, sort_keys=True)
    merge_doc_entries(entries, incoming)
    merge_doc_entries(
        entries,
        {**incoming, "url": f"{PAGE}#two", "sections": [{"heading": "Examples", "blocks": []}]},
    )
    # The second merge folded into the appended copy, not into incoming.
    assert json.dumps(incoming, sort_keys=True) == snapshot
    assert entries[0]["sections"] is not incoming["sections"]
    entries[0]["sections"][0]["blocks"][0]["text"] = "mutated"
    assert incoming["sections"][0]["blocks"][0]["text"] == "one.b32 d;"
