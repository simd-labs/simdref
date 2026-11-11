"""Tests for the one PTX ref rule and the PTX row floor fallback."""

from __future__ import annotations

import json

from simdref.ingest_catalog import link_records
from simdref.models import InstructionRecord, IntrinsicRecord


def _ptx_row() -> InstructionRecord:
    return InstructionRecord(
        mnemonic="zyx.sync.b32",
        form="d, p, a, b, membermask",
        summary="zyx.sync.b32 instruction.",
        architecture="ptx",
        isa=["PTX"],
        metadata={
            "qualifiers": json.dumps(
                {"mode": ["up", "down", "idx", "bfly"], "type": ["b32", "b64"]}
            ),
            "sm": "sm_70",
            "ptx_isa": "6.0, 7.0",
        },
        description={"Syntax": "zyx.sync.mode.type d, p, a, b, membermask;"},
        source="ptx-isa",
    )


def _cuda_row(refs: list[str]) -> IntrinsicRecord:
    return IntrinsicRecord(
        name="__zyx_sync_v1",
        signature="int __zyx_sync_v1(int)",
        description="",
        header="__clang_cuda_intrinsics.h",
        url="",
        architecture="cuda",
        isa=["CUDA"],
        instructions=list(refs),
        instruction_refs=[{"name": name, "form": "", "architecture": "ptx"} for name in refs],
        source="clang-cuda-headers",
    )


def test_concrete_spelling_resolves_to_the_template_row() -> None:
    row = _cuda_row(["zyx.sync.idx.b32"])
    ptx = _ptx_row()
    link_records([row], [ptx])
    assert row.instruction_refs[0]["resolution"] == "ptx-spelling"
    assert row.instruction_refs[0]["match_count"] == "1"
    assert row.instructions == [ptx.key]
    assert "__zyx_sync_v1" in ptx.linked_intrinsics


def test_unresolvable_ref_stays_unresolved() -> None:
    row = _cuda_row(["zyx.nosuchthing.b32"])
    link_records([row], [_ptx_row()])
    assert row.instruction_refs[0]["resolution"] == "unresolved"


def test_floors_come_from_the_linked_ptx_row() -> None:
    row = _cuda_row(["zyx.sync.idx.b32"])
    link_records([row], [_ptx_row()])
    assert row.metadata["min_sm"] == "sm_70"
    assert row.metadata["min_ptx"] == "6.0"
    assert row.metadata["min_sm_from"] == "ptx-row"
    assert row.metadata["min_ptx_from"] == "ptx-row"


def test_existing_floor_survives() -> None:
    row = _cuda_row(["zyx.sync.idx.b32"])
    row.metadata["min_sm"] = "sm_90"
    link_records([row], [_ptx_row()])
    assert row.metadata["min_sm"] == "sm_90"
    assert row.metadata.get("min_sm_from") != "ptx-row"
