"""Tests for the sm floor keys in the web export."""

from __future__ import annotations

import json

from simdref.export import _detail_chunks, _intrinsic_detail_entry
from simdref.models import Catalog, InstructionRecord, IntrinsicRecord


def _catalog() -> Catalog:
    instruction = InstructionRecord(
        mnemonic="zyx.sync.b32",
        form="d, p, a, b, membermask",
        summary="zyx.sync.b32 instruction.",
        architecture="ptx",
        isa=["PTX"],
        metadata={"sm": "sm_70", "ptx_isa": "6.0", "qualifiers": "{}"},
        description={"Syntax": "zyx.sync.mode.type d, p, a, b, membermask;"},
    )
    intrinsic = IntrinsicRecord(
        name="__zyx_sync_v1",
        signature="int __zyx_sync_v1(int)",
        description="A zyx intrinsic.",
        header="__clang_cuda_intrinsics.h",
        url="https://example.invalid/zyx#zyx",
        architecture="cuda",
        isa=["CUDA"],
        metadata={"min_sm": "sm_70", "min_ptx": "6.0", "doc_source": "cuda-guide"},
    )
    return Catalog(
        intrinsics=[intrinsic],
        instructions=[instruction],
        sources=[],
        generated_at="2026-01-01T00:00:00+00:00",
    )


def test_instruction_detail_keeps_sm_metadata() -> None:
    chunks = _detail_chunks(_catalog())
    entry = next(iter(next(iter(chunks.values())).values()))
    assert entry["metadata"]["sm"] == "sm_70"
    assert entry["metadata"]["ptx_isa"] == "6.0"


def test_intrinsic_detail_keeps_min_sm_keys() -> None:
    from simdref.cuda import parse_cuda_intrinsics_payload

    bundle = json.dumps(
        {
            "format": "clang-cuda-headers-v2",
            "intrinsics": [
                {
                    "name": "__zyx_sync_v1",
                    "signature": "int __zyx_sync_v1(int)",
                    "header": "__clang_cuda_intrinsics.h",
                    "calls": [],
                    "arch_guard": "",
                    "classification": "unclassified",
                    "min_sm": "sm_70",
                    "min_ptx": "6.0",
                    "instruction_refs": [],
                    "metadata": {"classification": "unclassified"},
                }
            ],
        }
    )
    row = parse_cuda_intrinsics_payload(bundle)[0]
    entry = _intrinsic_detail_entry(row)
    assert entry["metadata"]["min_sm"] == "sm_70"
    assert entry["metadata"]["min_ptx"] == "6.0"
    assert entry["url"] == ""
