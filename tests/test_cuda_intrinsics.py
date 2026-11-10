"""Tests for CUDA C++ device intrinsics parsing and chain resolution."""

from __future__ import annotations

import json
from pathlib import Path

from simdref.cuda import parse_cuda_intrinsics_payload

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_fixture() -> str:
    return (FIXTURES_DIR / "cuda_intrinsics.json").read_text()


class TestCudaParser:
    """Parser tests on fixture data."""

    def test_parse_returns_records(self) -> None:
        rows = parse_cuda_intrinsics_payload(_load_fixture())
        assert len(rows) == 5

    def test_record_has_required_fields(self) -> None:
        rows = parse_cuda_intrinsics_payload(_load_fixture())
        for row in rows:
            assert row.name
            assert row.signature
            assert row.header
            assert row.architecture == "cuda"
            assert row.isa == ["CUDA"]
            assert row.source == "clang-cuda-headers"

    def test_classification_present(self) -> None:
        rows = parse_cuda_intrinsics_payload(_load_fixture())
        for row in rows:
            assert "classification" in row.metadata


class TestChainResolution:
    """Chain resolution tests for specific intrinsics.

    The fixture rows come from tools/vendor_cuda_sources.py output, so the
    PTX strings are what the .td record chain resolves, never a hand list.
    """

    def test_shfl_sync_has_ptx(self) -> None:
        rows = parse_cuda_intrinsics_payload(_load_fixture())
        shfl = next(r for r in rows if r.name == "__shfl_sync")
        assert shfl.instructions == ["shfl.sync.idx.b32"]
        assert shfl.metadata["classification"] == "full_chain"

    def test_ldg_is_unlinked(self) -> None:
        # llvm.nvvm.ldg.global.* auto-upgrades to a plain addrspace(1) load;
        # no record chain names a PTX opcode for it, so it stays unlinked.
        rows = parse_cuda_intrinsics_payload(_load_fixture())
        ldg = next(r for r in rows if r.name == "__ldg")
        assert ldg.instructions == []
        assert ldg.metadata["classification"] == "unchainable_nvvm"

    def test_popc_is_libdevice(self) -> None:
        rows = parse_cuda_intrinsics_payload(_load_fixture())
        popc = next(r for r in rows if r.name == "__popc")
        assert popc.instructions == []
        assert popc.metadata["classification"] == "libdevice"

    def test_atomic_is_unlinked(self) -> None:
        # __nvvm_atom_*_gen_* lowers straight to atomicrmw: no named
        # int_nvvm_* exists, so the class carries no PTX claim.
        rows = parse_cuda_intrinsics_payload(_load_fixture())
        atomic = next(r for r in rows if r.name == "__uAtomicAdd")
        assert atomic.instructions == []
        assert atomic.metadata["classification"] == "atomic"

    def test_libdevice_function_has_no_ptx(self) -> None:
        rows = parse_cuda_intrinsics_payload(_load_fixture())
        sinf = next(r for r in rows if r.name == "__sinf")
        assert sinf.instructions == []
        assert sinf.metadata["classification"] == "libdevice"


class TestHeaderAttribution:
    """Each record carries its source header."""

    def test_shfl_from_intrinsics_header(self) -> None:
        rows = parse_cuda_intrinsics_payload(_load_fixture())
        shfl = next(r for r in rows if r.name == "__shfl_sync")
        assert shfl.header == "__clang_cuda_intrinsics.h"

    def test_ldg_from_intrinsics_header(self) -> None:
        rows = parse_cuda_intrinsics_payload(_load_fixture())
        ldg = next(r for r in rows if r.name == "__ldg")
        assert ldg.header == "__clang_cuda_intrinsics.h"

    def test_sinf_from_device_functions_header(self) -> None:
        rows = parse_cuda_intrinsics_payload(_load_fixture())
        sinf = next(r for r in rows if r.name == "__sinf")
        assert sinf.header == "__clang_cuda_device_functions.h"


class TestFixtureMatchesToolOutput:
    """The fixture must stay a verbatim slice of the vendor tool output."""

    def test_metadata_counts_agree_with_rows(self) -> None:
        data = json.loads(_load_fixture())
        rows = data["intrinsics"]
        linked = [r for r in rows if r["instructions"]]
        assert data["metadata"]["ptx_linked_count"] == len(linked)
        assert data["metadata"]["total_intrinsics"] == len(rows)
