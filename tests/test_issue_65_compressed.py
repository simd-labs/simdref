"""Issue #65 item 27: riscv compressed rows get llvm-mca timing.

The row named stays the compressed encoding. Its timing comes from
llvm-mca's model of that instruction itself, which for most c.* rows
equals the base instruction's timing and for c.nop / c.li / c.j /
c.beqz / c.bnez is the compressed opcode's own scheduling class.
"""

from __future__ import annotations

import shutil

import pytest

from simdref.models import InstructionRecord
from simdref.perf_sources.cores import RISCV_CORES
from simdref.perf_sources.llvm_mca import ingest_compressed_rows

_MCA = next(
    (name for name in ("llvm-mca-23", "llvm-mca-24", "llvm-mca") if shutil.which(name)),
    None,
)
pytestmark = pytest.mark.skipif(_MCA is None, reason="needs llvm-mca")

_CORES = [c for c in RISCV_CORES if c.canonical_id in ("sifive-u74", "sifive-x280")]

COMPRESSED = ("c.mv", "c.li", "c.addi", "c.lw", "c.j", "c.beqz", "c.nop")
BASE = ("mv", "li", "addi", "lw")


def _records(*mnemonics: str) -> list[InstructionRecord]:
    return [
        InstructionRecord(mnemonic=m, form=m, summary="", architecture="riscv") for m in mnemonics
    ]


def test_compressed_rows_get_llvm_mca_timing() -> None:
    assert len(_CORES) == 2
    for core in _CORES:
        rows = ingest_compressed_rows(
            _records(*COMPRESSED), core, mca_version="test", executable=_MCA
        )
        by_mnemonic = {row.mnemonic: row for row in rows}
        assert sorted(by_mnemonic) == sorted(COMPRESSED)
        for row in rows:
            assert row.latency.isdigit()
            assert row.cpi
            assert row.source == "llvm-mca"
            assert row.architecture == "riscv"
            assert row.applies_to == "mnemonic"


def test_base_rows_are_not_consumed() -> None:
    # Base rows keep the timing the exegesis pipeline already gives them;
    # this pass must not touch them (byte-identical output on both trees).
    for core in _CORES:
        rows = ingest_compressed_rows(_records(*BASE), core, mca_version="test", executable=_MCA)
        assert rows == []
