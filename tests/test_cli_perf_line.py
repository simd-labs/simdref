"""Bare-query summary line for a row with no perf data.

`isa <query>` on a row no arch holds perf data for must print one line,
"performance: no data" (the TUI wording), never the empty aggregate.
"""

from __future__ import annotations

import pytest

from simdref.cli import _print_non_interactive_summary
from simdref.models import InstructionRecord


def _record(mnemonic: str, architecture: str, isa: str) -> InstructionRecord:
    return InstructionRecord(
        mnemonic=mnemonic,
        form="d, [a]",
        summary="Greets one widget against another.",
        architecture=architecture,
        isa=[isa],
        source="test",
    )


@pytest.mark.parametrize(
    "mnemonic,architecture,isa",
    [("flob", "x86", "X86"), ("flob.b32", "ptx", "PTX")],
)
def test_no_perf_data_prints_one_no_data_line(mnemonic, architecture, isa, capsys):
    # One shared formatter serves every ISA family.
    record = _record(mnemonic, architecture, isa)
    status = _print_non_interactive_summary(mnemonic, records=[record])
    out = capsys.readouterr().out
    assert status == 0
    assert "avg over" not in out
    assert out.splitlines().count("  performance: no data") == 1
