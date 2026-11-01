"""Instruction categories in the RISC-V vendoring tool.

The load/store check must match whole mnemonics. A letter prefix classified
sh1add, sha256sum0, sctrclr, vsetvli and vslideup as loads and stores.
The grammar must also cover FP, compressed, hypervisor, cache-block and
prefetch memory operations.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _vendor_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "vendor_riscv_sources.py"
    spec = importlib.util.spec_from_file_location("vendor_riscv_sources", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TOOL = _vendor_tool()


@pytest.mark.parametrize(
    ("mnemonic", "isa", "expected"),
    [
        ("sh1add", ["Zba"], "Arithmetic"),
        ("sh2add.uw", ["Zba"], "Arithmetic"),
        ("sh3add", ["Zba"], "Arithmetic"),
        ("sha256sum0", ["Zknh"], "Arithmetic"),
        ("sctrclr", ["Smctr"], "Arithmetic"),
        ("vsetvli", ["V"], "Arithmetic"),
        ("vsadd.vv", ["V"], "Arithmetic"),
        ("vsub.vx", ["V"], "Arithmetic"),
        ("vsmul.vv", ["V"], "Arithmetic"),
        ("vslideup.vx", ["V"], "Permute/Move"),
        ("vsext.vf2", ["V"], "Widening/Narrowing"),
        ("slli", ["I"], "Arithmetic"),
        ("sll", ["I"], "Arithmetic"),
        ("sbset", ["Zbs"], "Arithmetic"),
        ("sh", ["I"], "Loads and Stores"),
        ("lw", ["I"], "Loads and Stores"),
        ("lbu", ["I"], "Loads and Stores"),
        ("lb.aq", ["Zalasr"], "Loads and Stores"),
        ("sd.rl", ["Zalasr"], "Loads and Stores"),
        ("sc.w", ["Zalrsc"], "Loads and Stores"),
        ("lr.w.aq", ["Zalrsc"], "Loads and Stores"),
        ("amoadd.w", ["Zaamo"], "Loads and Stores"),
        ("amoadd.b.aqrl", ["Zabha"], "Loads and Stores"),
        ("vle32.v", ["V"], "Loads and Stores"),
        ("vle8ff.v", ["V"], "Loads and Stores"),
        ("vse8.v", ["V"], "Loads and Stores"),
        ("vlseg2e8.v", ["V"], "Loads and Stores"),
        ("vlsseg2e8.v", ["V"], "Loads and Stores"),
        ("vluxseg2ei8.v", ["V"], "Loads and Stores"),
        ("vl1re8.v", ["V"], "Loads and Stores"),
        ("vlm.v", ["V"], "Loads and Stores"),
        ("vsm.v", ["V"], "Loads and Stores"),
        ("flw", ["F"], "Loads and Stores"),
        ("fsd", ["D"], "Loads and Stores"),
        ("c.lwsp", ["Zca"], "Loads and Stores"),
        ("c.sd", ["Zca"], "Loads and Stores"),
        ("hlv.w", ["H"], "Loads and Stores"),
        ("hsv.d", ["H"], "Loads and Stores"),
        ("hlvx.hu", ["H"], "Loads and Stores"),
        ("cbo.zero", ["Zicbom"], "Loads and Stores"),
        ("prefetch.r", ["Zicbop"], "Loads and Stores"),
        ("c.add", ["Zca"], "Arithmetic"),
        ("c.slli", ["Zca"], "Arithmetic"),
        ("fadd.s", ["F"], "Arithmetic"),
        ("fmv.x.w", ["F"], "Arithmetic"),
    ],
)
def test_instruction_category(mnemonic: str, isa: list[str], expected: str) -> None:
    assert _TOOL.instruction_category(mnemonic, isa) == expected
