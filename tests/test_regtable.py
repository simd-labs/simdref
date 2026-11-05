"""The generated register table and the lsp.py edge-case rules over it."""

from pathlib import Path

from simdref import lsp
from simdref.regtable import LLVM_TAG, REG_TABLE

# The register spellings in tests/fixtures/isa_detect and hello_simd.s, with
# their (klass, width, family) rows. The scan below must find exactly these.
FIXTURE_ROWS = {
    "rax": (("gpr", 64, "x86"),),
    "rbp": (("gpr", 64, "x86"),),
    "rbx": (("gpr", 64, "x86"),),
    "rdi": (("gpr", 64, "x86"),),
    "rsi": (("gpr", 64, "x86"),),
    "rsp": (("gpr", 64, "x86"),),
    "xmm1": (("vec", 128, "x86"),),
    "ymm0": (("vec", 256, "x86"),),
    "ymm1": (("vec", 256, "x86"),),
    "d0": (("fp", 64, "a64"),),
    "d1": (("fp", 64, "a64"),),
    "d2": (("fp", 64, "a64"),),
    "q0": (("fp", 128, "a64"),),
    "s0": (("fp", 32, "a64"), ("rvabi", 64, "riscv")),
    "s1": (("fp", 32, "a64"), ("rvabi", 64, "riscv")),
    "s2": (("fp", 32, "a64"), ("rvabi", 64, "riscv")),
    "sp": (("gpr", 16, "x86"), ("rvabi", 64, "riscv"), ("sp", 64, "a64")),
    "x1": (("gpr", 64, "riscv"), ("xw", 64, "a64")),
    "x5": (("gpr", 64, "riscv"), ("xw", 64, "a64")),
    "x6": (("gpr", 64, "riscv"), ("xw", 64, "a64")),
    "x7": (("gpr", 64, "riscv"), ("xw", 64, "a64")),
}

# Rows outside the fixtures: every table class plus the keep-rule boundaries.
EDGE_ROWS = {
    "r7": (("arm", 32, "arm32"),),
    "r8": (("arm", 32, "arm32"), ("gpr", 64, "x86")),
    "r8d": (("gpr", 32, "x86"),),
    "r15w": (("gpr", 16, "x86"),),
    "spl": (("gpr", 8, "x86"),),
    "r16": (("gpr", 64, "x86"),),
    "rip": (("gpr", 64, "x86"),),
    "ip": (("arm", 32, "arm32"),),
    "pc": (("arm", 32, "arm32"),),
    "k0": (("k", 64, "x86"),),
    "mm7": (("mm", 64, "x86"),),
    "xmm31": (("vec", 128, "x86"),),
    "zmm16": (("vec", 512, "x86"),),
    "z0": (("z", 128, "a64"),),
    "p0": (("p", 16, "a64"),),
    "pn8": (("pn", 16, "a64"),),
    "v0": (("v", 128, "a64"), ("v", 512, "riscv")),
    "xzr": (("xw", 64, "a64"),),
    "wzr": (("xw", 32, "a64"),),
    "wsp": (("sp", 32, "a64"),),
    "s11": (("fp", 32, "a64"), ("rvabi", 64, "riscv")),
    "x31": (("gpr", 64, "riscv"),),
    "zero": (("rvabi", 64, "riscv"),),
    "fp": (),
    "lr": (),
}


def fixture_spellings() -> dict:
    """The table rows of every register spelling the fixture files use."""
    root = Path(__file__).parent / "fixtures"
    paths = [p for p in (root / "isa_detect").rglob("*") if p.is_file()]
    paths.append(root / "hello_simd.s")
    found = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in text.split():
            for piece in "".join(c if c.isalnum() or c in "._" else " " for c in token).split():
                name = piece.strip(".").casefold()
                if name in REG_TABLE:
                    found[name] = REG_TABLE[name]
    return found


def test_llvm_tag_is_recorded():
    assert LLVM_TAG == "llvmorg-23.1.4"


def test_fixture_register_rows():
    """The fixtures' register spellings keep their class rows."""
    assert fixture_spellings() == FIXTURE_ROWS


def test_edge_register_rows():
    """Every table class keeps its rows at the keep-rule boundaries."""
    for name, rows in EDGE_ROWS.items():
        assert REG_TABLE.get(name, ()) == rows, name


def test_keep_rule_x86_numbers_stop_at_r8():
    """r0-r7 are ARM registers and r16-r31 are APX names: neither is x86."""
    assert lsp._x86_gpr_width("r7") is None
    assert lsp._x86_gpr_width("r8") == 64
    assert lsp._x86_gpr_width("r15") == 64
    assert lsp._x86_gpr_width("r16") is None
    assert lsp._x86_gpr_width("rip") is None
    assert lsp._x86_gpr_width("r8d") == 32
    assert lsp._x86_gpr_width("r15w") == 16
    assert lsp._x86_gpr_width("spl") == 8
    assert lsp._x86_gpr_width("sp") == 16


def test_keep_rule_si_is_x86_only_prefixed():
    """A bare si is a symbol or the ARM stack index; %si is the r16."""
    assert lsp._operand_class("si", False, frozenset()) == "m"
    assert lsp._operand_class("%si", False, frozenset()) == "r16"


def test_keep_rule_s1_is_scalar_fp_and_rv_abi():
    """s0-s11 are AArch64 scalar FP and RISC-V saved registers at once."""
    assert lsp._is_a64_fp("s1") and lsp._is_rv_scalar("s1")
    assert lsp._is_a64_fp("s12") and not lsp._is_rv_scalar("s12")
