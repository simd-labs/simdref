"""Canonical core-id table.

Upstream sources disagree about core naming: LLVM uses ``neoverse-n1``,
rvv-bench labels its rows ``c908``/``c910``. This module
maps every upstream alias to a single canonical id used inside
``InstructionRecord.arch_details`` so lookups and filters are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoreSpec:
    """A canonical microarchitecture entry."""

    canonical_id: str
    architecture: str  # "x86", "aarch64", "riscv"
    llvm_triple: str
    llvm_cpu: str
    aliases: frozenset[str]


# AArch64 cores covered by LLVM scheduling models. Aliases include the
# exact strings emitted by the upstream sources.
AARCH64_CORES: tuple[CoreSpec, ...] = (
    CoreSpec(
        "cortex-a72",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "cortex-a72",
        frozenset({"cortex-a72", "A72"}),
    ),
    CoreSpec(
        "cortex-a76",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "cortex-a76",
        frozenset({"cortex-a76", "A76"}),
    ),
    CoreSpec(
        "cortex-a78",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "cortex-a78",
        frozenset({"cortex-a78", "A78"}),
    ),
    CoreSpec(
        "cortex-x1",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "cortex-x1",
        frozenset({"cortex-x1", "X1"}),
    ),
    CoreSpec(
        "cortex-x2",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "cortex-x2",
        frozenset({"cortex-x2", "X2"}),
    ),
    CoreSpec(
        "neoverse-n1",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "neoverse-n1",
        frozenset({"neoverse-n1", "N1", "Neoverse-N1"}),
    ),
    CoreSpec(
        "neoverse-n2",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "neoverse-n2",
        frozenset({"neoverse-n2", "N2", "Neoverse-N2"}),
    ),
    CoreSpec(
        "neoverse-v1",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "neoverse-v1",
        frozenset({"neoverse-v1", "V1", "Neoverse-V1"}),
    ),
    CoreSpec(
        "neoverse-v2",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "neoverse-v2",
        frozenset({"neoverse-v2", "V2", "Neoverse-V2"}),
    ),
    CoreSpec(
        "neoverse-n3",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "neoverse-n3",
        frozenset({"neoverse-n3", "N3", "Neoverse-N3"}),
    ),
    CoreSpec(
        "neoverse-v3",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "neoverse-v3",
        frozenset({"neoverse-v3", "V3", "Neoverse-V3"}),
    ),
    CoreSpec(
        "a64fx", "aarch64", "aarch64-unknown-linux-gnu", "a64fx", frozenset({"a64fx", "A64FX"})
    ),
    # Apple cores reuse the Linux AArch64 triple so llvm-exegesis can
    # assemble on a non-Darwin host. The scheduling data we consume is
    # keyed off --mcpu, not --mtriple, so the numbers are identical.
    CoreSpec(
        "apple-m1",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "apple-m1",
        frozenset({"apple-m1", "M1"}),
    ),
    CoreSpec(
        "apple-m2",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "apple-m2",
        frozenset({"apple-m2", "M2"}),
    ),
    CoreSpec(
        "thunderx2t99",
        "aarch64",
        "aarch64-unknown-linux-gnu",
        "thunderx2t99",
        frozenset({"thunderx2t99", "ThunderX2"}),
    ),
)

# RISC-V cores with measured (rvv-bench) and modeled (LLVM) coverage.
RISCV_CORES: tuple[CoreSpec, ...] = (
    CoreSpec(
        "sifive-u74",
        "riscv",
        "riscv64-unknown-linux-gnu",
        "sifive-u74",
        frozenset({"sifive-u74", "U74"}),
    ),
    CoreSpec(
        "sifive-x280",
        "riscv",
        "riscv64-unknown-linux-gnu",
        "sifive-x280",
        frozenset({"sifive-x280", "X280"}),
    ),
    # LLVM 22 exposes the sifive-p400 and sifive-p600 *families* under the
    # specific part names p450 and p670 (the default LLVM CPU names for
    # those families). The canonical id is kept generic so catalog
    # consumers can filter by family rather than part.
    CoreSpec(
        "sifive-p400",
        "riscv",
        "riscv64-unknown-linux-gnu",
        "sifive-p450",
        frozenset({"sifive-p400", "sifive-p450", "P400", "P450"}),
    ),
    CoreSpec(
        "sifive-p600",
        "riscv",
        "riscv64-unknown-linux-gnu",
        "sifive-p670",
        frozenset({"sifive-p600", "sifive-p670", "P600", "P670"}),
    ),
    CoreSpec(
        "c908", "riscv", "riscv64-unknown-linux-gnu", "xiangshan-nanhu", frozenset({"c908", "C908"})
    ),
    CoreSpec(
        "c910", "riscv", "riscv64-unknown-linux-gnu", "xiangshan-nanhu", frozenset({"c910", "C910"})
    ),
    CoreSpec(
        "x60",
        "riscv",
        "riscv64-unknown-linux-gnu",
        "sifive-x280",
        frozenset({"x60", "X60", "Spacemit-X60"}),
    ),
)

# x86 cores. Canonical id matches the uops.info short code used as the
# arch_details dict key in InstructionRecord. Aliases cover common LLVM
# -march names, vendor marketing names, and uops.info short codes.
# Note: uops.info has no "SPR" row; Sapphire Rapids aliases map to EMR
# (Emerald Rapids) which uses the same Golden Cove P-core microarchitecture.
X86_CORES: tuple[CoreSpec, ...] = (
    CoreSpec(
        "EMR",
        "x86",
        "x86_64-unknown-linux-gnu",
        "sapphirerapids",
        frozenset(
            {"EMR", "emeraldrapids", "emerald-rapids", "SPR", "sapphirerapids", "sapphire-rapids"}
        ),
    ),
    CoreSpec(
        "SKX",
        "x86",
        "x86_64-unknown-linux-gnu",
        "skylake-avx512",
        frozenset({"SKX", "skylake-x", "skylake-avx512", "skx"}),
    ),
    CoreSpec(
        "CLX",
        "x86",
        "x86_64-unknown-linux-gnu",
        "cascadelake",
        frozenset({"CLX", "cascadelake", "cascade-lake", "clx"}),
    ),
    CoreSpec(
        "ICL",
        "x86",
        "x86_64-unknown-linux-gnu",
        "icelake-client",
        frozenset({"ICL", "icelake", "icelake-client", "icelake-server", "icl"}),
    ),
    CoreSpec(
        "TGL",
        "x86",
        "x86_64-unknown-linux-gnu",
        "tigerlake",
        frozenset({"TGL", "tigerlake", "tgl"}),
    ),
    CoreSpec(
        "RKL",
        "x86",
        "x86_64-unknown-linux-gnu",
        "rocketlake",
        frozenset({"RKL", "rocketlake", "rkl"}),
    ),
    CoreSpec(
        "ADL-P",
        "x86",
        "x86_64-unknown-linux-gnu",
        "alderlake",
        frozenset({"ADL-P", "alderlake", "alderlake-p", "adl-p", "raptorlake-p"}),
    ),
    CoreSpec(
        "ADL-E",
        "x86",
        "x86_64-unknown-linux-gnu",
        "gracemont",
        frozenset({"ADL-E", "gracemont", "alderlake-e", "adl-e"}),
    ),
    CoreSpec(
        "MTL-P",
        "x86",
        "x86_64-unknown-linux-gnu",
        "meteorlake",
        frozenset({"MTL-P", "meteorlake", "meteorlake-p", "mtl-p", "redwoodcove"}),
    ),
    CoreSpec(
        "MTL-E",
        "x86",
        "x86_64-unknown-linux-gnu",
        "crestmont",
        frozenset({"MTL-E", "crestmont", "meteorlake-e", "mtl-e"}),
    ),
    CoreSpec(
        "ARL-P",
        "x86",
        "x86_64-unknown-linux-gnu",
        "arrowlake",
        frozenset({"ARL-P", "arrowlake", "arrowlake-p", "arl-p", "lioncove"}),
    ),
    CoreSpec(
        "ARL-E",
        "x86",
        "x86_64-unknown-linux-gnu",
        "skymont",
        frozenset({"ARL-E", "skymont", "arrowlake-e", "arl-e"}),
    ),
    # Raptor Lake, Lunar Lake, Granite Rapids and Sierra Forest have no
    # uops.info rows in the vendored instructions.xml, so they carry no
    # uops.info aliases.
    CoreSpec(
        "RPL",
        "x86",
        "x86_64-unknown-linux-gnu",
        "raptorlake",
        frozenset({"RPL", "raptorlake", "raptor-lake", "rpl"}),
    ),
    CoreSpec(
        "LNL-P",
        "x86",
        "x86_64-unknown-linux-gnu",
        "lioncove",
        frozenset({"LNL-P", "lunarlake", "lunarlake-p", "lnl-p"}),
    ),
    CoreSpec(
        "LNL-E",
        "x86",
        "x86_64-unknown-linux-gnu",
        "skymont",
        frozenset({"LNL-E", "lunarlake-e", "lnl-e"}),
    ),
    CoreSpec(
        "GNR",
        "x86",
        "x86_64-unknown-linux-gnu",
        "graniterapids",
        frozenset({"GNR", "graniterapids", "granite-rapids", "gnr"}),
    ),
    CoreSpec(
        "SRF",
        "x86",
        "x86_64-unknown-linux-gnu",
        "sierraforest",
        frozenset({"SRF", "sierraforest", "sierra-forest", "srf"}),
    ),
    CoreSpec(
        "CNL",
        "x86",
        "x86_64-unknown-linux-gnu",
        "cannonlake",
        frozenset({"CNL", "cannonlake", "cnl"}),
    ),
    CoreSpec(
        "CFL",
        "x86",
        "x86_64-unknown-linux-gnu",
        "coffeelake",
        frozenset({"CFL", "coffeelake", "cfl"}),
    ),
    CoreSpec(
        "KBL", "x86", "x86_64-unknown-linux-gnu", "kabylake", frozenset({"KBL", "kabylake", "kbl"})
    ),
    CoreSpec(
        "SKL", "x86", "x86_64-unknown-linux-gnu", "skylake", frozenset({"SKL", "skylake", "skl"})
    ),
    CoreSpec(
        "BDW",
        "x86",
        "x86_64-unknown-linux-gnu",
        "broadwell",
        frozenset({"BDW", "broadwell", "bdw"}),
    ),
    CoreSpec(
        "HSW", "x86", "x86_64-unknown-linux-gnu", "haswell", frozenset({"HSW", "haswell", "hsw"})
    ),
    CoreSpec(
        "IVB",
        "x86",
        "x86_64-unknown-linux-gnu",
        "ivybridge",
        frozenset({"IVB", "ivybridge", "ivb"}),
    ),
    CoreSpec(
        "SNB",
        "x86",
        "x86_64-unknown-linux-gnu",
        "sandybridge",
        frozenset({"SNB", "sandybridge", "snb"}),
    ),
    CoreSpec(
        "WSM", "x86", "x86_64-unknown-linux-gnu", "westmere", frozenset({"WSM", "westmere", "wsm"})
    ),
    CoreSpec(
        "NHM", "x86", "x86_64-unknown-linux-gnu", "nehalem", frozenset({"NHM", "nehalem", "nhm"})
    ),
    CoreSpec(
        "GLP",
        "x86",
        "x86_64-unknown-linux-gnu",
        "goldmont-plus",
        frozenset({"GLP", "goldmont-plus", "glp"}),
    ),
    CoreSpec(
        "GLM", "x86", "x86_64-unknown-linux-gnu", "goldmont", frozenset({"GLM", "goldmont", "glm"})
    ),
    CoreSpec(
        "TRM", "x86", "x86_64-unknown-linux-gnu", "tremont", frozenset({"TRM", "tremont", "trm"})
    ),
    CoreSpec(
        "AMT", "x86", "x86_64-unknown-linux-gnu", "airmont", frozenset({"AMT", "airmont", "amt"})
    ),
    CoreSpec(
        "BNL", "x86", "x86_64-unknown-linux-gnu", "bonnell", frozenset({"BNL", "bonnell", "bnl"})
    ),
    CoreSpec(
        "WOL", "x86", "x86_64-unknown-linux-gnu", "wolfdale", frozenset({"WOL", "wolfdale", "wol"})
    ),
    CoreSpec(
        "CON",
        "x86",
        "x86_64-unknown-linux-gnu",
        "core2",
        frozenset({"CON", "conroe", "core2", "con"}),
    ),
    CoreSpec(
        "ZEN5", "x86", "x86_64-unknown-linux-gnu", "znver5", frozenset({"ZEN5", "zen5", "znver5"})
    ),
    CoreSpec(
        "ZEN4", "x86", "x86_64-unknown-linux-gnu", "znver4", frozenset({"ZEN4", "zen4", "znver4"})
    ),
    CoreSpec(
        "ZEN3", "x86", "x86_64-unknown-linux-gnu", "znver3", frozenset({"ZEN3", "zen3", "znver3"})
    ),
    CoreSpec(
        "ZEN2", "x86", "x86_64-unknown-linux-gnu", "znver2", frozenset({"ZEN2", "zen2", "znver2"})
    ),
    CoreSpec(
        "ZEN1", "x86", "x86_64-unknown-linux-gnu", "znver1", frozenset({"ZEN1", "zen1", "zen"})
    ),
    CoreSpec(
        "ZEN+",
        "x86",
        "x86_64-unknown-linux-gnu",
        "znver1",
        frozenset({"ZEN+", "zen+", "zenplus", "zen-plus"}),
    ),
)

CANONICAL_CORES: tuple[CoreSpec, ...] = AARCH64_CORES + RISCV_CORES + X86_CORES

_ALIAS_INDEX: dict[str, CoreSpec] = {}
for _core in CANONICAL_CORES:
    for _alias in _core.aliases:
        _ALIAS_INDEX[_alias.casefold()] = _core
    _ALIAS_INDEX[_core.canonical_id.casefold()] = _core


def canonical_core_id(name: str) -> str | None:
    """Return the canonical id for *name* or ``None`` if unknown."""
    if not name:
        return None
    core = _ALIAS_INDEX.get(name.casefold())
    return core.canonical_id if core is not None else None


def core_architecture(canonical_id: str) -> str | None:
    """Return the architecture family (x86/aarch64/riscv) for a canonical id."""
    core = _ALIAS_INDEX.get(canonical_id.casefold())
    return core.architecture if core is not None else None


def supported_core_ids() -> list[str]:
    """Return the sorted list of canonical core ids carried in the catalog."""
    return sorted(core.canonical_id for core in CANONICAL_CORES)
