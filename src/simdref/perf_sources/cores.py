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


# llvm target triple per architecture family. Apple cores reuse the Linux
# AArch64 triple so llvm-exegesis can assemble on a non-Darwin host. The
# scheduling data we consume is keyed off --mcpu, not --mtriple, so the
# numbers are identical.
_TRIPLE = {
    "x86": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
    "riscv": "riscv64-unknown-linux-gnu",
}


def _specs(arch: str, rows: tuple[tuple[str, str, str], ...]) -> tuple[CoreSpec, ...]:
    """Build core specs for *arch* from (canonical id, llvm cpu, extra alias) rows."""
    # The canonical id is itself an alias, so rows omit it.
    return tuple(
        CoreSpec(cid, arch, _TRIPLE[arch], cpu, frozenset(extra.split()) | {cid})
        for cid, cpu, extra in rows
    )


# AArch64 cores covered by LLVM scheduling models. Aliases include the
# exact strings emitted by the upstream sources. Apple cores reuse the
# Linux AArch64 triple (see _TRIPLE).
AARCH64_CORES: tuple[CoreSpec, ...] = _specs(
    "aarch64",
    (
        ("cortex-a72", "cortex-a72", "A72"),
        ("cortex-a76", "cortex-a76", "A76"),
        ("cortex-a78", "cortex-a78", "A78"),
        ("cortex-x1", "cortex-x1", "X1"),
        ("cortex-x2", "cortex-x2", "X2"),
        ("neoverse-n1", "neoverse-n1", "N1 Neoverse-N1"),
        ("neoverse-n2", "neoverse-n2", "N2 Neoverse-N2"),
        ("neoverse-v1", "neoverse-v1", "Neoverse-V1 V1"),
        ("neoverse-v2", "neoverse-v2", "Neoverse-V2 V2"),
        ("neoverse-n3", "neoverse-n3", "N3 Neoverse-N3"),
        ("neoverse-v3", "neoverse-v3", "Neoverse-V3 V3"),
        ("neoverse-e1", "neoverse-e1", "E1 Neoverse-E1"),
        ("neoverse-v3ae", "neoverse-v3ae", "Neoverse-V3AE V3AE"),
        ("a64fx", "a64fx", "A64FX"),
        ("apple-m1", "apple-m1", "M1"),
        ("apple-m2", "apple-m2", "M2"),
        ("thunderx2t99", "thunderx2t99", "ThunderX2"),
    ),
)

# RISC-V cores with measured (rvv-bench) and modeled (LLVM) coverage.
# LLVM 22 exposes the sifive-p400 and sifive-p600 *families* under the
# specific part names p450 and p670 (the default LLVM CPU names for
# those families). The canonical id is kept generic so catalog
# consumers can filter by family rather than part.
RISCV_CORES: tuple[CoreSpec, ...] = _specs(
    "riscv",
    (
        ("sifive-u74", "sifive-u74", "U74"),
        ("sifive-x280", "sifive-x280", "X280"),
        ("sifive-p400", "sifive-p450", "P400 P450 sifive-p450"),
        ("sifive-p600", "sifive-p670", "P600 P670 sifive-p670"),
        ("c908", "xiangshan-nanhu", "C908"),
        ("c910", "xiangshan-nanhu", "C910"),
        ("x60", "sifive-x280", "Spacemit-X60 X60"),
    ),
)

# x86 cores. Canonical id matches the uops.info short code used as the
# arch_details dict key in InstructionRecord. Aliases cover common LLVM
# -march names, vendor marketing names, and uops.info short codes.
# Note: uops.info has no "SPR" row; Sapphire Rapids aliases map to EMR
# (Emerald Rapids) which uses the same Golden Cove P-core microarchitecture.
# Raptor Lake, Lunar Lake, Granite Rapids and Sierra Forest have no
# uops.info rows in the vendored instructions.xml, so they carry no
# uops.info aliases.
X86_CORES: tuple[CoreSpec, ...] = _specs(
    "x86",
    (
        (
            "EMR",
            "sapphirerapids",
            "SPR emerald-rapids emeraldrapids sapphire-rapids sapphirerapids",
        ),
        ("SKX", "skylake-avx512", "skx skylake-avx512 skylake-x"),
        ("CLX", "cascadelake", "cascade-lake cascadelake clx"),
        ("ICL", "icelake-client", "icelake icelake-client icl"),
        ("ICX", "icelake-server", "icelake-server icelake-xeon"),
        ("TGL", "tigerlake", "tgl tigerlake"),
        ("RKL", "rocketlake", "rkl rocketlake"),
        ("ADL-P", "alderlake", "adl-p alderlake alderlake-p raptorlake-p"),
        ("ADL-E", "gracemont", "adl-e alderlake-e gracemont"),
        ("MTL-P", "meteorlake", "meteorlake meteorlake-p mtl-p redwoodcove"),
        ("MTL-E", "crestmont", "crestmont meteorlake-e mtl-e"),
        ("ARL-P", "arrowlake", "arl-p arrowlake arrowlake-p lioncove"),
        ("ARL-E", "skymont", "arl-e arrowlake-e skymont"),
        ("RPL", "raptorlake", "raptor-lake raptorlake rpl"),
        ("LNL-P", "lioncove", "lnl-p lunarlake lunarlake-p"),
        ("LNL-E", "skymont", "lnl-e lunarlake-e"),
        ("GNR", "graniterapids", "gnr granite-rapids graniterapids"),
        ("SRF", "sierraforest", "sierra-forest sierraforest srf"),
        ("CNL", "cannonlake", "cannonlake cnl"),
        ("CFL", "coffeelake", "cfl coffeelake"),
        ("KBL", "kabylake", "kabylake kbl"),
        ("SKL", "skylake", "skl skylake"),
        ("BDW", "broadwell", "bdw broadwell"),
        ("HSW", "haswell", "haswell hsw"),
        ("IVB", "ivybridge", "ivb ivybridge"),
        ("SNB", "sandybridge", "sandybridge snb"),
        ("WSM", "westmere", "westmere wsm"),
        ("NHM", "nehalem", "nehalem nhm"),
        ("GLP", "goldmont-plus", "glp goldmont-plus"),
        ("GLM", "goldmont", "glm goldmont"),
        ("SLM", "silvermont", "silvermont"),
        ("TRM", "tremont", "tremont trm"),
        ("AMT", "airmont", "airmont amt"),
        ("BNL", "bonnell", "bnl bonnell"),
        ("WOL", "wolfdale", "wol wolfdale"),
        ("CON", "core2", "con conroe core2"),
        ("ZEN5", "znver5", "zen5 znver5"),
        ("ZEN4", "znver4", "zen4 znver4"),
        ("ZEN3", "znver3", "zen3 znver3"),
        ("ZEN2", "znver2", "zen2 znver2"),
        ("ZEN1", "znver1", "zen zen1"),
        ("ZEN+", "znver1", "zen+ zen-plus zenplus"),
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
