"""Shared filter specification for intrinsic/instruction lookups.

Single source of truth for the ISA-family / sub-ISA / category facets used by
the TUI, web SPA, CLI, and LLM subcommands. The Python constants here are
re-exported through :mod:`simdref.display` for back-compat and serialised to
``filter_spec.json`` for the static web bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3
from typing import Any, Iterable


ISA_FAMILY_ORDER: dict[str, int] = {
    "x86": 0,
    "MMX": 1,
    "SSE": 2,
    "AVX": 3,
    "AVX-512": 4,
    "AVX10": 5,
    "AMX": 6,
    "APX": 7,
    "Arm": 8,
    "RISC-V": 9,
    "SVML": 10,
    "Other": 11,
    "PTX": 12,
    "CUDA": 13,
}

DEFAULT_ENABLED_ISAS: tuple[str, ...] = ("SSE", "AVX", "AVX-512")

# Families reachable only through a preset, never as a family chip.
PRESET_ONLY_FAMILIES: tuple[str, ...] = ("PTX", "CUDA")

FAMILY_SUB_ORDER: dict[str, list[str]] = {
    "SSE": ["SSE", "SSE2", "SSE3", "SSSE3", "SSE4.1", "SSE4.2"],
    "AVX": [
        "AVX",
        "AVX2",
        "FMA",
        "F16C",
        "AVX_VNNI",
        "AVX_VNNI_INT8",
        "AVX_VNNI_INT16",
        "AVX_IFMA",
        "AVX_NE_CONVERT",
    ],
    "AVX-512": [
        "AVX512F",
        "AVX512VL",
        "AVX512BW",
        "AVX512DQ",
        "AVX512CD",
        "AVX512_VNNI",
        "AVX512_FP16",
        "AVX512_BF16",
        "AVX512_VBMI",
        "AVX512_VBMI2",
        "AVX512_BITALG",
        "AVX512IFMA52",
        "AVX512VPOPCNTDQ",
        "AVX512_VP2INTERSECT",
        "VAES",
        "VPCLMULQDQ",
        "GFNI",
    ],
    "AMX": ["AMX-TILE", "AMX-INT8", "AMX-BF16", "AMX-FP16", "AMX-COMPLEX"],
    "Arm": ["NEON", "SVE", "SVE2", "SME"],
    "RISC-V": ["V", "ZVE", "ZV"],
}

DEFAULT_SUBS: dict[str, set[str]] = {
    "SSE": {"SSE", "SSE2", "SSE3", "SSSE3", "SSE4.1", "SSE4.2"},
    "AVX": {"AVX", "AVX2", "FMA", "F16C"},
    "AVX-512": {"AVX512F", "AVX512VL", "AVX512BW", "AVX512DQ", "AVX512CD"},
    "AMX": {"AMX-TILE", "AMX-INT8", "AMX-BF16"},
    "Arm": {"NEON", "SVE", "SVE2", "SME"},
    "RISC-V": {"V", "ZVE", "ZV"},
}


# Strict x86-64-v4 sub-ISAs — what ships on every psABI-v4 target.
# Explicitly excludes: AVX-512 _VNNI/_FP16/_BF16/_VBMI*/_VPOPCNTDQ/_BITALG/
# _IFMA/_VP2INTERSECT, VAES, VPCLMULQDQ, GFNI, AMX, APX, AVX10, SVML.
X86_64_V4_SUBS: set[str] = {
    # psABI x86-64-v2
    "SSE",
    "SSE2",
    "SSE3",
    "SSSE3",
    "SSE4.1",
    "SSE4.2",
    "POPCNT",
    # psABI x86-64-v3
    "AVX",
    "AVX2",
    "F16C",
    "FMA",
    "BMI1",
    "BMI2",
    "LZCNT",
    "MOVBE",
    # psABI x86-64-v4
    "AVX512F",
    "AVX512VL",
    "AVX512BW",
    "AVX512DQ",
    "AVX512CD",
}

ARM_ALL_SUBS: set[str] = {"NEON", "SVE", "SVE2", "SME", "MVE"}
ARM32_SUBS: set[str] = {"NEON", "MVE"}
ARM64_SUBS: set[str] = {"NEON", "SVE", "SVE2", "SME"}
RVV_SUBS: set[str] = {"V", "ZVE", "ZV"}

# Default preset mirrors DEFAULT_ENABLED_ISAS and uses strict-v4 on x86.
DEFAULT_PRESET_SUBS: set[str] = set(X86_64_V4_SUBS)

# arm_arch facet values (NULL excluded; filter logic handles missing values).
ARM_ARCH_VALUES: tuple[str, ...] = ("A32", "A64", "BOTH")


def sm_number(value: str) -> int:
    """The numeric level of an "sm_XX" floor string.

    The parse splits on "_", so a suffix letter stays in the label but
    not in the number: "sm_90a" is level 90. The plain number works too.
    """
    text = value.strip().split("_")[-1]
    digits = ""
    for ch in text:
        if not ch.isdigit():
            break
        digits += ch
    if not digits:
        raise ValueError(f"not an sm level: {value!r}")
    return int(digits)


def runs_on_sm(min_sm: str | None, chosen_sm: str | None) -> bool:
    """Return whether a record floor passes the chosen sm level.

    One rule for the TUI, the CLI and the web export. An empty side
    passes: a record with no floor runs on every sm level, and no choice
    keeps every record. Both sides use the "sm_XX" form; the plain
    number works too.
    """
    if not min_sm or not chosen_sm:
        return True
    return sm_number(min_sm) <= sm_number(chosen_sm)


def record_sm_floor(record: Any) -> str:
    """The sm floor of one record. Empty means every sm level.

    CUDA intrinsics carry metadata["min_sm"]. PTX rows list their
    target levels in metadata["sm"]; the smallest one is the floor.
    Records of other families carry no floor key and always pass.
    """
    metadata = (
        record.get("metadata") if isinstance(record, dict) else getattr(record, "metadata", None)
    )
    metadata = metadata or {}
    floor = metadata.get("min_sm") or ""
    if floor:
        return str(floor)
    sm = metadata.get("sm") or ""
    tokens = (sm.split(",") if isinstance(sm, str) else [str(t) for t in sm]) if sm else []
    tokens = [t.strip() for t in tokens if str(t).strip()]
    return min(tokens, key=sm_number) if tokens else ""


def sm_levels(floors: Iterable[str]) -> list[str]:
    """The distinct non-empty floors, sorted by level, suffix kept."""
    return sorted({f for f in floors if f}, key=lambda f: (sm_number(f), f))


def sm_floors_from_db(conn: sqlite3.Connection) -> dict[str, str]:
    """Record key to sm floor, for the CUDA intrinsic and PTX rows only.

    The floors live inside the msgpack payload, so one scan per catalog
    feeds both the sm level list and the search-side filter.
    """
    from simdref.storage import _unpack_payload  # local import avoids cycle

    floors: dict[str, str] = {}
    queries = (
        ("SELECT name, payload FROM intrinsics_data WHERE architecture = 'cuda'"),
        ("SELECT db_key, payload FROM instructions_data WHERE architecture = 'ptx'"),
    )
    for sql in queries:
        try:
            rows = conn.execute(sql).fetchall()
        except sqlite3.Error:
            continue
        for row in rows:
            floor = record_sm_floor(_unpack_payload(row[1]))
            if floor:
                floors[row[0]] = floor
    return floors


@dataclass(frozen=True)
class PresetSpec:
    """Declarative preset: families + subs + arm_arch + kind applied atomically."""

    families: frozenset[str]
    subs: frozenset[str]
    arm_arch: frozenset[str] | None  # None = no arm_arch filter
    kind: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "families": sorted(self.families),
            "subs": sorted(self.subs),
            "arm_arch": sorted(self.arm_arch) if self.arm_arch is not None else None,
            "kind": sorted(self.kind),
        }

    @classmethod
    def empty(cls) -> "PresetSpec":
        return cls(frozenset(), frozenset(), None, frozenset())

    @classmethod
    def all(cls) -> "PresetSpec":
        return cls(
            families=frozenset(ISA_FAMILY_ORDER.keys()),
            subs=frozenset(sub for subs in FAMILY_SUB_ORDER.values() for sub in subs)
            | ARM_ALL_SUBS,
            arm_arch=None,
            kind=frozenset({"intrinsic", "instruction"}),
        )


ARCH_PRESETS: dict[str, PresetSpec] = {
    "default": PresetSpec(
        families=frozenset({"SSE", "AVX", "AVX-512"}),
        subs=frozenset(DEFAULT_PRESET_SUBS),
        arm_arch=None,
        kind=frozenset({"intrinsic"}),
    ),
    "intel": PresetSpec(
        # Include "x86" so BMI/POPCNT/LZCNT intrinsics (family=x86) are reachable.
        families=frozenset({"x86", "SSE", "AVX", "AVX-512"}),
        subs=frozenset(X86_64_V4_SUBS),
        arm_arch=None,
        kind=frozenset({"intrinsic", "instruction"}),
    ),
    "arm32": PresetSpec(
        families=frozenset({"Arm"}),
        subs=frozenset(ARM32_SUBS),
        arm_arch=frozenset({"A32", "BOTH"}),
        kind=frozenset({"intrinsic", "instruction"}),
    ),
    "arm64": PresetSpec(
        families=frozenset({"Arm"}),
        subs=frozenset(ARM64_SUBS),
        arm_arch=frozenset({"A64", "BOTH"}),
        kind=frozenset({"intrinsic", "instruction"}),
    ),
    "riscv": PresetSpec(
        families=frozenset({"RISC-V"}),
        subs=frozenset(RVV_SUBS),
        arm_arch=None,
        kind=frozenset({"intrinsic", "instruction"}),
    ),
    # CUDA has no sub-ISA breakdown in FAMILY_SUB_ORDER, so subs stays empty;
    # the family filter alone scopes results to CUDA and PTX rows.
    "cuda": PresetSpec(
        families=frozenset({"CUDA", "PTX"}),
        subs=frozenset(),
        arm_arch=None,
        kind=frozenset({"intrinsic", "instruction"}),
    ),
    "none": PresetSpec.empty(),
    "all": PresetSpec.all(),
}


@dataclass(frozen=True)
class CategorySpec:
    """A category facet derived from the catalog database."""

    family: str
    category: str
    subcategory: str = ""
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "category": self.category,
            "subcategory": self.subcategory,
            "count": self.count,
        }


@dataclass
class FilterSpec:
    """Filterable facets usable by TUI, web, CLI, and LLM consumers."""

    family_order: dict[str, int] = field(default_factory=lambda: dict(ISA_FAMILY_ORDER))
    # The family chips a surface shows: every family but the preset-only ones.
    chip_families: tuple[str, ...] = field(
        default_factory=lambda: tuple(n for n in ISA_FAMILY_ORDER if n not in PRESET_ONLY_FAMILIES)
    )
    family_sub_order: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in FAMILY_SUB_ORDER.items()}
    )
    default_enabled: tuple[str, ...] = DEFAULT_ENABLED_ISAS
    default_subs: dict[str, set[str]] = field(
        default_factory=lambda: {k: set(v) for k, v in DEFAULT_SUBS.items()}
    )
    categories: list[CategorySpec] = field(default_factory=list)
    presets: dict[str, PresetSpec] = field(default_factory=lambda: dict(ARCH_PRESETS))
    arm_arch_values: tuple[str, ...] = ARM_ARCH_VALUES
    sm_levels: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """JSON-friendly projection used by the web SPA."""
        return {
            "family_order": dict(self.family_order),
            "chip_families": list(self.chip_families),
            "family_sub_order": {k: list(v) for k, v in self.family_sub_order.items()},
            "default_enabled": list(self.default_enabled),
            "default_subs": {k: sorted(v) for k, v in self.default_subs.items()},
            "categories": [c.to_dict() for c in self.categories],
            "presets": {name: spec.to_dict() for name, spec in self.presets.items()},
            "arm_arch_values": list(self.arm_arch_values),
            "sm_levels": list(self.sm_levels),
        }

    # --- in-memory predicate --------------------------------------------------

    def matches(
        self,
        record: Any,
        enabled_families: Iterable[str] | None = None,
        enabled_categories: Iterable[str] | None = None,
    ) -> bool:
        """Return whether *record* passes the given family/category filter.

        Accepts anything with an ``isa`` iterable and optional ``category``.
        """
        families = set(enabled_families) if enabled_families is not None else None
        categories = set(enabled_categories) if enabled_categories is not None else None
        if families is not None:
            from simdref.display import isa_family  # local import avoids cycle

            record_families = {isa_family(v) for v in (getattr(record, "isa", None) or [])}
            if not record_families & families:
                return False
        if categories is not None:
            record_category = getattr(record, "category", None)
            if not record_category:
                record_category = (getattr(record, "metadata", {}) or {}).get("category", "")
            if record_category not in categories:
                return False
        return True

    # --- SQL pushdown helper --------------------------------------------------

    def sql_predicate(
        self,
        table: str,
        enabled_families: Iterable[str] | None = None,
        enabled_categories: Iterable[str] | None = None,
        enabled_arm_arch: Iterable[str] | None = None,
    ) -> tuple[str, list[Any]]:
        """Return a SQL ``WHERE`` fragment and bind values for *table*.

        Both ``intrinsics_data`` and ``instructions_data`` carry an indexed
        ``category`` column (schema v10+). ISA family filtering relies on
        substring match against the space-joined ``isa`` column.
        """
        clauses: list[str] = []
        binds: list[Any] = []
        if enabled_families:
            family_tokens: set[str] = set()
            for fam in enabled_families:
                family_tokens.update(self.family_sub_order.get(fam, []))
                family_tokens.add(fam)
            if family_tokens:
                clauses.append(
                    "(" + " OR ".join(f"{table}.isa LIKE ?" for _ in family_tokens) + ")"
                )
                binds.extend(f"%{tok}%" for tok in family_tokens)
        if enabled_categories and table in ("intrinsics_data", "instructions_data"):
            categories_list = list(enabled_categories)
            if categories_list:
                placeholders = ",".join("?" for _ in categories_list)
                clauses.append(f"{table}.category IN ({placeholders})")
                binds.extend(categories_list)
        if enabled_arm_arch and table == "intrinsics_data":
            arch_list = list(enabled_arm_arch)
            if arch_list:
                placeholders = ",".join("?" for _ in arch_list)
                clauses.append(f"{table}.arm_arch IN ({placeholders})")
                binds.extend(arch_list)
        return (" AND ".join(clauses), binds)


# ---------------------------------------------------------------------------
# Filter state
# ---------------------------------------------------------------------------


@dataclass
class FilterState:
    """The filter choices of one surface: ISA families and the sm level.

    ``min_sm`` is an "sm_XX" level from the catalog floor list, or ""
    for no sm filter.
    """

    enabled_families: frozenset[str] = frozenset()
    min_sm: str = ""

    def matches(self, record: Any, spec: FilterSpec | None = None) -> bool:
        """Whether one record passes the family and the sm filter."""
        spec = spec if spec is not None else FilterSpec()
        families = self.enabled_families or None
        return spec.matches(record, enabled_families=families) and runs_on_sm(
            record_sm_floor(record), self.min_sm
        )


# ---------------------------------------------------------------------------
# DB-backed category aggregation
# ---------------------------------------------------------------------------


def load_categories_from_db(conn: sqlite3.Connection) -> list[CategorySpec]:
    """Derive the category facet list from ``intrinsics_data``.

    Runs at build time so categories are data-driven rather than hardcoded.
    Instruction-side categories are reached through ``metadata->>'category'``
    on the payload and are skipped here (schema lacks an indexed column).
    """
    from simdref.display import isa_family  # local import avoids cycle

    queries = [
        """
        SELECT category, subcategory, isa, COUNT(*) AS n
        FROM intrinsics_data
        WHERE category != ''
        GROUP BY category, subcategory, isa
        """,
        """
        SELECT category, '' AS subcategory, isa, COUNT(*) AS n
        FROM instructions_data
        WHERE category != ''
        GROUP BY category, isa
        """,
    ]
    aggregate: dict[tuple[str, str, str], int] = {}
    for sql in queries:
        try:
            rows = conn.execute(sql).fetchall()
        except sqlite3.Error:
            continue
        for row in rows:
            isa_str = row["isa"] if hasattr(row, "keys") else row[2]
            category = row["category"] if hasattr(row, "keys") else row[0]
            subcategory = row["subcategory"] if hasattr(row, "keys") else row[1]
            count = row["n"] if hasattr(row, "keys") else row[3]
            tokens = [t for t in str(isa_str or "").split() if t]
            families = {isa_family(t) for t in tokens} or {"Other"}
            for family in families:
                key = (family, category, subcategory)
                aggregate[key] = aggregate.get(key, 0) + int(count)
    specs = [
        CategorySpec(family=fam, category=cat, subcategory=sub, count=n)
        for (fam, cat, sub), n in sorted(
            aggregate.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])
        )
    ]
    return specs


def build_filter_spec(conn: sqlite3.Connection | None = None) -> FilterSpec:
    """Build a :class:`FilterSpec`, deriving categories from *conn* if given."""
    spec = FilterSpec()
    if conn is not None:
        try:
            spec.categories = load_categories_from_db(conn)
            spec.sm_levels = tuple(sm_levels(sm_floors_from_db(conn).values()))
        except sqlite3.Error:
            spec.categories = []
    return spec
