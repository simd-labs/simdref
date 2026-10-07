"""Terminal display and formatting for simdref CLI output.

This module owns the shared :class:`~rich.console.Console` instance and all
functions that render intrinsics, instructions, performance tables, and ISA
metadata to the terminal.  The CLI command handlers in :mod:`simdref.cli`
delegate to these functions for all Rich-based output.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from simdref.perf import (
    latency_cycle_values,
    variant_perf_summary,
    variant_perf_summary_labeled,
)
from simdref.pdfrefs import normalize_pdf_refs, pdf_ref_label
from simdref.queries import linked_instruction_records

if TYPE_CHECKING:
    import sqlite3

    from simdref.models import Catalog, InstructionRecord, IntrinsicRecord
    from simdref.search import SearchResult

console = Console()

# ---------------------------------------------------------------------------
# Microarchitecture constants
# ---------------------------------------------------------------------------

UARCH_ORDER = [
    "ARL-P",
    "ARL-E",
    "MTL-P",
    "MTL-E",
    "EMR",
    "ADL-P",
    "ADL-E",
    "RKL",
    "TGL",
    "ICL",
    "CLX",
    "CNL",
    "SKX",
    "CFL",
    "KBL",
    "SKL",
    "BDW",
    "HSW",
    "IVB",
    "SNB",
    "ZEN5",
    "ZEN4",
    "ZEN3",
    "ZEN2",
    "ZEN+",
]

UARCH_LABELS: dict[str, tuple[str, str]] = {
    "ARL-P": ("Arrow Lake-P", "2024"),
    "ARL-E": ("Arrow Lake-E", "2024"),
    "MTL-P": ("Meteor Lake-P", "2023"),
    "MTL-E": ("Meteor Lake-E", "2023"),
    "EMR": ("Emerald Rapids", "2023"),
    "ADL-P": ("Alder Lake-P", "2021"),
    "ADL-E": ("Alder Lake-E", "2021"),
    "RKL": ("Rocket Lake", "2021"),
    "TGL": ("Tiger Lake", "2020"),
    "ICL": ("Ice Lake", "2019"),
    "CLX": ("Cascade Lake", "2019"),
    "CNL": ("Cannon Lake", "2018"),
    "SKX": ("Skylake-X", "2017"),
    "CFL": ("Coffee Lake", "2017"),
    "KBL": ("Kaby Lake", "2016"),
    "SKL": ("Skylake", "2015"),
    "BDW": ("Broadwell", "2014"),
    "HSW": ("Haswell", "2013"),
    "IVB": ("Ivy Bridge", "2012"),
    "SNB": ("Sandy Bridge", "2011"),
    "ZEN5": ("Zen 5", "2024"),
    "ZEN4": ("Zen 4", "2022"),
    "ZEN3": ("Zen 3", "2020"),
    "ZEN2": ("Zen 2", "2019"),
    "ZEN+": ("Zen+", "2018"),
    "AMT": ("Atom", "2012"),
    "BNL": ("Bonnell", "2008"),
    "CON": ("Conroe", "2006"),
    "GLM": ("Goldmont", "2016"),
    "GLP": ("Goldmont Plus", "2017"),
    "NHM": ("Nehalem", "2008"),
    "TRM": ("Tremont", "2019"),
    "WOL": ("Wolfdale", "2007"),
    "WSM": ("Westmere", "2010"),
}

# ISA chronological ordering for sort keys.
ISA_CHRONOLOGY: dict[str, tuple[int, int]] = {
    "I86": (0, 0),
    "MMX": (1, 0),
    "SSE": (2, 0),
    "SSE2": (2, 1),
    "SSE3": (2, 2),
    "SSSE3": (2, 3),
    "SSE4A": (2, 4),
    "SSE4.1": (2, 5),
    "SSE4.2": (2, 6),
    "AES": (2, 7),
    "PCLMULQDQ": (2, 8),
    "F16C": (3, 0),
    "FMA": (3, 1),
    "AVX": (4, 0),
    "NEON": (4, 1),
    "AVX2": (5, 0),
    "AVX512F": (6, 0),
    "AVX512DQ": (6, 1),
    "AVX512IFMA": (6, 2),
    "AVX512PF": (6, 3),
    "AVX512ER": (6, 4),
    "AVX512CD": (6, 5),
    "AVX512BW": (6, 6),
    "AVX512VL": (6, 7),
    "AVX512VBMI": (6, 8),
    "AVX512VBMI2": (6, 9),
    "AVX512VNNI": (6, 10),
    "AVX512BITALG": (6, 11),
    "AVX512VPOPCNTDQ": (6, 12),
    "AVX5124VNNIW": (6, 13),
    "AVX5124FMAPS": (6, 14),
    "AVX512VP2INTERSECT": (6, 15),
    "AVX512BF16": (6, 16),
    "AVX512FP16": (6, 99),
    "AVX10": (7, 0),
    "AMX": (7, 0),
    "SVE": (8, 0),
    "SVE2": (8, 1),
    "V": (8, 2),
    "ZVE": (8, 3),
    "ZV": (8, 4),
    "APX": (9, 0),
}

_ISA_PREFIXES_BY_LEN = sorted(ISA_CHRONOLOGY.items(), key=lambda kv: len(kv[0]), reverse=True)

# ISA family/sub-ISA constants live in simdref.filters (single source of truth).
# Re-exported here for back-compat with TUI, web export, and external callers.
from simdref.filters import (  # noqa: E402  (re-export)
    DEFAULT_ENABLED_ISAS,
    DEFAULT_SUBS,
    FAMILY_SUB_ORDER,
    ISA_FAMILY_ORDER,
)

X86_BASE_ISAS: frozenset[str] = frozenset(
    {
        "I86",
        "I186",
        "I286",
        "I386",
        "I486",
        "I586",
        "X87",
        "CMOV",
    }
)

# ---------------------------------------------------------------------------
# Compiled patterns (simple helpers avoid regex where possible)
# ---------------------------------------------------------------------------

_NORMALIZE_TEXT_RE = re.compile(r"[^a-z0-9]+")

# Matches display-only instruction decorators like ``{evex}``, ``{load}``,
# ``{disp8}``, etc. at the beginning of a form/mnemonic.
_LEADING_INSTR_TAG_RE = re.compile(r"^(?:\s*\{[^}]+\}\s*)+")

# ---------------------------------------------------------------------------
# Microarchitecture helpers
# ---------------------------------------------------------------------------


def uarch_sort_key(name: str) -> tuple[int, str]:
    """Sort key that places known architectures in chronological order."""
    try:
        return (UARCH_ORDER.index(name), name)
    except ValueError:
        return (len(UARCH_ORDER), name)


def _uarch_display_mode() -> str:
    width = console.size.width
    if width >= 145:
        return "full"
    if width >= 115:
        return "year"
    return "short"


def display_uarch(name: str, mode: str | None = None) -> str:
    """Format a microarchitecture code for display."""
    if not name:
        return "-"
    mode = mode or _uarch_display_mode()
    label = UARCH_LABELS.get(name)
    if label is None:
        return name
    full_name, year = label
    if mode == "full":
        return f"{full_name} ({year})"
    if mode == "year":
        return f"{name} ({year})"
    return name


def _uarch_display_mode_for_table(
    rows: list[dict], columns: list[str], label_map: dict[str, str]
) -> str:
    if "uarch" not in columns:
        return "short"
    available = console.size.width - 8
    sample_rows = rows[:64]
    for column in columns:
        if column == "uarch":
            continue
        header = label_map.get(column, column)
        max_cell = max((len(str(row.get(column, "-"))) for row in sample_rows), default=1)
        available -= min(max(len(header), max_cell), 22) + 3
    uarch_names = [str(row.get("uarch", "-")) for row in sample_rows]
    full_needed = max((len(display_uarch(name, mode="full")) for name in uarch_names), default=0)
    year_needed = max((len(display_uarch(name, mode="year")) for name in uarch_names), default=0)
    if available >= full_needed + 2:
        return "full"
    if available >= year_needed + 2:
        return "year"
    return "short"


def _column_width_budget(
    rows: list[dict], columns: list[str], label_map: dict[str, str], uarch_mode: str
) -> dict[str, int]:
    """Compute column widths that fit the terminal, shrinking least important first."""
    sample_rows = rows[:64]
    desired: dict[str, int] = {}
    minimum: dict[str, int] = {}
    for column in columns:
        header = label_map.get(column, column)
        if column == "uarch":
            values = [
                display_uarch(str(row.get("uarch", "-")), mode=uarch_mode) for row in sample_rows
            ]
            desired[column] = max([len(header), *[len(v) for v in values]] or [len(header)])
            minimum[column] = 5
            continue
        values = [str(row.get(column, "-")) for row in sample_rows]
        widest = max([len(header), *[len(v) for v in values]] or [len(header)])
        desired[column] = min(widest, 34 if column == "ports" else 22)
        if column == "ports":
            minimum[column] = 10
        elif column.startswith("TP") or column.startswith("cycle/") or column == "latency":
            minimum[column] = 8
        elif column == "uops":
            minimum[column] = 4
        else:
            minimum[column] = min(max(len(header), 4), 10)

    separator_cost = max(0, (len(columns) - 1) * 3)
    available = max(40, console.size.width - 8 - separator_cost)
    current = sum(desired.values())
    if current <= available:
        return desired

    widths = dict(desired)
    shrink_order = ["ports", "TP_unrolled", "TP_loop", "TP_ports", "TP", "latency", "uarch", "uops"]
    shrink_order += [c for c in columns if c not in shrink_order]
    deficit = current - available
    while deficit > 0:
        changed = False
        for column in shrink_order:
            if column not in widths:
                continue
            if widths[column] > minimum[column]:
                widths[column] -= 1
                deficit -= 1
                changed = True
                if deficit <= 0:
                    break
        if not changed:
            break
    return widths


# ---------------------------------------------------------------------------
# ISA display helpers
# ---------------------------------------------------------------------------


def strip_instruction_decorators(text: str) -> str:
    """Remove display-only leading decorators and EVEX suffix markers."""
    result = text or ""
    while True:
        stripped = _LEADING_INSTR_TAG_RE.sub("", result, count=1)
        if stripped == result:
            break
        result = stripped
    return result.replace("_EVEX", "").strip()


def display_isa(values: list[str]) -> str:
    """Normalize and deduplicate ISA extension names for display."""

    def normalize_token(value: str) -> str:
        # Strip width and scalar suffixes.
        base = value
        for suffix in ("_128", "_256", "_512", "_SCALAR"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        upper = base.upper()
        if upper in {"ADV_SIMD", "ADVSIMD", "NEON"}:
            return "NEON"
        if upper.startswith("SVE2"):
            return "SVE2"
        if upper.startswith("SVE"):
            return "SVE"
        if upper == "AARCH64":
            return "AArch64"
        if upper.startswith("AVX10_"):
            parts = base.split("_")
            if len(parts) >= 2:
                version = parts[1]
                suffix_str = " ".join(parts[2:]) if len(parts) > 2 else ""
                return f"AVX10.{version}{(' ' + suffix_str) if suffix_str else ''}"
        if upper.startswith("AVX512"):
            tail = base[len("AVX512") :]
            if tail.startswith("_"):
                return f"AVX512 {tail[1:].replace('_', ' ')}"
            return f"AVX512{tail}"
        if upper.startswith("AMX_"):
            return f"AMX-{base.split('_', 1)[1].replace('_', '-')}"
        return base

    rendered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_token(value)
        if normalized not in seen:
            seen.add(normalized)
            rendered.append(normalized)
    return ", ".join(rendered) or "-"


def display_architecture(architecture: str) -> str:
    return {
        "x86": "x86",
        "arm": "Arm",
        "riscv": "RISC-V",
    }.get((architecture or "").lower(), architecture or "-")


def normalize_isa_token(isa: str) -> str:
    """Normalize ISA tokens for family and sub-ISA matching."""
    return isa.upper().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")


def isa_family(isa: str) -> str:
    """Map a single ISA token to its high-level family."""
    d = normalize_isa_token(display_isa([isa]) or isa)
    if not d or d == "-":
        return "Other"
    if d in X86_BASE_ISAS or d.startswith("BMI") or d in ("ADX", "AES", "PCLMULQDQ", "CRC32"):
        return "x86"
    if d == "SVML":
        return "SVML"
    if d in {"AARCH64", "NEON"} or d.startswith("SVE"):
        return "Arm"
    if d == "V" or d.startswith("RV") or d.startswith("ZV") or d.startswith("ZVE"):
        return "RISC-V"
    if d.startswith("APX"):
        return "APX"
    if d.startswith("AMX"):
        return "AMX"
    if d.startswith("AVX10"):
        return "AVX10"
    if d.startswith("AVX512") or d in ("VAES", "VPCLMULQDQ", "GFNI"):
        return "AVX-512"
    if (
        d in ("AVX", "AVX2", "AVX2GATHER")
        or d.startswith("AVX")
        or d in ("FMA", "FMA4", "F16C", "XOP")
    ):
        return "AVX"
    if d.startswith("SSE") or d.startswith("SSSE"):
        return "SSE"
    if d.startswith("MMX") or d in ("3DNOW", "PENTIUMMMX"):
        return "MMX"
    return "Other"


def isa_families(values: list[str]) -> list[str]:
    """Deduplicated ISA families for a list of ISA tokens."""
    seen: set[str] = set()
    families: list[str] = []
    for value in values:
        family = isa_family(value)
        if family and family not in seen:
            seen.add(family)
            families.append(family)
    return families


def isa_to_sub_isa(raw_isa: str) -> str | None:
    """Map a raw ISA token to its configured family sub-ISA.

    Longest matching sub wins so that ``SSE2``/``SSE4.1`` don't collapse
    to ``SSE`` just because SSE appears earlier in FAMILY_SUB_ORDER.
    """
    family = isa_family(raw_isa)
    subs = FAMILY_SUB_ORDER.get(family)
    if not subs:
        return None
    normalized = normalize_isa_token(display_isa([raw_isa]) or raw_isa)
    # Exact match first, then longest-prefix match.
    for sub in subs:
        if normalize_isa_token(sub) == normalized:
            return sub
    best: str | None = None
    best_len = -1
    for sub in subs:
        candidate = normalize_isa_token(sub)
        if normalized.startswith(candidate) and len(candidate) > best_len:
            best = sub
            best_len = len(candidate)
    return best


def isa_sort_key(values: list[str]) -> tuple[int, int, str]:
    """Chronological sort key for a list of ISA extensions."""
    display = display_isa(values)
    normalized_values = [display_isa([v]).upper() for v in values] or [display.upper()]

    def isa_rank(value: str) -> tuple[int, int]:
        compact = value.replace(" ", "")
        for prefix, rank in _ISA_PREFIXES_BY_LEN:
            if compact.startswith(prefix):
                return rank
        for family in ("AVX10", "AMX", "APX"):
            if compact.startswith(family):
                return ISA_CHRONOLOGY[family]
        return (8, 0)

    return min((isa_rank(v) for v in normalized_values), default=(8, 0)) + (display,)


def is_apx_isa(values: list[str]) -> bool:
    return display_isa(values).upper().startswith("APX")


def is_fp16_or_bf16_isa(values: list[str]) -> bool:
    display = display_isa(values).upper()
    return "FP16" in display or "BF16" in display


def isa_visible(values: list[str], show_fp16: bool = False) -> bool:
    """Whether an ISA variant should be shown (hides APX and optionally FP16/BF16)."""
    if is_apx_isa(values):
        return False
    if not show_fp16 and is_fp16_or_bf16_isa(values):
        return False
    return True


# ---------------------------------------------------------------------------
# Instruction text helpers
# ---------------------------------------------------------------------------


_WEB_APP_URL = "https://simdref.diamondinoia.com"


def web_deep_link(kind: str, key: str) -> str:
    """Deep link into the simdref web app for a record key."""
    # Match the web app's encodeURIComponent(entry.key): parens stay literal.
    return f"{_WEB_APP_URL}/#{urllib.parse.quote(str(key), safe='()')}"


def instruction_markdown(item, conn=None) -> str:
    """Render the full instruction page as Markdown (hover docs, exports).

    Carries the same data as :func:`render_instruction` prints to the
    terminal: metadata, every description section, the intrinsic mapping
    and the per-microarchitecture performance table, plus reference URLs.
    """
    lines = [f"```asm\n{item.key}\n```"]
    lines.append(f"**isa.** {display_isa(item.isa)}")
    for key, value in instruction_metadata_rows(item):
        lines.append(f"**{key}.** {value}")
    if item.description:
        shown: set[str] = set()
        for section in _DESCRIPTION_ORDER:
            if section in item.description:
                lines.append(f"## {section}\n\n{item.description[section].strip()}")
                shown.add(section)
        for section, body in item.description.items():
            if section not in shown:
                lines.append(f"## {section}\n\n{body.strip()}")
    if item.linked_intrinsics:
        lines.append(
            "## instruction to intrinsic mapping\n\n"
            + "\n".join(f"- `{name}`" for name in item.linked_intrinsics)
        )
    lines.extend(_perf_markdown_lines(measurement_rows(item)))
    return "\n\n".join(lines)


def _perf_markdown_lines(rows: list[dict]) -> list[str]:
    """Markdown table for measurement rows, one section per source kind."""
    sections: list[str] = []
    for kind, group in split_perf_rows(rows):
        group = [row for row in group if row]
        columns = [
            column
            for column in _MEASUREMENT_PREFERRED_ORDER
            if column not in _MEASUREMENT_EXCLUDE_KEYS and any(column in row for row in group)
        ]
        if not columns:
            continue
        header = [_GENERIC_TABLE_LABEL_MAP.get(column, column) for column in columns]
        table = ["| " + " | ".join(header) + " |", "|" + "---|" * len(columns)]
        for row in sorted(group, key=lambda r: uarch_sort_key(r.get("uarch", ""))):
            cells = []
            for column in columns:
                value = row.get(column, "-")
                cells.append(display_uarch(str(value)) if column == "uarch" else str(value))
            table.append("| " + " | ".join(cells) + " |")
        sections.append(f"## {perf_panel_title(kind)}\n\n" + "\n".join(table))
    return sections


def canonical_url(path: str) -> str:
    if not path:
        return ""
    if path.startswith("http"):
        return path
    return f"https://www.{path.lstrip('/')}"


def instruction_query_text(item) -> str:
    """Build a clean query string from an instruction's form."""
    form = strip_instruction_decorators(item.form or "")
    name = strip_instruction_decorators(item.mnemonic or "")
    if "(" in form:
        form_name = form.split("(", 1)[0].strip()
        if form_name:
            name = form_name
            form = form[len(form_name) :].strip()
    elif form.casefold().startswith(item.mnemonic.casefold()):
        form = form[len(item.mnemonic) :].strip()
    if form.startswith("(") and form.endswith(")"):
        form = form[1:-1]
    tokens = [t for t in re.split(r"[\s,()]+", form) if t]
    return " ".join([name, *tokens]).strip()


def display_instruction_form(form: str) -> str:
    return strip_instruction_decorators(form or "") or "-"


def display_instruction_title(item) -> str:
    return display_instruction_form(item.key)


def normalize_instruction_query(value: str) -> str:
    """Normalize an instruction query to lowercase alphanumeric tokens."""
    text = value.replace("_", " ").replace(",", " ").replace("{", " ").replace("}", " ")
    tokens = []
    for ch in text:
        if ch.isalnum():
            tokens.append(ch.lower())
        elif tokens and tokens[-1] != " ":
            tokens.append(" ")
    return "".join(tokens).strip()


def natural_query_sort_key(value: str) -> tuple[object, ...]:
    """Sort key that orders numbers numerically within text."""
    parts = re.findall(r"[A-Za-z]+|\d+", value.casefold())
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append((1, int(part)))
        else:
            key.append((0, part))
    return tuple(key)


def instruction_variant_items(items):
    """Sort instruction variants by ISA chronology then natural form order."""
    return sorted(
        items,
        key=lambda item: (
            isa_sort_key(item.isa),
            natural_query_sort_key(instruction_query_text(item)),
            item.key,
        ),
    )


def normalized_sentence(text: str) -> str:
    return _NORMALIZE_TEXT_RE.sub(" ", text.casefold()).strip()


# ---------------------------------------------------------------------------
# Row extraction helpers
# ---------------------------------------------------------------------------


def measurement_rows(item) -> list[dict]:
    """Extract per-microarchitecture measurement rows from an instruction.

    Each row carries a ``source`` column (``measured`` / ``modeled``) so
    downstream renderers can show provenance without re-reading the
    ``arch_details`` entry.
    """
    rows: list[dict] = []
    for arch, details in item.arch_details.items():
        measurement = details.get("measurement") or {}
        if measurement:
            row = {"uarch": arch, **measurement}
            values = latency_cycle_values(details.get("latencies") or [])
            if values:
                row["latency"] = values[0]
            row["source"] = details.get("source_kind") or "measured"
            rows.append(row)
    return rows


def doc_rows(item) -> list[dict]:
    rows: list[dict] = []
    for arch, details in item.arch_details.items():
        doc = details.get("doc") or {}
        if doc:
            rows.append({"uarch": arch, **doc})
    return rows


def iaca_rows(item) -> list[dict]:
    rows: list[dict] = []
    for arch, details in item.arch_details.items():
        for iaca in details.get("iaca") or []:
            rows.append({"uarch": arch, **iaca})
    return rows


def latency_rows(item) -> list[dict]:
    rows: list[dict] = []
    for arch, details in item.arch_details.items():
        values = latency_cycle_values(details.get("latencies") or [])
        if values:
            rows.append({"uarch": arch, "cycles": ", ".join(values)})
    return rows


# ---------------------------------------------------------------------------
# Rich table printing
# ---------------------------------------------------------------------------

_GENERIC_TABLE_LABEL_MAP = {
    "uarch": "microarch",
    "latency": "LAT",
    "TP_loop": "CPI",
    "TP_unrolled": "CPI unroll",
    "TP_ports": "CPI ports",
    "TP": "CPI",
    "TP_no_interiteration": "cycle/instr (no interiteration)",
    "kind": "op",
}


_PERF_PANEL_ORDER = ("measured", "modeled")
_PERF_PANEL_TITLES = {
    "measured": "perf (measured)",
    "modeled": "perf (modeled)",
}
_PERF_PANEL_BORDER = {
    "measured": "green",
    "modeled": "yellow",
}


def split_perf_rows(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group measurement rows by ``source`` kind in a stable order.

    Returns ``[(kind, rows), ...]`` with measured first, then modeled, so
    callers can render one panel per kind without re-sorting. Rows
    missing a ``source`` field fall under ``"measured"`` for back-compat
    with older arch_details entries.
    """
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        kind = row.get("source") or "measured"
        buckets.setdefault(kind, []).append(row)
    ordered = [(k, buckets[k]) for k in _PERF_PANEL_ORDER if k in buckets]
    # Any unexpected kinds (e.g. a new source kind) appear after the
    # known ones, sorted for determinism.
    for kind in sorted(buckets):
        if kind not in _PERF_PANEL_ORDER:
            ordered.append((kind, buckets[kind]))
    return ordered


def perf_panel_title(kind: str) -> str:
    """Human-readable panel title for a given source kind."""
    return _PERF_PANEL_TITLES.get(kind, f"perf ({kind})")


def perf_panel_border(kind: str) -> str:
    """Rich border style that distinguishes measured from modeled panels."""
    return _PERF_PANEL_BORDER.get(kind, "green")


def print_generic_table(
    rows: list[dict],
    title: str,
    preferred_order: list[str] | None = None,
    border_style: str = "green",
    exclude_keys: set[str] | None = None,
    include_extras: bool = True,
) -> None:
    """Render a performance data table inside a Rich Panel."""
    if not rows:
        return
    preferred_order = preferred_order or []
    exclude_keys = exclude_keys or set()
    keys = [k for k in preferred_order if any(k in row for row in rows)]
    extras = (
        sorted({k for row in rows for k in row if k not in keys and k not in exclude_keys})
        if include_extras
        else []
    )
    columns = keys + extras
    table = Table(header_style=f"bold {border_style}", expand=True)
    uarch_mode = _uarch_display_mode_for_table(rows, columns, _GENERIC_TABLE_LABEL_MAP)
    widths = _column_width_budget(rows, columns, _GENERIC_TABLE_LABEL_MAP, uarch_mode)
    for column in columns:
        label = _GENERIC_TABLE_LABEL_MAP.get(column, column)
        if column == "uarch":
            table.add_column(label, no_wrap=True, width=widths[column])
        elif column == "ports":
            table.add_column(label, no_wrap=True, width=widths[column], overflow="ellipsis")
        else:
            table.add_column(label, width=widths[column], overflow="fold")

    def row_sort(row: dict):
        return uarch_sort_key(row.get("uarch", "")) if "uarch" in row else ("",)

    for row in sorted(rows, key=row_sort):
        rendered = []
        for column in columns:
            value = row.get(column, "-")
            if column == "uarch":
                rendered.append(display_uarch(str(value), mode=uarch_mode))
            else:
                rendered.append(str(value))
        table.add_row(*rendered)
    console.print(Panel(table, title=title, border_style=border_style))


def print_perf_tables(rows: list[dict]) -> None:
    """Render one Rich panel per source kind (measured vs modeled).

    Splits *rows* on the ``source`` column and prints each group as its
    own table with a distinct title and border — no ``source`` column
    inside the table itself, since the panel header already names it.
    """
    for kind, group in split_perf_rows(rows):
        print_generic_table(
            group,
            perf_panel_title(kind),
            preferred_order=_MEASUREMENT_PREFERRED_ORDER,
            border_style=perf_panel_border(kind),
            exclude_keys=_MEASUREMENT_EXCLUDE_KEYS,
            include_extras=False,
        )


def print_operand_block(item) -> None:
    """Render an operand details table."""
    if not item.operand_details:
        return
    table = Table(header_style="bold blue")
    table.add_column("idx", width=4)
    table.add_column("rw", width=4)
    table.add_column("type", width=8)
    table.add_column("width", width=6)
    table.add_column("xtype", width=8)
    table.add_column("name", width=10)
    for operand in item.operand_details:
        rw = "".join(flag for flag in ("r", "w") if operand.get(flag) == "1")
        table.add_row(
            operand.get("idx", "-"),
            rw or "-",
            operand.get("type", "-"),
            operand.get("width", "-"),
            operand.get("xtype", "-"),
            operand.get("name", "-"),
        )
    console.print(Panel(table, title="operands", border_style="blue"))


def print_instruction_mapping(catalog, intrinsic, conn=None) -> None:
    """Show which instructions implement a given intrinsic."""
    linked = linked_instruction_records(catalog, intrinsic, conn=conn)
    if not linked:
        return
    table = Table(header_style="bold cyan")
    table.add_column("instruction", style="cyan")
    table.add_column("arch", width=6)
    table.add_column("summary")
    table.add_column("isa", width=12)
    for instr in linked:
        table.add_row(
            instr.key,
            display_architecture(instr.architecture),
            instr.summary or "-",
            ", ".join(instr.isa) or "-",
        )
    console.print(table)


def print_intrinsic_mapping(catalog, item, conn=None, find_intrinsic_fn=None) -> None:
    """Show which intrinsics use a given instruction."""
    if not item.linked_intrinsics:
        return
    from simdref.storage import load_intrinsic_from_db
    from simdref.search import find_intrinsic as _find_intrinsic

    find_fn = find_intrinsic_fn or _find_intrinsic
    table = Table(header_style="bold cyan")
    table.add_column("intrinsic", style="cyan")
    table.add_column("arch", width=6)
    table.add_column("summary")
    table.add_column("isa", width=12)
    for intrinsic_name in item.linked_intrinsics:
        intrinsic = (
            load_intrinsic_from_db(conn, intrinsic_name)
            if conn is not None
            else find_fn(catalog, intrinsic_name)
        )
        if intrinsic is None:
            table.add_row(intrinsic_name, "-", "-", "-")
            continue
        table.add_row(
            intrinsic.signature or intrinsic.name,
            display_architecture(intrinsic.architecture),
            intrinsic.description or "-",
            ", ".join(intrinsic.isa) or "-",
        )
    console.print(table)


_MEASUREMENT_PREFERRED_ORDER = [
    "uarch",
    "latency",
    "TP_loop",
    "TP_ports",
    "uops",
    "ports",
    "kind",
]
_MEASUREMENT_EXCLUDE_KEYS = {
    "uops_retire_slots",
    "uops_MITE",
    "uops_MS",
    "macro_fusible",
    "source",
}


def print_instruction_metadata(item) -> None:
    table = Table(show_header=False, box=None)
    for key, value in instruction_metadata_rows(item):
        table.add_row(key, value)
    console.print(Panel(table, title="instruction metadata", border_style="magenta"))


def instruction_metadata_rows(item) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [("summary", item.summary or "-")]
    url = item.metadata.get("url", "")
    if url:
        rows.append(("url", canonical_url(url)))
    if item.metadata.get("url-ref"):
        rows.append(("reference", canonical_url(item.metadata["url-ref"])))
    for ref in normalize_pdf_refs(getattr(item, "pdf_refs", []), item.metadata):
        url = ref.get("url", "")
        label = ref.get("label", "pdf").lower()
        page_label = pdf_ref_label(ref)
        if page_label.casefold() != (ref.get("label", "") or "").casefold():
            rows.append((label, f"{url} [{page_label}]"))
        else:
            rows.append((label, url))
    if item.metadata.get("extension"):
        rows.append(("isa", item.metadata["extension"]))
    if item.metadata.get("category"):
        rows.append(("category", item.metadata["category"]))
    if item.metadata.get("cpl"):
        rows.append(("cpl", item.metadata["cpl"]))
    return rows


_DESCRIPTION_ORDER = [
    "Description",
    "Operation",
    "Intrinsic Equivalents",
    "Flags Affected",
    "FPU Flags Affected",
    "Exceptions",
    "SIMD Floating-Point Exceptions",
    "Floating-Point Exceptions",
    "x87 FPU and SIMD Floating-Point Exceptions",
    "Numeric Exceptions",
    "Other Exceptions",
    "Other Mode Exceptions",
    "Protected Mode Exceptions",
    "Real-Address Mode Exceptions",
    "Real Address Mode Exceptions",
    "Virtual-8086 Mode Exceptions",
    "Virtual-8086 Exceptions",
    "Virtual 8086 Mode Exceptions",
    "Compatibility Mode Exceptions",
    "64-Bit Mode Exceptions",
]

# Sections that are always shown expanded.
_EXPANDED_SECTIONS: set[str] = set()

# Syntax language per code section.
_CODE_SECTION_LANG = {
    "Operation": "asm",
    "ACLE Operation": "asm",
    "Intrinsic Equivalents": "c",
}


def print_description_sections(
    description: dict[str, str],
    full: bool = False,
) -> None:
    """Render instruction description sections as Rich panels.

    By default, only Description and Flags Affected are expanded.
    Other sections show a collapsed summary line.  Pass *full=True*
    to expand everything.
    """
    if not description:
        return
    shown: set[str] = set()
    for key in _DESCRIPTION_ORDER:
        if key in description:
            expand = full or key in _EXPANDED_SECTIONS
            _print_section(key, description[key], expand=expand)
            shown.add(key)
    for key, value in description.items():
        if key not in shown:
            expand = full or key in _EXPANDED_SECTIONS
            _print_section(key, value, expand=expand)


def _print_section(title: str, body: str, expand: bool = True) -> None:
    from rich.syntax import Syntax
    from rich.text import Text

    if not expand:
        line_count = body.count("\n") + 1
        summary = Text(f"  \u25b8 {title} ({line_count} lines)", style="dim")
        console.print(summary)
        return

    lang = _CODE_SECTION_LANG.get(title)
    if lang:
        syntax = Syntax(body, lang, theme="monokai", word_wrap=True)
        console.print(Panel(syntax, title=title, border_style="dim"))
    else:
        console.print(Panel(body, title=title, border_style="dim"))


# ---------------------------------------------------------------------------
# High-level render functions
# ---------------------------------------------------------------------------


def render_intrinsic(catalog, item, conn=None, short: bool = False, full: bool = False) -> None:
    """Render full intrinsic detail view to the terminal."""
    table = Table(show_header=False, box=None)
    table.add_row("signature", item.signature or "-")
    table.add_row("header", item.header or "-")
    if item.url:
        table.add_row("source", item.url)
    table.add_row("architecture", display_architecture(item.architecture))
    table.add_row("isa", ", ".join(item.isa) or "-")
    table.add_row("category", item.category or "-")
    for key in (
        "reference_url",
        "argument_preparation",
        "result",
        "supported_architectures",
        "classification_path",
    ):
        if item.metadata.get(key):
            table.add_row(key, item.metadata[key])
    table.add_row("notes", "; ".join(item.notes) or "-")
    linked = linked_instruction_records(catalog, item, conn=conn)
    primary = linked[0] if linked else None
    if primary:
        for key, value in instruction_metadata_rows(primary):
            if key == "summary":
                continue
            table.add_row(key, value)
    console.print(Panel(table, title=f"intrinsic: {item.name}", border_style="cyan"))
    if not short and item.doc_sections:
        print_description_sections(item.doc_sections, full=full)
    if not short and primary and primary.description:
        print_description_sections(primary.description, full=full)
    if linked:
        console.print(Rule("intrinsic to instruction mapping", style="cyan"))
        print_instruction_mapping(catalog, item, conn=conn)
        console.print(
            Rule(f"instruction details: {display_instruction_title(primary)}", style="magenta")
        )
        print_operand_block(primary)
        print_perf_tables(measurement_rows(primary))


def render_instruction_sections(
    catalog, item, include_title: bool = True, conn=None, short: bool = False, full: bool = False
) -> None:
    """Render instruction detail with optional title panel."""
    if include_title:
        table = Table(show_header=False, box=None)
        table.add_row("mnemonic", item.mnemonic)
        table.add_row("form", display_instruction_form(item.form))
        table.add_row("architecture", display_architecture(item.architecture))
        table.add_row("isa", display_isa(item.isa))
        for key, value in instruction_metadata_rows(item):
            table.add_row(key, value)
        console.print(
            Panel(
                table,
                title=f"instruction: {display_instruction_title(item)}",
                border_style="magenta",
            )
        )
    else:
        print_instruction_metadata(item)
    if not short and item.description:
        print_description_sections(item.description, full=full)
    console.print(Rule("instruction to intrinsic mapping", style="cyan"))
    print_intrinsic_mapping(catalog, item, conn=conn)
    print_operand_block(item)
    print_perf_tables(measurement_rows(item))


def render_instruction(catalog, item, conn=None, short: bool = False, full: bool = False) -> None:
    """Render full instruction detail view."""
    render_instruction_sections(
        catalog, item, include_title=True, conn=conn, short=short, full=full
    )


def render_instruction_variants(query: str, items, show_fp16: bool = False) -> None:
    """Render a variant selection table for a mnemonic with multiple forms."""
    all_items = instruction_variant_items(items)
    visible_items = [item for item in all_items if isa_visible(item.isa, show_fp16=show_fp16)]
    items = visible_items or all_items
    table = Table(header_style="bold cyan")
    table.add_column("#", width=3, style="cyan")
    table.add_column("query", style="cyan")
    table.add_column("arch", width=6)
    table.add_column("isa", width=14)
    table.add_column("lat", width=5)
    table.add_column("cpi", width=5)
    table.add_column("summary")
    for index, item in enumerate(items, start=1):
        lat, cpi = variant_perf_summary_labeled(item.arch_details)
        table.add_row(
            str(index),
            instruction_query_text(item),
            display_architecture(item.architecture),
            display_isa(item.isa),
            _label_perf(lat),
            _label_perf(cpi),
            item.summary or "-",
        )


def _label_perf(value) -> str:
    """Render a :class:`PerfValue` as ``"val (measured, core)"`` for tables.

    Empty / missing values render as ``"-"`` to match the legacy behaviour
    the unit tests depend on.
    """
    raw = getattr(value, "value", value)
    if not raw or raw == "-":
        return "-"
    kind = getattr(value, "source_kind", "")
    core = getattr(value, "core", "")
    if kind and core:
        return f"{raw} ({kind[0]}, {core})"
    return str(raw)
    console.print(Panel(table, title=f"instruction variants: {query}", border_style="magenta"))
    base_query = query.split()[0] if query.split() else query
    if items:
        hidden_count = len(all_items) - len(items)
        hidden_note = (
            f" Hidden {hidden_count} APX/FP16/BF16 forms by default." if hidden_count > 0 else ""
        )
        console.print(
            f"[dim]Open one with `simdref {instruction_query_text(items[0])}` or `simdref {base_query} <index>`. Showing {len(items)} forms.{hidden_note}[/dim]"
        )


def render_search_results(
    results: list[tuple],
) -> None:
    """Render search results table from pre-computed rows.

    Each element in *results* is ``(SearchResult, arch_str, isa_str, lat, cpi)``.
    """
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("#", width=3, style="cyan")
    table.add_column("query", width=30, no_wrap=True, overflow="ellipsis")
    table.add_column("Arch", width=6)
    table.add_column("ISA", width=14)
    table.add_column("lat", width=5)
    table.add_column("cpi", width=5)
    table.add_column("summary", min_width=18, overflow="fold")
    for index, (result, arch, isa, lat, cpi) in enumerate(results, start=1):
        table.add_row(str(index), result.title, arch, isa, lat, cpi, result.subtitle)
    console.print(table)
