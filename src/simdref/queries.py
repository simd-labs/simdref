"""Shared record-linking and lookup helpers.

Functions here resolve relationships between intrinsics and instructions,
optionally using a SQLite connection for fast lookups or falling back to
in-memory catalog scans.  They are used by the CLI, LSP, and man-page
modules.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from simdref.perf import record_iclass, variant_perf_summary
from simdref.perf_sources.merge import _FIELD_SOURCE_KEYS, source_kind_for
from simdref.storage import load_instruction_from_db


def filter_arch_details_by_source_kind(
    arch_details: dict[str, dict[str, object]],
    source_kind: str,
) -> dict[str, dict[str, object]]:
    """Drop ``arch_details`` entries whose provenance does not match.

    ``"any"`` (or empty) keeps everything; ``"measured"`` / ``"modeled"``
    keep entries holding a value of that kind. Each field counts under its
    own source (a borrowed latency under its ``latency_source`` donor), so
    a mixed slot matches both kinds. Legacy entries without a
    ``source_kind`` key are treated as measured.
    """
    if source_kind in ("", "any"):
        return arch_details
    kept: dict[str, dict[str, object]] = {}
    for core, details in arch_details.items():
        kinds = {details.get("source_kind") or "measured"}
        kinds |= {
            source_kind_for(details.get(key))
            for key in _FIELD_SOURCE_KEYS.values()
            if details.get(key)
        }
        if source_kind in kinds:
            kept[core] = details
    return kept


if TYPE_CHECKING:
    from simdref.models import Catalog, InstructionRecord, IntrinsicRecord


def linked_instruction_records(
    catalog: Catalog | None,
    intrinsic: IntrinsicRecord,
    conn: sqlite3.Connection | None = None,
    cache: dict[str, object] | None = None,
) -> list[InstructionRecord]:
    """Return instruction records linked to *intrinsic*.

    When *conn* is provided, performs fast DB lookups by instruction key.
    Otherwise falls back to scanning ``catalog.instructions``. Pass *cache*
    to share already-loaded records with the caller and skip repeat queries.
    """
    ref_keys = [
        ref.get("key", "").strip()
        for ref in intrinsic.instruction_refs
        if isinstance(ref, dict) and ref.get("key", "").strip()
    ]
    if conn is not None:
        linked: list[InstructionRecord] = []
        keys = ref_keys or intrinsic.instructions
        for instruction_key in keys:
            instruction = cache.get(instruction_key) if cache is not None else None
            if instruction is None:
                instruction = load_instruction_from_db(conn, instruction_key)
                if instruction is not None and cache is not None:
                    cache[instruction_key] = instruction
            if instruction is not None:
                linked.append(instruction)
        return linked
    if catalog is None:
        return []
    return [
        instruction
        for instruction in catalog.instructions
        if intrinsic.name in instruction.linked_intrinsics
    ]


def build_intrinsic_instruction_index(
    catalog: Catalog,
) -> dict[str, list[InstructionRecord]]:
    """Invert ``catalog.instructions`` into an ``intrinsic name → instructions`` map.

    One linear pass replaces per-intrinsic scans of the instruction list,
    turning ``O(intrinsics × instructions)`` into ``O(intrinsics + instructions)``.
    """
    index: dict[str, list[InstructionRecord]] = {}
    for instruction in catalog.instructions:
        for name in instruction.linked_intrinsics:
            index.setdefault(name, []).append(instruction)
    return index


def _rows_from_linked_instructions(
    linked: list[InstructionRecord],
) -> list[dict]:
    rows: list[dict] = []
    for instruction in linked:
        if instruction.metrics:
            for arch, values in sorted(instruction.metrics.items()):
                row = {"instruction": instruction.key, "uarch": arch}
                row.update(values)
                rows.append(row)
        else:
            rows.append(
                {
                    "instruction": instruction.key,
                    "uarch": "-",
                    "latency": "-",
                    "throughput": "-",
                    "uops": "-",
                    "ports": "-",
                }
            )
    return rows


def instruction_rows_for_intrinsic_indexed(
    linked: list[InstructionRecord],
) -> list[dict]:
    """Build per-microarchitecture metric rows from a prebuilt linked list."""
    return _rows_from_linked_instructions(linked)


def intrinsic_perf_summary_runtime(
    conn: sqlite3.Connection,
    intrinsic: IntrinsicRecord,
    instruction_map: dict[str, object],
) -> tuple[str, str]:
    """Return ``(best_latency, best_cpi)`` across linked instruction variants.

    Resolves linked instructions via *instruction_map* (populated during
    search) and falls back to ``load_instruction_from_db`` for cache misses.
    """
    linked = linked_instruction_records(None, intrinsic, conn, cache=instruction_map)
    if not linked:
        return "-", "-"
    perf_pairs = [
        variant_perf_summary(record.arch_details, record_iclass(record)) for record in linked
    ]
    latencies = [float(lat) for lat, _ in perf_pairs if lat not in {"-", ""}]
    cpis = [float(cpi_value) for _, cpi_value in perf_pairs if cpi_value not in {"-", ""}]
    lat = str(min(latencies)).rstrip("0").rstrip(".") if latencies else "-"
    cpi = f"{min(cpis):.2f}" if cpis else "-"
    return lat, cpi
