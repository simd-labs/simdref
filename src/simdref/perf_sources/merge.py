"""Attach ingested perf rows onto existing :class:`InstructionRecord` data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from simdref.models import InstructionRecord

# Perf-source ids, best first. When several sources give perf for one
# instruction form on one core, the catalog keeps exactly one value: the
# one from the highest-priority source.
SOURCE_UOPS_INFO = "uops.info"
SOURCE_INSTLATX64 = "instlatx64"
SOURCE_INTEL_GUIDE = "intel-guide"
SOURCE_INTEL_OPTREF = "intel-optref"
SOURCE_AMD_SWOPT = "amd-swopt"
SOURCE_ARM_SWOPT = "arm-swopt"
SOURCE_LLVM_MCA = "llvm-mca"
SOURCE_LLVM_SCHEDULING = "llvm_scheduling"

SOURCE_PRIORITY: tuple[str, ...] = (
    SOURCE_UOPS_INFO,
    SOURCE_INSTLATX64,
    SOURCE_INTEL_GUIDE,
    SOURCE_INTEL_OPTREF,
    SOURCE_AMD_SWOPT,
    SOURCE_ARM_SWOPT,
    SOURCE_LLVM_MCA,
    SOURCE_LLVM_SCHEDULING,
)
_SOURCE_RANK = {source: rank for rank, source in enumerate(SOURCE_PRIORITY)}


def _entry_rank(entry: dict[str, Any]) -> int | None:
    # Entries written by parse_uops_xml carry no ``source`` key; they are
    # uops.info rows and take the top rank.
    return _SOURCE_RANK.get(entry.get("source") or SOURCE_UOPS_INFO)


@dataclass(frozen=True)
class PerfRow:
    """A single perf observation produced by a perf_sources ingester.

    Attributes map onto the provenance keys stored in
    ``InstructionRecord.arch_details[core]`` so :func:`merge_perf_rows` can
    drop them in verbatim.
    """

    mnemonic: str
    core: str
    source: str
    source_kind: str  # "measured" | "modeled"
    source_version: str
    architecture: str = ""
    form: str = ""
    latency: str = ""
    cpi: str = ""
    applies_to: str = "mnemonic"
    citation_url: str = ""
    extra_measurement: dict[str, str] = field(default_factory=dict)

    def as_arch_details_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "source": self.source,
            "source_kind": self.source_kind,
            "source_version": self.source_version,
            "applies_to": self.applies_to,
            "citation_url": self.citation_url,
        }
        if self.latency:
            entry["latencies"] = [{"cycles": self.latency}]
        measurement: dict[str, str] = dict(self.extra_measurement)
        if self.cpi:
            # ``TP_loop`` is the column the TUI and CLI measurement tables
            # already render under the "CPI" header, so modeled rows must
            # land there to surface at all. ``TP`` is retained as a
            # duplicate for any downstream consumer that still reads the
            # legacy key.
            measurement.setdefault("TP_loop", self.cpi)
            measurement.setdefault("TP", self.cpi)
        if measurement:
            entry["measurement"] = measurement
        return entry


def _record_key(record: InstructionRecord) -> tuple[str, str, str]:
    return (
        record.architecture.casefold(),
        record.mnemonic.casefold(),
        record.form.strip().casefold(),
    )


def merge_perf_rows(
    records: Iterable[InstructionRecord],
    rows: Iterable[PerfRow],
    *,
    overwrite: bool = False,
) -> int:
    """Merge *rows* into matching records in-place.

    Matching rule: a row attaches to every record with the same architecture
    and mnemonic. If ``row.form`` is non-empty, only records with the
    matching form are updated (case-insensitive). Otherwise the row applies
    to every variant of the mnemonic (``applies_to="mnemonic"``).

    ``overwrite=False`` (default) keeps one value per ``arch_details[core]``
    slot: an incoming row replaces the existing entry only when its source
    ranks higher in :data:`SOURCE_PRIORITY`. Slots whose source is unknown
    to the priority table are never displaced, so passes that predate the
    table keep their data.

    Returns the number of ``arch_details[core]`` entries newly written.
    """
    records = list(records)
    by_arch_mnemonic: dict[tuple[str, str], list[InstructionRecord]] = {}
    for record in records:
        by_arch_mnemonic.setdefault(
            (record.architecture.casefold(), record.mnemonic.casefold()),
            [],
        ).append(record)

    written = 0
    for row in rows:
        targets = by_arch_mnemonic.get(
            (row.architecture.casefold() or _arch_guess(row.core), row.mnemonic.casefold()),
            [],
        )
        if not targets:
            continue
        form_key = row.form.strip().casefold()
        matched = [r for r in targets if not form_key or r.form.strip().casefold() == form_key]
        if not matched and form_key:
            matched = targets  # graceful fallback: attach to every variant
        entry = row.as_arch_details_entry()
        new_rank = _SOURCE_RANK.get(row.source)
        for record in matched:
            existing = record.arch_details.get(row.core)
            if existing is not None and not overwrite:
                old_rank = _entry_rank(existing)
                if new_rank is None or old_rank is None or new_rank >= old_rank:
                    continue
            record.arch_details[row.core] = entry
            written += 1
    return written


def _arch_guess(core: str) -> str:
    """Best-effort architecture lookup used when :class:`PerfRow` lacks one."""
    from simdref.perf_sources.cores import core_architecture

    arch = core_architecture(core)
    if arch == "aarch64":
        return "arm"
    if arch == "riscv":
        return "riscv"
    return arch or ""
