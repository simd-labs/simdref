"""Attach ingested perf rows onto existing :class:`InstructionRecord` data."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
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


# A ZA tile ("<ZAT...><HV>.S[W1, #0]") or ZA array ("<ZDA>.S",
# "Z3.S[<WV>, #0...]") operand. No perf source measures ZA forms.
_ZA_OPERAND = re.compile(r"<ZAT|<ZAN|<ZAD|<ZDA|\[<WV>|\[W1")


def _has_za_operand(form: str) -> bool:
    return _ZA_OPERAND.search(form) is not None


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
    # Operand classes per key position (e.g. frozenset({"xmm"}) for an AMD
    # workbook ``xmm1``). Empty means the row matches by the form string.
    form_classes: tuple[frozenset[str], ...] = ()

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


_KEY_GPR_WIDTHS = {
    "al": "gpr8",
    "cl": "gpr8",
    "r8l": "gpr8",
    "r8h": "gpr8",
    "ax": "gpr16",
    "dx": "gpr16",
    "r16": "gpr16",
    "eax": "gpr32",
    "ecx": "gpr32",
    "edx": "gpr32",
    "r32": "gpr32",
    "rax": "gpr64",
    "rcx": "gpr64",
    "rdx": "gpr64",
    "r64": "gpr64",
    "r": "gpr0",
}


def _key_operand_class(token: str) -> str:
    # Map one catalog key operand ("XMM", "M128", "I8", "Rel32") to the
    # class an AMD workbook row names for it. ``0`` marks unknown width.
    c = token.strip().casefold()
    if c.startswith("vsib") or c.startswith("moffs") or c == "agen":
        return "mem0"
    mem = re.fullmatch(r"m(\d+).*", c)
    if mem:
        return f"mem{int(mem.group(1))}"
    if c == "m":
        return "mem0"
    if c in ("xmm", "ymm", "zmm", "mm", "k"):
        return "mmx" if c == "mm" else c
    if c.startswith("st"):
        return "x87"
    if c in ("i8", "i16", "i32", "i64"):
        return "imm" + c[1:]
    if c in ("rel8", "rel16", "rel32", "abs64"):
        return "imm" + c[3:]
    if c.isdigit():
        return "imm0"
    if c in _KEY_GPR_WIDTHS:
        return _KEY_GPR_WIDTHS[c]
    return "other"


@lru_cache(maxsize=65536)
def _record_form_classes(form: str) -> tuple[str, ...]:
    # Operand classes of one catalog form, read from the key text so every
    # source (uops.info, SDM, vendored) maps the same way.
    match = re.match(r"^[^(]+\((.*)\)", form)
    if not match:
        return ()
    return tuple(_key_operand_class(part) for part in match.group(1).split(","))


def _form_class_match(
    row_classes: tuple[frozenset[str], ...], record_classes: tuple[str, ...]
) -> bool:
    if len(row_classes) != len(record_classes):
        return False
    for allowed, actual in zip(row_classes, record_classes):
        base = re.match(r"[a-z]+", actual).group(0) + "*"
        if actual not in allowed and base not in allowed:
            return False
    return True


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
    to every variant of the mnemonic (``applies_to="mnemonic"``), except ZA
    tile and ZA array forms, which no source measures.

    ``overwrite=False`` (default) keeps one value per ``arch_details[core]``
    slot: an incoming row replaces the existing entry only when its source
    ranks higher in :data:`SOURCE_PRIORITY`. Slots whose source is unknown
    to the priority table are never displaced, so passes that predate the
    table keep their data. A slot with empty ``latencies`` is not blocking:
    a lower-priority row fills its latency without displacing the slot.

    Returns the number of ``arch_details[core]`` entries newly written.
    """
    records = list(records)
    # amd_swopt imports this module, so the import stays function-local.
    # Mnemonic keys normalize the way the byte matcher does: the workbook
    # "JMP (near)" and the catalog "JMP" share one key.
    from simdref.perf_sources.amd_swopt import _norm_mnemonic

    by_arch_mnemonic: dict[tuple[str, str], list[InstructionRecord]] = {}
    for record in records:
        by_arch_mnemonic.setdefault(
            (record.architecture.casefold(), _norm_mnemonic(record.mnemonic)),
            [],
        ).append(record)

    rows = list(rows)
    amd_at = [i for i, row in enumerate(rows) if row.source == SOURCE_AMD_SWOPT]
    amd_matches: dict[int, Any] = {}
    if amd_at:
        # AMD workbook rows name operand classes, so at ingest the encoded
        # bytes pick the catalog rows. None keeps the operand-class matcher
        # (tool missing or no encoding identity in the catalog).
        from simdref.perf_sources.amd_swopt import match_rows_by_bytes

        resolved = match_rows_by_bytes([rows[i] for i in amd_at], records)
        if resolved is not None:
            amd_matches = dict(zip(amd_at, resolved))

    written = 0
    for i, row in enumerate(rows):
        # Only the AMD workbook carries parenthetical mnemonics that must
        # normalize like the byte matcher ("JMP (near)" keys "jmp"); other
        # sources keep their raw key so nothing else reattaches.
        key_mnemonic = (
            _norm_mnemonic(row.mnemonic)
            if row.source == SOURCE_AMD_SWOPT
            else row.mnemonic.casefold()
        )
        targets = by_arch_mnemonic.get(
            (row.architecture.casefold() or _arch_guess(row.core), key_mnemonic),
            [],
        )
        if not targets:
            continue
        form_key = row.form.strip().casefold()
        if i in amd_matches:
            # Bytes decide; a row that names no catalog row is dropped
            # here and never spread over every variant.
            named = set(amd_matches[i].targets)
            matched = [r for r in targets if r.db_key in named]
        else:
            if row.form_classes:
                # Class rows (AMD workbook operand codes) land on the catalog
                # forms whose operands match the classes position by position.
                matched = [
                    r
                    for r in targets
                    if _form_class_match(row.form_classes, _record_form_classes(r.form))
                ]
            else:
                matched = [
                    r for r in targets if not form_key or r.form.strip().casefold() == form_key
                ]
                if not form_key:
                    # Mnemonic rows spread over every form except ZA forms:
                    # no source measures ZA tiles or ZA arrays (#65 M11).
                    matched = [r for r in matched if not _has_za_operand(r.form)]
            # Plain form rows keep the mnemonic-level spread when the form
            # text matches nothing (intel-guide and friends name iforms).
            # An AMD class row the classes do not match attaches nowhere.
            if not matched and form_key and not row.form_classes:
                matched = [r for r in targets if not _has_za_operand(r.form)]
        entry = row.as_arch_details_entry()
        new_rank = _SOURCE_RANK.get(row.source)
        for record in matched:
            existing = record.arch_details.get(row.core)
            if existing is not None and not overwrite:
                old_rank = _entry_rank(existing)
                if new_rank is None or old_rank is None or new_rank >= old_rank:
                    # An empty latency slot must not block a lower source
                    # from filling latency (#65 M12); other fields stay.
                    if entry.get("latencies") and not existing.get("latencies"):
                        existing["latencies"] = entry["latencies"]
                        written += 1
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
