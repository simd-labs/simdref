"""Attach ingested perf rows onto existing :class:`InstructionRecord` data."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable

from simdref.models import InstructionRecord
from simdref.tablecols import mixed_number, strip_footnote

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


# Every perf source maps to one ``source_kind`` in the catalog; the
# modeled ones are named here and every other source (uops.info,
# instlatx64, intel-optref) is measured.
_MODELED_SOURCES = frozenset(
    {
        SOURCE_INTEL_GUIDE,
        SOURCE_AMD_SWOPT,
        SOURCE_ARM_SWOPT,
        SOURCE_LLVM_MCA,
        SOURCE_LLVM_SCHEDULING,
    }
)


def source_kind_for(source: str | None) -> str:
    """``"measured"`` or ``"modeled"`` for one perf-source id."""
    return "modeled" if (source or SOURCE_UOPS_INFO) in _MODELED_SOURCES else "measured"


def fmt_cycles(value: float) -> str:
    """CPI text: three decimals with trailing zeros and the point dropped."""
    return f"{value:.3f}".rstrip("0").rstrip(".")


def ipc_value(raw: str) -> Fraction | None:
    """Parse an ops-per-cycle cell (footnote, mixed number or plain number)."""
    raw = strip_footnote(raw)
    mixed = mixed_number(raw)
    try:
        return mixed if mixed is not None else Fraction(raw)
    except ValueError:
        return None


def cpi_for(ipc: Fraction) -> str:
    """Invert ops-per-cycle into the cycles-per-instruction text."""
    return fmt_cycles(float(1 / ipc))


def write_json(payload: Any, target: Path) -> None:
    """Write one vendored perf JSON document: sorted keys, compact separators."""
    target.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# One provenance key per merged field, written only when the value comes
# from a source other than the slot's own (#65 field-wise merge).
_FIELD_VALUE_KEYS = {
    "latencies": (),
    "cpi": ("TP_loop", "TP"),
    "uops": ("uops",),
    "ports": ("ports", "TP_ports"),
}
_FIELD_SOURCE_KEYS = {
    "latencies": "latency_source",
    "cpi": "cpi_source",
    "uops": "uops_source",
    "ports": "ports_source",
}
_FIELD_LABELS = {"latencies": "latency", "cpi": "cpi", "uops": "uops", "ports": "ports"}


def _has_field(entry: dict[str, Any], field: str) -> bool:
    if field == "latencies":
        return bool(entry.get("latencies"))
    measurement = entry.get("measurement") or {}
    return any(measurement.get(key) for key in _FIELD_VALUE_KEYS[field])


def field_source(details: dict[str, Any], field: str) -> str:
    """Source id that owns one perf field of a slot."""
    return details.get(_FIELD_SOURCE_KEYS[field]) or details.get("source") or SOURCE_UOPS_INFO


def field_kind(details: dict[str, Any], field: str) -> str:
    """Provenance kind of one perf field; a borrowed value keeps its donor's."""
    donor = details.get(_FIELD_SOURCE_KEYS[field])
    if donor:
        return source_kind_for(donor)
    return details.get("source_kind") or "measured"


def slot_kind(details: dict[str, Any]) -> str:
    """Kind label for a slot's perf fields: ``"mixed"`` when they disagree."""
    kinds = {
        field_kind(details, field) for field in _FIELD_VALUE_KEYS if _has_field(details, field)
    }
    if len(kinds) == 1:
        return kinds.pop()
    return "mixed" if kinds else (details.get("source_kind") or "measured")


def donor_source_kinds(details: dict[str, Any]) -> set[str]:
    """Provenance kinds of one slot: its owner kind plus each donor's."""
    kinds = {details.get("source_kind") or "measured"}
    kinds |= {
        source_kind_for(details.get(key)) for key in _FIELD_SOURCE_KEYS.values() if details.get(key)
    }
    return kinds


def borrowed_field_sources(details: dict[str, Any]) -> dict[str, str]:
    """``<field>_source`` donor labels of a slot's borrowed fields (#65)."""
    return {
        key: details[key]
        for field, key in _FIELD_SOURCE_KEYS.items()
        if details.get(key) and _has_field(details, field)
    }


def slot_kind_label(details: dict[str, Any]) -> str:
    """Slot kind plus the source labels of its fields when they differ.

    ``"measured (uops.info, latency from instlatx64)"`` names the owner of
    the slot and the donor of each borrowed field, so a reader never
    attributes one source's latency to another's ports (#65).
    """
    label = slot_kind(details)
    borrowed = [
        f"{_FIELD_LABELS[field]} from {details[_FIELD_SOURCE_KEYS[field]]}"
        for field in _FIELD_VALUE_KEYS
        if details.get(_FIELD_SOURCE_KEYS[field])
    ]
    if not borrowed:
        return label
    owner = details.get("source") or SOURCE_UOPS_INFO
    return f"{label} (owner {owner}, {', '.join(borrowed)})"


def _fill_missing(winner: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Return *winner* with the fields it lacks copied from *other*.

    Each copied field records its donor in its own ``<field>_source`` key,
    so a borrowed latency never carries the donor's ports or uops along.
    A field *winner* borrowed keeps the highest-priority donor seen so far.
    Returns *winner* itself when nothing is copied.
    """
    result = winner
    for field, keys in _FIELD_VALUE_KEYS.items():
        if not _has_field(other, field):
            continue
        donor = field_source(other, field)
        if _has_field(winner, field):
            if not winner.get(_FIELD_SOURCE_KEYS[field]):
                continue  # the owner's own value
            old_rank = _SOURCE_RANK.get(field_source(winner, field))
            if old_rank is None or _SOURCE_RANK.get(donor, old_rank) >= old_rank:
                continue
        if result is winner:
            result = dict(winner)
        if field == "latencies":
            result["latencies"] = other["latencies"]
        else:
            if result.get("measurement") is winner.get("measurement"):
                result["measurement"] = dict(winner.get("measurement") or {})
            for key in keys:
                if (other.get("measurement") or {}).get(key):
                    result["measurement"][key] = other["measurement"][key]
        # A mixed slot may donate again; keep the value's own donor.
        result[_FIELD_SOURCE_KEYS[field]] = donor
    return result


def _is_sme_only(record: InstructionRecord) -> bool:
    # SME-only forms (their tags name SME and nothing else) are ZA tiles,
    # ZA arrays and ZT0 tables: no perf source measures them (#65 M11).
    isa = [str(tag).upper() for tag in record.isa or ()]
    return bool(isa) and all(tag.startswith("SME") for tag in isa)


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
    if c.startswith("m"):
        end = 1
        while end < len(c) and c[end].isdecimal():
            end += 1
        if end > 1:
            return f"mem{int(c[1:end])}"
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
    before, opened, rest = form.partition("(")
    inner, closed, _ = rest.rpartition(")")
    if not before or not opened or not closed:
        return ()
    return tuple(_key_operand_class(part) for part in inner.split(","))


def _form_class_match(
    row_classes: tuple[frozenset[str], ...], record_classes: tuple[str, ...]
) -> bool:
    if len(row_classes) != len(record_classes):
        return False
    for allowed, actual in zip(row_classes, record_classes):
        end = 0
        while end < len(actual) and "a" <= actual[end] <= "z":
            end += 1
        base = actual[:end] + "*"
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
    to every variant of the mnemonic (``applies_to="mnemonic"``), except
    SME-only forms (ZA tiles, ZA arrays, ZT0 tables), which no source
    measures.

    ``overwrite=False`` (default) keeps one value per ``arch_details[core]``
    slot: an incoming row replaces the existing entry only when its source
    ranks higher in :data:`SOURCE_PRIORITY`. Slots whose source is unknown
    to the priority table are never displaced or filled, so passes that
    predate the table keep their data. Each field merges on its own: a
    field the winner lacks comes from the best source that has it, and the
    slot records that donor as ``latency_source`` / ``cpi_source`` /
    ``uops_source`` / ``ports_source``. Ports and uops therefore never ride
    along with a borrowed latency (#65).

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
                    # Mnemonic rows spread over every form except SME-only
                    # ones: no source measures ZA tiles, ZA arrays or ZT0
                    # tables (#65 M11).
                    matched = [r for r in matched if not _is_sme_only(r)]
            # Plain form rows keep the mnemonic-level spread when the form
            # text matches nothing (intel-guide and friends name iforms).
            # An AMD class row the classes do not match attaches nowhere.
            if not matched and form_key and not row.form_classes:
                matched = [r for r in targets if not _is_sme_only(r)]
        entry = row.as_arch_details_entry()
        new_rank = _SOURCE_RANK.get(row.source)
        for record in matched:
            existing = record.arch_details.get(row.core)
            if existing is not None and not overwrite:
                old_rank = _entry_rank(existing)
                if new_rank is None or old_rank is None:
                    continue
                if new_rank >= old_rank:
                    # The winner keeps the slot; fields it lacks come from
                    # this row, each under its own donor key (#65).
                    filled = _fill_missing(existing, entry)
                    if filled is not existing:
                        record.arch_details[row.core] = filled
                    continue
                # Higher rank wins; the displaced entry fills what the new
                # one lacks, so its ports and uops are never dropped.
                slot = _fill_missing(entry, existing)
            else:
                slot = entry
            record.arch_details[row.core] = slot
            written += 1
    # An alias row is the same encoding as its target row, so it carries the
    # target's timing. metadata "aliasof" holds the target row's form key;
    # the copy runs after every merge pass so later sources refresh it.
    by_form = {(r.architecture, r.form.strip().casefold()): r for r in records}
    for record in records:
        aliasof = (record.metadata.get("aliasof") or "").strip()
        if not aliasof:
            continue
        target = by_form.get((record.architecture, aliasof.casefold()))
        if target is not None and target is not record:
            record.arch_details = deepcopy(target.arch_details)
    return written


def _arch_guess(core: str) -> str:
    """Best-effort architecture lookup used when :class:`PerfRow` lacks one."""
    from simdref.perf_sources.cores import core_architecture, perf_architecture

    return perf_architecture(core_architecture(core))
