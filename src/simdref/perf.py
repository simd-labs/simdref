"""Shared performance-metric extraction helpers.

Functions in this module extract latency, throughput (cycles-per-instruction),
and other microarchitecture-level performance data from the nested
``arch_details`` dictionaries stored on :class:`~simdref.models.InstructionRecord`.

They are consumed by the CLI, LSP, web-export, and man-page modules so
that the extraction logic lives in exactly one place.

Each ``arch_details[core]`` entry may carry provenance keys added by the
ingesters (``source``, ``source_kind``, ``source_version``, ``applies_to``,
``citation_url``). ``source_kind`` is either ``"measured"`` or
``"modeled"``; when absent it defaults to ``"measured"`` for compatibility
with the original uops.info rows which are measured.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from simdref.perf_sources.cores import core_architecture

# x86 vendor sets mirror the web's current labels so old rows keep the same
# group names. Non-x86 cores group by architecture, never by a fixed vendor.
_X86_AMD_CORES = {"ZEN5", "ZEN4", "ZEN3", "ZEN2", "ZEN1", "ZEN+"}
_X86_SERVER_CORES = {"EMR", "CLX", "SKX", "CNL"}


def perf_family_label(uarch: str) -> str:
    """Vendor/family label for a perf-group header, per core_architecture.

    Returns "" when the core is unknown so callers show no label rather
    than guess one.
    """
    if not uarch:
        return ""
    core = uarch.strip().upper()
    if core in _X86_AMD_CORES:
        return "AMD"
    if core in _X86_SERVER_CORES:
        return "Intel Server"
    architecture = core_architecture(uarch)
    if architecture is None:
        return ""
    if architecture == "x86":
        return "Intel Client"
    if architecture == "aarch64":
        return "Arm"
    if architecture == "riscv":
        return "RISC-V"
    return ""


class PerfValue(NamedTuple):
    """A labeled performance value.

    ``value`` is the cycle-count string (``"-"`` when missing).
    ``source_kind`` is ``"measured"`` or ``"modeled"``.
    ``core`` is the microarchitecture id the value came from (empty when none).
    """

    value: str
    source_kind: str
    core: str

    def __str__(self) -> str:
        return self.value


MISSING_PERF = PerfValue("-", "", "")


def _is_numeric(value: str) -> bool:
    """Check if a string is a non-negative number (integer or decimal)."""
    if not value:
        return False
    parts = value.split(".", 1)
    return all(p.isdigit() for p in parts) and parts[0] != ""


def _source_kind(details: dict[str, Any]) -> str:
    """Return the provenance kind for an ``arch_details`` entry.

    Defaults to ``"measured"`` when unset so legacy uops.info rows carry
    the correct label without migration.
    """
    kind = details.get("source_kind")
    if kind in ("measured", "modeled"):
        return kind
    return "measured"


def latency_cycle_values(latencies: list[dict[str, Any]]) -> list[str]:
    """Collect unique cycle-count strings from a list of latency dicts.

    Each *latency* dict may contain keys like ``cycles``, ``cycles_mem``,
    ``cycles_addr``, or ``cycles_addr_index``.  Values are collected in
    order, skipping duplicates.

    Returns:
        De-duplicated list of cycle-count strings (may be empty).
    """
    values: list[str] = []
    for latency in latencies:
        mem_is_upper = latency.get("cycles_mem_is_upper_bound") == "1"
        for key in ("cycles", "cycles_mem", "cycles_addr", "cycles_addr_index"):
            value = latency.get(key)
            if not value:
                continue
            if key == "cycles_mem" and mem_is_upper and _is_numeric(value) and float(value) == 0:
                # uops.info convention: cycles_mem=0 with the upper-bound flag
                # is a "not measured" marker, not a real zero-cycle latency.
                continue
            if value not in values:
                values.append(value)
    return values


def best_numeric(values: list[str]) -> str:
    """Return the smallest numeric string from *values*, or ``"-"`` if empty.

    Non-numeric entries (e.g. ``"variable"``) are ignored when a numeric
    alternative exists.  If every entry is non-numeric the first value is
    returned as-is.
    """
    if not values:
        return "-"
    numeric = [v for v in values if _is_numeric(str(v))]
    if numeric:
        return min(numeric, key=lambda v: float(v))
    return values[0]


# Upstream uops.info measurements of control-flow instructions (RET, some
# CALL/JMP forms) are recorded inside a call/ret loop harness, so the raw
# ``uops`` count and ``TP_unrolled``/``TP_loop`` figures describe the
# entire harness rather than the single measured instruction — RET_NEAR on
# HSW reports ``uops=20``, ``TP_unrolled=33.35``. When the measurement
# clearly can't be a single-instruction figure, drop the measured CPI
# fields so callers fall back to modeled/IACA sources (or display ``-``).
_HARNESS_UOPS_THRESHOLD = 8


def _measurement_is_harness_contaminated(measurement: dict[str, Any]) -> bool:
    try:
        uops = int(float(measurement.get("uops", 0) or 0))
    except (TypeError, ValueError):
        return False
    return uops >= _HARNESS_UOPS_THRESHOLD


def _cpi_values(details: dict[str, Any]) -> list[str]:
    measurement = details.get("measurement") or {}
    if _measurement_is_harness_contaminated(measurement):
        return []
    out: list[str] = []
    for key in ("TP_unrolled", "TP_loop", "TP_ports", "TP"):
        value = measurement.get(key)
        if value and value not in out:
            out.append(value)
    return out


def _best_labeled(
    arch_details: dict[str, dict[str, Any]],
    value_fn,
    kind_filter: str | None = None,
) -> PerfValue:
    """Return the lowest numeric value across arch entries, with provenance.

    *value_fn* is called on each ``details`` dict and must return a list of
    candidate cycle-count strings. When *kind_filter* is ``"measured"`` or
    ``"modeled"`` only entries with that ``source_kind`` are considered.
    """
    best: tuple[float, str, str, str] | None = None
    first_non_numeric: PerfValue | None = None
    for core, details in arch_details.items():
        kind = _source_kind(details)
        if kind_filter is not None and kind != kind_filter:
            continue
        for value in value_fn(details):
            value_str = str(value)
            if _is_numeric(value_str):
                numeric = float(value_str)
                if best is None or numeric < best[0]:
                    best = (numeric, value_str, kind, core)
            elif first_non_numeric is None:
                first_non_numeric = PerfValue(value_str, kind, core)
    if best is not None:
        return PerfValue(best[1], best[2], best[3])
    if first_non_numeric is not None:
        return first_non_numeric
    return MISSING_PERF


def best_latency_labeled(
    arch_details: dict[str, dict[str, Any]],
    kind_filter: str | None = None,
) -> PerfValue:
    """Lowest latency across arch entries, as a :class:`PerfValue`.

    Prefers measured rows: if any measured row exposes a numeric latency
    it is picked over all modeled rows. Pass ``kind_filter="modeled"`` to
    select only modeled data.
    """

    def lat_values(details: dict[str, Any]) -> list[str]:
        return latency_cycle_values(details.get("latencies") or [])

    if kind_filter is None:
        measured = _best_labeled(arch_details, lat_values, kind_filter="measured")
        if measured.value != "-":
            return measured
        return _best_labeled(arch_details, lat_values, kind_filter="modeled")
    return _best_labeled(arch_details, lat_values, kind_filter=kind_filter)


def best_cpi_labeled(
    arch_details: dict[str, dict[str, Any]],
    kind_filter: str | None = None,
) -> PerfValue:
    """Lowest cycles-per-instruction across arch entries, as a :class:`PerfValue`."""
    if kind_filter is None:
        measured = _best_labeled(arch_details, _cpi_values, kind_filter="measured")
        if measured.value != "-":
            return measured
        return _best_labeled(arch_details, _cpi_values, kind_filter="modeled")
    return _best_labeled(arch_details, _cpi_values, kind_filter=kind_filter)


def best_latency(arch_details: dict[str, dict[str, Any]]) -> str:
    """Return the best (lowest) latency across all microarchitectures.

    Prefers measured rows over modeled rows (see :func:`best_latency_labeled`).
    """
    return best_latency_labeled(arch_details).value


def best_cpi(arch_details: dict[str, dict[str, Any]]) -> str:
    """Return the best (lowest) cycles-per-instruction across all microarchitectures.

    Prefers measured rows over modeled rows (see :func:`best_cpi_labeled`).
    """
    return best_cpi_labeled(arch_details).value


def best_latency_measured(arch_details: dict[str, dict[str, Any]]) -> PerfValue:
    """Lowest latency drawn exclusively from ``source_kind="measured"`` rows."""
    return best_latency_labeled(arch_details, kind_filter="measured")


def best_latency_modeled(arch_details: dict[str, dict[str, Any]]) -> PerfValue:
    """Lowest latency drawn exclusively from ``source_kind="modeled"`` rows."""
    return best_latency_labeled(arch_details, kind_filter="modeled")


def best_cpi_measured(arch_details: dict[str, dict[str, Any]]) -> PerfValue:
    """Lowest CPI drawn exclusively from ``source_kind="measured"`` rows."""
    return best_cpi_labeled(arch_details, kind_filter="measured")


def best_cpi_modeled(arch_details: dict[str, dict[str, Any]]) -> PerfValue:
    """Lowest CPI drawn exclusively from ``source_kind="modeled"`` rows."""
    return best_cpi_labeled(arch_details, kind_filter="modeled")


def variant_perf_summary(arch_details: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Return ``(best_latency, best_cpi)`` strings for a single variant."""
    return best_latency(arch_details), best_cpi(arch_details)


def variant_perf_summary_labeled(
    arch_details: dict[str, dict[str, Any]],
) -> tuple[PerfValue, PerfValue]:
    """Return labeled ``(latency, cpi)`` for a single variant.

    Each element carries its ``source_kind`` and originating ``core`` so
    renderers can show "(measured, SKL)" / "(modeled, neoverse-n1)" suffixes.
    """
    return best_latency_labeled(arch_details), best_cpi_labeled(arch_details)


def perf_row_values(details: dict[str, Any]) -> tuple[str, str]:
    """Return the displayed ``(LAT, CPI)`` for one ``arch_details`` entry.

    This function is the only place that picks the displayed values of a
    perf row.

    - LAT is the minimum latency over the operand pairs of the row.
    - CPI is the rule the CLI uses today: ``TP_loop`` when present, else
      ``TP``, else ``"-"``. ``TP_unrolled`` and ``TP_ports`` are not picked.
    """
    lat = best_numeric(latency_cycle_values(details.get("latencies") or []))
    measurement = details.get("measurement") or {}
    cpi = measurement.get("TP_loop") or measurement.get("TP") or "-"
    return lat, cpi


# ---------------------------------------------------------------------------
# Latency per operand pair (uops.info start_op -> target_op detail)
# ---------------------------------------------------------------------------

_LATENCY_VALUE_KEYS = ("cycles", "cycles_mem", "cycles_addr", "cycles_addr_index")
_VECTOR_LABELS = {"128": "xmm", "256": "ymm", "512": "zmm", "64": "mm"}
_GPR_LABELS = {"32": "r32"}


def _pair_value(latency: dict[str, Any]) -> str | None:
    """One bounded cycle string for a latency entry (``"3"``, ``"3-5"``, ``"<=7"``).

    Range rows (``min_cycles``/``max_cycles``) join the bounds with ``-``.
    A bound flagged ``*_is_upper_bound`` gets the ``<=`` mark uops.info
    shows. Returns ``None`` when the entry has no cycle data.
    """
    mem_is_upper = latency.get("cycles_mem_is_upper_bound") == "1"
    for key in _LATENCY_VALUE_KEYS:
        value = latency.get(key)
        if value:
            if key == "cycles_mem" and mem_is_upper and _is_numeric(value) and float(value) == 0:
                # Same "not measured" zero convention as latency_cycle_values.
                continue
            if latency.get(f"{key}_is_upper_bound") == "1":
                return f"<={value}"
            return value
        lo = latency.get(f"min_{key}")
        hi = latency.get(f"max_{key}")
        if lo or hi:
            lo_t = f"<={lo}" if lo and latency.get(f"min_{key}_is_upper_bound") == "1" else lo
            hi_t = f"<={hi}" if hi and latency.get(f"max_{key}_is_upper_bound") == "1" else hi
            if lo and hi:
                return f"{lo_t}-{hi_t}"
            return f"{lo_t}-" if lo else f"-{hi_t}"
    return None


def _operand_label(operands: list[dict[str, Any]] | None, idx: Any) -> str:
    """Compact operand label for a pair endpoint, matching the uops.info style.

    Register operands map by width to ``xmm``/``ymm``/``zmm``/``mm``/``r32``
    plus a zero-based number (start_op 2 -> xmm1). Other types use the raw
    type (``mem``, ``imm``, ``flags``). Unknown operands fall back to the
    raw start_op index.
    """
    idx_str = str(idx)
    for operand in operands or []:
        if str(operand.get("idx")) != idx_str:
            continue
        typ = operand.get("type") or ""
        width = str(operand.get("width") or "")
        if typ == "reg":
            base = _VECTOR_LABELS.get(width) or _GPR_LABELS.get(width) or "reg"
            if base != "reg" and idx_str.isdigit():
                return f"{base}{int(idx_str) - 1}"
            return base
        return typ or idx_str
    return idx_str


def latency_pairs(details: dict[str, Any], operands=None) -> list[dict[str, str]]:
    """Per-operand-pair latency entries for one ``arch_details`` row.

    Each entry is ``{"start", "target", "value"}`` with an optional
    ``same_reg`` key when the source row carries ``cycles_same_reg``.

    shortcut: show only the first cycle key with data (cycles, mem, addr,
    addr_index) per pair. Upgrade if a pair ever needs both.
    """
    pairs: list[dict[str, str]] = []
    for latency in details.get("latencies") or []:
        value = _pair_value(latency)
        if value is None:
            continue
        pair = {
            "start": _operand_label(operands, latency.get("start_op", "")),
            "target": _operand_label(operands, latency.get("target_op", "")),
            "value": value,
        }
        same_reg = latency.get("cycles_same_reg")
        if same_reg:
            pair["same_reg"] = same_reg
        pairs.append(pair)
    return pairs


def _pair_segment(pair: dict[str, str]) -> str:
    segment = f"{pair['value']} ({pair['start']}->{pair['target']}"
    if pair.get("same_reg"):
        segment += f", same reg {pair['same_reg']}"
    return segment + ")"


def latency_pair_text(details: dict[str, Any], operands=None) -> str:
    """Compact pair line like ``3-5 (xmm1->xmm0), <=7 (mem->xmm0)``.

    Empty when the row has no pair data, so callers show no label at all.
    """
    return ", ".join(_pair_segment(pair) for pair in latency_pairs(details, operands))
