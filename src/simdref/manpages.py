"""Roff manpage generation for intrinsics and instructions.

``simdref man <name>`` renders a single page on demand; ``simdref
install-manpages`` pre-generates man7 pages into a manpath-visible dir so
plain ``man <name>`` works.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from simdref.display import display_architecture
from simdref.models import Catalog, InstructionRecord, IntrinsicRecord
from simdref.queries import (
    build_intrinsic_instruction_index,
    instruction_rows_for_intrinsic_indexed,
    linked_instruction_records,
)


def _roff_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("-", "\\-")


def _section(title: str, body: str) -> str:
    return f".SH {title}\n{body}\n"


def _metric_lines(record: InstructionRecord) -> list[str]:
    lines = []
    for arch, values in sorted(record.metrics.items()):
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(values.items()))
        lines.append(f"{arch}: {rendered}")
    return lines


def _instruction_perf_lines(linked: list[InstructionRecord]) -> list[str]:
    """Format linked instruction performance data as text lines for man pages."""
    lines = []
    for row in instruction_rows_for_intrinsic_indexed(linked):
        instruction = row.get("instruction", "-")
        uarch = row.get("uarch", "-")
        metrics = {k: v for k, v in row.items() if k not in {"instruction", "uarch"}}
        if metrics and uarch != "-":
            rendered = ", ".join(f"{key}={value}" for key, value in sorted(metrics.items()))
            lines.append(f"{instruction} | {uarch}: {rendered}")
        else:
            lines.append(f"{instruction} | no performance metrics available")
    return lines


def intrinsic_page(
    record: IntrinsicRecord,
    catalog: Catalog | None = None,
    linked_instructions: list[InstructionRecord] | None = None,
) -> str:
    if linked_instructions is None:
        linked_instructions = linked_instruction_records(catalog, record)
    parts = [f'.TH "{record.name}" "7" "simdref" "simdref" "SIMD Intrinsic Reference"\n']
    parts.append(
        _section(
            "NAME",
            f"{_roff_escape(record.name)} \\- {_roff_escape(record.description or 'intrinsic')}",
        )
    )
    parts.append(_section("SYNOPSIS", f".nf\n{_roff_escape(record.signature)}\n.fi"))
    parts.append(
        _section("DESCRIPTION", _roff_escape(record.description or "No description available."))
    )
    parts.append(_section("HEADER", _roff_escape(record.header or "Unknown")))
    if record.url:
        parts.append(_section("SOURCE", _roff_escape(record.url)))
    parts.append(
        _section(
            "ARCHITECTURE", _roff_escape(display_architecture(record.architecture or "Unknown"))
        )
    )
    parts.append(_section("ISA", _roff_escape(", ".join(record.isa) or "Unknown")))
    parts.append(_section("CATEGORY", _roff_escape(record.category or "Unknown")))
    parts.append(
        _section("INSTRUCTIONS", _roff_escape(", ".join(record.instructions) or "None linked"))
    )
    perf_text = _roff_escape(
        "\n".join(_instruction_perf_lines(linked_instructions))
        or "No performance metrics available."
    )
    parts.append(_section("PERFORMANCE SUMMARY", perf_text))
    parts.append(_section("PERFORMANCE DETAILS", perf_text))
    parts.append(_section("NOTES", _roff_escape("; ".join(record.notes) or "None")))
    parts.append(
        _section("SEE ALSO", _roff_escape(", ".join(record.instructions) or "simdref-search(7)"))
    )
    return "".join(parts)


def instruction_page(record: InstructionRecord) -> str:
    parts = [f'.TH "{record.mnemonic}" "7" "simdref" "simdref" "SIMD Instruction Reference"\n']
    name = _roff_escape(record.key)
    if record.summary:
        name += f" \\- {_roff_escape(record.summary)}"
    parts.append(_section("NAME", name))
    if record.description.get("Description"):
        parts.append(_section("DESCRIPTION", _roff_escape(record.description["Description"])))
    else:
        parts.append(_section("DESCRIPTION", _roff_escape(record.summary)))
    if record.description.get("Operation"):
        parts.append(
            _section("OPERATION", f".nf\n{_roff_escape(record.description['Operation'])}\n.fi")
        )
    parts.append(
        _section(
            "ARCHITECTURE", _roff_escape(display_architecture(record.architecture or "Unknown"))
        )
    )
    parts.append(_section("ISA", _roff_escape(", ".join(record.isa) or "Unknown")))
    parts.append(
        _section(
            "OPERANDS", _roff_escape("\n".join(record.operands) or "No operand details available.")
        )
    )
    parts.append(
        _section("INTRINSICS", _roff_escape(", ".join(record.linked_intrinsics) or "None linked"))
    )
    parts.append(
        _section(
            "PERFORMANCE DETAILS",
            _roff_escape("\n".join(_metric_lines(record)) or "No performance metrics available."),
        )
    )
    if record.description.get("Flags Affected"):
        parts.append(_section("FLAGS AFFECTED", _roff_escape(record.description["Flags Affected"])))
    for exc_key in (
        "Exceptions",
        "SIMD Floating-Point Exceptions",
        "Numeric Exceptions",
        "Other Exceptions",
    ):
        if record.description.get(exc_key):
            parts.append(_section(exc_key.upper(), _roff_escape(record.description[exc_key])))
    return "".join(parts)


def write_manpages(
    catalog: Catalog,
    man_dir: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    section_dir = man_dir / "man7"
    section_dir.mkdir(parents=True, exist_ok=True)
    index = build_intrinsic_instruction_index(catalog)
    total = len(catalog.intrinsics) + len(catalog.instructions)
    done = 0
    for intrinsic in catalog.intrinsics:
        linked = index.get(intrinsic.name, [])
        (section_dir / f"{intrinsic.name}.7").write_text(intrinsic_page(intrinsic, catalog, linked))
        done += 1
        if on_progress is not None:
            on_progress(done, total)
    for instruction in catalog.instructions:
        page = instruction_page(instruction)
        filename = (
            f"instruction-{record_slug(instruction.architecture)}-{record_slug(instruction.key)}.7"
        )
        (section_dir / filename).write_text(page)
        if instruction.architecture == "x86":
            (section_dir / f"{instruction.mnemonic}.7").write_text(page)
        (
            section_dir / f"{record_slug(instruction.architecture)}-{instruction.mnemonic}.7"
        ).write_text(page)
        done += 1
        if on_progress is not None:
            on_progress(done, total)


def record_slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
