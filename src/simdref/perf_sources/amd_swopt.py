"""Parser and vendored loader for the AMD Software Optimization Guide latency workbooks.

AMD ships the per-instruction latency and throughput tables as xlsx
workbooks next to the optimization guide PDFs. This module parses them
with the standard library only (xlsx is a zip of XML) and rewrites the
derived rows as JSON under ``vendor/amd_swopt/``.

Download URLs (put the xlsx files in one raw directory, then run
``python -m simdref.perf_sources.amd_swopt refresh RAWDIR``):

- Zen 5, AMD doc 58455 Rev 1.00. Zip with the guide PDF and workbook:
  https://docs.amd.com/api/khub/documents/RCLISlZoOyWkKA9SYwLLDQ/content?Ft-Calling-App=ft%2Fturnkey-portal&Ft-Calling-App-Version=5.4.1&download=true&locationValue=viewer
  Viewer page: https://docs.amd.com/v/u/en-US/58455_1.00
- Zen 4, AMD doc 57647 Rev 1.00. Internet Archive capture of AMD's zip:
  https://web.archive.org/web/20230108094512id_/https://www.amd.com/system/files/TechDocs/57647.zip
  Viewer page: https://docs.amd.com/v/u/en-US/57647_1.00
- Zen 3, AMD doc 56665. Workbook (no longer served by amd.com):
  https://raw.githubusercontent.com/bitRAKE/amd_zen_sogs/main/56665/Family_19h_Instruction_Latencies_version_1-00.xlsx
  Viewer page: https://docs.amd.com/v/u/en-US/56665
- Zen 2, AMD doc 56305. Workbook (no longer served by amd.com):
  https://raw.githubusercontent.com/bitRAKE/amd_zen_sogs/main/56305/Family%2017h%20Models%2030h%20and%20Greater%20Instruction%20Latencies%20version_1.00.xlsx
  Viewer page: https://docs.amd.com/v/u/en-US/56305
- Zen 1, AMD doc 55723. Workbook (no longer served by amd.com):
  https://raw.githubusercontent.com/bitRAKE/amd_zen_sogs/main/55723/Family%2017h%20Instruction%20Latencies%20version_1.00.xlsx
  Viewer page: https://docs.amd.com/v/u/en-US/55723_3.01

The vendored JSON keeps the vendor's latency and throughput text
verbatim (ranges such as "3-5" stay strings). The Throughput column is
operations per cycle, not cycles per instruction, so it is stored under
``amd_ops_per_cycle`` in ``extra_measurement``; ``PerfRow.cpi`` holds
1 / ops-per-cycle when that value parses as a number.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from simdref.perf_sources.cores import canonical_core_id
from simdref.perf_sources.merge import SOURCE_AMD_SWOPT, PerfRow

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VENDOR_DIR = _REPO_ROOT / "vendor" / "amd_swopt"

_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_M = "{%s}" % _NS
_R_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

# AMD marks the workbook values as estimates ("based on estimates and is
# subject to change", doc 58455 Appendix A), so rows are modeled.
_SOURCE_KIND = "modeled"

THROUGHPUT_KEY = "amd_ops_per_cycle"
CPUID_KEY = "cpuid"


@dataclass(frozen=True)
class Workbook:
    """How to find and label one AMD latency workbook inside a raw directory."""

    match: str  # lowercase filename fragment with underscores for spaces
    core: str
    source_version: str
    citation_url: str
    output: str

    def find(self, rawdir: Path) -> Path | None:
        for path in sorted(rawdir.iterdir()):
            if path.suffix.lower() != ".xlsx":
                continue
            if self.match in path.name.lower().replace(" ", "_"):
                return path
        return None


WORKBOOKS: tuple[Workbook, ...] = (
    Workbook(
        match="zen5_instruction_latencies",
        core="ZEN5",
        source_version="1.00",
        citation_url="https://docs.amd.com/v/u/en-US/58455_1.00",
        output="zen5.json",
    ),
    Workbook(
        match="zen4_instruction_latencies",
        core="ZEN4",
        source_version="1.00",
        citation_url="https://docs.amd.com/v/u/en-US/57647_1.00",
        output="zen4.json",
    ),
    Workbook(
        match="family_19h_instruction_latencies",
        core="ZEN3",
        source_version="1.00",
        citation_url="https://docs.amd.com/v/u/en-US/56665",
        output="zen3.json",
    ),
    Workbook(
        match="family_17h_models_30h",
        core="ZEN2",
        source_version="1.00",
        citation_url="https://docs.amd.com/v/u/en-US/56305",
        output="zen2.json",
    ),
    Workbook(
        match="family_17h_instruction_latencies",
        core="ZEN1",
        source_version="1.00",
        citation_url="https://docs.amd.com/v/u/en-US/55723_3.01",
        output="zen1.json",
    ),
)


@dataclass
class ParseStats:
    """Row counts from one workbook, for the refresh report."""

    total: int = 0
    kept: int = 0
    skipped_empty: int = 0
    dropped_unknown_core: int = 0


def _column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def _read_sheet_rows(zf: zipfile.ZipFile, target: str) -> list[list[str]]:
    shared: list[str] = []
    if "xl/sharedStrings.xml" in zf.namelist():
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in si.iter(_M + "t")) for si in root.iter(_M + "si")]
    rows: list[list[str]] = []
    root = ET.fromstring(zf.read("xl/" + target))
    for row in root.iter(_M + "row"):
        vals: dict[int, str] = {}
        for cell in row.findall(_M + "c"):
            ref = cell.get("r")
            if ref is None:
                continue
            kind = cell.get("t")
            value = cell.find(_M + "v")
            if kind == "s" and value is not None:
                text = shared[int(value.text)]
            elif kind == "inlineStr":
                inline = cell.find(_M + "is")
                text = (
                    "".join(t.text or "" for t in inline.iter(_M + "t"))
                    if inline is not None
                    else ""
                )
            else:
                text = value.text if value is not None and value.text is not None else ""
            vals[_column_index(ref)] = text
        if vals:
            rows.append([vals.get(i, "") for i in range(max(vals) + 1)])
    return rows


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _data_sheet_rows(path: Path) -> list[list[str]]:
    """Return the rows of the sheet whose header row starts with "Instruction"."""
    with zipfile.ZipFile(path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        targets = {rel.get("Id"): rel.get("Target") for rel in rels}
        for sheet in workbook.iter(_M + "sheet"):
            rows = _read_sheet_rows(zf, targets[sheet.get(_R_ID)])
            for row in rows[:10]:
                if row and _norm(row[0]) == "instruction":
                    return rows
    raise ValueError(f"no instruction table sheet found in {path}")


def _header_columns(header: list[str], name: str) -> list[int]:
    # The later workbooks add columns (masking operand, opmask), so the
    # column positions come from the header text, not from fixed offsets.
    return [i for i, cell in enumerate(header) if name in _norm(cell)]


def _clean(text: str) -> str:
    # "NA" marks absent data; store an empty string so merge writes nothing.
    text = text.strip()
    return "" if text.casefold() == "na" else text


# AMD workbook operand codes name classes, not catalog forms. ``*`` marks a
# wildcard class: it matches the record class with its digits stripped
# (``gpr*`` takes gpr8 through gpr64 and the unknown width, ``mem*`` every
# memory form). See merge._form_class_match.
def _vendor_alt_classes(alt: str) -> frozenset[str]:
    # One ``/`` alternative of one workbook operand ("mem128{k0}", "reg32").
    a = alt.strip().casefold().split(" ", 1)[0]
    a = re.sub(r"\{.*$", "", a).strip("<>_`")
    if re.fullmatch(r"xmm\d*|x|s", a):
        return frozenset({"xmm"})
    if re.fullmatch(r"ymm\d*|y", a):
        return frozenset({"ymm"})
    if re.fullmatch(r"zmm\d*|z", a):
        return frozenset({"zmm"})
    if a == "xy":
        return frozenset({"xmm", "ymm"})
    if a == "xyz":
        return frozenset({"xmm", "ymm", "zmm"})
    if a == "yz":
        return frozenset({"ymm", "zmm"})
    if re.fullmatch(r"mmx\d*", a):
        return frozenset({"mmx"})
    if re.fullmatch(r"k\d*", a):
        return frozenset({"k"})
    if a == "sti":
        return frozenset({"x87"})
    if a == "xi":
        return frozenset({"xmm", "imm*"})
    if a == "cl":
        return frozenset({"gpr8"})
    regs = re.fullmatch(r"(?:\[)?r(?:eg)?((?:\d+(?:/(?:r(?:eg)?)?\d+)*)?)(?:\])?", a)
    if a in ("reg", "r", "g", "gx"):
        return frozenset({"gpr*"})
    if regs and regs.group(1):
        return frozenset(f"gpr{w}" for w in re.findall(r"\d+", regs.group(1)))
    if a in ("i", "imm"):
        return frozenset({"imm*"})
    if a in ("m", "mi", "mem", "vm32x"):
        return frozenset({"mem*"})
    widths = re.fullmatch(r"(?:mem)?(?:em)?(\d+)(?:bcst)?(?:/(\d+))?", a)
    if a.startswith(("mem", "memem")) and widths:
        found = [w for w in widths.groups() if w]
        return frozenset(f"mem{w}" for w in found)
    imms = re.fullmatch(r"imm(\d+(?:/\d+)*)", a)
    if imms:
        return frozenset(
            re.findall(r"imm\d+", "imm" + imms.group(1).replace("/", "/imm"))
        ) | frozenset({"imm0"})
    return frozenset({"other"})


def form_classes(form: str) -> tuple[frozenset[str], ...]:
    # Workbook "Instruction Operands" text ("xmm1,xmm2/mem128") as one
    # allowed-class set per operand position.
    out: list[frozenset[str]] = []
    for operand in form.split(","):
        operand = operand.strip()
        if not operand:
            continue
        # Trailing width alternatives take the family of the first token:
        # "reg8/16/32" reads as "reg8/reg16/reg32".
        widen = re.fullmatch(r"([a-z\[\]_]*[a-z_\]])(\d+)((?:/\d+)+)", operand)
        if widen:
            alts = [
                f"{widen.group(1)}{w}" for w in [widen.group(2), *widen.group(3)[1:].split("/")]
            ]
        else:
            alts = operand.split("/")
        allowed: set[str] = set()
        for alt in alts:
            allowed |= _vendor_alt_classes(alt)
        out.append(frozenset(allowed))
    return tuple(out)


def _cpi(throughput: str) -> str:
    """Invert the ops-per-cycle value to cycles per instruction.

    Ranges ("0.25-0.5"), "variable" and "?" give an empty string; the raw
    text stays in ``amd_ops_per_cycle``.
    """
    try:
        ops = float(throughput)
    except ValueError:
        return ""
    if ops <= 0:
        return ""
    cpi = 1 / ops
    return f"{cpi:.3f}".rstrip("0").rstrip(".")


def parse_workbook(
    path: Path | str,
    *,
    core: str,
    source_version: str,
    citation_url: str,
) -> tuple[list[PerfRow], ParseStats]:
    """Parse one AMD latency workbook into :class:`PerfRow` s.

    Reads the sheet whose header row starts with "Instruction". Rows
    whose CPU has no canonical core id are dropped and counted in
    ``ParseStats.dropped_unknown_core``.
    """
    stats = ParseStats()
    canonical = canonical_core_id(core)
    rows = _data_sheet_rows(Path(path))
    header_at = next(i for i, row in enumerate(rows) if row and _norm(row[0]) == "instruction")
    header = rows[header_at]
    latency_at = _header_columns(header, "latency")
    throughput_at = _header_columns(header, "throughput")
    cpuid_at = _header_columns(header, "cpuid")
    form_at = _header_columns(header, "instruction operands")
    operand_at = [i for i, cell in enumerate(header) if re.fullmatch(r"op\d", _norm(cell))]

    def cell(row: list[str], indices: list[int]) -> str:
        return _clean(row[indices[0]]) if indices and indices[0] < len(row) else ""

    parsed: list[PerfRow] = []
    for row in rows[header_at + 1 :]:
        stats.total += 1
        mnemonic = row[0].strip() if row else ""
        if not mnemonic:
            stats.skipped_empty += 1
            continue
        latency = cell(row, latency_at)
        throughput = cell(row, throughput_at)
        if not latency and not throughput:
            stats.skipped_empty += 1
            continue
        if canonical is None:
            stats.dropped_unknown_core += 1
            continue
        if form_at:
            form = cell(row, form_at)
        else:
            form = ",".join(_clean(row[i]) for i in operand_at if i < len(row) and _clean(row[i]))
        measurement = {}
        cpuid = cell(row, cpuid_at)
        if cpuid:
            measurement[CPUID_KEY] = cpuid
        if throughput:
            measurement[THROUGHPUT_KEY] = throughput
        parsed.append(
            PerfRow(
                mnemonic=mnemonic,
                core=canonical,
                source=SOURCE_AMD_SWOPT,
                source_kind=_SOURCE_KIND,
                source_version=source_version,
                architecture="x86",
                form=form,
                latency=latency,
                cpi=_cpi(throughput),
                citation_url=citation_url,
                extra_measurement=measurement,
                form_classes=form_classes(form),
            )
        )
        stats.kept += 1
    return parsed, stats


def _row_payload(row: PerfRow) -> dict[str, str]:
    return {
        "mnemonic": row.mnemonic,
        "form": row.form,
        "latency": row.latency,
        CPUID_KEY: row.extra_measurement.get(CPUID_KEY, ""),
        THROUGHPUT_KEY: row.extra_measurement.get(THROUGHPUT_KEY, ""),
    }


def write_vendored(
    rows: Iterable[PerfRow], spec: Workbook, workbook_name: str, outdir: Path
) -> int:
    rows = list(rows)
    payload = {
        "source": SOURCE_AMD_SWOPT,
        "source_kind": _SOURCE_KIND,
        "source_version": spec.source_version,
        "core": spec.core,
        "architecture": "x86",
        "citation_url": spec.citation_url,
        "workbook": workbook_name,
        "rows": [_row_payload(row) for row in rows],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / spec.output
    target.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def load_vendored(path: Path | str | None = None) -> list[PerfRow]:
    """Rebuild :class:`PerfRow` s from ``vendor/amd_swopt/*.json``."""
    vendor_dir = Path(path) if path is not None else DEFAULT_VENDOR_DIR
    if not vendor_dir.is_dir():
        return []
    rows: list[PerfRow] = []
    for file in sorted(vendor_dir.glob("*.json")):
        data = json.loads(file.read_text(encoding="utf-8"))
        for raw in data["rows"]:
            measurement = {}
            if raw.get(CPUID_KEY):
                measurement[CPUID_KEY] = raw[CPUID_KEY]
            if raw.get(THROUGHPUT_KEY):
                measurement[THROUGHPUT_KEY] = raw[THROUGHPUT_KEY]
            rows.append(
                PerfRow(
                    mnemonic=raw["mnemonic"],
                    core=data["core"],
                    source=data["source"],
                    source_kind=data["source_kind"],
                    source_version=data["source_version"],
                    architecture=data.get("architecture", "x86"),
                    form=raw.get("form", ""),
                    latency=raw.get("latency", ""),
                    cpi=_cpi(raw.get(THROUGHPUT_KEY, "")),
                    citation_url=data["citation_url"],
                    extra_measurement=measurement,
                    form_classes=form_classes(raw.get("form", "")),
                )
            )
    return rows


@dataclass
class RefreshReport:
    written: dict[str, int] = field(default_factory=dict)
    skipped_empty: dict[str, int] = field(default_factory=dict)
    dropped_unknown_core: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def refresh(rawdir: Path | str, outdir: Path | str | None = None) -> RefreshReport:
    """Parse the raw workbooks in *rawdir* and rewrite ``vendor/amd_swopt/*.json``."""
    rawdir = Path(rawdir)
    outdir = Path(outdir) if outdir is not None else DEFAULT_VENDOR_DIR
    report = RefreshReport()
    for spec in WORKBOOKS:
        found = spec.find(rawdir)
        if found is None:
            report.missing.append(spec.match)
            continue
        rows, stats = parse_workbook(
            found,
            core=spec.core,
            source_version=spec.source_version,
            citation_url=spec.citation_url,
        )
        report.written[spec.core] = write_vendored(rows, spec, found.name, outdir)
        report.skipped_empty[spec.core] = stats.skipped_empty
        if stats.dropped_unknown_core:
            report.dropped_unknown_core[spec.core] = stats.dropped_unknown_core
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m simdref.perf_sources.amd_swopt")
    sub = parser.add_subparsers(dest="command", required=True)
    refresh_cmd = sub.add_parser("refresh", help="parse raw workbooks and rewrite vendor/amd_swopt")
    refresh_cmd.add_argument(
        "rawdir", type=Path, help="directory holding the downloaded xlsx files"
    )
    refresh_cmd.add_argument(
        "--outdir", type=Path, default=None, help="override the vendor output directory"
    )
    args = parser.parse_args(argv)
    if args.command == "refresh":
        report = refresh(args.rawdir, args.outdir)
        for core, count in report.written.items():
            print(f"{core}: {count} rows vendored, {report.skipped_empty[core]} empty rows skipped")
        if report.dropped_unknown_core:
            print(f"dropped (no core id): {report.dropped_unknown_core}")
        if report.missing:
            print(f"missing workbooks: {', '.join(report.missing)}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
