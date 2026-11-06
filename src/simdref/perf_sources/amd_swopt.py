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
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from simdref.perf_sources.cores import canonical_core_id
from simdref.perf_sources.merge import SOURCE_AMD_SWOPT, PerfRow, cpi_for, ipc_value

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
    i = 0
    while "A" <= cell_ref[i : i + 1] <= "Z":
        i += 1
    n = 0
    for ch in cell_ref[:i]:
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


def _op_column(name: str) -> bool:
    # "Op1" .. "Op9" name operand columns in the workbooks without an
    # "Instruction Operands" column.
    return len(name) == 3 and name[:2] == "op" and name[2].isdecimal()


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
def _num_tail(token: str, name: str) -> str | None:
    # "xmm2" is *name* plus a digit run; the bare name counts as no digits.
    if token == name:
        return ""
    if token.startswith(name) and token[len(name) :].isdecimal():
        return token[len(name) :]
    return None


def _width_alts(tail: str) -> list[str]:
    # "32/64" and "32/reg64" both name the widths 32 and 64.
    out: list[str] = []
    for i, part in enumerate(tail.split("/")):
        if i:
            for pfx in ("reg", "r"):
                num = _num_tail(part, pfx)
                if num is not None:
                    part = num
                    break
        if not part.isdecimal():
            return []
        out.append(part)
    return out


def _vendor_alt_classes(alt: str) -> frozenset[str]:
    # One ``/`` alternative of one workbook operand ("mem128{k0}", "reg32").
    a = alt.strip().casefold().split(" ", 1)[0].partition("{")[0].strip("<>_`")
    single = {"x": "xmm", "s": "xmm", "y": "ymm", "z": "zmm"}
    if a in single:
        return frozenset({single[a]})
    for name, cls in (("xmm", "xmm"), ("ymm", "ymm"), ("zmm", "zmm"), ("mmx", "mmx"), ("k", "k")):
        if _num_tail(a, name) is not None:
            return frozenset({cls})
    if a == "xy":
        return frozenset({"xmm", "ymm"})
    if a == "xyz":
        return frozenset({"xmm", "ymm", "zmm"})
    if a == "yz":
        return frozenset({"ymm", "zmm"})
    if a == "sti":
        return frozenset({"x87"})
    if a == "xi":
        return frozenset({"xmm", "imm*"})
    if a == "cl":
        return frozenset({"gpr8"})
    if a in ("reg", "r", "g", "gx"):
        return frozenset({"gpr*"})
    core = a[1:] if a.startswith("[") else a
    if core.endswith("]"):
        core = core[:-1]
    for pfx in ("reg", "r"):
        if core.startswith(pfx):
            widths = _width_alts(core[len(pfx) :])
            if widths:
                return frozenset(f"gpr{w}" for w in widths)
            break
    if a in ("i", "imm"):
        return frozenset({"imm*"})
    if a in ("m", "mi", "mem", "vm32x"):
        return frozenset({"mem*"})
    if a.startswith("mem"):
        body = a[3:]
        if body.startswith("em"):
            body = body[2:]
        first, slash, second = body.partition("/")
        if first.endswith("bcst"):
            first = first[:-4]
        if first.isdecimal() and (not slash or second.isdecimal()):
            found = [first] + ([second] if slash else [])
            return frozenset(f"mem{w}" for w in found)
    if a.startswith("imm"):
        parts = a[3:].split("/")
        if all(p.isdecimal() for p in parts):
            return frozenset(f"imm{p}" for p in parts) | frozenset({"imm0"})
    return frozenset({"other"})


def form_classes(form: str) -> tuple[frozenset[str], ...]:
    # Workbook "Instruction Operands" text ("xmm1,xmm2/mem128") as one
    # allowed-class set per operand position.
    out: list[frozenset[str]] = []
    for operand in form.split(","):
        operand = operand.strip()
        if not operand:
            continue
        allowed: set[str] = set()
        for alt in _operand_alts(operand):
            allowed |= _vendor_alt_classes(alt)
        out.append(frozenset(allowed))
    return tuple(out)


def _cpi(throughput: str) -> str:
    """Invert the ops-per-cycle value to cycles per instruction.

    Ranges ("0.25-0.5"), "variable" and "?" give an empty string; the raw
    text stays in ``amd_ops_per_cycle``.
    """
    ipc = ipc_value(throughput)
    if ipc is None or ipc <= 0:
        return ""
    return cpi_for(ipc)


def _row(
    mnemonic: str,
    form: str,
    core: str,
    source_version: str,
    citation_url: str,
    latency: str,
    throughput: str,
    cpuid: str,
    *,
    source: str = SOURCE_AMD_SWOPT,
    source_kind: str = _SOURCE_KIND,
    architecture: str = "x86",
) -> PerfRow:
    """One perf row for a workbook line, raw values kept verbatim."""
    measurement: dict[str, str] = {}
    if cpuid:
        measurement[CPUID_KEY] = cpuid
    if throughput:
        measurement[THROUGHPUT_KEY] = throughput
    return PerfRow(
        mnemonic=mnemonic,
        core=core,
        source=source,
        source_kind=source_kind,
        source_version=source_version,
        architecture=architecture,
        form=form,
        latency=latency,
        cpi=_cpi(throughput),
        citation_url=citation_url,
        extra_measurement=measurement,
        form_classes=form_classes(form),
    )


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
    operand_at = [i for i, cell in enumerate(header) if _op_column(_norm(cell))]

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
        parsed.append(
            _row(
                mnemonic,
                form,
                canonical,
                source_version,
                citation_url,
                latency,
                throughput,
                cell(row, cpuid_at),
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
            rows.append(
                _row(
                    raw["mnemonic"],
                    raw.get("form", ""),
                    data["core"],
                    data["source_version"],
                    data["citation_url"],
                    raw.get("latency", ""),
                    raw.get(THROUGHPUT_KEY, ""),
                    raw.get(CPUID_KEY, ""),
                    source=data["source"],
                    source_kind=data["source_kind"],
                    architecture=data.get("architecture", "x86"),
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


# ---------------------------------------------------------------------------
# concrete instructions: workbook rows -> catalog rows by encoded bytes
# ---------------------------------------------------------------------------

# Memory and register names for the workbook width numbers. The bare
# shapes (``m``, ``reg``) name every width the old class wildcard matched;
# the assembler rejects the widths the instruction does not take.
_MEM_SIZE = {
    8: "byte",
    16: "word",
    32: "dword",
    64: "qword",
    80: "tbyte",
    128: "xmmword",
    256: "ymmword",
    512: "zmmword",
}
_GPR_NAME = {8: "al", 16: "ax", 32: "eax", 64: "rax"}
_SHAPES = {
    "x": ("xmm",),
    "y": ("ymm",),
    "z": ("zmm",),
    "xy": ("xmm", "ymm"),
    "xyz": ("xmm", "ymm", "zmm"),
    "yz": ("ymm", "zmm"),
}
# One small immediate takes the imm8 encoding, the wide ones force the
# imm16/imm32 encodings, so the immediate-width forms stay distinct.
_IMM = ("2", "0x1234", "0x12345678")


def _operand_alts(operand: str) -> list[str]:
    # Trailing width alternatives take the family of the first token:
    # "reg8/16/32" reads as "reg8/reg16/reg32".
    alts = operand.split("/")
    head = alts[0]
    i = len(head)
    while i and head[i - 1].isdecimal():
        i -= 1
    prefix, width = head[:i], head[i:]
    widen = (
        len(alts) > 1
        and width.isdecimal()
        and all(w.isdecimal() for w in alts[1:])
        and prefix
        and all(ch in "abcdefghijklmnopqrstuvwxyz[]_" for ch in prefix)
        and not prefix.endswith("[")
    )
    if widen:
        return [prefix + w for w in [width, *alts[1:]]]
    return alts


def _alt_token(alt: str) -> str:
    # ``{k1}`` mask and ``{z}`` suffixes and the workbook's quote/bracket
    # artifacts carry no operand meaning.
    return alt.partition("{")[0].strip().strip("'`_<>")


def _alt_width(alt: str) -> int | None:
    a = _alt_token(alt)
    if a in _SHAPES:
        return {"x": 128, "y": 256, "z": 512}.get(a[-1], 512)
    for name, width in (("xmm", 128), ("ymm", 256), ("zmm", 512)):
        if _num_tail(a.lower(), name) is not None:
            return width
    if a.startswith("mem"):
        rest = a[5:] if a.startswith("memem") else a[3:]
        if rest.endswith("bcst"):
            rest = rest[:-4]
        if rest.isdecimal():
            return int(rest)
    return None


def _alt_lines(alt: str, pos: int, op_width: int) -> list[str]:
    if alt.strip() == "s_":
        return ["st(0)"]  # x87 ST(0); see the memory rule in concrete_lines
    if alt.strip() == "m_":
        return ["qword ptr [rax]"]
    a = _alt_token(alt)
    n = min(pos + 1, 7)
    if not a:
        return []
    if a == "cl":
        return ["cl"]
    if a in ("s", "sti"):
        return ["st(1)"]
    if a in _SHAPES:
        return [f"{family}{n}" for family in _SHAPES[a]]
    if a == "M":
        return [f"mm{n}"]  # workbook capital M is the MMX register
    if a == "Mi":
        return [f"mm{n}", _IMM[0]]  # MMX register or immediate
    if a == "xi":
        return [f"xmm{n}", _IMM[0]]  # xmm register or immediate
    low = a.lower()
    for family in ("xmm", "ymm", "zmm", "mmx", "mm"):
        num = _num_tail(low, family)
        if num is not None:
            family = "mm" if family == "mmx" else family
            return [f"{family}{num or str(n)}"]
    core = a[1:] if a.startswith("[") else a
    if core.endswith("]"):
        core = core[:-1]
    if core in ("r", "reg", "g", "gx"):
        return [_GPR_NAME[w] for w in (8, 16, 32, 64)]
    for pfx in ("reg", "r"):
        num = _num_tail(core, pfx)
        if num:
            return [_GPR_NAME[int(num)]]
    num = _num_tail(low, "imm")
    if num is not None:
        if num == "8":
            return [_IMM[0]]
        if num == "16":
            return [_IMM[1]]
        if not num:  # a bare imm names every immediate width
            return list(_IMM)
        return [_IMM[2]]
    if a == "i":
        return list(_IMM)
    num = _num_tail(low, "k")
    if num is not None:
        return ["k" + (num or str(n))]
    if a in ("m", "mem"):
        return [f"{_MEM_SIZE[w]} ptr [rax]" for w in (8, 16, 32, 64, 80, 128, 256, 512)]
    if a == "vm32x":  # VSIB memory; the index register width follows the form
        return [f"dword ptr [rax + {family}2]" for family in ("xmm", "ymm", "zmm")]
    if a == "mem (complex addressing)":
        return ["qword ptr [rax + rcx*2 + 0x12345678]"]
    if a == "mem (simple addressing)":
        return ["qword ptr [rax]"]
    if a.startswith("mem") and a.endswith("bcst") and a[3:-4].isdecimal():
        width = int(a[3:-4])
        factor = max(op_width // width, 1)
        return [f"{_MEM_SIZE[width]} ptr [rax]{{1to{factor}}}"]
    if a.startswith("mem"):
        rest = a[5:] if a.startswith("memem") else a[3:]
        if rest.isdecimal():
            return [f"{_MEM_SIZE[int(rest)]} ptr [rax]"]
    return []


def concrete_lines(row: PerfRow) -> list[str]:
    """Concrete Intel-syntax instructions that name the forms of one row.

    One workbook operand shape can name several instructions (``xy`` is the
    xmm and the ymm instruction, ``xmm2/mem128`` the register and the memory
    form); every generated line must map to one catalog row on its own.
    """
    # "JMP (near)" writes as "jmp"; the parenthetical names a variant
    mnemonic = row.mnemonic.strip().split("(", 1)[0].strip()
    operands = [op.strip() for op in row.form.split(",") if op.strip()]
    if mnemonic.casefold() in ("blendvpd", "blendvps", "pblendvb") and len(operands) == 3:
        operands[2] = "xmm0"  # the legacy selector register is fixed to xmm0
    combos: list[list[str]] = [[]]
    for pos, operand in enumerate(operands):
        alts = _operand_alts(operand)
        widths = [_alt_width(alt) for alt in alts]
        op_width = next((w for w in widths if w), 512)
        expansions = [line for alt in alts for line in _alt_lines(alt, pos, op_width)]
        if not expansions:
            return []
        combos = [combo + [line] for combo in combos for line in expansions]
    out: list[str] = []
    for combo in combos:
        if not combo:
            out.append(mnemonic)
            continue
        # x87 ST(0) is implicit on the memory form ("fadd s/m,s_" also
        # names "fadd dword ptr [rax]").
        if combo[-1] == "st(0)" and any("ptr" in part for part in combo[:-1]):
            out.append(f"{mnemonic} " + ", ".join(combo[:-1]))
        out.append(f"{mnemonic} " + ", ".join(combo))
    return list(dict.fromkeys(out))


@dataclass(frozen=True)
class RowMatch:
    """What the encoded bytes say one workbook row names."""

    targets: tuple[str, ...] = ()  # catalog db_keys
    assembled: bool = False  # at least one concrete line encoded


def _norm_mnemonic(name: str) -> str:
    return name.split("(", 1)[0].strip().casefold().split("/", 1)[0].strip()


def match_rows_by_bytes(rows: list[PerfRow], records: Iterable[Any]) -> list[RowMatch] | None:
    """Resolve each workbook row to the catalog rows its bytes name.

    One llvm-mc run per core, XED decode and the iform pick through
    asm_identity's x86 path. A row whose lines all fail to encode, or name
    no catalog row, gets no target and is dropped by the caller. Returns
    None when the tools or the catalog encoding identity are missing; the
    caller then keeps the operand-class matcher (today's behavior).
    """
    from simdref.asm_identity import (
        Row,
        ToolUnavailableError,
        identify_x86_instructions,
    )

    catalog_rows = [
        Row(
            db_key=record.db_key,
            key=record.key,
            architecture="x86",
            mnemonic=record.mnemonic,
            form=record.form,
            enc_bits=0,
            enc_mask=0,
            enc_match=0,
            enc_extra=(),
            iform=record.metadata.get("iform", ""),
            split_attrs=json.loads(record.metadata.get("split_attrs") or "{}"),
        )
        for record in records
        if record.architecture.casefold() == "x86" and record.metadata.get("iform")
    ]
    if not catalog_rows:
        # A catalog without encoding identity cannot be byte-matched, the
        # schema-13 keyless case in asm_identity: the tool cannot decide.
        return None
    out: list[RowMatch] = [RowMatch()] * len(rows)
    by_core: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_core.setdefault(row.core, []).append(i)
    for indices in by_core.values():
        texts: list[str] = []
        owner: list[int] = []
        for i in indices:
            for line in concrete_lines(rows[i]):
                texts.append(line)
                owner.append(i)
        if not texts:
            continue
        try:
            results = identify_x86_instructions(texts, catalog_rows)
        except (ToolUnavailableError, OSError):
            return None
        targets: dict[int, set[str]] = {i: set() for i in indices}
        assembled: dict[int, bool] = {i: False for i in indices}
        for i, result in zip(owner, results):
            assembled[i] = (
                assembled[i]
                or result.status == "row"
                or result.reason.startswith(("no catalog row", "ambiguous"))
            )
            if result.status != "row":
                continue
            # Only a row of the same instruction counts; an expansion can
            # encode as a sibling mnemonic ("movd mm, rax" encodes MOVQ).
            if _norm_mnemonic(rows[i].mnemonic) == _norm_mnemonic(result.row.mnemonic):
                targets[i].add(result.row.db_key)
        for i in indices:
            out[i] = RowMatch(targets=tuple(sorted(targets[i])), assembled=assembled[i])
    return out


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
