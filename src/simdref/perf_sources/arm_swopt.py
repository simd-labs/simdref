"""Parse Arm Neoverse software optimization guides (SWOG) into PerfRows.

The guides are PDFs the user downloads; refresh converts them with
``pdftotext -layout`` (poppler) and rewrites ``vendor/arm_swopt/*.json``.
Only the AArch64 and SVE tables of chapter 3 are used. One instruction-group
row yields one :class:`PerfRow` per mnemonic with the pipeline list in
``extra_measurement``.

Downloads (probe report, probes/arm-swopt/report.md):

- Neoverse V2  https://documentation-service.arm.com/static/668bc0a369e89f01e39c4668
- Neoverse N2  https://documentation-service.arm.com/static/66880d189082ad344b14c342?token=
- Neoverse V1  https://documentation-service.arm.com/static/69c30b41351d0c3476bddb16
- Neoverse N1  https://documentation-service.arm.com/static/6687d73a69e89f01e39c4507?token=
- Neoverse V3  https://documentation-service.arm.com/static/69c3f3c7351d0c3476bddc36?token=
- Neoverse N3  https://documentation-service.arm.com/static/67be2073fdbd9d54ee93ff15?token=
- Neoverse E1  https://documentation-service.arm.com/static/6687ce1469e89f01e39c44d0?token=

Refresh: ``python -m simdref.perf_sources.arm_swopt refresh RAWDIR`` where
RAWDIR holds the PDFs under the file names in :data:`CPUS` (``v2.pdf`` ...).
Rows whose CPU has no id in cores.py are dropped.

The Neoverse V3AE guide (Issue 3.0, 110075_0000_3.0_en) is vendored from
the same documentation-service.arm.com download as the other guides.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from simdref.perf_sources.cores import canonical_core_id
from simdref.perf_sources.merge import SOURCE_ARM_SWOPT, PerfRow
from simdref.tablecols import mixed_number, row_tail, strip_footnote, word_spans

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VENDOR_DIR = _REPO_ROOT / "vendor" / "arm_swopt"

SOURCE_KIND = "modeled"  # vendor-documented figures, not measured by simdref


@dataclass(frozen=True)
class CpuSpec:
    pdf: str
    cpu: str  # name used for the cores.py lookup
    issue: str
    citation_url: str


CPUS: tuple[CpuSpec, ...] = (
    CpuSpec(
        "v2.pdf",
        "neoverse-v2",
        "Issue 3.0 (r0p2)",
        "https://documentation-service.arm.com/static/668bc0a369e89f01e39c4668",
    ),
    CpuSpec(
        "n2.pdf",
        "neoverse-n2",
        "Issue 5.0 (r0p3)",
        "https://documentation-service.arm.com/static/66880d189082ad344b14c342?token=",
    ),
    CpuSpec(
        "v1.pdf",
        "neoverse-v1",
        "Issue 7.0 (r1p2)",
        "https://documentation-service.arm.com/static/69c30b41351d0c3476bddb16",
    ),
    CpuSpec(
        "n1.pdf",
        "neoverse-n1",
        "Issue 4.0",
        "https://documentation-service.arm.com/static/6687d73a69e89f01e39c4507?token=",
    ),
    CpuSpec(
        "v3.pdf",
        "neoverse-v3",
        "Issue 4.0 (r0p2)",
        "https://documentation-service.arm.com/static/69c3f3c7351d0c3476bddc36?token=",
    ),
    CpuSpec(
        "n3.pdf",
        "neoverse-n3",
        "Issue 3.0 (r0p1)",
        "https://documentation-service.arm.com/static/67be2073fdbd9d54ee93ff15?token=",
    ),
    CpuSpec(
        "e1.pdf",
        "neoverse-e1",
        "Issue 1.0 (r1p1)",
        "https://documentation-service.arm.com/static/6687ce1469e89f01e39c44d0?token=",
    ),
    CpuSpec(
        "v3ae.pdf",
        "neoverse-v3ae",
        "Issue 3.0 (r0p2)",
        "https://documentation-service.arm.com/static/6734ebb3c7fc0d1f211d931c?token=",
    ),
)
_CPU_BY_NAME = {c.cpu: c for c in CPUS}

# A table row ends in: latency, throughput, pipelines and optional notes.
# tablecols.row_tail parses that tail.
_MNEM = re.compile(r"^[A-Z][A-Z0-9]*(?:\(\d\))?(?:\{[A-Z]+\})?$")
_MNEM_CHAR = re.compile(r"[A-Z0-9{(\]},]")
_NOISE = re.compile(r"Copyright|Non-Confidential|Page \d|Issue \d|Arm®|PJDOC|^\s*$")
_FOOT = re.compile(r"^\s*\d+\.\s+[A-Z§]")
_SECT = re.compile(r"^\s*3\.\d+ [A-Z]")


def _toks(seg: str) -> list[str]:
    return [t for t in seg.replace(",", " ").split() if _MNEM.match(t) and (len(t) > 1 or t == "B")]


def _snap_a(line: str, col: int) -> int:
    # col may cut one char into the first mnemonic; walk back to its start.
    if col <= 0 or col > len(line) or line[col - 1] == " ":
        return col
    start = col
    while start > 0 and line[start - 1] != " " and _MNEM_CHAR.match(line[start - 1]):
        start -= 1
        if col - start > 10:
            return col
    tok = line[start:col].split(",")[0]
    if _MNEM.match(tok.lstrip("{(")) or _MNEM.match(tok):
        return start
    return col


def _snap_e(line: str, col: int) -> int:
    # col may cut the last mnemonic mid-token; extend to the next space.
    if col <= 0 or col >= len(line) or line[col] == " " or line[col - 1] == " ":
        return col
    while col < len(line) and line[col] != " ":
        col += 1
    return col


def _peel(head: str, seg: list[str]) -> tuple[str, list[str]]:
    # Some rows place mnemonics just left of the derived column; move trailing
    # mnemonic tokens of head into seg when seg is empty.
    if seg:
        return head.strip(), seg
    moved = []
    rest = head.rstrip()
    while rest:
        word, _, tail = rest.rpartition(" ")
        tok = (tail if tail else word).rstrip(",")
        if not _MNEM.match(tok):
            break
        moved.insert(0, tok)
        rest = word if tail else ""
    return (rest.strip() if moved else head.strip()), moved


def _cont_toks(head: str) -> list[str]:
    # A wrapped mnemonic column can sit fully left of the cut column, leaving
    # only mnemonic tokens in head.
    toks = [t.rstrip(",") for t in head.split() if t.rstrip(",")]
    if toks and all(_MNEM.match(t) and (len(t) > 1 or t == "B") for t in toks):
        return toks
    return []


def parse_groups(text: str) -> list[dict]:
    """Parse ``pdftotext -layout`` output into AArch64 group rows.

    Each row keeps the group name, raw latency, raw throughput (instructions
    per cycle), pipeline letters, notes and the mnemonic list.
    """
    lines = text.split("\n")
    start = next(
        (i for i, l in enumerate(lines) if re.match(r"^\s*3\.1 Instruction tables\s*$", l)), None
    )
    if start is None:
        return []
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if re.match(r"^\s*4\.1 Dispatch constraints\s*$", lines[i])
        ),
        None,
    )
    if end is None:
        end = next(
            (
                i
                for i in range(start + 1, len(lines))
                if re.match(r"^\s*4 Special considerations\s*$", lines[i])
            ),
            len(lines),
        )
    out: list[dict] = []
    arch = col_a = col_e = cur = in_foot = None
    in_foot = False
    for i in range(start, end):
        line = lines[i]
        s = line.strip()
        if re.match(r"^\s*Table \S+", line):
            m = re.search(r"AArch(32|64)", line)
            arch = ("AArch" + m.group(1)) if m else "AArch64"
            in_foot, cur, col_a, col_e = False, None, None, None
            continue
        # Column headers repeat at each page break; re-derive the cut columns
        # each time. They may drift by a few chars between pages. The mnemonic
        # column is the last word before "Exec" (AArch64, Instructions or
        # "SVE Instruction" depending on the guide and the table).
        if "Exec" in line and re.search(r"AArch|Instruction", line):
            words = word_spans(line[: line.find("Exec")])
            if words:
                col_a, col_e = words[-1][1], line.find("Exec")
            continue
        if "Latency" in line and "Throughput" in line:
            continue
        if _SECT.match(line):
            in_foot = False
            continue
        if _FOOT.match(line):
            in_foot = True
        if in_foot or _NOISE.search(line) or not s:
            continue
        m = row_tail(s)
        if m and col_a is not None and col_e is not None and col_a < col_e:
            a = _snap_a(line, col_a)
            if a >= col_e:
                a = col_a
            e = _snap_e(line, col_e)
            head, first = _peel(line[:a], _toks(line[a:e]))
            if not head:
                continue
            lat, tp, pipe, notes = m
            cur = {
                "group": head,
                "latency": lat,
                "throughput": tp,
                "pipelines": " ".join(pipe.split()),
                "notes": notes.strip(),
                "arch": arch,
                "mnemonics": first,
            }
            out.append(cur)
            continue
        if cur is None or col_a is None:
            continue
        a = min(_snap_a(line, col_a), len(line))
        e = _snap_e(line, col_e)
        head = line[:a].strip()
        seg = _toks(line[a:e])
        if not seg:
            moved = _cont_toks(head)
            if moved:
                seg, head = moved, ""
        if head or seg:
            if head:
                cur["group"] = (cur["group"] + " " + head).strip()
            cur["mnemonics"] += seg
    return [r for r in out if r["arch"] == "AArch64"]


def _clean_num(raw: str) -> str:
    # Drop a footnote marker: "4(1)" becomes "4".
    return strip_footnote(raw)


def _cpi(throughput: str) -> str:
    """Invert the throughput (instructions per cycle) to cycles per instruction."""
    raw = _clean_num(throughput)
    mixed = mixed_number(raw)
    try:
        if mixed is not None:
            ipc = mixed
        else:
            ipc = Fraction(raw)
    except ValueError:
        return ""  # ranges such as "1/4 to 1/2" stay in extra_measurement
    cpi = 1 / ipc
    if cpi == cpi.numerator // cpi.denominator:
        return str(cpi.numerator // cpi.denominator)
    return f"{float(cpi):.3f}".rstrip("0").rstrip(".")


def perf_rows(groups: list[dict], cpu: str) -> tuple[list[PerfRow], int]:
    """Expand group rows to one PerfRow per mnemonic.

    Returns (rows, dropped) where dropped counts groups skipped because the
    CPU has no core id in cores.py.
    """
    core = canonical_core_id(cpu)
    if core is None:
        return [], len(groups)
    spec = _CPU_BY_NAME.get(cpu)
    issue = spec.issue if spec else ""
    url = spec.citation_url if spec else ""
    rows: list[PerfRow] = []
    for g in groups:
        latency = _clean_num(g["latency"])
        extra = {
            "pipelines": g["pipelines"].rstrip(","),
            "throughput_ipc": g["throughput"],
            "instruction_group": g["group"],
        }
        if g["notes"] and g["notes"] != "-":
            extra["notes"] = g["notes"]
        cpi = _cpi(g["throughput"])
        for mnem in dict.fromkeys(g["mnemonics"]):
            rows.append(
                PerfRow(
                    mnemonic=mnem,
                    core=core,
                    source=SOURCE_ARM_SWOPT,
                    source_kind=SOURCE_KIND,
                    source_version=issue,
                    architecture="arm",
                    latency=latency,
                    cpi=cpi,
                    applies_to="mnemonic",
                    citation_url=url,
                    extra_measurement=extra,
                )
            )
    return rows, 0


def parse_text(text: str, cpu: str) -> list[PerfRow]:
    """Pure parser: guide text to PerfRows for one CPU."""
    rows, _ = perf_rows(parse_groups(text), cpu)
    return rows


def write_vendored(groups: list[dict], cpu: str, vendor_dir: Path | None = None) -> Path:
    """Write vendor/arm_swopt/<cpu>.json for one CPU and return its path."""
    spec = _CPU_BY_NAME[cpu]
    vendor_dir = vendor_dir or DEFAULT_VENDOR_DIR
    vendor_dir.mkdir(parents=True, exist_ok=True)
    store = [
        {
            "group": g["group"],
            "latency": g["latency"],
            "throughput": g["throughput"],
            "pipelines": g["pipelines"],
            "notes": g["notes"],
            "mnemonics": list(dict.fromkeys(g["mnemonics"])),
        }
        for g in groups
    ]
    payload = {
        "cpu": cpu,
        "source_version": spec.issue,
        "citation_url": spec.citation_url,
        "groups": store,
    }
    path = vendor_dir / f"{cpu}.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    )
    return path


def load_vendored(path: str | Path | None = None) -> list[PerfRow]:
    """Read vendor/arm_swopt/*.json and return the PerfRows.

    *path* may be the vendor directory or a single JSON file; the default is
    the repo's vendor directory. CPUs with no core id in cores.py are dropped.
    """
    root = Path(path) if path is not None else DEFAULT_VENDOR_DIR
    files = [root] if root.is_file() else sorted(root.glob("*.json"))
    rows: list[PerfRow] = []
    for f in files:
        payload = json.loads(f.read_text())
        per, _ = perf_rows(payload["groups"], payload["cpu"])
        rows.extend(per)
    return rows


def refresh(rawdir: str | Path, vendor_dir: str | Path | None = None) -> None:
    """Parse the PDFs in *rawdir* and rewrite vendor/arm_swopt/*.json."""
    rawdir = Path(rawdir)
    vendor_dir = Path(vendor_dir) if vendor_dir is not None else None
    for spec in CPUS:
        pdf = rawdir / spec.pdf
        if not pdf.is_file():
            print(f"{spec.pdf}: not found, skipped", file=sys.stderr)
            continue
        text = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        groups = parse_groups(text)
        rows, dropped = perf_rows(groups, spec.cpu)
        path = write_vendored(groups, spec.cpu, Path(vendor_dir) if vendor_dir else None)
        print(
            f"{spec.cpu}: {len(groups)} groups, {len(rows)} rows vendored"
            + (f", {dropped} groups dropped (no core id)" if dropped else "")
            + f" -> {path}"
        )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "refresh":
        if len(argv) != 2:
            print("usage: python -m simdref.perf_sources.arm_swopt refresh RAWDIR", file=sys.stderr)
            return 2
        refresh(argv[1])
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
