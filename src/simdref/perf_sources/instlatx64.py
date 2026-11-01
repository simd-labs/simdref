"""InstLatX64 dump ingester.

InstLatX64 measures per-instruction latency and reciprocal throughput on
real CPUs with AIDA64's instlatx64 module. Download the dump text files
from the GitHub mirror:

https://raw.githubusercontent.com/InstLatx64/InstLatx64/master/AuthenticAMD/AuthenticAMD0800F11_K17_Zen3_InstLatX64.txt
https://raw.githubusercontent.com/InstLatx64/InstLatx64/master/AuthenticAMD/AuthenticAMD0830F10_K17_Rome_InstLatX64.txt
https://raw.githubusercontent.com/InstLatx64/InstLatx64/master/AuthenticAMD/AuthenticAMD0A20F10_K19_Vermeer_InstLatX64.txt
https://raw.githubusercontent.com/InstLatx64/InstLatx64/master/AuthenticAMD/AuthenticAMD0A60F12_K19_Raphael_04_InstLatX64.txt
https://raw.githubusercontent.com/InstLatx64/InstLatx64/master/AuthenticAMD/AuthenticAMD0B40F40_K20_GraniteRidge_02_512b_InstLatX64.txt
https://raw.githubusercontent.com/InstLatx64/InstLatx64/master/GenuineIntel/GenuineIntel00C0662_ArrowLake_08_BC_InstLatX64.txt

The file name does not always name the measured CPU: the K17_Zen3 file
measures a Zen 1 Ryzen 5 1500X. The CPU therefore comes from the dump
header, not the file name. Refresh with::

    python -m simdref.perf_sources.instlatx64 refresh RAWDIR

The refresh parses every ``*.txt`` dump in RAWDIR and rewrites
``vendor/instlatx64/*.json``. The license decision: the source is used
under the README removal-request policy, so only the derived JSON is
vendored, never the dumps.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

from simdref.perf_sources.merge import SOURCE_INSTLATX64, PerfRow

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VENDOR_DIR = _REPO_ROOT / "vendor" / "instlatx64"

_ARCHITECTURE = "x86"
_SOURCE_KIND = "measured"

_CITATION_BASE = "https://github.com/InstLatx64/InstLatx64/blob/master"

# Two row layouts occur. Layout A is one row per line. Layout B (the
# Raphael dump) starts each row with "Inst".
_ROW_LAYOUT_A = re.compile(r"^\s*(\d+) (\S+)\s*:(.*?)\s+L:\s*(\S.*?)\s+T:\s*(\S.*)$")
_ROW_LAYOUT_B = re.compile(r"^Inst\s+(\d+) (\S+)\s*:\s*(.*?)\s+L:\s*(\S.*?)\s+T:\s*(\S.*)$")

_USED_CPUS = re.compile(r"Instruction Latency: Used CPUs:\s*\d+ ProcMask:\s*0x([0-9A-Fa-f:]+)")
_CPU_TYPE = re.compile(r'CPU#(\d+) Type: "([^"]+)"')
_CPUID_NAME = re.compile(r"^CPUID CPU Name\s+:\s*(.+?)\s*$", re.MULTILINE)
_BUILD = re.compile(r"aida_bench64\.dll build:\s*(\S+)\s+([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{4})")
_AIDA64_VERSION = re.compile(r"AIDA64 Engineer (v[0-9.]+)")

_NUMERIC_FIELD = re.compile(r"[-\d.]+\s*ns=\s*([-\d.]+)c")
_TAG = re.compile(r"^(?:\{(?:EVEX|REX|VEX|XOP)\}\s*)+")
_JUNK_TOKEN = re.compile(r"(?:\d+x|0x[0-9A-Fa-f]+)$")

# Ordered (regex on the CPU type string, canonical core id). The first
# match wins. The marketing name maps to the microarchitecture generation.
# When no pattern matches, the refresh drops the dump and counts it.
_CPU_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"Ryzen [3579] 1\d{3}", "ZEN1"),
    (r"Ryzen [3579] 2\d{3}", "ZEN+"),
    (r"Ryzen [3579] 3\d{3}|EPYC 7\d{2}2", "ZEN2"),
    (r"Ryzen [3579] 5\d{3}", "ZEN3"),
    (r"Ryzen [3579] 7\d{3}|EPYC 9\d{2}4", "ZEN4"),
    (r"Ryzen [3579] 9\d{3}", "ZEN5"),
    (r"Core\(TM\) Ultra \w+ 2\d\dK", "ARL-P"),
)


def _measured_cpu_name(text: str) -> str | None:
    """Return the type string of the CPU the dump actually measured."""
    match = _USED_CPUS.search(text)
    if match:
        mask = 0
        for group in match.group(1).split(":"):
            mask = (mask << 64) | int(group, 16)
        # The dump is single-CPU; the lowest set bit selects the CPU#
        # block whose ``Type:`` line names the measured part.
        idx = (mask & -mask).bit_length() - 1 if mask else None
        for number, name in _CPU_TYPE.findall(text):
            if int(number) == idx:
                return " ".join(name.split())
    match = _CPUID_NAME.search(text)
    return match.group(1) if match else None


def _core_for_cpu(cpu_name: str) -> str | None:
    for pattern, core_id in _CPU_PATTERNS:
        if re.search(pattern, cpu_name):
            return core_id
    return None


def _source_version(text: str) -> str:
    match = _BUILD.search(text)
    if match:
        version, month, day, year = match.groups()
        return f"aida_bench64.dll build: {version} {month} {int(day)} {year}"
    match = _AIDA64_VERSION.search(text)
    if match:
        return f"AIDA64 Engineer {match.group(1)}"
    return "unknown"


def _cycles(field: str) -> str:
    """Extract the cycle count from an ``L:`` or ``T:`` field.

    Non-numeric fields are markers such as ``[no true dep.]`` and give
    an empty string.
    """
    match = _NUMERIC_FIELD.search(field)
    return match.group(1) if match else ""


def _has_fused(text: str) -> bool:
    """True when a ``+`` outside brackets joins two instructions."""
    depth = 0
    for char in text:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        elif char == "+" and depth == 0:
            return True
    return False


def _split_instruction(text: str) -> tuple[str, str]:
    """Split the instruction text into (mnemonic, operand form).

    Encoding tags (``{EVEX}``, ``{REX}``, ``{VEX}``, ``{XOP}``),
    repetition counts like ``2x``, and numeric prefixes like ``0x66``
    carry no mnemonic. When one is dropped the row maps to the mnemonic
    alone, so the form is empty.
    """
    text = _TAG.sub("", text, count=1).lstrip()
    if text.startswith("V{EVEX}"):
        # ``V{EVEX} PMOVZXBW`` is the EVEX encoding of VPMOVZXBW.
        text = "V" + text[text.index("}") + 1 :].lstrip()
    tokens = text.split()
    original = list(tokens)
    while len(tokens) > 1 and _JUNK_TOKEN.fullmatch(tokens[0]):
        tokens.pop(0)
    if len(tokens) < len(original):
        # A repetition or numeric prefix was dropped: the operand form is
        # ambiguous, so the row applies to the mnemonic.
        form = ""
    else:
        form = " ".join(tokens[1:])
    if tokens and not tokens[0][0].isalpha():
        # Compound lines like ``4xADDPS xm1,xm1 4x xm2,xm2`` have no
        # single mnemonic; the caller drops the row.
        return "", ""
    return tokens[0] if tokens else "", form


def parse_dump(text: str, citation_url: str) -> tuple[list[PerfRow], dict[str, int]]:
    """Parse one InstLatX64 dump into PerfRows.

    Returns the rows plus counters: ``rows`` parsed, ``fused`` skipped,
    and ``dropped`` when the CPU has no known core id.
    """
    cpu_name = _measured_cpu_name(text)
    core = _core_for_cpu(cpu_name) if cpu_name else None
    version = _source_version(text)
    stats = {"rows": 0, "fused": 0, "dropped": 0}
    rows: list[PerfRow] = []
    for line in text.splitlines():
        match = _ROW_LAYOUT_B.match(line) or _ROW_LAYOUT_A.match(line)
        if match is None:
            continue
        stats["rows"] += 1
        if core is None:
            stats["dropped"] += 1
            continue
        instruction = match.group(3).strip()
        if _has_fused(instruction):
            stats["fused"] += 1
            continue
        mnemonic, form = _split_instruction(instruction)
        if not mnemonic:
            stats["dropped"] += 1
            continue
        rows.append(
            PerfRow(
                mnemonic=mnemonic,
                core=core,
                source=SOURCE_INSTLATX64,
                source_kind=_SOURCE_KIND,
                source_version=version,
                architecture=_ARCHITECTURE,
                form=form,
                latency=_cycles(match.group(4)),
                cpi=_cycles(match.group(5)),
                citation_url=citation_url,
            )
        )
    return rows, stats


def _citation_url(path: Path) -> str:
    subdir = "AuthenticAMD" if path.name.startswith("AuthenticAMD") else "GenuineIntel"
    return f"{_CITATION_BASE}/{subdir}/{path.name}"


def refresh(rawdir: Path, outdir: Path | None = None) -> dict[str, object]:
    """Parse every dump in *rawdir* and rewrite the vendored JSON files."""
    rawdir = Path(rawdir)
    outdir = DEFAULT_VENDOR_DIR if outdir is None else Path(outdir)
    by_core: dict[str, list[PerfRow]] = {}
    totals = {"rows": 0, "fused": 0, "dropped": 0, "files": 0}
    for dump in sorted(rawdir.glob("*.txt")):
        rows, stats = parse_dump(
            dump.read_text(encoding="utf-8", errors="replace"), _citation_url(dump)
        )
        totals["files"] += 1
        for key in ("rows", "fused", "dropped"):
            totals[key] += stats[key]
        if rows:
            by_core.setdefault(rows[0].core, []).extend(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    for stale in outdir.glob("*.json"):
        stale.unlink()
    for core, rows in sorted(by_core.items()):
        payload = [dataclasses.asdict(row) for row in rows]
        (outdir / f"{core}.json").write_text(
            json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return {"per_core": {core: len(rows) for core, rows in sorted(by_core.items())}, **totals}


def load_vendored(path: Path | str | None = None) -> list[PerfRow]:
    """Load the vendored rows written by :func:`refresh`."""
    root = DEFAULT_VENDOR_DIR if path is None else Path(path)
    rows: list[PerfRow] = []
    for file in sorted(root.glob("*.json")):
        rows.extend(PerfRow(**entry) for entry in json.loads(file.read_text(encoding="utf-8")))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] != ["refresh"] or len(args) != 2:
        print(__doc__)
        return 2
    report = refresh(Path(args[1]))
    print(f"files parsed: {report['files']}")
    print(
        f"rows parsed: {report['rows']}, fused skipped: {report['fused']}, "
        f"dropped (unknown core): {report['dropped']}"
    )
    for core, count in report["per_core"].items():
        print(f"{core}: {count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
