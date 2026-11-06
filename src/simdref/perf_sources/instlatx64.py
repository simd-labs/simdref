"""InstLatX64 dump ingester.

InstLatX64 measures per-instruction latency and reciprocal throughput on
real CPUs with AIDA64's instlatx64 module. Download the dump text files
from the GitHub mirror:

https://raw.githubusercontent.com/InstLatx64/InstLatx64/master/AuthenticAMD/AuthenticAMD0800F11_K17_Zen3_InstLatX64.txt
https://raw.githubusercontent.com/InstLatx64/InstLatx64/master/AuthenticAMD/AuthenticAMD0830F10_K17_Rome_InstLatX64.txt
https://raw.githubusercontent.com/InstLatx64/InstLatX64/master/AuthenticAMD/AuthenticAMD0A20F10_K19_Vermeer_InstLatX64.txt
https://raw.githubusercontent.com/InstLatx64/InstLatX64/master/AuthenticAMD/AuthenticAMD0A60F12_K19_Raphael_04_InstLatX64.txt
https://raw.githubusercontent.com/InstLatx64/InstLatX64/master/AuthenticAMD/AuthenticAMD0B40F40_K20_GraniteRidge_02_512b_InstLatX64.txt
https://raw.githubusercontent.com/InstLatx64/InstLatX64/master/GenuineIntel/GenuineIntel00C0662_ArrowLake_08_BC_InstLatX64.txt

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
import sys
from pathlib import Path

from simdref.perf_sources.merge import SOURCE_INSTLATX64, PerfRow

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VENDOR_DIR = _REPO_ROOT / "vendor" / "instlatx64"

_ARCHITECTURE = "x86"
_SOURCE_KIND = "measured"

_CITATION_BASE = "https://github.com/InstLatx64/InstLatx64/blob/master"

_TAGS = ("{EVEX}", "{REX}", "{VEX}", "{XOP}")
# The lead digit of the parsed marketing model number names the Zen
# generation. When no number matches, the refresh drops the dump.
_ZEN_BY_MODEL = {"1": "ZEN1", "2": "ZEN+", "3": "ZEN2", "5": "ZEN3", "7": "ZEN4", "9": "ZEN5"}


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


# Two row layouts occur. Layout A is one row per line. Layout B (the
# Raphael dump) starts each row with "Inst".
def _row_fields(line: str) -> tuple[str, str, str] | None:
    """Split one dump row into (instruction, L field, T field)."""
    # shortcut: a colon inside the ISA token or tabs before L:/T: parse
    # wrong, upgrade when a dump ever shows one
    if line.startswith("Inst") and line[4:5].isspace():
        line = line[4:]
    index, space, rest = line.lstrip().partition(" ")
    if not space or not index.isdecimal() or not rest or rest[0].isspace():
        return None
    isa, colon, rest = rest.partition(":")
    if not colon or len(isa.split()) != 1:
        return None
    instruction, lmark, rest = rest.partition(" L:")
    latency, tmark, throughput = rest.partition(" T:")
    if not lmark or not tmark or not latency.strip() or not throughput.strip():
        return None
    return instruction.strip(), latency.strip(), throughput.strip()


def _proc_mask(text: str) -> str | None:
    """The ProcMask hex field of the first latency section, if any."""
    marker = "Instruction Latency: Used CPUs:"
    # shortcut: the marker must start its line, upgrade when a dump moves it
    for line in text.splitlines():
        head, sep, rest = line.partition(" ProcMask:")
        if not sep or not head.startswith(marker):
            continue
        if not head[len(marker) :].strip().isdecimal():
            continue
        mask = rest.lstrip().removeprefix("0x")
        end = 0
        while end < len(mask) and (mask[end].isdecimal() or mask[end] in "abcdefABCDEF:"):
            end += 1
        if end:
            return mask[:end]
    return None


def _cpu_types(text: str) -> list[tuple[str, str]]:
    """The ``CPU#`` index and type string of each CPU in the dump."""
    # shortcut: the marker must start its line, upgrade when a dump moves it
    found: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("CPU#"):
            continue
        number, sep, rest = line.partition(' Type: "')
        name, close, _ = rest.partition('"')
        if sep and close and name and number[4:].isdecimal():
            found.append((number[4:], name))
    return found


def _cpuid_name(text: str) -> str | None:
    """The ``CPUID CPU Name`` field of the dump, if present."""
    for line in text.splitlines():
        if line.startswith("CPUID CPU Name"):
            i = _skip_ws(line, 14)
            if i > 14 and line[i : i + 1] == ":":
                name = line[i + 1 :].strip()
                if name:
                    return name
    return None


def _measured_cpu_name(text: str) -> str | None:
    """Return the type string of the CPU the dump actually measured."""
    mask = _proc_mask(text)
    if mask is not None:
        value = 0
        for group in mask.split(":"):
            value = (value << 64) | int(group, 16)
        # The dump is single-CPU; the lowest set bit selects the CPU#
        # block whose ``Type:`` line names the measured part.
        idx = (value & -value).bit_length() - 1 if value else None
        for number, name in _cpu_types(text):
            if int(number) == idx:
                return " ".join(name.split())
    return _cpuid_name(text)


def _core_for_cpu(cpu_name: str) -> str | None:
    """Map the marketing name's model numbers to a canonical core id."""
    words = cpu_name.split()
    for i, word in enumerate(words):
        tail = words[i + 1 :]
        if word == "Ryzen" and len(tail) > 1 and tail[0] in ("3", "5", "7", "9"):
            model = tail[1][:4]
            if len(model) == 4 and model.isdecimal() and model[0] in _ZEN_BY_MODEL:
                return _ZEN_BY_MODEL[model[0]]
        elif word == "EPYC":
            model = tail[0][:4] if tail else ""
            if len(model) == 4 and model.isdecimal():
                if model[0] == "7" and model[3] == "2":
                    return "ZEN2"
                if model[0] == "9" and model[3] == "4":
                    return "ZEN4"
        elif (
            word == "Core(TM)"
            and len(tail) > 2
            and tail[0] == "Ultra"
            and all(c.isalnum() or c == "_" for c in tail[1])
        ):
            model = tail[2][:4]
            if len(model) == 4 and model[0] == "2" and model[1:3].isdecimal() and model[3] == "K":
                return "ARL-P"
    return None


def _source_version(text: str) -> str:
    # shortcut: only the first build marker counts, upgrade when a dump has two
    _, sep, rest = text.partition("aida_bench64.dll build:")
    if sep:
        version, month, day, year = (rest.split() + [""] * 4)[:4]
        if (
            len(month) == 3
            and month[:1].isupper()
            and month[1:].islower()
            and 0 < len(day) <= 2
            and day.isdecimal()
            and len(year) == 4
            and year.isdecimal()
        ):
            return f"aida_bench64.dll build: {version} {month} {int(day)} {year}"
    _, sep, rest = text.partition("AIDA64 Engineer v")
    end = 0
    while end < len(rest) and (rest[end].isdecimal() or rest[end] == "."):
        end += 1
    if sep and end:
        return f"AIDA64 Engineer v{rest[:end]}"
    return "unknown"


def _cycles(field: str) -> str:
    """Extract the cycle count from an ``L:`` or ``T:`` field.

    Non-numeric fields are markers such as ``[no true dep.]`` and give
    an empty string.
    """
    # shortcut: only the first ns= in a field counts, upgrade when a field has two
    before, sep, after = field.partition("ns=")
    run = after.lstrip()
    end = 0
    while end < len(run) and (run[end].isdecimal() or run[end] in "-."):
        end += 1
    tail = before.rstrip()
    if (
        sep
        and end
        and tail
        and run[end : end + 1] == "c"
        and (tail[-1].isdecimal() or tail[-1] in "-.")
    ):
        return run[:end]
    return ""


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


def _strip_tags(text: str) -> str:
    for tag in _TAGS:
        if text.startswith(tag):
            return _strip_tags(text[len(tag) :].lstrip())
    return text


def _is_junk(token: str) -> bool:
    """True for repetition counts (``2x``) and numeric prefixes (``0x66``)."""
    if token.endswith("x"):
        return token[:-1].isdecimal()
    return (
        len(token) > 2
        and token.startswith("0x")
        and all(c in "0123456789abcdefABCDEF" for c in token[2:])
    )


def _split_instruction(text: str) -> tuple[str, str]:
    """Split the instruction text into (mnemonic, operand form).

    Encoding tags (``{EVEX}``, ``{REX}``, ``{VEX}``, ``{XOP}``), the
    repetition counts like ``2x``, and numeric prefixes like ``0x66``
    carry no mnemonic. When one is dropped the row maps to the mnemonic
    alone, so the form is empty.
    """
    text = _strip_tags(text).lstrip()
    if text.startswith("V{EVEX}"):
        # ``V{EVEX} PMOVZXBW`` is the EVEX encoding of VPMOVZXBW.
        text = "V" + text[text.index("}") + 1 :].lstrip()
    tokens = text.split()
    original = list(tokens)
    while len(tokens) > 1 and _is_junk(tokens[0]):
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
        fields = _row_fields(line)
        if fields is None:
            continue
        stats["rows"] += 1
        if core is None:
            stats["dropped"] += 1
            continue
        instruction, latency, throughput = fields
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
                latency=_cycles(latency),
                cpi=_cycles(throughput),
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
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
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
