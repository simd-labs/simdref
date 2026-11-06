"""Perf rows from the Intel Optimization Reference Manual, Volume 2.

Source: Intel 64 and IA-32 Architectures Optimization Reference Manual,
Volume 2, document 356477-051US (October 2026), Chapter 7 Tables 7-3 to
7-17 (latency and throughput by CPUID column) and Chapter 6.9 Table 6-19
(Atom). Download URL for the raw file:

- https://cdrdv2-public.intel.com/930022/356477-051-optimization-reference-manual-vol2.pdf

``refresh`` runs ``pdftotext -layout`` on a downloaded PDF and rewrites
this repo's ``vendor/intel_optref/rows.json``. ``load_vendored`` reads
that JSON. The PDF itself is not vendored; the rows are used under the
README removal-request policy.

Column-to-core mapping comes from the manual's Table 7-1, with one rule
from the tables themselves: a column shared by two microarchitectures
(SNB with IVB, GLM with GLP) yields one row per core. Columns whose CPU
has no cores.py id (Silvermont) are dropped and counted.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from simdref.perf_sources.merge import SOURCE_INTEL_OPTREF, PerfRow, write_json
from simdref.tablecols import value_shape, word_spans

SOURCE_VERSION = "356477-051US, October 2026"
CITATION_URL = (
    "https://cdrdv2-public.intel.com/930022/356477-051-optimization-reference-manual-vol2.pdf"
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_DIR = _REPO_ROOT / "vendor" / "intel_optref"


@dataclass(frozen=True)
class _TableSpec:
    # Column cores, latency first in text order. A tuple lists the
    # canonical ids one column covers. Six tables share one SNB+IVB
    # column; 6-19 shares one GLM+GLP column and puts the throughput
    # block before the latency block. Its first column is Silvermont
    # (SLM).
    latency: tuple[tuple[str, ...] | None, ...]
    throughput: tuple[tuple[str, ...] | None, ...]
    throughput_first: bool = False
    allow_partial: bool = True

    @property
    def total(self) -> int:
        return len(self.latency) + len(self.throughput)


def _spec(cols: tuple[tuple[str, ...], ...]) -> _TableSpec:
    return _TableSpec(latency=cols, throughput=cols)


_SKL_BDW = (("SKL",), ("BDW",))
_SKL_BDW_HSW = (("SKL",), ("BDW",), ("HSW",))
_SKL_BDW_HSW_IVB = (("SKL",), ("BDW",), ("HSW",), ("IVB",))
_SKL_BDW_HSW_SNBIVB = (("SKL",), ("BDW",), ("HSW",), ("SNB", "IVB"))

_TABLES: dict[str, _TableSpec] = {
    "7-3": _spec(_SKL_BDW),
    "7-4": _spec(_SKL_BDW_HSW),
    "7-5": _spec(_SKL_BDW_HSW),
    "7-6": _spec(_SKL_BDW_HSW),
    "7-7": _spec(_SKL_BDW_HSW_IVB),
    "7-8": _spec(_SKL_BDW_HSW_IVB),
    "7-9": _spec(_SKL_BDW_HSW_IVB),
    "7-10": _spec(_SKL_BDW_HSW_SNBIVB),
    "7-11": _spec(_SKL_BDW_HSW_SNBIVB),
    "7-12": _spec(_SKL_BDW_HSW_SNBIVB),
    "7-13": _spec(_SKL_BDW_HSW_SNBIVB),
    "7-14": _spec(_SKL_BDW_HSW_SNBIVB),
    "7-15": _spec(_SKL_BDW_HSW_SNBIVB),
    "7-16": _spec(_SKL_BDW_HSW_SNBIVB),
    "7-17": _spec(_SKL_BDW_HSW_IVB),
    "6-19": _TableSpec(
        latency=(("SLM",), ("GLM", "GLP")),
        throughput=(("SLM",), ("GLM", "GLP")),
        throughput_first=True,
        allow_partial=False,
    ),
}


@dataclass
class ParseStats:
    """Counts a parse run produces for the refresh report."""

    rows: int = 0
    row_groups_per_table: dict[str, int] = field(default_factory=dict)
    rows_per_core: dict[str, int] = field(default_factory=dict)
    dropped_no_core: int = 0  # values in a column with no cores.py id (Silvermont)
    unparseable_rows: int = 0
    unparseable_samples: list[str] = field(default_factory=list)


@dataclass
class _Tok:
    text: str
    x: int
    line: int
    first: bool  # first token of its line


@dataclass
class _Candidate:
    names: list[str]
    vals: list[_Tok]
    orphans: int  # leading vals that arrived before the first name token
    name_line: int = -1  # line the name started on; later lines hold notes


_JUNK_MARKERS = (
    "Document #",
    "INSTRUCTION LATENCY AND THROUGHPUT",
    "EARLIER GENERATIONS OF INTEL ATOM",
    "DisplayFamily_DisplayModel",
    "•",
    "NOTES:",
    "(GLP)",
    "(EP GLP)",
)
_JUNK_PREFIXES = ("7.3.", "*", "Processors")
_NAME_DROP = {"N", "Y", "NA"}  # 6-19 MSROM flags trailing every Atom row


def _is_value(tok: str) -> bool:
    if "_" in tok:
        return False  # CPUID signatures and placeholders, never cell values
    if tok.count("(") != tok.count(")"):
        return False  # parenthetical prose such as "0)", never a cell value
    return tok == "-" or value_shape(tok)


def _digit_dot_units(s: str) -> tuple[int, int]:
    # Leading "digit-run." units of *s*; (count, index past the last dot).
    i, n, units = 0, len(s), 0
    while i < n:
        j = i
        while j < n and s[j].isdecimal():
            j += 1
        if j == i or j >= n or s[j] != ".":
            return units, i
        units += 1
        i = j + 1
    return units, i


def _letter_after_blank(s: str, i: int) -> bool:
    # One or more blanks from index i, then an uppercase letter.
    n = len(s)
    if i >= n or not s[i].isspace():
        return False
    i += 1
    while i < n and s[i].isspace():
        i += 1
    return i < n and "A" <= s[i] <= "Z"


def _is_junk_line(line: str) -> bool:
    if "Throughput" in line and "Latency" in line:
        return True  # repeated column header on every page
    if any(mark in line for mark in _JUNK_MARKERS):
        return True
    s = line.lstrip()
    if any(s.startswith(p) for p in _JUNK_PREFIXES):
        return True
    if len(s) >= 3 and s[0] in "abc" and s[1] == ":" and s[2].isspace():
        return True  # lettered footnote "a: "
    if s.strip() in ("a", "b", "c"):
        return True  # bare footnote letter
    units, i = _digit_dot_units(s)
    return units == 1 and _letter_after_blank(s, i)


def _section_head(line: str) -> bool:
    # "5.2.3. Notes" section headers and "NOTES:" blocks end the open table.
    # Anchored at column 0: an indented "1. ..." is not a section header.
    if line.startswith("NOTES:"):
        return True
    units, i = _digit_dot_units(line)
    return units >= 1 and _letter_after_blank(line, i)


def _table_key(line: str) -> str | None:
    # "Table 7-16." starts a tracked table; return the number, else None.
    i = 0
    while True:
        i = line.find("Table", i)
        if i < 0:
            return None
        j = i + 5
        while j < len(line) and line[j].isspace():
            j += 1
        if j > i + 5:
            token, _, _ = line[j:].partition(" ")
            core, dot, _ = token.partition(".")
            head, dash, tail = core.partition("-")
            if dot and dash and head.isdecimal() and tail.isdecimal():
                return core
        i += 1


def _split_group_form(tokens: list[str]) -> tuple[list[str], str]:
    group: list[str] = []
    rest: list[str] = []
    for tok in tokens:
        if not rest and not group and tok == "Lock":
            continue  # lock-prefixed variant is folded into the plain row
        if rest or tok[0].islower() or tok[0] in "[(" or "," in tok:
            rest.append(tok)
        else:
            group.append(tok)
    if group and group[0] == "Lock":
        group = group[1:]
    return group, " ".join(rest)


def _glues(prev: str, tok: str) -> bool:
    # Wrapped rows split a slashed name across lines: "PCMPGTB/PCMPGTD/PC"
    # then "MPGTW". Join when the fragment is demonstrably incomplete.
    if prev.endswith(("/", ",", ";")):
        return True
    frag = prev[max(prev.rfind("/"), prev.rfind(";")) + 1 :].rstrip(",")
    if not frag.isupper():
        return False
    parts = tok.split("/")
    continues = any(len(part) > len(frag) and part.startswith(frag) for part in parts)
    return len(frag) <= 2 or continues or (len(parts[0]) <= 2 and parts[0].isupper())


def _glue(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        if out and _glues(out[-1], tok):
            out[-1] = out[-1] + tok
        else:
            out.append(tok)
    return out


def _chain_mnemonics(segment: str) -> list[str]:
    parts = [p for p in segment.split("/") if p]
    if not parts:
        return []
    out = [parts[0]]
    for part in parts[1:]:
        if len(part) < len(out[0]):
            candidate = out[0][: len(out[0]) - len(part)] + part
            prefix = len(_common_prefix(candidate, out[0]))
            out.append(candidate if prefix >= 2 else part)
        else:
            out.append(part)
    return out


def _common_prefix(a: str, b: str) -> str:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


def _strip_footnote_digits(mnemonic: str) -> str:
    # Rows glue their table footnote number onto the name: LFENCE3,
    # CLFLUSH12, DIVPD1. Real trailing digits (SHA1RNDS4, F2XM1) only
    # survive when the mnemonic carries an earlier digit.
    i = len(mnemonic)
    while i > 0 and mnemonic[i - 1].isdecimal():
        i -= 1
    stripped = mnemonic[:i]
    return stripped if stripped and not any(c.isdecimal() for c in stripped) else mnemonic


def _mnemonics(group: list[str]) -> list[str]:
    out: list[str] = []
    for tok in _glue(group):
        for segment in tok.split(";"):
            for mnemonic in _chain_mnemonics(segment):
                mnemonic = _strip_footnote_digits(mnemonic)
                if mnemonic and mnemonic[0].isupper():
                    out.append(mnemonic)
    return out


def _parse_text_lines(text: str) -> dict[str, list[_Candidate]]:
    by_table: dict[str, list[_Candidate]] = {}
    key: str | None = None
    cur_names: list[str] = []
    cur_vals: list[_Tok] = []
    orphans = 0
    name_line = -1

    def flush() -> None:
        nonlocal cur_names, cur_vals, orphans, name_line
        if key is not None and (cur_names or cur_vals):
            cand = _Candidate(cur_names, cur_vals, orphans, name_line)
            by_table.setdefault(key, []).append(cand)
        cur_names, cur_vals, orphans, name_line = [], [], 0, -1

    for lineno, raw in enumerate(text.splitlines()):
        line = raw.lstrip("\f")
        tkey = _table_key(line)
        if tkey is not None:
            flush()
            key = tkey if tkey in _TABLES else None
            continue
        if _is_junk_line(line):
            flush()
            continue
        if _section_head(line):
            # Section headers and footnote blocks end the open table.
            flush()
            key = None
            continue
        if key is None:
            continue
        toks = [
            _Tok(text, x, lineno, first=(i == 0)) for i, (text, x) in enumerate(word_spans(line))
        ]
        # 6-19 MSROM flags trail every Atom row and are not values.
        while len(toks) > 2 and toks[-1].text in _NAME_DROP:
            toks.pop()
        vals = [t for t in toks if _is_value(t.text)]
        names = [t.text for t in toks if not _is_value(t.text)]
        if names and not vals:
            if len(names) > 5:
                flush()  # free-flowing prose between tables
                continue
            if cur_names and not _glues(cur_names[-1], names[0]):
                flush()
            # Name continuation of a wrapped row, even after a value line.
            cur_names.extend(names)
            continue
        if not vals:
            continue  # blank line
        if any(len(t) > 4 and all("a" <= c <= "z" for c in t[:5]) for t in names):
            flush()  # prose carrying a stray number between tables
            continue
        # A line with values closes a row that already has values. Rows
        # that wrap their stacked cells onto another line (6-19) stay open
        # while the row still misses values.
        if cur_names and (
            (cur_vals and (len(cur_vals) >= 4 or names))
            or (names and not cur_vals and not _glues(cur_names[-1], names[0]))
        ):
            flush()
        orphans = len(vals) if not cur_names and not names else orphans
        if not cur_names and names:
            name_line = lineno
        cur_names.extend(names)
        cur_vals.extend(vals)
    flush()
    return by_table


def _emit_candidate(
    cand: _Candidate,
    spec: _TableSpec,
    anchors: tuple[list[int], list[int]] | None,
    out: list[PerfRow],
    stats: ParseStats,
    table_key: str,
) -> None:
    if not cand.names:
        return  # straggler footnote number on its own line
    vals = cand.vals
    total = spec.total
    n_tp = len(spec.throughput)
    if len(vals) < total and cand.orphans and len(vals) - cand.orphans == n_tp:
        vals = vals[cand.orphans :]  # footnote stragglers rode the orphan line
    if len(vals) > total:
        kept = vals[-total:]
        lines = {t.line for t in kept}
        comma = any("," in t.text for t in kept)
        if not comma and len(lines) == 1:
            vals = kept  # stacked decor on another line
        elif not comma and len(lines) == 2 and max(lines) <= cand.name_line:
            vals = kept  # cells above and on the name line
    cells: list[str] | None = None
    lat_blank = False
    tp_blank = False
    if len(vals) == total:
        svals = sorted(vals, key=lambda t: t.x)  # stacked cell on another line
        collision = any(b.x - a.x < 3 for a, b in itertools.pairwise(svals))
        if collision and cand.orphans:
            vals = vals[cand.orphans :]  # two rows' cells merged; keep the row's own
        else:
            vals = svals
    if len(vals) == total:
        cells = [t.text for t in vals]
    elif (
        spec.allow_partial
        and anchors is not None
        and len(vals) in (len(spec.latency), len(spec.throughput))
        and all(t.line == cand.name_line for t in vals)
    ):
        # Row with a fully blank half (CLC, LAHF/SAHF): assign by the
        # column the first value sits closest to.
        lat_xs, tp_xs = anchors
        d_tp = min(abs(vals[0].x - x) for x in tp_xs)
        d_lat = min(abs(vals[0].x - x) for x in lat_xs)
        cells = [t.text for t in vals]
        lat_blank = d_tp <= d_lat
        tp_blank = not lat_blank
    if cells is None or "to" in cand.names or any(len(c) > 1 and c.endswith("-") for c in cells):
        stats.unparseable_rows += 1
        if len(stats.unparseable_samples) < 30:
            stats.unparseable_samples.append(f"{table_key}: {' '.join(cand.names)[:60]}")
        return

    names = [t for t in cand.names if t not in _NAME_DROP]
    group, form = _split_group_form(names)
    mnemonics = _mnemonics(group)
    if not mnemonics:
        stats.unparseable_rows += 1
        if len(stats.unparseable_samples) < 30:
            stats.unparseable_samples.append(f"{table_key}: {' '.join(cand.names)[:60]}")
        return

    dash = ["" if c == "-" else c for c in cells]
    n_lat, n_tp = len(spec.latency), len(spec.throughput)
    if lat_blank:
        lat_cells, tp_cells = [""] * n_lat, dash
    elif tp_blank:
        lat_cells, tp_cells = dash, [""] * n_tp
    elif spec.throughput_first:
        tp_cells, lat_cells = dash[:n_tp], dash[n_tp:]
    else:
        lat_cells, tp_cells = dash[:n_lat], dash[n_lat:]

    stats.row_groups_per_table[table_key] = stats.row_groups_per_table.get(table_key, 0) + 1
    for i, lat_cores in enumerate(spec.latency):
        tp_cores = spec.throughput[i]
        cores = lat_cores or tp_cores
        if cores is None:
            stats.dropped_no_core += len(mnemonics)
            continue
        for core in cores:
            for mnemonic in mnemonics:
                out.append(
                    PerfRow(
                        mnemonic=mnemonic,
                        core=core,
                        source=SOURCE_INTEL_OPTREF,
                        source_kind="measured",
                        source_version=SOURCE_VERSION,
                        architecture="x86",
                        form=form,
                        latency=lat_cells[i],
                        cpi=tp_cells[i],
                        citation_url=CITATION_URL,
                    )
                )
                stats.rows += 1
                stats.rows_per_core[core] = stats.rows_per_core.get(core, 0) + 1


def parse_text(text: str) -> tuple[list[PerfRow], ParseStats]:
    """Parse ``pdftotext -layout`` output into perf rows and parse counts."""
    stats = ParseStats()
    out: list[PerfRow] = []
    for table_key, cands in _parse_text_lines(text).items():
        spec = _TABLES[table_key]
        anchors = None
        if spec.allow_partial:
            full = [c for c in cands if len(c.vals) == spec.total and c.names]
            if full:
                anchors = (
                    [c.vals[0].x for c in full],
                    [c.vals[len(spec.latency)].x for c in full],
                )
        for cand in cands:
            _emit_candidate(cand, spec, anchors, out, stats, table_key)
    return out, stats


def load_vendored(path: str | Path | None = None) -> list[PerfRow]:
    """Read ``vendor/intel_optref/*.json`` into PerfRow objects."""
    root = Path(path) if path is not None else _VENDOR_DIR
    rows: list[PerfRow] = []
    for file in sorted(root.glob("*.json")):
        payload = json.loads(file.read_text(encoding="utf-8"))
        rows.extend(PerfRow(**data) for data in payload["rows"])
    return rows


def refresh(rawdir: Path) -> None:
    """Parse the downloaded PDF in *rawdir* and rewrite the vendored JSON."""
    pdfs = sorted(rawdir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDF found in {rawdir}")
    rows: list[PerfRow] = []
    stats = ParseStats()
    for pdf in pdfs:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        file_rows, file_stats = parse_text(proc.stdout)
        rows.extend(file_rows)
        for key, value in file_stats.row_groups_per_table.items():
            stats.row_groups_per_table[key] = stats.row_groups_per_table.get(key, 0) + value
        for key, value in file_stats.rows_per_core.items():
            stats.rows_per_core[key] = stats.rows_per_core.get(key, 0) + value
        stats.rows += file_stats.rows
        stats.dropped_no_core += file_stats.dropped_no_core
        stats.unparseable_rows += file_stats.unparseable_rows
        stats.unparseable_samples.extend(file_stats.unparseable_samples)
    _VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": SOURCE_INTEL_OPTREF,
        "source_version": SOURCE_VERSION,
        "citation_url": CITATION_URL,
        "rows": [asdict(row) for row in rows],
    }
    target = _VENDOR_DIR / "rows.json"
    write_json(payload, target)
    print(f"pdf: {', '.join(pdf.name for pdf in pdfs)}")
    print(f"wrote: {target} ({target.stat().st_size} bytes)")
    print(f"rows: {stats.rows}")
    for core in sorted(stats.rows_per_core):
        print(f"  {core}: {stats.rows_per_core[core]}")
    print("row groups per table:")
    for table_key in sorted(stats.row_groups_per_table):
        print(f"  {table_key}: {stats.row_groups_per_table[table_key]}")
    print(f"dropped (no cores.py id, Silvermont column): {stats.dropped_no_core}")
    print(f"unparseable rows: {stats.unparseable_rows}")
    for sample in stats.unparseable_samples:
        print(f"  {sample}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] != "refresh":
        print("usage: python -m simdref.perf_sources.intel_optref refresh RAWDIR", file=sys.stderr)
        return 2
    refresh(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
