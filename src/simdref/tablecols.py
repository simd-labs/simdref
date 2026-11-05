"""Column helpers for vendor table text: split cells and classify value shapes.

The vendor tables (Intel SDM opcode pages, Intel optref, Arm SWOG) are parsed
by the structure of the line: runs of spaces cut the cells, and the value
tokens follow the vendors' own notation. No regular expressions.
"""

from __future__ import annotations

from fractions import Fraction

_VALUE_CHARS = frozenset("0123456789.,~<>()/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-")
_PIPE_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
_PIPE_SEP = frozenset(" ,")
_NOTES_CHARS = frozenset("-0123456789, ")


def split_cells(line: str) -> list[str]:
    """Cut at runs of two or more whitespace; single spaces stay in the cell."""
    cells: list[str] = []
    cur = ""
    i, n = 0, len(line)
    while i < n:
        j = i
        while j < n and line[j].isspace():
            j += 1
        if j == i:
            while j < n and not line[j].isspace():
                j += 1
            cur += line[i:j]
        elif j - i >= 2:
            cells.append(cur)
            cur = ""
        else:
            cur += line[i:j]
        i = j
    cells.append(cur)
    return cells


def word_spans(line: str) -> list[tuple[str, int]]:
    """Return (word, start) for every run of non-space characters."""
    words: list[tuple[str, int]] = []
    i, n = 0, len(line)
    while i < n:
        if line[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not line[j].isspace():
            j += 1
        words.append((line[i:j], i))
        i = j
    return words


def value_shape(tok: str) -> bool:
    """True for a cell value: optional "-~<>" prefix, one digit, value chars."""
    i = 1 if tok.startswith(("-", "~", "<", ">")) else 0
    return len(tok) > i and tok[i].isdecimal() and all(c in _VALUE_CHARS for c in tok[i + 1 :])


def strip_footnote(raw: str) -> str:
    """Drop a footnote marker: "4(1)" becomes "4"."""
    head, sep, tail = raw.partition("(")
    if sep and head.isdecimal() and len(tail) == 2 and tail[0].isdecimal() and tail[1] == ")":
        return head
    return raw


def mixed_number(raw: str) -> Fraction | None:
    """Parse a mixed number "3 1/2" as one Fraction; None when it is not one."""
    whole, sep, frac = raw.partition(" ")
    if (
        sep
        and whole.isdecimal()
        and frac.count("/") == 1
        and all(p.isdecimal() for p in frac.split("/"))
    ):
        return Fraction(int(whole)) + Fraction(frac)
    return None


def _digit_run(s: str, i: int) -> int:
    j = i
    while j < len(s) and s[j].isdecimal():
        j += 1
    return j


def _marker(s: str, i: int) -> int:
    # One footnote marker "(d)" right after a digit run; end index or -1.
    if s[i : i + 1] == "(" and i + 2 < len(s) and s[i + 1].isdecimal() and s[i + 2] == ")":
        return i + 3
    return -1


def _lat_forms(s: str, i: int) -> list[tuple[str, int]]:
    # Latency shapes in the old alternation order: d(d), d, "d to d",
    # "d d/d", "d/d". Each shape has one candidate at a position.
    d = _digit_run(s, i)
    out: list[tuple[str, int]] = []
    m = _marker(s, d)
    if m > 0:
        out.append((s[i:m], m))
    out.append((s[i:d], d))
    if s[d : d + 4] == " to ":
        d2 = _digit_run(s, d + 4)
        if d2 > d + 4:
            out.append((s[i:d2], d2))
    if s[d : d + 1] == " ":
        d2 = _digit_run(s, d + 1)
        if s[d2 : d2 + 1] == "/":
            d3 = _digit_run(s, d2 + 1)
            if d3 > d2 + 1:
                out.append((s[i:d3], d3))
    if s[d : d + 1] == "/":
        d2 = _digit_run(s, d + 1)
        if d2 > d + 1:
            out.append((s[i:d2], d2))
    return out


def _tp_forms(s: str, i: int) -> list[tuple[str, int]]:
    # Throughput shapes in the old alternation order: d(d), d, "d/d", "d.d",
    # "d d/d", "d/d to d/d".
    d = _digit_run(s, i)
    out: list[tuple[str, int]] = []
    m = _marker(s, d)
    if m > 0:
        out.append((s[i:m], m))
    out.append((s[i:d], d))
    if s[d : d + 1] == "/":
        d2 = _digit_run(s, d + 1)
        if d2 > d + 1:
            out.append((s[i:d2], d2))
    if s[d : d + 1] == ".":
        d2 = _digit_run(s, d + 1)
        if d2 > d + 1:
            out.append((s[i:d2], d2))
    if s[d : d + 1] == " ":
        d2 = _digit_run(s, d + 1)
        if s[d2 : d2 + 1] == "/":
            d3 = _digit_run(s, d2 + 1)
            if d3 > d2 + 1:
                out.append((s[i:d3], d3))
    if s[d : d + 1] == "/":
        d2 = _digit_run(s, d + 1)
        if s[d2 : d2 + 4] == " to ":
            d3 = _digit_run(s, d2 + 4)
            if s[d3 : d3 + 1] == "/":
                d4 = _digit_run(s, d3 + 1)
                if d4 > d3 + 1:
                    out.append((s[i:d4], d4))
    return out


def _pipe_end(s: str, i: int) -> int:
    # Greedy pipe list: [A-Z][A-Z0-9]* tokens joined by spaces or commas.
    n = len(s)
    end = i
    while i < n and s[i] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        j = i + 1
        while j < n and s[j] in _PIPE_CHARS:
            j += 1
        end = j
        k = j
        while k < n and s[k] in _PIPE_SEP:
            k += 1
        if k == j or k >= n or s[k] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            break
        i = k
    return end


def row_tail(s: str) -> tuple[str, str, str, str] | None:
    """Parse the row tail of a stripped table line.

    Returns (latency, throughput, pipelines, notes) or None. The tail is
    "latency throughput pipelines" with optional notes at the end; the scan
    starts at the leftmost digit whose tail parses to the line end.
    shortcut: assumes a stripped line (its only caller strips), upgrade if a
    caller passes trailing spaces.
    """
    n = len(s)
    for i, ch in enumerate(s):
        if not ch.isdecimal():
            continue
        for lat, p in _lat_forms(s, i):
            q = p
            while q < n and s[q].isspace():
                q += 1
            if q == p:
                continue
            for tp, r in _tp_forms(s, q):
                t = r
                while t < n and s[t].isspace():
                    t += 1
                if t == r:
                    continue
                e = _pipe_end(s, t)
                if e == t:
                    continue
                rest = s[e:]
                if not rest:
                    return lat, tp, s[t:e], ""
                w = 0
                while w < len(rest) and rest[w].isspace():
                    w += 1
                if w and w < len(rest) and all(c in _NOTES_CHARS for c in rest[w:]):
                    return lat, tp, s[t:e], rest[w:]
    return None
