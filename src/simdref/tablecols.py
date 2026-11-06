"""Column helpers for vendor table text: split cells and classify value shapes.

The vendor tables (Intel SDM opcode pages, Intel optref, Arm SWOG) are parsed
by the structure of the line: runs of spaces cut the cells, and the value
tokens follow the vendors' own notation. No regular expressions.
"""

from __future__ import annotations

from fractions import Fraction

_VALUE_CHARS = frozenset("0123456789.,~<>()/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-")
_PIPE_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
_NOTES_CHARS = frozenset("-0123456789, ")


def split_cells(line: str) -> list[str]:
    """Cut at runs of two or more whitespace; single spaces stay in the cell."""
    cells: list[str] = []
    cur, gap, prev_space = "", 0, ""
    for ch in line:
        if ch.isspace():
            prev_space = ch
            gap += 1
            if gap == 2:
                cells.append(cur)
                cur = ""
        else:
            if gap == 1:
                cur += prev_space
            cur += ch
            gap = 0
    if gap == 1:
        cur += prev_space
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


# Value templates: each letter is one (separator, digits) step after the
# leading digit run, in the old alternation order. "T"/"t" is " to ", a
# lowercase or space step tolerates missing digits after the separator.
_LAT_STEPS = ("T", " /", "/")  # d to d; d d/d; d/d
_TP_STEPS = ("/", ".", " /", "/t/")  # d/d; d.d; d d/d; d/d to d/d


def _value_forms(s: str, i: int, templates: tuple[str, ...]) -> list[tuple[str, int]]:
    # Candidate values at position i, in the old alternation order: the
    # footnote marker d(d), the plain digit run, then each template.
    d = _digit_run(s, i)
    out: list[tuple[str, int]] = []
    m = _marker(s, d)
    if m > 0:
        out.append((s[i:m], m))
    out.append((s[i:d], d))
    for form in templates:
        p = d
        for step in form:
            sep = " to " if step in "Tt" else step
            if not s.startswith(sep, p):
                break
            q = _digit_run(s, p + len(sep))
            if step not in " t" and q == p + len(sep):
                break
            p = q
        else:
            out.append((s[i:p], p))
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
        while k < n and s[k] in " ,":
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
        for lat, p in _value_forms(s, i, _LAT_STEPS):
            q = p
            while q < n and s[q].isspace():
                q += 1
            if q == p:
                continue
            for tp, r in _value_forms(s, q, _TP_STEPS):
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
