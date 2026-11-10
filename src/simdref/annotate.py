"""Annotate assembly (``.s``) files with per-instruction summaries and perf.

Given GAS/AT&T-syntax assembly, emit an annotated ``.sa`` file where each
recognised instruction line carries a trailing ``# ...`` comment describing
what the instruction does plus latency / CPI figures pulled from the
simdref catalog.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator

from simdref import perf
from simdref.arm_instructions import _CONDITION_CODES
from simdref.asm_identity import isa_by_acceptance
from simdref.catkey import ISA_FAMILY, isa_family as _isa_family
from simdref.lsp import NO_CATALOG_ROW, _document_arch, identity_keys
from simdref.models import InstructionRecord
from simdref.regtable import REG_TABLE, reg_rows
from simdref.storage import load_instruction_from_db, load_instructions_by_mnemonic_from_db


class LineKind(str, Enum):
    BLANK = "blank"
    LABEL = "label"
    DIRECTIVE = "directive"
    COMMENT = "comment"
    INSTRUCTION = "instruction"


@dataclass(slots=True)
class AsmLine:
    kind: LineKind
    raw: str
    indent: str = ""
    mnemonic: str = ""
    operands: str = ""
    trailing_comment: str = ""
    address: int | None = None
    source_file: str | None = None
    source_line: int | None = None


# Instruction head without trailing comment: the '#' comment (and GAS-style
# annotations) is split off with str.partition — per the regex HOWTO, string
# methods beat a lazy re group for a fixed single-character cut.
def _instr_head(text: str) -> tuple[str, str, str] | None:
    """An instruction head: (indent, mnemonic, operands), or None."""
    n = len(text)
    i = 0
    while i < n and text[i] in " \t":
        i += 1
    if i >= n or not ("a" <= text[i] <= "z" or "A" <= text[i] <= "Z"):
        return None
    j = i + 1
    while j < n and (text[j].isascii() and (text[j].isalnum() or text[j] in "._")):
        j += 1
    if j < n and text[j] not in " \t":
        return None
    k = j
    while k < n and text[k] in " \t":
        k += 1
    return text[:i], text[i:j], text[k:]


def _is_label(text: str) -> bool:
    """A label line: a name then ':' (hex-letter addresses too, issue #61)."""
    n = len(text)
    i = 0
    while i < n and text[i] in " \t":
        i += 1
    if i >= n:
        return False
    head = text[i]
    if not (head.isascii() and (head.isalpha() or head in "._$")):
        return False
    j = i + 1
    while j < n and (text[j].isalnum() or text[j] in "._$"):
        j += 1
    return j < n and text[j] == ":"


# objdump listings (issue #65): one format detection per listing, then one
# rule per address line: the byte column holds next_address - address bytes.
# objdump right-aligns the address with spaces; a TAB right before it marks a
# relocation line ("-r") or a plain-asm label ("\tadd: ret"), not an address.
_HEX_DIGITS = frozenset(string.hexdigits)
_MNEMONIC_CHARS = frozenset(string.ascii_letters + string.digits + "._")


def _addr_line(line: str) -> tuple[int, int] | None:
    """An address line: ``(address, index after the ':')``, else None."""
    i = 0
    while i < len(line) and line[i] in " \t":
        i += 1
    if i and line[i - 1] == "\t":
        return None  # a TAB right before the address: a relocation or label line
    j = i
    while j < len(line) and line[j] in _HEX_DIGITS:
        j += 1
    if j == i or line[j : j + 1] != ":" or line[j + 1 : j + 2].strip():
        return None
    return int(line[i:j], 16), j + 1


def _is_hex_group(tok: str) -> bool:
    """A byte group: 2-digit byte, 4-digit halfword or 8-digit word."""
    return len(tok) in (2, 4, 8) and all(c in _HEX_DIGITS for c in tok)


# A byte column: a hex run before objdump's TAB, or before 2+ spaces once a
# terminal paste expanded the TAB. One group needs padding before the TAB
# ("fadd\td0" has none); a full llvm x86 column of 10+ bytes has none either.
def _padded_run(body: str) -> bool:
    """The hex run before objdump's TAB closes the byte column."""
    s = body.lstrip(" \t")
    for k in range(2, min(8, len(s)) + 1):
        if s[k : k + 1] == " " and all(c in _HEX_DIGITS for c in s[:k]) and _tab_tail(s[k:]):
            return True
    return False


def _tab_tail(t: str) -> bool:
    """Byte groups and padding after one space, then objdump's closing TAB."""
    pos = 0
    while True:
        j = pos
        while j < len(t) and t[j] == " ":
            j += 1
        if j < len(t) and t[j] == "\t":
            return True
        if t[pos : pos + 1] != " ":
            return False
        j = pos + 1
        while j < len(t) and t[j] in _HEX_DIGITS:
            j += 1
        if not 2 <= j - pos - 1 <= 8 or t[j : j + 1] not in (" ", "\t"):
            return False
        pos = j


def _pasted_run(body: str) -> bool:
    """The hex run before the 2+ space gap that a pasted byte column leaves."""
    s = body.lstrip(" ")
    for k in range(2, min(8, len(s)) + 1):
        if s[k : k + 1] == " " and all(c in _HEX_DIGITS for c in s[:k]) and _gap_tail(s[k:]):
            return True
    return False


def _gap_tail(t: str) -> bool:
    """Byte groups after one space, then a 2+ space gap and a non-space char."""
    pos = 0
    while True:
        j = pos
        while j < len(t) and t[j] == " ":
            j += 1
        if j - pos >= 2 and t[j : j + 1] and not t[j].isspace():
            return True
        if t[pos : pos + 1] != " ":
            return False
        j = pos + 1
        while j < len(t) and t[j] in _HEX_DIGITS:
            j += 1
        if not 2 <= j - pos - 1 <= 8 or t[j : j + 1] != " ":
            return False
        pos = j


def _token_starts(text: str) -> list[int]:
    """Start offsets of the whitespace-separated tokens of ``text``."""
    starts = []
    pos = 0
    for tok in text.split():
        pos = text.find(tok, pos)
        starts.append(pos)
        pos += len(tok)
    return starts


def _is_instr_head(head: str) -> bool:
    """An instruction head: a mnemonic, then optional whitespace operands."""
    if not head or head[0] not in string.ascii_letters:
        return False
    i = 1
    while i < len(head) and head[i] in _MNEMONIC_CHARS:
        i += 1
    return i == len(head) or head[i] in " \t"


def _file_format_name(line: str) -> str | None:
    """The objdump ``file format NAME`` header: the format name, or None."""
    start = 0
    while True:
        i = line.find("file format", start)
        if i < 0:
            return None
        start = i + 1
        if i and (line[i - 1] == "_" or line[i - 1].isalnum()):
            continue
        j = k = i + len("file format")
        while k < len(line) and line[k].isspace():
            k += 1
        if k <= j or k >= len(line):
            continue
        j = k
        while j < len(line) and not line[j].isspace():
            j += 1
        return line[k:j]


# objdump -S injects "file:line" comment lines before the instruction block.
def _objdump_source_marker(line: str) -> tuple[str, int] | None:
    """An objdump -S source marker "file:line": (file, line), or None."""
    s = line.strip()
    colon = s.rfind(":")
    if colon < 1 or s[0] == "/":
        return None
    number = s[colon + 1 :]
    if not number.isdecimal():
        return None
    file = s[:colon]
    # the old file first char class allowed ':', the rest could not contain it
    colons = file.count(":")
    if colons > 1 or (colons == 1 and file[0] != ":"):
        return None
    return file.strip(), int(number)


# Legal x86 prefix tokens objdump prints as words before the mnemonic
# ("data16 cs nopw ..."). The mnemonic is the first token after them.
_X86_PREFIX_TOKENS = frozenset(
    {
        "data16",
        "addr32",
        "cs",
        "ds",
        "es",
        "ss",
        "fs",
        "gs",
        "lock",
        "rep",
        "repe",
        "repne",
        "notrack",
        "bnd",
    }
)


# A bare line holding one prefix token is the prefix instruction by itself
# (issue #52 prefix padding, issue #60 class 4 silent drop). addr16/data32
# and the REX spellings join the set objdump prints before a mnemonic.
def _is_lone_rex(token: str) -> bool:
    """A lone REX prefix spelling ("rex", "rex.w"), full token."""
    rest = token[3:] if token.startswith("rex") else None
    if rest is None:
        return False
    return rest == "" or (
        rest.startswith(".") and rest[1:].isascii() and rest[1:].isalpha() and len(rest) <= 5
    )


_LONE_PREFIX_TOKENS = _X86_PREFIX_TOKENS | {"addr16", "data32"}


def _strip_prefix_tokens(head: str) -> str:
    """Drop leading x86 prefix tokens; return the text from the mnemonic on."""
    rest = head.lstrip()
    while True:
        parts = rest.split(None, 1)
        if len(parts) != 2 or parts[0].lower() not in _X86_PREFIX_TOKENS:
            return rest
        rest = parts[1]


def _objdump_body(line: str) -> str:
    """The text after the address colon, or the whole line when it has no address."""
    addr = _addr_line(line)
    return line[addr[1] :] if addr else line


def _opens_byte_column(body: str) -> bool:
    toks = body.split()
    run = _padded_run if "\t" in body else _pasted_run
    # A byte-only body is a GNU wrap continuation line, also a byte column.
    return run(body) or (bool(toks) and all(map(_is_hex_group, toks)))


def _objdump_shaped(body: str) -> bool:
    """objdump's address-column form: a TAB or a byte column after the colon."""
    return body.lstrip(" ")[:1] == "\t" or _opens_byte_column(body)


def _group_bytes(group: str) -> bytes:
    # shortcut: a 4-/8-digit group is a little-endian halfword/word value; upgrade for big-endian listings.
    return int(group, 16).to_bytes(len(group) // 2, "little")


_ObjdumpRecord = tuple[int | None, bytes | None, str, str | None]


def _objdump_records(lines: list[str]) -> dict[int, _ObjdumpRecord]:
    """Parse an objdump listing: ``{line_index: (address, bytes, rest, bytehex)}``.

    ``rest`` starts at the mnemonic (prefix tokens kept); ``bytes`` is the
    encoding in memory order, or None when the listing has no byte column.
    ``bytehex`` is the column in the asm_identity contract: one 4-/8-digit
    group as printed (the word value), else memory-order 2-digit bytes.
    Returns ``{}`` when the lines are not an objdump listing (plain asm).
    Detection runs once per listing; then the byte column of each address
    line is the run of hex groups that holds next_address - address bytes.
    """
    content = [line for line in lines if line[:1] in " \t" and line.strip()]
    addr_lines = [line for line in content if _addr_line(line)]
    # A pasted fragment has no header: it is objdump-shaped when most content
    # lines are "addr:" lines with a TAB or a byte column after the colon (a
    # label with a same-line instruction, "\tadd: ret", has neither).
    shaped = [line for line in addr_lines if _objdump_shaped(_objdump_body(line))]
    has_header = any(map(_file_format_name, lines[:10]))
    if not has_header and not (content and len(shaped) * 2 > len(content)):
        return {}
    has_addr = len(addr_lines) * 2 > len(content)  # --no-leading-addr still prints data addresses
    probe = [_objdump_body(line) for line in (addr_lines if has_addr else content)[:50]]
    has_bytes = sum(map(_opens_byte_column, probe)) * 2 > len(probe)

    rows: list[tuple[int, int | None, str]] = []
    for index, line in enumerate(lines):
        addr = _addr_line(line)
        if addr:
            body = line[addr[1] :]
            toks = body.split()[:2]
            # A pasted relocation or label line lost its TABs but has no byte column
            # either; a pasted long llvm x86 column has no gap, so two groups count.
            if (
                not has_bytes
                or _objdump_shaped(body)
                or (len(toks) == 2 and all(map(_is_hex_group, toks)))
            ):
                rows.append((index, addr[0], body))
        elif not has_addr and line[:1] in " \t" and line.strip() and not line.strip().endswith(":"):
            rows.append((index, None, line))

    records: dict[int, _ObjdumpRecord] = {}
    last = None  # line index of the open instruction, for wrap folding
    for k, (index, addr, text) in enumerate(rows):
        starts = _token_starts(text)
        toks = text.split()
        run = []
        for tok in toks:
            if not _is_hex_group(tok):
                break
            run.append(tok)
        cum = [0]
        for tok in run:
            cum.append(cum[-1] + len(tok) // 2)
        nxt = rows[k + 1][1] if k + 1 < len(rows) else None
        want = nxt - addr if addr is not None and nxt is not None and nxt > addr else None
        # Without a byte column only data lines print bytes: the split must leave text.
        if want in cum[1:] and (has_bytes or cum.index(want) < len(toks)):
            n = cum.index(want)
        elif not has_bytes:
            n = 0
        elif cum[-1] and len(run) == len(toks):
            n = len(run)  # only hex groups: a continuation line or a last data line
        elif "\t" in text.strip(" "):
            # Last line of a block or no address: objdump ends the column with a TAB.
            n = min(len(text.lstrip(" \t").split("\t")[0].split()), len(run))
        else:
            # Pasted text: the column ends at the padding gap; no gap, no bytes.
            gap = len(text.lstrip(" ").split("  ")[0].split())
            n = gap if gap <= len(run) else 0
        # The split must leave a mnemonic or a directive: "fadd %st(1),%st" and
        # "addb $1" have no byte column although "fadd" and "addb" are hex.
        if n < len(toks) and not (toks[n].startswith(".") or _is_instr_head(toks[n])):
            n = 0
        col = b"".join(map(_group_bytes, run[:n])) if has_bytes else None
        if n == len(toks):  # wrap continuation: fold into the open instruction when contiguous
            if last is not None and addr is not None and col:
                a0, b0, rest0, _hex0 = records[last]
                if a0 is not None and a0 + len(b0 or b"") == addr:
                    b0 = (b0 or b"") + col
                    records[last] = (a0, b0, rest0, b0.hex(" "))
            continue
        rest = text[starts[n] :].rstrip()
        if not _is_instr_head(_strip_prefix_tokens(_split_comment(rest)[0])):
            last = None  # data (".word"), "(bad)", "...": no instruction
            continue
        word = n == 1 and len(run[0]) > 2
        records[index] = (
            addr,
            col,
            rest,
            run[0].lower() if word else col.hex(" ") if col else None,
        )
        last = index
    return records


def _listing_records(lines: list[str]) -> dict[int, _ObjdumpRecord]:
    """_objdump_records for a whole input; plain asm with pasted objdump lines parses per line."""
    return _objdump_records(lines) or {
        index: rec
        for index, line in enumerate(lines)
        if (rec := _objdump_records([line]).get(0)) is not None
    }


def _parse_objdump_instr(line: str) -> tuple[int | None, str, str | None] | None:
    """One objdump line: ``(address, rest, bytehex)`` or None (single-line view of _objdump_records)."""
    rec = _objdump_records([line]).get(0)
    return None if rec is None else (rec[0], rec[2], rec[3])


def _split_comment(text: str) -> tuple[str, str]:
    """Split AT&T ``#`` trailing comment off; returns (head, comment)."""
    head, sep, comment = text.partition("#")
    return head, (sep + comment if sep else "")


def parse_asm_line(line: str, *, track_positions: bool = False) -> AsmLine:
    stripped = line.rstrip("\n")
    return _asm_line(stripped, _objdump_records([stripped]).get(0), track_positions)


def _asm_line(stripped: str, obj: _ObjdumpRecord | None, track_positions: bool) -> AsmLine:
    """Classify one line; ``obj`` is its _objdump_records entry, or None."""
    if not stripped.strip():
        return AsmLine(LineKind.BLANK, stripped)
    bare = stripped.lstrip()
    if bare.startswith("#") or bare.startswith("//"):
        return AsmLine(LineKind.COMMENT, stripped)
    if bare.startswith("."):
        return AsmLine(LineKind.DIRECTIVE, stripped)
    if bare.startswith("Disassembly of "):
        # objdump section banner: "Disassembly of section .text:"
        return AsmLine(LineKind.COMMENT, stripped)

    # objdump input (workflow §1d: `objdump -d ... > file.s`) is a first-class
    # input. This runs before the label check: a hex-letter address ("b:",
    # "c:") otherwise reads as a label name (issue #61).
    if obj is not None:
        address, _bytes, rest, _hex = obj
        head, comment = _split_comment(rest)
        m = _instr_head(_strip_prefix_tokens(head))
        if m:
            return AsmLine(
                kind=LineKind.INSTRUCTION,
                raw=stripped,
                indent=(stripped[: stripped.find(rest)] if rest in stripped else ""),
                mnemonic=m[1],
                operands=m[2].strip(),
                trailing_comment=comment.strip(),
                address=address,
            )
    if track_positions:
        # In objdump mode, any other line is an objdump header, `-S` source
        # interleave, or an inline symbol annotation, never an AT&T
        # instruction. Bailing out here avoids classifying C++ source
        # keywords ("return", "if", "for", ...) as mnemonics.
        return AsmLine(LineKind.COMMENT, stripped)

    if _is_label(stripped):
        return AsmLine(LineKind.LABEL, stripped)

    head, comment = _split_comment(stripped)
    indent = head[: len(head) - len(head.lstrip())]
    m = _instr_head(indent + _strip_prefix_tokens(head))
    if not m:
        return AsmLine(LineKind.COMMENT, stripped)
    return AsmLine(
        kind=LineKind.INSTRUCTION,
        raw=stripped,
        indent=m[0],
        mnemonic=m[1],
        operands=m[2].strip(),
        trailing_comment=comment.strip(),
        address=None,
    )


# ---------------------------------------------------------------------------
# Catalog lookup
# ---------------------------------------------------------------------------


# Common AT&T size suffixes that may be absent from the catalog's Intel form.
_ATT_SUFFIXES = ("b", "w", "l", "q", "s", "d", "t")


def _lookup_variants(mnemonic: str) -> list[str]:
    """Candidate mnemonics to try against the catalog, in preference order."""
    base = mnemonic.lower()
    out = [base]
    # Strip a single trailing size suffix (AT&T style) if present.
    if len(base) > 2 and base[-1] in _ATT_SUFFIXES:
        trimmed = base[:-1]
        if trimmed not in out:
            out.append(trimmed)
    return out


# AArch64 condition-code branches: b.<cond>, no register table covers them.
# The vocabulary is the first spelling of each cond value in the A64 row table
# (cs/cc alias the hs/lo spellings the llvm-objdump printers print).
_A64_BCOND_CODES = frozenset(
    dict(reversed([(cond, code) for code, cond in _CONDITION_CODES])).values()
)


def _bcond_hits(text: str) -> int:
    """Count b.<cond> branch mnemonics with word boundaries on both sides."""
    count = 0
    i = text.find("b.")
    while i >= 0:
        tail = text[i + 2 : i + 4]
        before = text[i - 1 : i]
        after = text[i + 4 : i + 5]
        if (
            tail in _A64_BCOND_CODES
            and not (before and (before.isalnum() or before == "_"))
            and not (after and (after.isalnum() or after == "_"))
        ):
            count += 1
            i = text.find("b.", i + 4)
        else:
            i = text.find("b.", i + 2)
    return count


def _drop_symbol_annotations(text: str) -> str:
    """Drop objdump symbol annotations ("<x0+0x50>"): a target, never input."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "<":
            out.append(text[i])
            i += 1
            continue
        j = text.find(">", i + 1)
        if j < 0:
            out.append(text[i:])
            break
        out.append(" ")
        i = j + 1
    return "".join(out)


def _drop_symbol_target(text: str) -> str:
    """Keep the symbol expression of "114 <f+0x40>", drop the target address.

    lane C6: the bare numeric offset must not reach llvm-mc."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "<":
            out.append(text[i])
            i += 1
            continue
        j = text.find(">", i + 1)
        if j < 0:
            out.append(text[i:])
            break
        k = i
        while k > 0 and text[k - 1].isspace():
            k -= 1
        m = k
        while m > 0 and text[m - 1] in "0123456789abcdefABCDEF":
            m -= 1
        start = k
        if m < k:
            start = m
            if m >= 2 and text[m - 2 : m] == "0x":
                start = m - 2
        if i > start:
            out = out[: -(i - start)]
        out.append(text[i + 1 : j])
        i = j + 1
    return "".join(out)


def _voter_names():
    """The ISA vote spellings, derived from the a64/riscv rows of the table.

    The digit rules exclude wzr/xzr like the old regexes; v0-v31 stay
    arm-exclusive (bug-compatible: finding C3)."""
    excl_arm: set[str] = set()
    shared: set[str] = set()
    for name, rows in REG_TABLE.items():
        for klass, _width, family in rows:
            if family != "a64":
                continue
            if klass == "xw" and name[1:].isdigit() and int(name[1:]) <= 30:
                (excl_arm if name[0] == "w" else shared).add(name)
            elif klass == "v" and name[1:].isdigit() and int(name[1:]) <= 31:
                excl_arm.add(name)
    shared.add("sp")
    riscv = {
        name
        for name, rows in REG_TABLE.items()
        if any(k == "rvabi" and f == "riscv" for k, _w, f in rows)
        and not any(k == "fp" and f == "a64" for k, _w, f in rows)
    }
    riscv.discard("sp")
    riscv |= {"fp", "x31"}
    return frozenset(excl_arm), frozenset(shared), frozenset(riscv)


_EXCL_ARM, _SHARED, _RISCV = _voter_names()


def _isa_votes(text: str) -> tuple[int, int, int]:
    """(excl_arm, excl_riscv, shared) register votes of one instruction text."""
    excl_arm = excl_riscv = shared = 0
    i = 0
    n = len(text)
    while i < n:
        if not (text[i] == "_" or text[i].isalnum()):
            i += 1
            continue
        j = i
        while j < n and (text[j] == "_" or text[j].isalnum()):
            j += 1
        token = text[i:j]
        if i == 0 or text[i - 1] != "%":
            if token in _EXCL_ARM:
                excl_arm += 1
            elif token in _SHARED:
                shared += 1
            elif token in _RISCV:
                excl_riscv += 1
        i = j
    return excl_arm, excl_riscv, shared


def detect_isa(lines: Iterable[str]) -> str | None:
    """Name the ISA the input speaks: ``"x86"``, ``"arm"`` or ``"riscv"``.

    An objdump ``file format`` line names the ISA outright. Otherwise the
    AArch64 register names x0-x30, w0-w30, v0-v31, sp, the ``b.<cond>``
    branches and the RISC-V register names vote. Returns None when the
    input carries no ISA signal.
    Fallback only: annotate_stream prefers asm_identity.isa_by_acceptance
    when llvm-mc runs.
    shortcut: a RISC-V listing that uses only xN names and sp votes arm,
    upgrade when a form pick on rv64 rows needs those listings split
    """
    excl_arm = excl_riscv = shared = 0
    for line in lines:
        name = _file_format_name(line)
        if name:
            name = name.lower()
            if "riscv" in name:
                return "riscv"
            return "arm" if "aarch64" in name or "arm" in name else "x86"
        parsed = parse_asm_line(line)
        if parsed.kind != LineKind.INSTRUCTION:
            continue
        # Strip objdump symbol annotations: "<x0+0x50>" names a target, not
        # a register (review finding 1).
        text = _drop_symbol_annotations(f"{parsed.mnemonic} {parsed.operands}")
        vote_arm, vote_riscv, vote_shared = _isa_votes(text)
        excl_arm += vote_arm + _bcond_hits(text)
        excl_riscv += vote_riscv
        shared += vote_shared
    if excl_arm or excl_riscv:
        return "arm" if excl_arm >= excl_riscv else "riscv"
    return "arm" if shared else None


def _line_isa(text: str) -> str | None:
    """Name the family one instruction line's own evidence shows.

    detect_isa's register and ``b.<cond>`` votes come first, then the
    strong operand votes ``_document_arch`` feeds byte identity. Returns
    None for a neutral line: its family is the stream's.
    """
    text = _drop_symbol_annotations(text)
    vote = detect_isa([text])
    if vote:
        return vote
    picked = _document_arch((text,), False, False)
    return ISA_FAMILY.get(picked[0]) if picked else None


def lookup(
    mnemonic: str, conn: sqlite3.Connection, isa: str | None = None
) -> list[InstructionRecord]:
    keep = _isa_family(isa)
    for cand in _lookup_variants(mnemonic):
        records = load_instructions_by_mnemonic_from_db(conn, cand)
        if keep is not None:
            records = [r for r in records if _isa_family(r.architecture) == keep]
        if records:
            return records
    return []


_SIZE_WORDS = frozenset({"byte", "word", "dword", "qword", "xmmword", "ymmword", "zmmword"})


def _has_size_ptr(text: str) -> bool:
    """True for a size keyword with a trailing "ptr" ("byte ptr [rdx]")."""
    n = len(text)
    i = 0
    while i + 3 <= n:
        if text[i : i + 3].lower() != "ptr":
            i += 1
            continue
        if i + 3 < n and (text[i + 3].isalnum() or text[i + 3] == "_"):
            i += 1
            continue
        j = i
        while j > 0 and text[j - 1].isspace():
            j -= 1
        if j == i:
            i += 1
            continue
        for word in _SIZE_WORDS:
            k = j - len(word)
            if (
                k >= 0
                and text[k:j].lower() == word
                and not (k > 0 and (text[k - 1].isalnum() or text[k - 1] == "_"))
            ):
                return True
        i += 1
    return False


def _x86_reg_class(name: str) -> str | None:
    """The x86 register class token of one spelling, or None.

    klass "mm" stays unclassified: the old regexes never listed mmx."""
    for klass, width, family in reg_rows(name):
        if family != "x86":
            continue
        if klass == "gpr":
            return f"R{width}"
        if klass == "vec":
            return {128: "XMM", 256: "YMM", 512: "ZMM"}[width]
        if klass == "k":
            return "K"
    return None


def _is_vec_shaped(name: str) -> bool:
    # shortcut: the old regexes classed table-less vector names like xmm99, the table answers "?" (approved change C2); upgrade if a form pick ever needs them classed
    return name[:3] in ("xmm", "ymm", "zmm") and name[3:].isdecimal()


# ---- Operand-shape parser for variant selection ----------------------------
#
# We split an operand string like ``%eax, %eax`` or ``(%rdi), %xmm0`` into
# typed slots so we can reject variants whose shape doesn't match (e.g. a
# memory-destination form when there's no memory operand in the input) and
# recognise zeroing idioms (``xor %eax, %eax``) by operand aliasing.


def _split_operands(operands: str) -> list[str]:
    """Split by commas that are outside parentheses/brackets."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in operands:
        if ch in "([":
            depth += 1
            buf.append(ch)
        elif ch in ")]":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _imm_kind(value: int) -> str:
    """Immediate slot kind sized like the narrowest encoding that holds *value*."""
    if value == 0:
        return "I0"
    for bits in (8, 16, 32, 64):
        if -(1 << (bits - 1)) <= value < (1 << bits):
            return f"I{bits}"
    return "I64"


def _is_number(text: str) -> bool:
    """True for the literal ``-?0x<hex>`` or ``-?<digits>`` the scans matched."""
    body = text[1:] if text[:1] == "-" else text
    if body[:2] == "0x":
        body = body[2:]
        return bool(body) and all(char in string.hexdigits for char in body)
    return bool(body) and body.isdecimal()


def _is_symbol(text: str) -> bool:
    """True for ``[A-Za-z_.$]`` heads with ``\\w.$@`` tails (a plain symbol)."""
    head = text[:1]
    return (
        bool(head)
        and head.isascii()
        and (head.isalpha() or head in "_.$")
        and all(char.isalnum() or char in "_.$@" for char in text[1:])
    )


def _classify_operand(op: str) -> tuple[str, str | None]:
    """Return ``(kind, reg_name_lower)`` for one operand.

    ``kind`` is one of ``R8``/``R16``/``R32``/``R64``/``XMM``/``YMM``/
    ``ZMM``/``K``/``M``/``I0``/``I8``/``I16``/``I32``/``I64``/``I``/
    ``REL``/``?``. Immediate kinds carry the width the value needs.
    ``reg_name_lower`` is the lowercase register name when ``kind`` is a
    register class (for alias detection), else ``None``.
    """
    s = op.strip()
    if not s:
        return "?", None
    # objdump branch target: an address followed by a <symbol> annotation.
    lt = s.find("<")
    if lt > 0 and s.endswith(">") and ">" not in s[lt + 1 : -1]:
        k = lt
        while k > 0 and s[k - 1].isspace():
            k -= 1
        if k < lt and _is_number(s[:k]):
            return "REL", None
    # AT&T immediate: $... ; Intel immediate: bare number / 0x... / '$'-less literal
    if s.startswith("$") or _is_number(s):
        try:
            return _imm_kind(int(s.lstrip("$"), 0)), None
        except ValueError:
            return "I", None
    # AT&T memory: ``(%reg)``, ``disp(%reg,%idx,scale)``, or plain symbol
    # with a ``(...)`` section; Intel memory: ``[...]`` or explicit size ptr.
    if "(" in s or "[" in s or _has_size_ptr(s):
        return "M", None
    # Pure register?
    name = (s[1:] if s[:1] == "%" else s).casefold()
    tok = _x86_reg_class(name)
    if tok is not None or _is_vec_shaped(name):
        return tok or "?", name
    # Plain symbol (branch target) or unknown.
    if _is_symbol(s):
        return "REL", None
    return "?", None


def _operand_shape(operands: str) -> tuple[list[tuple[str, str | None]], bool]:
    """Parse *operands* into ``(kind, reg_name)`` slots and a "aliased" flag.

    ``aliased`` is true iff the expression has >= 2 register operands and
    they all refer to the same register (e.g. ``%eax, %eax`` or
    ``%ymm1, %ymm1, %ymm1``). This is the zeroing-idiom signal for
    xor/pxor/vpxor/vxorps/subtract-self families on modern x86.
    """
    if not operands:
        return [], False
    slots = [_classify_operand(p) for p in _split_operands(operands)]
    reg_names = [n for k, n in slots if n is not None and k not in ("M",)]
    aliased = len(reg_names) >= 2 and len(set(reg_names)) == 1
    return slots, aliased


def _form_operand_tokens(form_or_key: str) -> list[str]:
    """Extract the comma-separated operand tokens from a catalog form/key.

    For ``"XOR (R32, 0)"`` this returns ``["R32", "0"]``; for
    ``"RET_NEAR"`` returns ``[]``. Tokens are upper-cased.
    """
    s = str(form_or_key or "")
    tail = s.rstrip()
    if not tail or tail[-1] != ")":
        return []
    close = len(tail) - 1
    # The inner text starts at the first "(" after the last ")" before the
    # closing one, so no ")" splits it (the old group matched the same span).
    start = tail.find("(", tail.rfind(")", 0, close) + 1)
    if start < 0:
        return []
    inner = tail[start + 1 : close]
    return [t.strip().upper() for t in inner.split(",") if t.strip()]


def _is_broadcast_token(token: str) -> bool:
    """True for a form token with a ``_1toN`` tail (``M16_1to8``)."""
    start = len(token)
    while start > 0 and token[start - 1].isdecimal():
        start -= 1
    return start < len(token) and start >= 4 and token[start - 4 : start] == "_1TO"


def _token_matches_slot(form_tok: str, slot_kind: str) -> bool:
    """True when a catalog operand token matches a parsed input slot kind."""
    ft = form_tok.upper()
    sk = slot_kind.upper()
    if ft == sk:
        return True
    # Accumulator shorthand: AL/AX/EAX/RAX match Rn of same width.
    if ft in ("AL",) and sk == "R8":
        return True
    if ft in ("AX",) and sk == "R16":
        return True
    if ft in ("EAX",) and sk == "R32":
        return True
    if ft in ("RAX",) and sk == "R64":
        return True
    # R8h/R8l both map to R8.
    if ft in ("R8H", "R8L") and sk == "R8":
        return True
    # A plain R row takes a general register of any width.
    if ft == "R" and sk in ("R8", "R16", "R32", "R64"):
        return True
    # Mn forms: M32 matches a memory slot regardless of inferred width.
    if ft.startswith("M") and sk == "M":
        return True
    # Relative branch targets match the Rel8/Rel32 rows.
    if sk == "REL" and ft.startswith("REL"):
        return True
    # Immediate rows: a literal-0 row takes only a zero immediate, and an
    # In row takes an immediate whose value fits the encoding width.
    if sk == "I0":
        return ft == "0" or (ft.startswith("I") and ft[1:].isdigit())
    if sk.startswith("I"):
        if not (ft.startswith("I") and ft[1:].isdigit()):
            return False
        return sk == "I" or int(ft[1:]) >= int(sk[1:])
    return False


def _slot_token_score(form_tok: str, slot_kind: str) -> int:
    """Match strength for one form token against one input slot kind."""
    ft = form_tok.upper()
    sk = slot_kind.upper()
    if not _token_matches_slot(ft, sk):
        # Weak near-misses: a broadcast row wants a {1toN} annotation, and a
        # register can ride a memory row (mov r,r against MOV (M32, R32)).
        if sk == "M" and _is_broadcast_token(ft):
            return 2
        if sk.startswith("R") and ("M" + sk[1:]) == ft:
            return 2
        return 0
    score = 10
    if sk == "I0" and ft == "0":
        score += 2
    elif sk not in ("I", "I0") and ft.startswith("I") and int(ft[1:]) == int(sk[1:]):
        score += 2
    elif sk == "REL" and ft == "REL32":
        # The text carries no encoding width; the near jump is the common row.
        score += 2
    return score


def _operand_match_score(
    record: InstructionRecord,
    slots: list[tuple[str, str | None]],
    *,
    aliased: bool = False,
) -> int:
    """Score how well *record*'s form matches the parsed input *slots*.

    Exact operand-shape match dominates. Length mismatches, extra
    memory/immediate operands the input doesn't carry, etc. are penalised.
    When the input is an aliased-register zeroing idiom (e.g. ``xor
    %eax, %eax``) and the form's trailing operand is the literal ``0``,
    a strong bonus is applied so uops.info's zero-idiom rows
    (``XOR (R32, 0)``, ``VPXOR (YMM, YMM, YMM)`` etc.) win the tiebreak.
    """
    form_tokens = _form_operand_tokens(getattr(record, "key", "") or "")
    if not slots and not form_tokens:
        return 0
    if not slots:
        # Preserve a tiny signal so n-ary forms lose to zero-ary ones.
        return -len(form_tokens)

    score = 0
    matched_slots = 0
    for (kind, _), form_tok in zip(slots, form_tokens):
        pair = _slot_token_score(form_tok, kind)
        score += pair
        if pair >= 10:
            matched_slots += 1

    # Penalise operand-count mismatch — heavily for extra slots in the form.
    len_delta = len(form_tokens) - len(slots)
    if len_delta > 0:
        score -= 20 * len_delta
    elif len_delta < 0:
        score -= 5 * (-len_delta)

    # If the form carries a memory operand but the input has none, reject
    # hard — this is what made "xor %eax,%eax" resolve to "XOR (M32, R32)".
    input_has_mem = any(k == "M" for k, _ in slots)
    form_has_mem = any(t.startswith("M") and t not in {"MM"} for t in form_tokens)
    if form_has_mem and not input_has_mem:
        score -= 50

    # Zero-idiom: aliased register operands, form's tail operand is 0.
    if aliased and form_tokens and form_tokens[-1] == "0":
        score += 30

    # Fully matched every slot is meaningful: small bonus on top.
    if matched_slots == len(slots) and matched_slots > 0:
        score += 5

    return score


# shortcut: the form pick answers only when byte identity has no row
# (schema-13 catalogs or missing tools); remove it at the schema-14 data
# release with simdref-llvm-mc on PyPI as a required dependency
def pick_record(
    records: list[InstructionRecord],
    *,
    arch: str | None = None,
    operands: str = "",
    isa: str | None = None,
) -> InstructionRecord | None:
    if not records:
        return None
    candidates = records
    keep = _isa_family(isa)
    if keep is not None:
        candidates = [r for r in records if _isa_family(r.architecture) == keep]
        if not candidates:
            return None
    if arch is not None:
        pinned = [r for r in candidates if arch in (r.arch_details or {})]
        if pinned:
            candidates = pinned

    slots, aliased = _operand_shape(operands)
    if "%" in operands or "$" in operands:
        # AT&T lists sources first; catalog form tokens are in Intel order.
        slots = list(reversed(slots))

    def score(rec: InstructionRecord) -> tuple[int, int, int, int]:
        measured = sum(
            1
            for d in (rec.arch_details or {}).values()
            if (d.get("source_kind") or "measured") == "measured"
        )
        op_score = _operand_match_score(rec, slots, aliased=aliased)
        # Operand match dominates; a plain row beats encoding-variant rows
        # (ADD_NF, MOV_8B, ...) on ties, then measurement coverage.
        head = str(getattr(rec, "key", "") or "").split("(", 1)[0]
        variant = 1 if "_" in head else 0
        return (op_score, -variant, measured, len(rec.arch_details or {}))

    return max(candidates, key=score)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PerfSummary:
    latency: float | None
    cpi: float | None
    n_archs: int
    source_kind: str  # "measured", "modeled", "mixed"
    archs_used: list[str] = field(default_factory=list)


def _per_arch_value(details: dict[str, Any], value_fn) -> float | None:
    for v in value_fn(details):
        try:
            # A "min-max" string keeps its first number so the float stays numeric.
            return float(str(v).split("-", 1)[0])
        except (TypeError, ValueError):
            continue
    return None


def _latency_for(
    details: dict[str, Any], operands: list[dict[str, Any]] | None = None
) -> list[str]:
    # One value per row: the largest latency over the read-operand pairs
    # (simdref#65), never the first listed pair.
    # Without operands (arch_perf, aggregate_perf) the old first-value pick stays.
    bounds = None if operands is None else perf.latency_bounds(details, operands)
    if bounds is None:
        # No read-start pair has data (KMOVQ (M64, K)): keep the old first value.
        return perf.latency_cycle_values(details.get("latencies") or [])
    return [bounds[1]]


def arch_latency(record: InstructionRecord, arch: str) -> float | None:
    """Latency of the row of *arch*, with the write-only-start pairs skipped."""
    details = (record.arch_details or {}).get(arch) or {}
    return _per_arch_value(details, lambda d: _latency_for(d, record.operand_details))


def arch_latency_range(record: InstructionRecord, arch: str) -> tuple[float, float] | None:
    """``(low, high)`` when the row of *arch* stores a min-max latency range, else None."""
    details = (record.arch_details or {}).get(arch) or {}
    bounds = perf.latency_bounds(details, record.operand_details)
    if bounds is None or bounds[0] == bounds[1]:
        return None
    return float(bounds[0]), float(bounds[1])


def fmt_arch_latency(record: InstructionRecord, arch: str, lat: float | None) -> str:
    """The latency text of a pinned row: ``MIN-MAX`` for a stored range, else ``lat``."""
    lat_range = arch_latency_range(record, arch)
    if lat_range is None:
        return _fmt_num(lat)
    return f"{_fmt_num(lat_range[0])}-{_fmt_num(lat_range[1])}"


def _cpi_for(details: dict[str, Any], iclass: str | None = None) -> list[str]:
    return perf._cpi_values(details, iclass)


def aggregate_perf(
    record: InstructionRecord,
    *,
    mode: str = "avg",
    include_modeled: bool = False,
) -> PerfSummary:
    """Aggregate latency and CPI across ``record``'s measured microarches."""
    from simdref.perf_sources.merge import field_kind

    arch_details = record.arch_details or {}
    iclass = perf.record_iclass(record)

    def collect(kinds: tuple[str, ...]) -> tuple[list[float], list[float], list[str]]:
        lats: list[float] = []
        cpis: list[float] = []
        archs: list[str] = []
        for core, details in arch_details.items():
            lat = _per_arch_value(details, _latency_for)
            cpi = _per_arch_value(details, lambda d: _cpi_for(d, iclass))
            # Each value joins the bucket of its own source: a borrowed
            # latency counts as its donor's kind, never as the owner's.
            lat_ok = lat is not None and field_kind(details, "latencies") in kinds
            cpi_ok = cpi is not None and field_kind(details, "cpi") in kinds
            if not lat_ok and not cpi_ok:
                continue
            archs.append(core)
            if lat_ok:
                lats.append(lat)
            if cpi_ok:
                cpis.append(cpi)
        return lats, cpis, archs

    lats, cpis, archs = collect(("measured",))
    source_kind = "measured"
    if not archs and include_modeled:
        lats, cpis, archs = collect(("modeled",))
        source_kind = "modeled"
    if not archs:
        # Last-ditch: anything at all.
        lats, cpis, archs = collect(("measured", "modeled"))
        source_kind = "mixed"

    def reduce(values: list[float]) -> float | None:
        if not values:
            return None
        if mode == "avg":
            return statistics.fmean(values)
        if mode == "median":
            return statistics.median(values)
        if mode == "best":
            return min(values)
        if mode == "worst":
            return max(values)
        raise ValueError(f"unknown --agg mode: {mode}")

    return PerfSummary(
        latency=reduce(lats),
        cpi=reduce(cpis),
        n_archs=len(archs),
        source_kind=source_kind,
        archs_used=archs,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt_num(x: float | None) -> str:
    if x is None:
        return "-"
    if float(x).is_integer():
        return f"{x:.1f}"
    return f"{x:.2f}"


def arch_perf(record: InstructionRecord, arch: str) -> tuple[float | None, float | None, str]:
    """Return ``(latency, cpi, source_kind)`` for *arch* on *record*.

    Missing values are ``None``. ``source_kind`` labels the values with
    their own sources: each field's kind follows its ``<field>_source``
    donor when it has one, and the label names the owner plus every
    borrowed field (``"measured (owner uops.info, latency from
    instlatx64)"``). Rows that predate the provenance fields fall back to
    ``"measured"``.
    """
    from simdref.perf_sources.merge import slot_kind_label

    details = (record.arch_details or {}).get(arch) or {}
    iclass = perf.record_iclass(record)
    lat = _per_arch_value(details, _latency_for)
    cpi = _per_arch_value(details, lambda d: _cpi_for(d, iclass))
    return lat, cpi, slot_kind_label(details)


def arch_perf_tag(arch: str, lat: float | None, cpi: float | None, kind: str) -> str:
    """Provenance label for a perf row pinned to *arch*.

    ``[missing:<arch>]`` when the catalog holds no latency and no CPI for
    *arch*, so an empty row never reads as "measured to cost nothing".
    """
    if lat is None and cpi is None:
        return f"[missing:{arch}]"
    return f"[{arch}, {kind}]"


def _ports_for(details: dict[str, Any], iclass: str | None = None) -> str | None:
    # A harness-wide measurement loses its ports together with its CPI.
    if perf._measurement_is_harness_contaminated(iclass):
        return None
    measurement = details.get("measurement") or {}
    ports = measurement.get("ports") or measurement.get("TP_ports")
    if ports is None or ports == "-":
        return None
    return str(ports)


def _uops_for(
    details: dict[str, Any], arch: str, iclass: str | None = None
) -> tuple[int | None, bool | None]:
    """Return ``(uops, microcoded)`` for one core's measurement.

    Intel rows count microcode-sequencer uops in ``uops_MS``; a positive
    value marks the form microcoded. AMD rows carry no ``uops_MS``; their
    ``uops`` count is macro-ops, and AMD 57647 p.14 states "Instructions
    are marked as fastpath single (one macro op), fastpath double (two
    macro ops), or microcode (greater than two macro ops).", so over two
    macro-ops marks the form microcoded. Rows without the data omit the
    flag so the output never guesses. A harness-contaminated row loses
    both, the same as its ports.
    """
    if perf._measurement_is_harness_contaminated(iclass):
        return None, None

    def _int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    measurement = details.get("measurement") or {}
    uops = _int(measurement.get("uops"))
    if arch.upper().startswith("ZEN"):
        # AMD 57647 p.14: "Instructions are marked as fastpath single (one macro op), fastpath double (two macro ops), or microcode (greater than two macro ops)."
        return uops, (None if uops is None else uops > 2)
    ms = _int(measurement.get("uops_MS"))
    return uops, (None if ms is None else ms > 0)


def collect_ports(
    record: InstructionRecord,
    *,
    arch: str | None,
    archs_used: list[str] | None,
) -> list[str] | None:
    """Gather distinct port-pressure strings for the contributing arches."""
    arch_details = record.arch_details or {}
    if arch is not None:
        keys: list[str] = [arch] if arch in arch_details else []
    elif archs_used:
        keys = archs_used
    else:
        keys = list(arch_details.keys())
    ports: list[str] = []
    iclass = perf.record_iclass(record)
    for core in keys:
        value = _ports_for(arch_details.get(core) or {}, iclass)
        if value and value not in ports:
            ports.append(value)
    return ports or None


def _summary_bits(summary: str) -> str | None:
    """The digits of the first "NNN - bit" phrase ("32-bit" -> "32"), else None.

    shortcut: str.lower folds no Turkish dotted i in "bit", upgrade if an
    SDM summary ever carries one
    """
    for j in range(len(summary) - 2):
        if summary[j : j + 3].lower() != "bit":
            continue
        k = j
        while k > 0 and summary[k - 1].isspace():
            k -= 1
        if k > 0 and summary[k - 1] == "-":
            k -= 1
            while k > 0 and summary[k - 1].isspace():
                k -= 1
        end = k
        while k > 0 and summary[k - 1].isdecimal():
            k -= 1
        if k < end:
            return summary[k:end]
    return None


def _form_widths(key: str) -> list[str]:
    """The digit groups of the ``M64``-style width tokens of one key."""
    out: list[str] = []
    i = 0
    n = len(key)
    while i < n:
        if key[i] in "MRI" and i + 1 < n and key[i + 1].isdecimal():
            j = i + 1
            while j < n and key[j].isdecimal():
                j += 1
            out.append(key[i + 1 : j])
            i = j
        else:
            i += 1
    return out


def _summary_matches_form(summary: str, record: InstructionRecord) -> bool:
    """Drop obviously mislabeled SDM summaries (e.g. the generic MOV
    blurb "Move 32-bit integer operands." attached to MOV (M64, R64)).

    We only flag a summary as wrong when it declares an explicit bit-width
    that appears nowhere among the form's operand widths."""
    declared = _summary_bits(summary or "")
    if declared is None:
        return True
    widths: list[str] = []
    for op in record.operand_details or []:
        w = op.get("width")
        if w:
            widths.append(str(w))
    if not widths:
        key = str(getattr(record, "key", "") or "")
        widths = _form_widths(key)
    if not widths:
        return True
    return declared in widths


def format_annotation(
    record: InstructionRecord,
    *,
    performance: bool,
    docs: bool,
    arch: str | None,
    agg: str,
    include_modeled: bool,
    perf_summary: PerfSummary | None = None,
) -> str:
    """Compose the comment fragment (without the leading ``# `` marker)."""
    parts: list[str] = []
    if docs and record.summary and _summary_matches_form(record.summary, record):
        parts.append(record.summary.strip())
    if performance:
        uops_frag = ""
        if arch is not None:
            lat, cpi, kind = arch_perf(record, arch)
            lat = arch_latency(record, arch)
            lat_text = fmt_arch_latency(record, arch, lat)
            tag = arch_perf_tag(arch, lat, cpi, kind)
            uops_val, microcoded_val = _uops_for(
                (record.arch_details or {}).get(arch) or {}, arch, perf.record_iclass(record)
            )
            if uops_val is not None:
                uops_frag = f" uops={uops_val}"
                if microcoded_val:
                    uops_frag += " microcoded"
        else:
            summary = (
                perf_summary
                if perf_summary is not None
                else aggregate_perf(record, mode=agg, include_modeled=include_modeled)
            )
            lat, cpi = summary.latency, summary.cpi
            if summary.n_archs == 0:
                tag = "[no data]"
            elif arch is None and agg == "avg":
                tag = f"[avg of {summary.n_archs} archs, {summary.source_kind}]"
            else:
                tag = f"[{agg} of {summary.n_archs} archs, {summary.source_kind}]"
            lat_text = _fmt_num(lat)
        perf_frag = f"lat={lat_text}c cpi={_fmt_num(cpi)}{uops_frag} {tag}"
        parts.append(perf_frag)
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Options & streaming
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AnnotateOptions:
    performance: bool = True
    docs: bool = True
    arch: str | None = None
    isa: str = "auto"  # "x86" | "arm" | "riscv" | "auto"
    agg: str = "avg"
    include_modeled: bool = False
    block: bool = False  # otherwise inline
    unknown: str = "mark"  # "keep" | "drop" | "mark"
    fmt: str = "sa"  # "sa" | "md" | "json"
    track_positions: bool = False


def _annotate_instruction(
    parsed: AsmLine,
    opts: AnnotateOptions,
    conn: sqlite3.Connection,
    lookup_cache: dict[tuple[str, str | None], list[InstructionRecord]] | None = None,
    stats: dict[str, int] | None = None,
    identity_key: str | None = None,
    line_isa: str | None = None,
    line_number: int | None = None,
    identity_cache: dict[str, tuple[InstructionRecord, PerfSummary | None]] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Return the rendered output line and an optional JSON record.

    ``line_isa`` names the family the line's own evidence shows (a pinned
    ``--isa`` counts as one). None marks a neutral line in auto mode: the
    stream family answers first, then the family that holds rows.
    """
    if stats is not None:
        stats["parsed"] = stats.get("parsed", 0) + 1
    token = parsed.mnemonic.lower()
    if not parsed.operands and (token in _LONE_PREFIX_TOKENS or _is_lone_rex(token)):
        # A lone x86 prefix assembles to its prefix byte and its row is the
        # prefix record: never a neighbour form and never a silent drop.
        summary = f"Instruction prefix ({token})."
        prefix_rec: dict[str, Any] = {
            "mnemonic": parsed.mnemonic,
            "known": True,
            "summary": summary,
            "annotation": summary,
        }
        if parsed.address is not None:
            prefix_rec["address"] = f"0x{parsed.address:x}"
        if parsed.source_file:
            prefix_rec["source_file"] = parsed.source_file
        if parsed.source_line is not None:
            prefix_rec["source_line"] = parsed.source_line
        if stats is not None:
            stats["recognized"] = stats.get("recognized", 0) + 1
        if parsed.trailing_comment:
            return parsed.raw, prefix_rec
        return f"{parsed.raw}   # {summary}", prefix_rec
    if identity_key == NO_CATALOG_ROW:
        # Byte identity identified the form and the catalog has no row for
        # it (issue #49: uops.info carries no AVX reg-reg VMOVAPD/VMOVUPS):
        # report the miss by name, never guess a neighbour form below.
        if stats is not None:
            stats["no_row"] = stats.get("no_row", 0) + 1
        no_row: dict[str, Any] = {
            "mnemonic": parsed.mnemonic,
            "known": False,
            "no_measured_row": True,
        }
        if parsed.address is not None:
            no_row["address"] = f"0x{parsed.address:x}"
        if parsed.source_file:
            no_row["source_file"] = parsed.source_file
        if parsed.source_line is not None:
            no_row["source_line"] = parsed.source_line
        no_row["line"] = line_number
        no_row["source"] = parsed.raw
        return f"{parsed.raw}   # {parsed.mnemonic}: no measured row", no_row
    # Byte identity wins over the form pick: the record is what the line
    # assembles to, and an empty row keeps the old pick unchanged. The perf
    # input repeats a few hundred keys over thousands of lines, so the load
    # and the aggregate are cached per key within one stream.
    record: InstructionRecord | None = None
    summary: PerfSummary | None = None
    cached = (
        identity_cache.get(identity_key) if (identity_key and identity_cache is not None) else None
    )
    if cached is not None:
        record, summary = cached
    else:
        record = load_instruction_from_db(conn, identity_key) if identity_key else None
    if record is not None:
        if stats is not None:
            stats["recognized"] = stats.get("recognized", 0) + 1
    else:
        isa = line_isa or opts.isa
        pick_isa = isa
        if lookup_cache is not None:
            records = lookup_cache.get((parsed.mnemonic, isa))
            if records is None:
                # Module-level call (not a captured reference) so tests can
                # keep monkeypatching ``annotate.lookup`` with a fake.
                records = lookup(parsed.mnemonic, conn, isa=isa)
                lookup_cache[(parsed.mnemonic, isa)] = records
        else:
            records = lookup(parsed.mnemonic, conn, isa=isa)
        if not records and line_isa is None:
            # A neutral line whose stream family holds no rows gets the rows
            # of the family that has them (issue #57 batch: jnle, bare b).
            pick_isa = None
            records = lookup(parsed.mnemonic, conn, isa=None)
            if lookup_cache is not None:
                lookup_cache[(parsed.mnemonic, None)] = records
        if stats is not None and records:
            stats["recognized"] = stats.get("recognized", 0) + 1
        record = pick_record(records, arch=opts.arch, operands=parsed.operands, isa=pick_isa)

    if record is None:
        if opts.unknown == "drop":
            return parsed.raw, None
        if opts.unknown == "mark":
            marker = "# ??"
            unknown_rec: dict[str, Any] = {
                "mnemonic": parsed.mnemonic,
                "known": False,
            }
            if parsed.address is not None:
                unknown_rec["address"] = f"0x{parsed.address:x}"
            if parsed.source_file:
                unknown_rec["source_file"] = parsed.source_file
            if parsed.source_line is not None:
                unknown_rec["source_line"] = parsed.source_line
            unknown_rec["line"] = line_number
            unknown_rec["source"] = parsed.raw
            return f"{parsed.raw}   {marker}", unknown_rec
        return parsed.raw, None

    if not (opts.performance or opts.docs):
        return parsed.raw, None

    # One aggregate per record: format_annotation and the JSON fields below
    # read the same summary (perf gate: this ran twice per line).
    if summary is None:
        summary = (
            aggregate_perf(record, mode=opts.agg, include_modeled=opts.include_modeled)
            if opts.arch is None
            else None
        )
        if identity_key and identity_cache is not None and record is not None:
            identity_cache[identity_key] = (record, summary)
    annotation = format_annotation(
        record,
        performance=opts.performance,
        docs=opts.docs,
        arch=opts.arch,
        agg=opts.agg,
        include_modeled=opts.include_modeled,
        perf_summary=summary,
    )
    if opts.arch is not None:
        lat_val, cpi_val, _ = arch_perf(record, opts.arch)
        lat_val = arch_latency(record, opts.arch)
        archs_used: list[str] = [opts.arch] if (opts.arch in (record.arch_details or {})) else []
    else:
        lat_val, cpi_val = summary.latency, summary.cpi
        archs_used = list(summary.archs_used)
    ports_val = collect_ports(record, arch=opts.arch, archs_used=archs_used)

    json_record: dict[str, Any] = {
        "mnemonic": parsed.mnemonic,
        "known": True,
        "summary": record.summary,
        "annotation": annotation,
        "latency": lat_val,
        "cpi": cpi_val,
        "ports": ports_val,
    }
    if opts.arch is not None:
        lat_range = arch_latency_range(record, opts.arch)
        if lat_range is not None:
            # "latency" keeps the max; the stored min-max range rides beside it.
            json_record["latency_min"], json_record["latency_max"] = lat_range
        # A single core is pinned, so its uop count is well defined. In the
        # aggregate case the count differs per core and both fields stay out.
        uops_val, microcoded_val = _uops_for(
            (record.arch_details or {}).get(opts.arch) or {},
            opts.arch,
            perf.record_iclass(record),
        )
        if uops_val is not None:
            json_record["uops"] = uops_val
        if microcoded_val is not None:
            json_record["microcoded"] = microcoded_val
    if parsed.address is not None:
        json_record["address"] = f"0x{parsed.address:x}"
    if parsed.source_file:
        json_record["source_file"] = parsed.source_file
    if parsed.source_line is not None:
        json_record["source_line"] = parsed.source_line
    json_record["line"] = line_number
    json_record["source"] = parsed.raw
    json_record["architecture"] = record.architecture
    json_record["isa"] = record.isa

    if not annotation:
        # An identified row keeps its record without text; the line stays
        # bare, no dangling "# " marker.
        return parsed.raw, json_record
    if opts.block:
        block_line = f"{parsed.indent}# {annotation}"
        return f"{block_line}\n{parsed.raw}", json_record

    # Inline: append after the raw line, respecting any pre-existing comment.
    if parsed.trailing_comment:
        return parsed.raw, json_record
    return f"{parsed.raw}   # {annotation}", json_record


def annotate_stream(
    lines: Iterable[str],
    *,
    opts: AnnotateOptions,
    conn: sqlite3.Connection,
    stats: dict[str, int] | None = None,
) -> Iterator[str]:
    """Yield annotated lines for each input line (newline-terminated).

    ``stats`` (optional) accumulates parse/recognition counters — keys
    ``parsed``, ``recognized``, ``content`` — so callers can warn when the
    input format was not understood (e.g. 0 of N instruction lines found).
    """
    json_records: list[dict[str, Any]] = []
    collecting_json = opts.fmt == "json"
    pending_src_file: str | None = None
    pending_src_line: int | None = None
    lookup_cache: dict[tuple[str, str | None], list[InstructionRecord]] = {}
    identity_cache: dict[str, tuple[InstructionRecord, PerfSummary | None]] = {}
    arch_out: list[str] = []

    stripped = [line.rstrip("\n") for line in lines]
    records = _listing_records(stripped)
    parsed_lines = [
        _asm_line(line, records.get(index), opts.track_positions)
        for index, line in enumerate(stripped)
    ]
    # Byte identity per instruction line. llvm-mc sees the raw instruction
    # text after the objdump address and byte columns and the <sym> suffix
    # are stripped; the trailing comment stays so each assembler applies its
    # own comment rule (an AArch64 "#2" immediate is not a comment).
    items: list[tuple[int, str]] = []
    byte_columns: list[str | None] = []
    for index, parsed in enumerate(parsed_lines):
        if parsed.kind != LineKind.INSTRUCTION:
            continue
        obj = records.get(index)
        text = obj[2] if obj is not None else parsed.raw
        items.append((index, _drop_symbol_target(text).strip()))
        byte_columns.append(obj[3] if obj is not None else None)
    identity = {
        index: key
        for (index, _text), key in zip(
            items,
            identity_keys(
                [text for _index, text in items],
                None,
                conn,
                arch_out=arch_out,
                byte_columns=byte_columns,
            ),
        )
        if key
    }
    auto_isa = opts.isa not in ("x86", "arm", "riscv")
    line_family: dict[int, str] = {}
    if auto_isa:
        # Assembler acceptance decides each line's family: one llvm-mc run
        # per candidate triple. The register vote below is the fallback when
        # llvm-mc is missing or a line no triple accepts.
        probe_texts = [text for _index, text in items]
        families, majority = isa_by_acceptance(probe_texts)
        line_family = {index: family for (index, _text), family in zip(items, families) if family}
        # auto: the acceptance majority names the ISA, then the input
        # signals, then the document arch from _document_arch; x86 stays
        # the guess when all stay silent.
        opts.isa = (
            majority
            or detect_isa(parsed.raw for parsed in parsed_lines)
            or (arch_out[0] if arch_out else None)
            or "x86"
        )

    for index, parsed in enumerate(parsed_lines):
        if opts.track_positions:
            if parsed.kind == LineKind.INSTRUCTION:
                parsed.source_file = pending_src_file
                parsed.source_line = pending_src_line
            else:
                # `-S` source interleave: objdump injects "file:line" marker
                # lines immediately before the block they describe. Only
                # non-instruction lines reach the (pricier) marker scan.
                src = _objdump_source_marker(parsed.raw)
                if src:
                    pending_src_file, pending_src_line = src
                    if not collecting_json:
                        yield parsed.raw + "\n"
                    continue
        if stats is not None and parsed.kind != LineKind.BLANK:
            # "content" = anything that is not blank, an assembler directive,
            # or a real comment — i.e. instructions, labels, and unrecognised
            # lines the parser rejected (so "recognised 0/N" can warn).
            s = parsed.raw.lstrip()
            if s and not s.startswith((".", "#", "//")):
                stats["content"] = stats.get("content", 0) + 1
        if parsed.kind != LineKind.INSTRUCTION:
            if not collecting_json:
                yield parsed.raw + "\n"
            continue
        out_line, record = _annotate_instruction(
            parsed,
            opts,
            conn,
            lookup_cache,
            stats,
            identity_key=identity.get(index),
            # A pin forces one family; auto mode splits the batch per line,
            # the assembler verdict first and the register vote as fallback.
            line_isa=opts.isa
            if not auto_isa
            else line_family.get(index) or _line_isa(f"{parsed.mnemonic} {parsed.operands}"),
            line_number=index + 1,
            identity_cache=identity_cache,
        )
        if collecting_json:
            if record is not None:
                json_records.append(record)
            continue
        yield out_line + "\n"

    if collecting_json:
        yield json.dumps(json_records, indent=2) + "\n"
