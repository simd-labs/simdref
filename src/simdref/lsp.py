from __future__ import annotations

import functools
import json
import os
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from simdref import asmline, catkey
from simdref.display import instruction_markdown, web_deep_link
from simdref.operands import (
    INTEL_SIZE_BITS,
    brace_members,
    first_member,
    intel_size_bits,
    is_int_literal,
    strip_braces,
)
from simdref.perf import best_cpi, best_latency, record_iclass
from simdref.ptx import pick_form as _ptx_pick_form
from simdref.ptx import pick_spelling as _ptx_pick_spelling
from simdref.queries import linked_instruction_records
from simdref.regtable import reg_has, reg_rows, reg_width
from simdref.search import search_records
from simdref.storage import (
    SQLITE_PATH,
    load_intrinsic_from_db,
    load_instruction_from_db,
    load_instructions_by_arch_from_db,
    load_instructions_by_mnemonic_from_db,
    open_db,
    search_intrinsic_candidates_from_db,
    search_instruction_candidates_from_db,
    sqlite_schema_is_current,
)


C_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx", ".c++", ".h", ".hpp", ".hh", ".hxx", ".cu", ".cuh")
PTX_ARCH = frozenset({"ptx"})
CATALOG_MISSING = "simdref catalog not found. Run: isa update, then restart the editor."


@dataclass
class Session:
    documents: dict[str, str]
    languages: dict[str, str] = field(default_factory=dict)


def _jsonrpc_write(payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _jsonrpc_read() -> dict | None:
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode("ascii").split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body)


def _word_at(text: str, line: int, character: int) -> str | None:
    lines = text.splitlines()
    if line >= len(lines):
        return None
    current = lines[line]
    return asmline.word_at(current, _code_point_index(current, character))


def _code_point_index(line: str, character: int) -> int:
    # LSP columns count UTF-16 code units; Python indexes code points.
    units, index = 0, 0
    while index < len(line) and units < character:
        units += 2 if ord(line[index]) > 0xFFFF else 1
        index += 1
    return index


def _utf16_column(line: str, index: int) -> int:
    # The reverse of _code_point_index: a code-point index to a UTF-16 column.
    return sum(2 if ord(c) > 0xFFFF else 1 for c in line[:index])


def _line_prefix(text: str, line: int, character: int) -> str:
    lines = text.splitlines()
    if line >= len(lines):
        return ""
    current = lines[line][: _code_point_index(lines[line], character)]
    return asmline.word_at(current, len(current)) or ""


_ATT_SUFFIX_WIDTH = {"b": 8, "w": 16, "l": 32, "q": 64}
# GAS string-move names the catalog holds under Intel names. The AT&T
# zero/sign-extend strip must not resolve movsbl to the string move movsb.
_STRING_OPS = frozenset({"movs", "cmps", "scas", "lods", "stos", "ins", "outs"})


def _is_string_op(stem: str) -> bool:
    """True for a string-move spelling: a name plus one width letter."""
    return (
        len(stem) >= 2
        and stem[-1:].casefold() in ("b", "w", "d", "q")
        and stem[:-1].casefold() in _STRING_OPS
    )


def _za_tile_element(name: str) -> bool:
    """True for one ZA tile element ("za0h.b[w12, 0]"), never a range."""
    text = name.casefold()
    if not text.startswith("za"):
        return False
    end = 2
    while text[end : end + 1].isdecimal():
        end += 1
    if end == 2:
        return False
    index = end + (1 if text[end : end + 1] in ("v", "h") else 0)
    if text[index : index + 2] not in (".b", ".h", ".s", ".d"):
        return False
    index += 2
    if text[index : index + 1] != "[":
        return False
    index = _skip_space(text, index + 1)
    if text[index : index + 1] != "w":
        return False
    end = index + 1
    while text[end : end + 1].isdecimal():
        end += 1
    if end == index + 1:
        return False
    index = _skip_space(text, end)
    if text[index : index + 1] != ",":
        return False
    index = _skip_space(text, index + 1)
    if text[index : index + 1] == "#":
        index += 1
    end = index
    while text[end : end + 1].isdecimal():
        end += 1
    close = _skip_space(text, end)
    return end > index and text[close : close + 1] == "]" and close + 1 == len(text)


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


# Register spellings come from the generated regtable: names, classes,
# widths and families from the pinned LLVM's register data. The edge-case
# rules stay in code: sp/bp/si/di are x86 only with a '%' prefix (the
# callers below), x86 number registers stop at r8-r15 (r0-r7 are ARM
# registers, r16-r31 are APX names this matcher does not class), and rip is
# an address operand, not a general register.


def _x86_gpr_width(name: str) -> int | None:
    """Width of an x86 general-register spelling, or None."""
    tail = name[1:]
    if name.startswith("r") and tail[:1].isdigit():
        number = _reg_number(tail, 15)
        if not number or int(number) < 8 or tail[len(number) :] not in ("", "d", "w", "b"):
            return None
    if name == "rip":
        return None
    return reg_width(name, "gpr", "x86")


def _x86_vector(name: str) -> bool:
    """True for an x86 vector, mask or MMX spelling (xmm/ymm/zmm/k/mm)."""
    return any(
        klass in ("vec", "k", "mm") and family == "x86" for klass, _width, family in reg_rows(name)
    )


def _arrangement_tail(tail: str) -> tuple[str | None, str]:
    """The element letter and lane index of an arrangement tail.

    "4h" -> ("h", ""), "d[1]" -> ("d", "1"). The parse takes the prefix, so
    trailing text after the letter or lane is the caller's business.
    """
    index = 0
    while tail[index : index + 1].isdigit():
        index += 1
    letter = tail[index : index + 1]
    if letter not in ("b", "h", "s", "d", "q"):
        return None, ""
    rest = tail[index + 1 :]
    if rest[:1] == "[":
        lane = 1
        while rest[lane : lane + 1].isdigit():
            lane += 1
        if lane > 1 and rest[lane : lane + 1] == "]":
            return letter, rest[1:lane]
    return letter, ""


def _is_arrangement(name: str) -> bool:
    """True for an arranged register spelling ("z0.4h", "q0.16b")."""
    head, dot, tail = name.partition(".")
    element, lane = _arrangement_tail(tail) if dot else (None, "")
    return (
        bool(dot)
        and element is not None
        and not lane
        and (tail[:-1] == "" or tail[:-1].isdigit())
        and any(
            klass in ("fp", "z", "v") and family == "a64"
            for klass, _width, family in reg_rows(head)
        )
    )


def _is_a64_scalar(name: str) -> bool:
    """True for an AArch64 general register (x0-x30, w0-w30, xzr, wzr)."""
    return reg_has(name, "xw", "a64")


def _is_a64_fp(name: str) -> bool:
    """True for an AArch64 scalar FP register (b0-q31)."""
    return reg_has(name, "fp", "a64")


def _is_bare_v(name: str) -> bool:
    """True for a bare vector spelling ("v3"); ARM and RISC-V share it."""
    return any(klass == "v" for klass, _width, _family in reg_rows(name))


def _is_pred(name: str) -> bool:
    """True for an SVE predicate spelling with its qualifier ("p0/z")."""
    base, slash, qual = name.partition("/")
    return (not slash or qual in ("z", "m")) and reg_has(base, "p", "a64")


def _is_png(name: str) -> bool:
    """True for an SME predicate counter with its qualifier ("pn8/z")."""
    base, slash, qual = name.partition("/")
    return (not slash or qual in ("z", "m")) and reg_has(base, "pn", "a64")


def _is_arm32_scalar(name: str) -> bool:
    """True for an ARM32 scalar spelling (r0-r15, ip, pc)."""
    return reg_has(name, "arm", "arm32")


def _is_rv_scalar(name: str) -> bool:
    """True for an RISC-V ABI scalar spelling.

    s0-s11 also name AArch64 scalar FP registers; sp and fp are shared ARM
    spellings and only take the weak votes below.
    """
    return name not in ("sp", "fp") and reg_has(name, "rvabi", "riscv")


def _reg_number(text: str, limit: int) -> str:
    """The register-number prefix of text: canonical 0 to limit, or ''."""
    digits = ""
    while text[len(digits) : len(digits) + 1].isdigit():
        digits += text[len(digits)]
    canonical = digits == "0" or not digits.startswith("0")
    return digits if digits and len(digits) <= 2 and canonical and int(digits) <= limit else ""


def _z_range_numbers(member: str) -> list[int]:
    """The z-register indices of a Z-range member ("z0.b-z3.b").

    Scans like the old end pattern: a spelled z register with an element can
    start anywhere in the member.
    """
    numbers = []
    for index, char in enumerate(member):
        if char == "z":
            tail = member[index + 1 :]
            number = _reg_number(tail, 31)
            if number and tail[len(number) : len(number) + 1] == ".":
                element, _lane = _arrangement_tail(tail[len(number) + 1 :])
                if element:
                    numbers.append(int(number))
    return numbers


def _is_a64_only_operand(name: str) -> bool:
    """True for an operand spelling only AArch64 uses: a vector lane
    ("v1.d[1]"), a predicate with an arrangement ("p0.b", "pn8.b"), or a
    ZA tile operand. Matches a prefix, like the parser's operand scan.
    """
    if name.removeprefix("{").lstrip().startswith("za"):
        return True
    head, dot, tail = name.partition(".")
    number = _reg_number(head[1:], 31) if head[:1] in ("z", "v") else ""
    if dot and number and head[1:] == number:
        element, lane = _arrangement_tail(tail)
        if element and lane:
            return True
    if name[:1] == "p":
        rest = name[1:]
        number = _reg_number(rest.removeprefix("n"), 15)
        if number and rest.removeprefix("n")[len(number) : len(number) + 2] in (
            ".b",
            ".h",
            ".s",
            ".d",
            ".q",
        ):
            return True
    return False


# shortcut: this vote still picks a wrong row for `ptest p0, p1.b` (x86 PTEST),
# `ptrue p0.b` (predicate-counter PTRUE) and `rev p0.b, p0.b` (GPR REV),
# upgrade when the planned llvm-mc lookup replaces this vote.


def _logical_immediate(value: int, bits: int) -> bool:
    """True when value is encodable as an AArch64 logical (bitmask) immediate
    of the given register width: a rotated run of ones, replicated at a
    power-of-two element size.
    """
    mask = (1 << bits) - 1
    value &= mask
    if value in (0, mask):
        return False
    for size in (64, 32, 16, 8, 4, 2):
        if size > bits:
            continue
        smask = (1 << size) - 1
        for run in range(1, size):
            base = smask >> (size - run)
            for rot in range(size):
                pattern = ((base >> rot) | (base << (size - rot))) & smask
                replicated = 0
                for chunk in range(bits // size):
                    replicated |= pattern << (size * chunk)
                if replicated == value:
                    return True
    return False


def _brace_list_class(inner: str) -> str | None:
    """The class of a whole-brace register list, by its shape, or None.

    A ZA tile list ({za0h.b[...]}) matches ZA rows only. A one-register list
    {zN.T} matches single-register rows. A multi-register list {zA.T, zB.T}
    or range {zA.T-zB.T} matches the multi-vector rows, keeping the count of
    registers ("av2" or "av4"). shortcut: a 3-register list keeps the old
    tie ("av"); only the SME2 counts 2 and 4 are cataloged.
    """
    members = [m.strip().casefold() for m in inner.split(",")]
    if members[0].startswith("za"):
        return "za"
    count = len(members)
    if count == 1:
        member = members[0]
        if "-" in member:
            numbers = _z_range_numbers(member)
            if len(numbers) == 2:
                count = numbers[1] - numbers[0] + 1
        elif _is_arrangement(member):
            return _av_class(member)
    return f"av{count}" if count in (2, 4) else "av"


def _av_class(arrangement: str) -> str:
    """The arrangement class of an SVE Z register ("z0.4h" -> "avh").

    The element suffix (b/h/s/d/q) disambiguates Z-register rows that differ
    only in the lane width, like LD1W's Z0.D and Z0.S register lists. V and
    non-Z arrangements stay bare "av", so an Advanced SIMD line never ties a
    same-lane SVE row.
    """
    if not arrangement.casefold().startswith("z"):
        return "av"
    index = arrangement.find(".")
    while index != -1:
        element, _lane = _arrangement_tail(arrangement[index + 1 :])
        if element:
            return f"av{element}"
        index = arrangement.find(".", index + 1)
    return "av"


# Shift/extend keywords that modify the last register operand. As a trailing
# ", <kw> ..." piece they belong to that operand, not to a new one.
_ASM_SHIFT_KEYWORDS = frozenset(
    "lsl lsr asr ror msl uxtb uxth uxtw uxtx sxtb sxth sxtw sxtx".split()
)
_EXTEND_SHIFT_KEYWORDS = frozenset("uxtb uxth uxtw uxtx sxtb sxth sxtw sxtx".split())


def _split_asm_operands(operands: str) -> tuple[list[str], list[str | None]]:
    """Split at top-level commas; fold a trailing shift/extend suffix piece.

    Returns (parts, marks). A piece that starts with a shift or extend
    keyword ("lsl #2", "uxtw") modifies the last register operand and is not
    an operand of its own: "add x0, x1, x2, lsl #2" reads as three operands,
    matching form keys that absorb the suffix into their placeholder. The
    marks keep the suffix kind per part: "e" for an extend (uxtb..sxtx), "s"
    for a shift (lsl/lsr/asr/ror/msl), so the extended-register and
    shifted-register rows stay distinct. shortcut: a label spelled like a
    keyword in operand position ("add x0, x1, lsl") folds away; it is not a
    register.
    """
    if not operands.strip():
        return [], []
    parts: list[str] = []
    marks: list[str | None] = []
    for index, piece in enumerate(asmline.split_line_operands(operands)):
        token = piece.split(None, 1)[0].casefold() if piece else ""
        if index and token in _ASM_SHIFT_KEYWORDS:
            if marks:
                marks[-1] = "e" if token in _EXTEND_SHIFT_KEYWORDS else "s"
            continue
        parts.append(piece)
        marks.append(None)
    return parts, marks


# GAS zero/sign-extend names the catalog holds only under Intel names. Each
# maps to (Intel mnemonic, dest bits, source bits); the converts have no
# operands. The class is the GNU set: binutils 2.47 as + objdump accept and
# print these names, and as rejects movzww/movsww (no 16-to-16 form exists).
# Those two stay in the class and resolve to nothing, as in GNU as.
_ATT_EXTEND_ALIAS = {
    **{
        f"movz{src}{dst}": ("MOVZX", _ATT_SUFFIX_WIDTH[dst], _ATT_SUFFIX_WIDTH[src])
        for src in "bw"
        for dst in "wlq"
    },
    **{
        f"movs{src}{dst}": ("MOVSX", _ATT_SUFFIX_WIDTH[dst], _ATT_SUFFIX_WIDTH[src])
        for src in "bw"
        for dst in "wlq"
    },
    "movslq": ("MOVSXD", 64, 32),
    "cbtw": ("CBW", None, None),
    "cwtl": ("CWDE", None, None),
    "cltq": ("CDQE", None, None),
    "cwtd": ("CWD", None, None),
    "cltd": ("CDQ", None, None),
    "cqto": ("CQO", None, None),
}


def _lookup_mnemonic(conn, mnemonic: str, arch: frozenset = frozenset()):
    """Return (forms, width). The exact mnemonic wins when one of its rows fits
    the line's architecture or the line is undecided, else an AT&T
    zero/sign-extend alias (movzbl -> MOVZX), else a dropped AT&T size suffix
    (addq -> add). On an x86-decided line a foreign exact key never beats the
    suffix strip: "addw %ax, %bx" resolves to x86 "add" even when the catalog
    carries a RISC-V "addw" row. On an undecided line the exact key wins as it
    always did: "mulw sym" keeps its RISC-V row, never x86 MUL.

    PTX line only: a dotted mnemonic with no catalog row uses the longest
    chain prefix that holds ptx rows.

    The alias and the suffix strip run on an x86 or undecided line only: "addw"
    on a RISC-V or ARM32 line must not land on x86 "add". When the strip names
    no x86 rows, the exact rows stay, so "bl target" keeps its ARM row. Rows of
    other architectures stay in the result; _match_form filters them by the
    line's candidates.
    """
    forms = load_instructions_by_mnemonic_from_db(conn, mnemonic)
    width = _ATT_SUFFIX_WIDTH.get(mnemonic[-1:].casefold())
    if not arch:
        # Undecided line (no register vote, no Intel document): the exact key
        # wins as it always did. Only an x86-decided line lets the suffix
        # strip beat a foreign exact key.
        if forms:
            return forms, None
    elif any(form.architecture in arch for form in forms):
        return forms, None
    if "ptx" in arch and "." in mnemonic:
        token = mnemonic
        while True:
            ptx_rows = [
                form
                for form in load_instructions_by_mnemonic_from_db(conn, token)
                if form.architecture == "ptx"
            ]
            if ptx_rows:
                return ptx_rows, None
            dot = token.rfind(".")
            if dot <= 0:
                break
            token = token[:dot]
    if arch and "x86" not in arch:
        return [], None
    alias = _ATT_EXTEND_ALIAS.get(mnemonic.casefold())
    rows = []
    if alias is not None:
        intel, dst_bits, src_bits = alias
        rows = [
            form
            for form in load_instructions_by_mnemonic_from_db(conn, intel)
            if form.architecture == "x86"
        ]
        # The name carries both widths: movzwq is MOVZX (R64, R16) or MOVZX
        # (R64, M16) only, never the M8 rows the Intel spelling can also name.
        if dst_bits is not None:
            rows = [
                form for form in rows if catkey.key_operand_bits(form.key) == (dst_bits, src_bits)
            ]
    elif width and not _is_string_op(mnemonic[:-1]):
        # movsbl is a sign-extend, not the string op movsb.
        rows = [
            form
            for form in load_instructions_by_mnemonic_from_db(conn, mnemonic[:-1])
            if form.architecture == "x86"
        ]
    return (rows, width) if rows else (forms, None)


def _classify_line(
    operands: str, intel: bool
) -> tuple[frozenset, list[str | None], bool, frozenset]:
    """One pass over the line: (candidate buckets, one class per operand, reverse, weak votes).

    A strong vote (a %- or Intel-style register, an x86 vector, an AArch64
    arrangement, an AArch64/ARM32/RISC-V-only scalar) names one bucket. A
    spelling two architectures share ("sp", "fp", s0-s11, a bare v) votes for
    every bucket it can mean and only counts when no strong vote fired. Empty
    candidates mean undecided; callers then keep today's unfiltered behaviour.
    "reverse" is True when the line's operand order flips into key order
    (AT&T x86). A register of a non-candidate architecture reads as a symbol.
    "weak votes" names the candidates that came from shared spellings alone;
    callers gate them against the mnemonic's forms. It is empty when a strong
    vote decided the line: a strong vote stays out of the gate.
    """
    parts, marks = _split_asm_operands(operands)
    strong, weak = set(), set()
    for raw in parts:
        op = raw.strip()
        name = op.lstrip("*").lstrip("%").casefold()
        prefixed = op.lstrip("*").startswith("%")
        reg = _x86_gpr_width(name)
        if op.startswith("$"):
            strong.add("x86")
        elif _x86_vector(name):
            strong.add("x86")
        elif prefixed and reg:
            strong.add("x86")
        elif (bare := brace_members(op)) and (
            _is_arrangement(first_member(bare)) or bare.strip().casefold().startswith("za")
        ):
            # An SVE register list ("{z0.b}"), range or ZA tile list votes AArch64.
            strong.add("arm")
        elif _is_png(name):
            strong.add("arm")
        elif not name or "(" in op or "[" in op or "{" in op:
            continue
        elif _is_arrangement(name) or _is_a64_scalar(name) or name == "wsp":
            strong.add("arm")
        elif _is_arm32_scalar(name):
            strong.add("arm")
        elif _is_a64_fp(name):
            # s0-s11 are also RISC-V saved registers: a weak vote there, and
            # the mnemonic's catalog rows arbitrate between arm and riscv.
            if _is_rv_scalar(name):
                weak.update(("arm", "riscv"))
            else:
                strong.add("arm")
        elif _is_rv_scalar(name):
            strong.add("riscv")
        elif _is_bare_v(name):
            weak.update(("arm", "riscv"))
        elif name in ("sp", "fp"):
            weak.update(("arm", "riscv"))
        elif name == "lr":
            weak.add("arm")
    if intel or "x86" in strong:
        arch, used_weak = frozenset({"x86"}), frozenset()
    elif strong:
        arch, used_weak = frozenset(strong), frozenset()
    else:
        arch, used_weak = frozenset(weak), frozenset(weak)
    classes = [_operand_class(op.strip(), intel, arch) for op in parts]
    classes = [
        f"{cls}{mark}" if mark and cls in ("w", "x") else cls for cls, mark in zip(classes, marks)
    ]
    reverse = not intel and ("x86" in arch or not arch)
    return arch, classes, reverse, used_weak


def _a64_address_class(op: str) -> str:
    """Sub-class of an AArch64 bracket address, distinguishing the index mode.

    "mpre" is pre-index ("[x9, #-32]!"); "mo" holds an immediate offset inside
    the brackets ("[sp, #16]"); "mreg" offsets by a register ("[x9, x10]" or an
    extended form); "m" is a plain base register. The post-index form splits at
    its top-level comma, so its bracket reads as plain "m" and the trailing
    "#imm" is its own immediate operand.
    """
    if op.rstrip().endswith("!"):
        return "mpre"
    if "[" not in op or "]" not in op:
        return "m"
    body = op[op.index("[") + 1 : op.find("]")].replace(" ", "")
    parts = body.split(",")
    if len(parts) > 1:
        return "mo" if parts[1].startswith("#") else "mreg"
    return "m"


def _operand_class(op: str, intel: bool, arch: frozenset) -> str | None:
    """Classify one operand on a line whose candidates are arch, or None.

    Classes: r8..r64, v128/v256/v512/k/mm, av (AArch64 vector with an
    arrangement), x/w/xzr/wzr/sp/wsp (AArch64 scalars; fp and lr read as x),
    b/h/s/d/q (AArch64 scalar FP by width), ar (ARM32), rv (RISC-V), and
    i, lit0/lit1, m, mpre/mo/mreg (AArch64 address index modes), rel. A bare
    symbol is a memory reference in AT&T and a rel target in Intel syntax.
    """
    if not op:
        return None
    name = op.lstrip("*").lstrip("%").casefold()
    digits = op[1:] if op.startswith("$") else op
    if digits and set(digits) == {"0"}:
        return "lit0"
    if digits.endswith("1") and set(digits[:-1]) <= {"0"}:
        return "lit1"
    if op.startswith("$") or is_int_literal(op):
        return "i"
    # AArch64 marks immediates with '#' ("#-16", "#0x10", "#:lo12:sym").
    if op.startswith("#"):
        return "i"
    if "{1to" in op:
        return "mbc"
    if "arm" in arch:
        inner = brace_members(op.strip())
        if inner is not None:
            # The class of a brace list is decided by its shape.
            return _brace_list_class(inner.strip())
        if _is_png(name):
            return "png"
        if _is_pred(name):
            return "pred"
    if "{" in op:
        op = strip_braces(op).strip()
        name = op.lstrip("*").lstrip("%").casefold()
    if "(" in op or "[" in op:
        sized = intel_size_bits(op)
        if sized:
            return f"m{sized}"
        if "[" in op and "arm" in arch:
            return _a64_address_class(op)
        return "m"
    reg = _x86_gpr_width(name)
    # In AT&T a bare "sp" is the ARM stack pointer or a symbol, never the r16.
    if reg and (intel or op.lstrip("*").startswith("%") or name not in ("sp", "bp", "si", "di")):
        if intel or arch.isdisjoint({"arm", "riscv"}):
            return f"r{reg}"
    if _x86_vector(name):
        if name[0] == "k":
            return "k"
        if name[0] == "m":
            return "mm"
        return catkey.X86_VECTOR_WIDTHS[name[:3]]
    if not intel and op.startswith("*"):
        return "m"
    if "arm" in arch:
        if _is_arrangement(name):
            return _av_class(name)
        if _is_bare_v(name):
            return "av"
        if _is_a64_fp(name):
            return name[0]
        if _is_a64_scalar(name):
            return {"xzr": "xzr", "wzr": "wzr"}.get(name, name[0])
        if name in ("sp", "wsp", "fp", "lr"):
            return {"sp": "sp", "wsp": "wsp"}.get(name, "x")
        if _is_arm32_scalar(name):
            return "ar"
    if "riscv" in arch and (_is_rv_scalar(name) or _is_bare_v(name)):
        return "rv"
    if "x86" in arch or not arch:
        return "rel" if intel else "m"
    return None


_CLASS_KIND = dict(
    [(c, "register") for c in ("r8", "r16", "r32", "r64")]
    + [(c, "vector") for c in ("v128", "v256", "v512", "k", "mm", "dfp", "pred", "png")]
    + [(c, "vector") for c in ("av", "avb", "avh", "avs", "avd", "avq", "av2", "av4")]
    + [(c, "memory") for c in ("m", "moffs", "mbc", "mo", "mpre", "mreg")]
    + [(f"m{bits}", "memory") for bits in set(INTEL_SIZE_BITS.values())]
    + [(c, "immediate") for c in ("i", "lit0", "lit1")]
    + [
        (c, "scalar")
        for c in (
            "x",
            "w",
            "xzr",
            "wzr",
            "sp",
            "wsp",
            "b",
            "h",
            "s",
            "d",
            "q",
            "ar",
            "rv",
            "we",
            "xe",
            "ws",
            "xs",
        )
    ]
    + [("rel", "rel")]
)

# The line classes names in _operand_class mint: an "m" digit class per
# size keyword, never a "mo" (memory offset) or "mbc" (broadcast) class.
_MEMORY_CLASSES = frozenset(f"m{bits}" for bits in INTEL_SIZE_BITS.values())


def _form_takes_line(form, classes: list) -> bool:
    """The form accepts the line's operand classes position by position.

    A bare key constrains nothing; neither does a placeholder catkey.form_accept
    leaves empty. Rejecting on an empty set hid the AArch64 scalar saturating
    family (<V_OPTION><D> maps to no class). An operand the classifier gave no
    class (None) constrains nothing either.
    """
    ops = catkey.operands_of_key(form.key)
    if not ops:
        return True
    if len(ops) != len(classes):
        return False
    for cls, op in zip(classes, ops):
        accept = catkey.form_accept(op)
        if cls is not None and accept and cls not in accept:
            return False
    return True


def _form_takes_line_lenient(form, classes: list) -> bool:
    """Alias-judged acceptance: every per-position accept touches the line.

    An alias inserts an implicit operand (`cmp x, y` reads `subs zr, x, y`),
    so a shared line's classes need not land on the exact alias row; the row
    admits the line when each of its operands accepts some class the line
    carries. A bare key still constrains nothing.
    """
    ops = catkey.operands_of_key(form.key)
    if not ops or len(ops) != len(classes):
        return False
    have = {cls for cls in classes if cls}
    for cls, op in zip(classes, ops):
        accept = catkey.form_accept(op)
        if cls is not None and accept and not (accept & have):
            return False
    return True


def _gate_weak_votes(matches, line: tuple, operands: str, intel: bool, transform=None):
    """Keep a weak-voted candidate only when its rows accept the line, or None.

    A register spelling two architectures share admits its other architecture
    only when one of the mnemonic's forms there accepts the line's operands
    position by position. A line made only of shared names shows nothing when
    no candidate accepts it: None vetoes the line, and callers return no entry
    instead of the unfiltered pool. Strong votes never reach this gate.
    The returned line tuple carries classes for the pruned candidates.
    TRANSFORM is the alias class insert for a mov/cmp/cmn/tst line: the gate
    then judges the alias mnemonic's rows, strictly first and with each
    operand's accept set intersected with the line's classes second.
    """
    arch, classes, reverse, weak = line
    if not weak:
        return line
    by_arch = {}
    for form in matches:
        by_arch.setdefault(form.architecture, []).append(form)
    parts, _marks = _split_asm_operands(operands)

    def takes(arch_name: str) -> bool:
        reading = [_operand_class(piece, intel, frozenset({arch_name})) for piece in parts]
        if transform is not None:
            transformed = transform(reading)
            forms = by_arch.get(arch_name, [])
            return any(_form_takes_line(form, transformed) for form in forms) or any(
                _form_takes_line_lenient(form, transformed) for form in forms
            )
        return any(_form_takes_line(form, reading) for form in by_arch.get(arch_name, []))

    kept = {a for a in arch if a not in weak}
    kept.update(a for a in weak & arch if takes(a))
    if not kept:
        return None
    pruned = frozenset(kept)
    if pruned == arch:
        return line
    return (
        pruned,
        [_operand_class(piece, intel, pruned) for piece in parts],
        not intel and ("x86" in pruned or not pruned),
        weak,
    )


# shortcut: the form matcher answers only when byte identity has no row
# (schema-13 catalogs or missing tools); remove it at the schema-14 data
# release with simdref-llvm-mc on PyPI as a required dependency
def _match_form(
    matches,
    operand_count: int | None,
    line: tuple,
    suffix_bits: int | None,
):
    """Pick the form the line names, under one total order.

    line is _classify_line's (arch candidates, ordered classes, reverse, weak
    votes). Rows of a non-candidate architecture drop out before ranking; an
    undecided line keeps every row. Rank: Intel size-keyword match, class-set
    score (accepted +1 / same kind 0 / else -1, a literal scores only on its
    own value), position score, iN over a literal 0/1, match with the AT&T
    suffix width, fewer operands, key name.
    The no-same-count fallback uses the same key.
    """
    arch, classes, reverse = line[0], line[1], line[2]
    if arch:
        matches = [m for m in matches if m.architecture in arch]
    # A size keyword may only add information: a sized memory class no row of
    # the mnemonic accepts ranks as a plain "m", so the pick equals the line
    # with the keyword removed (VMOVDQU has no M512 row, and "vmovdqu zmm0,
    # zmmword ptr [rdx]" keeps the unsized load form). The keyword reward in
    # _rank still prefers the form that names the width exactly.
    if classes and any(c in _MEMORY_CLASSES for c in classes):
        accepted_memory = {
            accept
            for form in matches
            for op in catkey.operands_of_key(form.key)
            for accept in catkey.form_accept(op)
        }
        classes = ["m" if c in _MEMORY_CLASSES and c not in accepted_memory else c for c in classes]
    # The line's classified parts are authoritative: a shift/extend suffix
    # folded into its register operand, so the parsed comma count can lag.
    count = len(classes) if classes else operand_count
    if count is None:
        return matches[0] if matches else None
    same = [m for m in matches if len(catkey.operands_of_key(m.key)) == count]
    pool = same or matches
    if not pool:
        return None
    ordered = list(reversed(classes)) if reverse else classes
    line_set = {c for c in classes if c}
    line_kinds = {_CLASS_KIND[c] for c in line_set if c in _CLASS_KIND}

    def _rank(form):
        accepts = [catkey.form_accept(op) for op in catkey.operands_of_key(form.key)]
        set_score = position = in_over_lit = width = keyword = 0
        for acc in accepts:
            if acc in catkey.LITERAL_ACCEPT:
                set_score += 0 if acc & line_set else -1
            elif acc & line_set:
                set_score += 1
            elif {_CLASS_KIND[a] for a in acc if a in _CLASS_KIND} & line_kinds:
                pass
            else:
                set_score -= 1
            if any(
                c in line_set and int(c[1:]) == suffix_bits
                for c in acc
                if c[:1] in ("r", "v") and c[1:].isdecimal()
            ):
                width += 1
            # An Intel size keyword names the memory width.
            if any(c in _MEMORY_CLASSES and c in line_set for c in acc):
                keyword += 1
        for cls, acc in zip(ordered, accepts):
            if cls in acc:
                position += 1
            if cls in ("i", "lit0", "lit1"):
                if acc == catkey.IMMEDIATE_ACCEPT:
                    in_over_lit += 1
                elif acc in catkey.LITERAL_ACCEPT:
                    in_over_lit -= 1
        return (-keyword, -set_score, -position, -in_over_lit, -width, len(accepts), form.key)

    return min(pool, key=_rank)


def _zreg(classes: list) -> str:
    """The width-apportioned zero register for the line's register family."""
    return "wzr" if classes and classes[0] in ("w", "wsp", "wzr") else "xzr"


def _a64_gpr_bits(name: str) -> int | None:
    """The width of a general-register spelling (sp, fp, lr included), or None."""
    bits = {"sp": 64, "fp": 64, "lr": 64, "xzr": 64, "wsp": 32, "wzr": 32}
    return bits.get(name) or ((64, 32)[name.startswith("w")] if _is_a64_scalar(name) else None)


def _a64_alias_for_line(mnemonic: str, arch: frozenset, operands: str):
    """(alias mnemonic, classes transform) for an AArch64 alias line.

    The catalog keys no MOV/CMP/CMN/TST row in arm, so the alias names the
    canonical instruction. Only forms the operand text alone decides are
    aliased: a register mov is ORR with the zero register as source, a mov
    with sp or wsp is ADD immediate #0, an immediate mov is MOVZ, MOVN or
    ORR by its exact encoding, one ZA tile under a predicate is MOVA, and
    cmp/cmn/tst are subs/adds/ands with the zero register as destination.
    Every other form returns ("", None): hover and hint show nothing, never
    a row of another instruction. An x86 or ARM32 line is no alias at all.
    shortcut: alias-target rows missing from the catalog (SUBS immediate)
    match the nearest row of the target mnemonic.
    """
    if "x86" in arch:
        return None
    parts, _marks = _split_asm_operands(operands)
    names = [piece.strip().lstrip("*").casefold() for piece in parts]
    if any(_is_arm32_scalar(name) for name in names):
        return None
    mnemonic = mnemonic.casefold()
    if mnemonic not in ("mov", "cmp", "cmn", "tst"):
        return None
    if "arm" not in arch and not any(_is_a64_only_operand(name) for name in names):
        return None
    if mnemonic == "mov":
        # MOVA: one ZA tile element, a governing predicate, one Z register.
        if (
            len(parts) == 3
            and any(_za_tile_element(name) for name in names)
            and any(_is_pred(name) for name in names)
            and any(_is_arrangement(name) for name in names)
        ):
            return ("mova", lambda classes: classes)
        if len(parts) == 2:
            dst, src = names
            dst_bits, src_bits = _a64_gpr_bits(dst), _a64_gpr_bits(src)
            if dst_bits and src_bits and dst_bits == src_bits:
                if dst in ("sp", "wsp") or src in ("sp", "wsp"):
                    # MOV to or from SP is ADD immediate with #0.
                    return ("add", lambda classes: [*classes, "lit0"])
                # A register move is ORR shifted register: the zero register
                # is the source. shortcut: nearest row when no ORR row takes it.
                return ("orr", lambda classes: [classes[0], _zreg(classes), *classes[1:]])
            if dst_bits and src.startswith("#"):
                # int(text, 0) inside a length cap: a parse failure means the
                # line shows nothing, and a huge immediate cannot crash the
                # server.
                text = src.strip().lstrip("#").strip()
                try:
                    value = int(text, 0) if len(text) <= 20 else None
                except ValueError:
                    value = None
                if value is None:
                    return ("", None)
                mask = (1 << dst_bits) - 1
                value &= mask
                steps = (0, 16, 32, 48)
                if dst not in ("sp", "wsp"):
                    # MOVZ when the value is a shifted halfword, else MOVN
                    # when the inverted value is.
                    if any(value & (mask & ~(0xFFFF << step)) == 0 for step in steps):
                        return ("movz", lambda classes: classes)
                    if any((value ^ mask) & (mask & ~(0xFFFF << step)) == 0 for step in steps):
                        return ("movn", lambda classes: classes)
                # ORR bitmask immediate: the only immediate MOV to SP.
                if _logical_immediate(value, dst_bits):
                    return ("orr", lambda classes: [classes[0], _zreg(classes), *classes[1:]])
        return ("", None)
    alias = {"cmp": "subs", "cmn": "adds", "tst": "ands"}[mnemonic]
    # The catalog holds register and immediate forms of the target only.
    if (
        len(parts) == 2
        and _a64_gpr_bits(names[0])
        and (_a64_gpr_bits(names[1]) or names[1].startswith("#"))
    ):
        return (alias, lambda classes: [_zreg(classes), *classes])
    return ("", None)


# ---------------------------------------------------------------------------
# byte identity: one llvm-mc run per document, shared by every consumer
# ---------------------------------------------------------------------------


def _document_arch(
    texts: tuple[str, ...], intel: bool, syntax: bool
) -> tuple[str, str, tuple[bool, ...]] | None:
    """Name one arch for the llvm-mc run from the strongest operand votes.

    Returns ``(arch, x86 syntax, keep)`` or None when the votes do not name
    one arch. ``keep`` marks the lines whose own strong votes fit that arch
    and the lines with no strong votes (they ride the document's arch); a
    line whose strong votes name another arch falls back to the old matcher.
    shortcut: arm32 and rv32 documents get no identity rows (arm32 rows carry
    no byte keys and rv32 runs on the rv64 base), upgrade when the catalogs
    carry rows that split those bases.
    """
    counts: dict[str, int] = {}
    strongs: list[frozenset] = []
    arm32 = False
    # Repeated lines share one parse and one classification (the 2000-line
    # hint document holds two distinct lines).
    memo: dict[str, tuple[frozenset, bool]] = {}
    for text in texts:
        hit = memo.get(text)
        if hit is None:
            parsed = _mnemonic_from_asm_line(text, intel)
            if parsed is None:
                hit = (frozenset(), False)
            else:
                arch, classes, _reverse, weak = _classify_line(parsed[2], intel)
                strong = frozenset() if weak else arch
                hit = (strong, "arm" in strong and "ar" in classes)
            memo[text] = hit
        strong, line_arm32 = hit
        strongs.append(strong)
        arm32 = arm32 or line_arm32
        for bucket in strong:
            counts[bucket] = counts.get(bucket, 0) + 1
    if arm32 or not counts:
        return None
    top = max(counts.values())
    winners = [bucket for bucket, count in counts.items() if count == top]
    if len(winners) != 1:
        return None
    name = {"x86": "x86", "arm": "aarch64", "riscv": "riscv64"}.get(winners[0])
    if name is None:
        return None
    keep = tuple(winners[0] in strong or not strong for strong in strongs)
    return name, ("intel" if syntax else "att"), keep


# identity_keys slot value when byte identity ran on a schema-14 catalog and
# identified the instruction, but the catalog has no row for that form. Not a
# db_key: callers show the line as having no measured row and never fall back
# to the legacy pick. The legacy pick stays only where byte identity could
# not run (missing tools, a schema-13 catalog, a declined document).
NO_CATALOG_ROW = "?:no-catalog-row"


@functools.lru_cache(maxsize=32)
def _identity_batch(
    texts: tuple[str, ...],
    intel: bool,
    syntax: bool,
    byte_cols: tuple[str | None, ...],
    catalog: str,
    mtime_ns: int,
) -> tuple[str | None, ...]:
    from simdref.asm_identity import ToolUnavailableError, identify_lines

    slots: list[str | None] = [None] * len(texts)
    # The document vote only names the family whose assembler to try for
    # lines no per-line acceptance answer exists for. No line is dropped.
    picked = _document_arch(texts, intel, syntax)
    fallback_family = catkey.ISA_FAMILY.get(picked[0]) if picked is not None else None
    try:
        results = identify_lines(
            list(zip(texts, byte_cols)),
            catalog_path=catalog,
            fallback_family=fallback_family,
        )
    except (ToolUnavailableError, ValueError, OSError, FileNotFoundError, sqlite3.Error):
        # No tool, no keys or no row: the old matcher answers these lines.
        return tuple(slots)
    if len(results) != len(texts):
        return tuple(slots)
    for index, result in enumerate(results):
        # The text check keeps a row on its own line even if a text carries
        # its own line break.
        if result.text != texts[index]:
            continue
        if result.row is not None:
            slots[index] = result.row.db_key
        elif result.reason == "no catalog row" or result.reason.startswith(
            ("no catalog row (iform ", "no catalog row (splits emptied ")
        ):
            # Byte identity identified the form and the catalog lacks its row
            # (e.g. uops.info carries no AVX reg-reg VMOVAPD): mark the miss.
            # A fixup placeholder or a foreign decode mode keeps the line on
            # the old matcher: the row can exist, it is just not byte-proven.
            slots[index] = NO_CATALOG_ROW
    return tuple(slots)


def _mnemonic_has_rows(conn, text: str, semicolon_is_comment: bool) -> bool:
    """Whether any catalog row carries the line's mnemonic.

    A byte-identity miss over a mnemonic the catalog does not hold at all
    (a RISC-V word op in a catalog without the word rows) stays unmarked:
    every legacy lookup is mnemonic-keyed and finds nothing there, so the
    line stays silent instead of reporting a miss by name. The same miss
    over a held mnemonic (issue #49 vmovapd) must stay marked so the line
    never joins a neighbour form.
    """
    parsed = _mnemonic_from_asm_line(text, semicolon_is_comment)
    return (
        bool(parsed)
        and conn.execute(
            "SELECT 1 FROM instructions_data WHERE mnemonic = ? LIMIT 1", (parsed[0],)
        ).fetchone()
        is not None
    )


def _exact_key_beats_identity(conn, text: str, intel: bool, slot: str) -> bool:
    """Whether the line keeps its exact catalog key over the byte-identity row.

    A line with no architecture vote and an exact key keeps that key when the
    bytes prove no row in the key's family (issue #60: "mulw sym" hovers the
    RISC-V key "mulw", never the x86 MUL its AT&T suffix spelling assembles
    to). A decided line keeps its byte row: there the bytes name the form.
    """
    parsed = _mnemonic_from_asm_line(text, intel)
    if not parsed:
        return False
    if _classify_line(parsed[2], intel)[0]:
        return False
    exact = load_instruction_from_db(conn, parsed[0])
    if exact is None:
        return False
    if slot == NO_CATALOG_ROW:
        return True
    return not slot.casefold().startswith(exact.architecture.casefold() + ":")


def identity_keys(
    texts: Sequence[str],
    intel: bool | None,
    conn: sqlite3.Connection,
    *,
    arch_out: list[str] | None = None,
    byte_columns: Sequence[str | None] | None = None,
) -> list[str | None]:
    """Map each text to the catalog db_key its encoded bytes name, or a mark.

    One shared entry for hover, inlay hints, annotate and the llm query: one
    llvm-mc run per family over the batch, decided per line. A None means
    byte identity could not run for the line (tools missing, a schema-13
    catalog, or a line the candidate assemblers reject) and the caller's old
    matcher answers that line unchanged. NO_CATALOG_ROW means byte identity
    identified the instruction and the catalog has no row for that form: the
    caller reports no measured row and never guesses a neighbour form. The
    mark is dropped when the catalog holds no row at all for the line's
    mnemonic; ``intel`` is the file-type syntax flag; None infers the x86
    syntax from the text, and keeps the operand votes neutral so a bare
    AArch64 or RISC-V line never reads as forced x86. ``byte_columns`` holds
    the objdump byte column hex per line, or None. ``arch_out`` receives the
    catalog architecture that _document_arch infers for the document, so the
    fallback matcher can keep rows of that architecture even without an
    --isa pin.
    """
    if not texts:
        return []
    items = tuple(texts)
    if intel is None:
        joined = "\n".join(items)
        intel = "%" not in joined and "$" not in joined
        vote = False
    else:
        vote = bool(intel)
    if arch_out is not None:
        # The doc arch feeds the fallback matcher when byte identity cannot run.
        picked = _document_arch(items, vote, bool(intel))
        if picked is not None:
            arch_out.append(catkey.ISA_FAMILY.get(picked[0], picked[0]))
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
    except (sqlite3.Error, AttributeError):
        row = None
    # A connection that cannot name its catalog gets no identity rows: a key
    # from another database would be loaded against this connection.
    if not row or not row[2]:
        return [None] * len(items)
    catalog = row[2]
    try:
        mtime_ns = Path(catalog).stat().st_mtime_ns
    except OSError:
        return [None] * len(items)
    byte_cols = tuple(byte_columns) if byte_columns is not None else (None,) * len(items)
    if len(byte_cols) != len(items):
        byte_cols = (None,) * len(items)
    slots = list(_identity_batch(items, vote, bool(intel), byte_cols, catalog, mtime_ns))
    out: list[str | None] = []
    # The gate is deterministic per (text, slot): a repeated line reuses the
    # first answer instead of re-running the matcher and the catalog read
    # per line (the 2000-line hint request holds two distinct lines).
    decided: dict[tuple[str, str | None], str | None] = {}
    for text, slot in zip(items, slots):
        key = (text, slot)
        if key not in decided:
            if slot == NO_CATALOG_ROW and not _mnemonic_has_rows(conn, text, bool(intel)):
                decided[key] = None
            elif slot and _exact_key_beats_identity(conn, text, bool(intel), slot):
                # The exact key is the only name the line has: the old matcher
                # answers it and hover and hint agree on the key's row.
                decided[key] = None
            else:
                decided[key] = slot
        out.append(decided[key])
    return out


def _resolve_line_form(
    db, mnemonic, count, operands, intel, exact=None, arch: frozenset | None = None
):
    """The form one asm line names: classify, alias, weak gate, match.

    db is a connection (hover) or a catalog path (hints, cached lookup).
    EXACT is the line's own exact-name row for hover's exact-name path; a
    hint passes None and only matches the form on the source line.
    """
    if arch is PTX_ARCH:
        # PTX line: skip classify, alias, weak gate and match. The exact key
        # can name a bare row from another architecture, so the code drops
        # it and picks the row by chain prefix and operand shape.
        lookup = _mnemonic_matches if isinstance(db, str) else _lookup_mnemonic
        if "." in mnemonic:
            # A concrete spelling resolves through the shared matcher first:
            # it picks the template line and the section that documents it.
            hit = _ptx_pick_spelling(_ptx_rows(db), mnemonic, operands)
            if hit is not None:
                return hit
        rows, _ = lookup(db, mnemonic, PTX_ARCH)
        return _ptx_pick_form(rows, mnemonic, operands)
    line = _classify_line(operands, intel)
    # The folded shift/extend suffix makes the classified parts shorter than
    # the parsed comma count; the parts are authoritative for the count
    # checks below.
    if line[1]:
        count = len(line[1])
    # One alias step for hover and hint: an AArch64 alias line (mov, cmp,
    # cmn, tst) is judged on the alias mnemonic's rows before any weak-gate
    # or catalog-miss decision. An empty alias mnemonic means the operand
    # text does not decide the form: show nothing.
    alias = _a64_alias_for_line(mnemonic, line[0], operands)
    mnemonic, transform = alias if alias else (mnemonic, None)
    if not mnemonic:
        return None
    lookup = _mnemonic_matches if isinstance(db, str) else _lookup_mnemonic
    matches, width = lookup(db, mnemonic, line[0])
    if exact is None and not matches:
        return None
    # Weak votes come from spellings two architectures share (a RISC-V `add
    # s0, s1, s2` vs the AArch64 scalar FP s-registers). Gate them so the
    # exact-name path and _match_form both honor the veto.
    if line[3]:
        line = _gate_weak_votes(matches, line, operands, intel, transform=transform)
        if line is None:
            return None
    if exact is not None:
        # An exact name can name a row with a different operand count (a bare
        # "movsd" is the string move), or a row from the wrong architecture
        # (a bare "nop" picks AArch64 NOP over x86 in an Intel file). In both
        # cases fall through to the form match, so hover and hint agree.
        chosen = exact
        if line[0] and chosen.architecture not in line[0]:
            chosen = None
        elif count is not None and len(catkey.operands_of_key(chosen.key)) != count:
            chosen = None
        if chosen is not None:
            return chosen
    if not matches:
        return None
    # Match the form on the source line, so "vaddps ymm0, ymm1, ymm2" gets
    # the YMM form, not the first row.
    match_line = (line[0], transform(line[1]), line[2], line[3]) if transform else line
    return _match_form(matches, count, match_line, width)


def _hover_markdown(
    conn,
    word: str,
    allow_instruction: bool = True,
    operand_count: int | None = None,
    operand_text: str = "",
    intel: bool = False,
    identity_key: str | None = None,
    arch: frozenset | None = None,
) -> str | None:
    intrinsic = load_intrinsic_from_db(conn, word)
    if intrinsic is not None:
        lines = [f"```c\n{intrinsic.signature}\n```"]
        if intrinsic.description:
            lines.append(intrinsic.description)
        meta = []
        if intrinsic.header:
            meta.append(f"header `{intrinsic.header}`")
        if intrinsic.isa:
            meta.append(f"ISA {', '.join(intrinsic.isa)}")
        if intrinsic.category:
            meta.append(f"category {intrinsic.category}")
        if intrinsic.metadata.get("tail_policy") or intrinsic.metadata.get("mask_policy"):
            meta.append(
                " ".join(
                    f"{key} {intrinsic.metadata[key]}"
                    for key in ("tail_policy", "mask_policy")
                    if intrinsic.metadata.get(key)
                )
            )
        if intrinsic.url:
            meta.append(f"[source]({intrinsic.url})")
        if meta:
            lines.append(" | ".join(meta))
        if intrinsic.instructions:
            lines.append(f"Instructions: {', '.join(intrinsic.instructions[:6])}")
        linked = linked_instruction_records(None, intrinsic, conn=conn)
        if linked:
            latencies = [
                best_latency(item.arch_details)
                for item in linked
                if best_latency(item.arch_details) != "-"
            ]
            throughputs = [
                best_cpi(item.arch_details, record_iclass(item))
                for item in linked
                if best_cpi(item.arch_details, record_iclass(item)) != "-"
            ]
            perf = []
            if latencies:
                perf.append(f"best latency {min(latencies, key=lambda value: float(value))} cycles")
            if throughputs:
                perf.append(f"best cycle/instr {min(throughputs, key=lambda value: float(value))}")
            if perf:
                lines.append("Performance: " + ", ".join(perf))
        return "\n\n".join(lines)

    if not allow_instruction:
        return None
    if identity_key == NO_CATALOG_ROW:
        # Byte identity identified the form and the catalog has no row for
        # it: say so by name, never guess a neighbour form below.
        return f"`{word}`: no measured row"
    # Byte identity wins over every matcher pick below: the row is what the
    # line assembles to. None keeps the old matcher on its exact-name path.
    instruction = load_instruction_from_db(conn, identity_key) if identity_key else None
    if instruction is None:
        exact = load_instruction_from_db(conn, word)
        instruction = _resolve_line_form(
            conn, word, operand_count, operand_text, intel, exact=exact, arch=arch
        )
    if instruction is not None:
        # The url and reference rows of the body already carry the source links.
        link = web_deep_link("instruction", instruction.key)
        return instruction_markdown(instruction, conn=conn) + f"\n\nFull page: {link}"
    return None


_INTRINSIC_KINDS = frozenset({3})
_MNEMONIC_KINDS = frozenset({14})


def _completion_candidates(
    conn, prefix: str, limit: int = 50, kinds: frozenset[int] | None = None, ptx: bool = False
) -> list[dict]:
    """Completion items for a word prefix, deduped by (kind, label).

    kinds restricts the emitted item kinds (3 intrinsic, 14 instruction);
    None keeps every kind, the pre-0.0.16 pool. ptx widens a dotted prefix
    (\"ld.\") to the bare op's rows; other documents keep the exact filter.
    """
    prefix_folded = prefix.casefold()
    emitted: set[tuple[str, str]] = set()
    items: list[dict] = []
    candidate_limit = max(limit * 3, 100)
    intrinsics = search_intrinsic_candidates_from_db(conn, prefix or "_mm", limit=candidate_limit)
    instructions = search_instruction_candidates_from_db(
        conn, prefix or "_mm", limit=candidate_limit
    )
    if kinds == _MNEMONIC_KINDS:
        intrinsics = []
    elif kinds == _INTRINSIC_KINDS:
        instructions = []
    for result in search_records(intrinsics, instructions, prefix or "_mm", limit=candidate_limit):
        label = result.title
        folded = label.casefold()
        if prefix_folded and not folded.startswith(prefix_folded):
            # A dotted prefix ("ld.") also names the bare op's rows: the
            # prefix's first dot segment followed by a space or open paren.
            if not (ptx and "." in prefix_folded):
                continue
            head = prefix_folded.partition(".")[0]
            if not (folded.startswith(head + " ") or folded.startswith(head + "(")):
                continue
        key = (result.kind, label)
        if key in emitted:
            continue
        emitted.add(key)
        kind = 3 if result.kind == "intrinsic" else 14
        items.append({"label": label, "kind": kind, "detail": result.subtitle, "insertText": label})
        if len(items) >= limit:
            return items
    return items


def _completion_items(
    conn, text: str, language_id: str, uri: str, prefix: str, line: int, character: int
) -> list[dict]:
    """Completion items for one cursor position.

    Assembly documents keep the full pool. In C and C++ the pool follows
    the cursor: mnemonics only inside an asm template string, an MSVC
    __asm block or an MSVC __asm line; no simdref items in the operand and
    clobber lists; intrinsics only everywhere else.
    """
    ptx = _is_ptx_doc(language_id, uri) or _is_cuda_doc(language_id, uri)
    if _is_ptx_doc(language_id, uri):
        return _completion_candidates(conn, prefix, kinds=_MNEMONIC_KINDS, ptx=True)
    if not _is_c_doc(language_id, uri):
        return _completion_candidates(conn, prefix, ptx=ptx)
    zone = asmline.c_asm_zone_at(text, _offset_at(text, line, character))
    if zone == "operands":
        return []
    kinds = _MNEMONIC_KINDS if zone in ("template", "msvc") else _INTRINSIC_KINDS
    return _completion_candidates(conn, prefix, kinds=kinds, ptx=ptx)


def _is_c_doc(language_id: str, uri: str) -> bool:
    return language_id in ("c", "cpp") or uri.lower().endswith(C_EXTENSIONS)


def _is_ptx_doc(language_id: str, uri: str) -> bool:
    return language_id == "ptx" or uri.lower().endswith(".ptx")


def _is_cuda_doc(language_id: str, uri: str) -> bool:
    return language_id == "cuda" or uri.lower().endswith((".cu", ".cuh"))


def _semicolon_is_comment(language_id: str, uri: str) -> bool:
    # A .s/.S extension means GAS syntax: ";" stays a statement separator there
    # even if the client tags the buffer with an assembly languageId.
    if _is_c_doc(language_id, uri):
        return False
    if uri.lower().endswith(".s"):
        return False
    return language_id in ("asm", "nasm", "masm") or uri.lower().endswith((".asm", ".nasm"))


def _asm_line_at(text: str, language_id: str, uri: str, offset: int) -> str | None:
    """Return the assembler segment that holds the text offset, or None.

    C/C++ documents hold assembler in string literals in asm(...) calls, so the
    segment comes from _asm_literals. In every other document the whole source
    line is the segment. Hover and inlay hints must call this one function so
    both parse the same text.
    """
    if _is_c_doc(language_id, uri):
        spaced = text.replace("\\t", " ")
        for start, body in asmline.c_asm_literals(spaced):
            if start <= offset <= start + len(body):
                spans = asmline.statement_spans(body)
                for low, high in spans:
                    if low <= offset - start < high:
                        return body[low:high]
                return body[spans[-1][0] :]
        return None
    lines = text.split("\n")
    for line, source_line in enumerate(lines):
        end = offset - sum(len(item) + 1 for item in lines[:line])
        if 0 <= end <= len(source_line):
            return source_line
    return None


def _offset_at(text: str, line: int, character: int) -> int:
    """Text offset of an LSP position. Out-of-range positions give len(text)."""
    lines = text.split("\n")
    if line >= len(lines):
        return len(text)
    return sum(len(item) + 1 for item in lines[:line]) + _code_point_index(lines[line], character)


def _in_asm_string(text: str, line: int, character: int) -> bool:
    return asmline.c_asm_zone_at(text, _offset_at(text, line, character)) == "template"


def _mnemonic_from_asm_line(line: str, semicolon_is_comment: bool = False, ptx: bool = False):
    """Return (mnemonic, operand_count, operands) for an asm line, or None."""
    line = asmline.cut_asm_comment(line, _hash_opens_operand)
    if ptx or semicolon_is_comment:
        line = asmline.cut_semicolon(line)
    line = line.strip()
    if not line or line.startswith("."):
        return None
    if ptx:
        tokens = line.split(None, 1)
        if tokens and tokens[0].startswith("@"):
            line = tokens[1].strip() if len(tokens) > 1 else ""
            if not line or line.startswith("."):
                return None
    parts = line.split(None, 1)
    if parts[0].endswith(":"):
        line = parts[1].strip() if len(parts) > 1 else ""
        if not line or line.startswith("."):
            return None
        parts = line.split(None, 1)
    if not asmline.is_mnemonic(parts[0]):
        return None
    operands = parts[1] if len(parts) > 1 else ""
    return parts[0], len(asmline.split_operands(operands)), operands


def _hash_opens_operand(prefix: str, rest: str) -> bool:
    """True when a '#' after PREFIX opens an operand immediate, not a comment."""
    words = prefix.split()
    if words and words[0].endswith(":"):
        words = words[1:]
    if prefix.rstrip().endswith(","):
        return True
    if words and words[-1].lower() in _ASM_SHIFT_KEYWORDS:
        return True
    # First operand right after the mnemonic ("svc #0"). The token after '#'
    # separates it from a zero-operand mnemonic's comment ("ret #c3").
    # shortcut: no mnemonic table, so "nop #2" reads as an immediate; upgrade
    # when the catalog can name first-immediate mnemonics.
    return len(words) == 1 and bool(rest) and rest[0] in "0123456789+-(:"


def _cut_label(text: str) -> str:
    return text[:57] + "..." if len(text) > 60 else text


@functools.lru_cache(maxsize=1024)
def _mnemonic_matches(db_path: str, mnemonic: str, arch: frozenset = frozenset()) -> tuple:
    # The inlay-hint loop asks the same mnemonics for every line, so query once
    # per (catalog, mnemonic, arch candidates) and reuse. The key carries the
    # path, not the connection, so a reloaded catalog at the same path would
    # keep stale rows; clear the cache if catalog reload lands.
    conn = open_db(path=Path(db_path))
    try:
        forms, width = _lookup_mnemonic(conn, mnemonic, arch)
        return tuple(forms), width
    finally:
        conn.close()


@functools.lru_cache(maxsize=32)
def _ptx_rows_by_path(db_path: str) -> tuple:
    # The inlay-hint loop resolves every PTX spelling through these rows, so
    # query the PTX rows once per catalog path and reuse, as for mnemonics.
    conn = open_db(path=Path(db_path))
    try:
        return tuple(load_instructions_by_arch_from_db(conn, "ptx"))
    finally:
        conn.close()


def _ptx_rows(db) -> list:
    """All PTX rows behind db: a connection or a catalog path."""
    if isinstance(db, str):
        return list(_ptx_rows_by_path(db))
    return load_instructions_by_arch_from_db(db, "ptx")


def _segment_is_ptx(conn, segment: str) -> bool:
    """True when one asm line of a .cu/.cuh document is PTX.

    A dotted op token is PTX. A bare one is PTX only when the catalog
    holds PTX rows for it and no host-ISA row; every other line resolves
    as the host ISA, as in .c/.cpp.
    """
    probe = _mnemonic_from_asm_line(segment, False, ptx=True)
    if not probe:
        return False
    if "." in probe[0]:
        return True
    if conn is None:
        return False
    rows = load_instructions_by_mnemonic_from_db(conn, probe[0])
    return bool(rows) and all(row.architecture == "ptx" for row in rows)


def _hint_label(
    db_path: str,
    mnemonic: str,
    operand_count: int,
    operands: str = "",
    intel: bool = False,
    arch: frozenset | None = None,
) -> str | None:
    chosen = _resolve_line_form(db_path, mnemonic, operand_count, operands, intel, arch=arch)
    return _cut_label(chosen.summary or "") if chosen else None


def _asm_candidates(text: str, language_id: str, uri: str) -> list[tuple[int, str]]:
    """The (source line, asm segment) items of a document, for hover and hints.

    C/C++ documents hold assembler in string literals; every other document
    is its own source lines. Hover and inlay hints call this one function so
    both judge the same texts.
    """
    if _is_c_doc(language_id, uri):
        return [
            (text.count("\n", 0, offset), spaced[low:high])
            for offset, body in asmline.c_asm_literals(text)
            for spaced in (body.replace("\\t", " "),)
            for low, high in asmline.statement_spans(spaced)
        ]
    return list(enumerate(line.rstrip("\r") for line in text.split("\n")))


def _inlay_hints(conn, text: str, language_id: str, uri: str, start: int, end: int) -> list[dict]:
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    source_lines = [line.rstrip("\r") for line in text.split("\n")]
    semicolon_is_comment = _semicolon_is_comment(language_id, uri)
    ptx_doc = _is_ptx_doc(language_id, uri)
    cuda_doc = _is_cuda_doc(language_id, uri)
    candidates = _asm_candidates(text, language_id, uri)
    # A .cu/.cuh line resolves as PTX or as the host ISA line by line.
    flags = [
        ptx_doc or (cuda_doc and _segment_is_ptx(conn, segment)) for _line, segment in candidates
    ]
    host_keys = iter(
        identity_keys(
            [segment for (_line, segment), flag in zip(candidates, flags) if not flag],
            semicolon_is_comment,
            conn,
        )
    )
    keys = [None if flag else next(host_keys) for flag in flags]
    records = {
        key: load_instruction_from_db(conn, key)
        for key in set(keys)
        if key is not None and key != NO_CATALOG_ROW
    }
    hints: dict[int, list[str]] = {}  # every instruction on the source line feeds its hint
    names: dict[int, list[str]] = {}  # the instruction names behind empty labels
    labels: dict[str, str | None] = {}  # one matcher run per distinct segment, not per line
    for (line, segment), key, ptx in zip(candidates, keys, flags):
        if not start <= line <= end:
            continue
        record = records.get(key) if key else None
        if key == NO_CATALOG_ROW:
            # Identified form the catalog lacks: name it, never the old pick.
            parsed = _mnemonic_from_asm_line(segment, semicolon_is_comment, ptx=ptx)
            label = f"{parsed[0]}: no measured row" if parsed else None
        elif record is not None:
            label = _cut_label(record.summary or "")
        elif segment in labels:
            label = labels[segment]
        else:
            parsed = _mnemonic_from_asm_line(segment, semicolon_is_comment, ptx=ptx)
            label = labels[segment] = (
                _hint_label(
                    db_path, *parsed, intel=semicolon_is_comment, arch=PTX_ARCH if ptx else None
                )
                if parsed
                else None
            )
        if label is not None:
            hints.setdefault(line, []).append(label)
            if not label:
                # The name is the fallback text for an empty label: it is
                # sourced data, never a generated placeholder.
                if record is not None:
                    name = record.mnemonic
                else:
                    parsed = _mnemonic_from_asm_line(segment, semicolon_is_comment, ptx=ptx)
                    name = parsed[0] if parsed else None
                if name:
                    names.setdefault(line, []).append(name)
    out: list[dict] = []
    for line in sorted(hints):
        # Empty labels drop out of the join; when they are all empty the hint
        # shows the instruction names, never an empty label.
        text = "; ".join(label for label in hints[line] if label) or "; ".join(names.get(line, []))
        if not text:
            continue
        out.append(
            {
                # LSP columns count UTF-16 code units.
                "position": {
                    "line": line,
                    "character": _utf16_column(source_lines[line], len(source_lines[line])),
                },
                "label": _cut_label(text),
                "paddingLeft": True,
            }
        )
    return out


def _open_catalog() -> sqlite3.Connection | None:
    path = Path(os.environ.get("SIMDREF_CATALOG") or SQLITE_PATH)
    try:
        # A stale schema opens fine but fails at the first payload read.
        return open_db(path=path) if sqlite_schema_is_current(path) else None
    except sqlite3.Error:
        return None


def main() -> int:
    conn = _open_catalog()
    session = Session(documents={})
    while True:
        message = _jsonrpc_read()
        if message is None:
            return 0
        method = message.get("method")
        if method == "initialize":
            _jsonrpc_write(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "capabilities": {
                            "hoverProvider": True,
                            "textDocumentSync": 1,
                            "inlayHintProvider": True,
                            "completionProvider": {
                                "resolveProvider": False,
                                "triggerCharacters": ["_", ".", "m", "v"],
                            },
                        }
                    },
                }
            )
        elif method == "initialized":
            if conn is None:
                _jsonrpc_write(
                    {
                        "jsonrpc": "2.0",
                        "method": "window/showMessage",
                        "params": {"type": 2, "message": CATALOG_MISSING},
                    }
                )
        elif method == "shutdown":
            _jsonrpc_write({"jsonrpc": "2.0", "id": message["id"], "result": None})
        elif method == "exit":
            return 0
        elif method == "textDocument/didOpen":
            doc = message["params"]["textDocument"]
            session.documents[doc["uri"]] = doc["text"]
            session.languages[doc["uri"]] = doc.get("languageId", "")
        elif method == "textDocument/didChange":
            params = message["params"]
            session.documents[params["textDocument"]["uri"]] = params["contentChanges"][-1]["text"]
        elif method == "textDocument/didClose":
            uri = message["params"]["textDocument"]["uri"]
            session.documents.pop(uri, None)
            session.languages.pop(uri, None)
        elif method == "textDocument/hover":
            params = message["params"]
            uri = params["textDocument"]["uri"]
            text = session.documents.get(uri, "")
            line, character = params["position"]["line"], params["position"]["character"]
            word = _word_at(text, line, character)
            # In C/C++ only the asm strings hold instructions; intrinsics hover everywhere.
            language_id = session.languages.get(uri, "")
            is_c_doc = _is_c_doc(language_id, uri)
            ptx_doc = _is_ptx_doc(language_id, uri)
            cuda_doc = _is_cuda_doc(language_id, uri)
            allow_instruction = not is_c_doc or (_in_asm_string(text, line, character))
            operand_count = None
            operand_text = ""
            segment = None
            ptx = ptx_doc
            if word and allow_instruction:
                offset = _offset_at(text, line, character)
                segment = _asm_line_at(text, language_id, uri, offset)
                if segment:
                    # A .cu/.cuh line resolves as PTX or as the host ISA line
                    # by line, never by document.
                    ptx = ptx_doc or (cuda_doc and _segment_is_ptx(conn, segment))
                    # The C-escape pair "\t" starts a segment in asm("...\n\t...").
                    # Strip the backslash so the mnemonic survives.
                    parsed = _mnemonic_from_asm_line(
                        segment.lstrip("\\"), _semicolon_is_comment(language_id, uri), ptx=ptx
                    )
                    if parsed:
                        operand_text = parsed[2]
                        if parsed[0].casefold() == word.casefold():
                            operand_count = parsed[1]
                        elif word.casefold().endswith(parsed[0].casefold()):
                            # The raw word can swallow a preceding escape character:
                            # on "\tvmulps" _word_at reads "tvmulps", but the parsed
                            # segment names the real mnemonic.
                            word = parsed[0]
                            operand_count = parsed[1]
            # Byte identity for the hovered mnemonic's segment, batched over
            # the whole document. A line holding several asm segments joins
            # its candidates only when exactly one text matches the segment.
            identity_key = None
            if conn and operand_count is not None and segment and not ptx:
                candidates = _asm_candidates(text, language_id, uri)
                keys = identity_keys(
                    [seg for _line, seg in candidates],
                    _semicolon_is_comment(language_id, uri),
                    conn,
                )
                wanted = [
                    key
                    for (cand_line, cand), key in zip(candidates, keys)
                    if cand_line == line and key and cand.rstrip("\r") == segment.rstrip("\r")
                ]
                identity_key = wanted[0] if len(wanted) == 1 else None
            body = (
                _hover_markdown(
                    conn,
                    word,
                    allow_instruction,
                    operand_count=operand_count,
                    operand_text=operand_text,
                    intel=_semicolon_is_comment(language_id, uri),
                    identity_key=identity_key,
                    arch=PTX_ARCH if ptx else None,
                )
                if conn and word
                else None
            )
            _jsonrpc_write(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"contents": {"kind": "markdown", "value": body}} if body else None,
                }
            )
        elif method == "textDocument/completion":
            params = message["params"]
            uri = params["textDocument"]["uri"]
            text = session.documents.get(uri, "")
            language_id = session.languages.get(uri, "")
            line, character = params["position"]["line"], params["position"]["character"]
            prefix = _line_prefix(text, line, character)
            items = (
                _completion_items(conn, text, language_id, uri, prefix, line, character)
                if conn
                else []
            )
            _jsonrpc_write(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"isIncomplete": False, "items": items},
                }
            )
        elif method == "textDocument/inlayHint":
            params = message["params"]
            uri = params["textDocument"]["uri"]
            rng = params["range"]
            hints = (
                _inlay_hints(
                    conn,
                    session.documents.get(uri, ""),
                    session.languages.get(uri, ""),
                    uri,
                    rng["start"]["line"],
                    rng["end"]["line"],
                )
                if conn
                else []
            )
            _jsonrpc_write({"jsonrpc": "2.0", "id": message["id"], "result": hints})
    return 0


if __name__ == "__main__":
    sys.exit(main())
