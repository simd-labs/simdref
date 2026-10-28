from __future__ import annotations

import functools
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from simdref.display import instruction_markdown, web_deep_link
from simdref.perf import best_cpi, best_latency
from simdref.queries import linked_instruction_records
from simdref.search import search_records
from simdref.storage import (
    SQLITE_PATH,
    load_intrinsic_from_db,
    load_instruction_from_db,
    load_instructions_by_mnemonic_from_db,
    open_db,
    search_intrinsic_candidates_from_db,
    search_instruction_candidates_from_db,
    sqlite_schema_is_current,
)


WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
MNEMONIC_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.]*$")
# Matches an asm comment marker: '#', '//' or a '/* ... */' block. The '#'
# context check lives in _strip_asm_comment, because a '#' can open an
# AArch64 immediate operand.
ASM_COMMENT_RE = re.compile(r"#|//|/\*.*?\*/")
# In a .asm file (NASM/MASM) ";" starts a comment. A .s/.S file in GAS syntax
# uses ";" as a statement separator, so it stays.
ASM_SEMICOLON_COMMENT_RE = re.compile(r";.*$")
# Matches a string literal or a comment. The scanner blanks comments and keeps strings.
# ponytail: character literals ('"') and raw strings are not handled.
C_TOKEN_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"|//[^\n]*|/\*.*?\*/', re.DOTALL)
# asm("..."), __asm__("..."), __asm("..."), with optional qualifiers. Adjacent
# string literals concatenate. Extended asm (": outputs : inputs") ends at ':'.
ASM_STRING_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:__asm__|__asm|asm)"
    r"(?:\s+(?:volatile|__volatile__|goto|inline))*\s*\(\s*"
    r'("(?:[^"\\]|\\.)*"(?:\s*"(?:[^"\\]|\\.)*")*)'
    r"(?=\s*[:)])"
)
STRING_PART_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
ASM_SPLIT_RE = re.compile(r"\\n|;")
C_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx", ".c++", ".h", ".hpp", ".hh", ".hxx", ".cu", ".cuh")
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
    character = _code_point_index(current, character)
    for match in WORD_RE.finditer(current):
        if match.start() <= character <= match.end():
            return match.group(0)
    return None


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
    match = re.search(WORD_RE.pattern + r"$", current)
    return match.group(0) if match else ""


_ATT_SUFFIX_WIDTH = {"b": 8, "w": 16, "l": 32, "q": 64}
# Intel size keywords ("byte ptr [rdx]", NASM "byte [rdx]") name the memory width.
_INTEL_SIZE_BITS = {
    "byte": 8,
    "word": 16,
    "dword": 32,
    "fword": 48,
    "qword": 64,
    "tbyte": 80,
    "oword": 128,
    "xmmword": 128,
    "ymmword": 256,
    "zmmword": 512,
}
_INTEL_SIZE_RE = re.compile(
    rf"({'|'.join(_INTEL_SIZE_BITS)})\s+(?:ptr\s+)?(?:[a-z]{{2,}}:\s*)?\[", re.IGNORECASE
)
_ACCUMULATOR = {"al": "r8", "ax": "r16", "eax": "r32", "rax": "r64"}
# x86 numbers only r8-r15; r0-r7 are ARM registers.
_GPR_PATTERNS = (
    (64, r"r(ax|bx|cx|dx|si|di|bp|sp|8|9|1[0-5])"),
    (32, r"e(ax|bx|cx|dx|si|di|bp|sp)|r(8|9|1[0-5])d"),
    (16, r"(ax|bx|cx|dx|si|di|bp|sp)|r(8|9|1[0-5])w"),
    (8, r"[a-d][lh]|(si|di|bp|sp)l|r(8|9|1[0-5])b"),
)
_STRING_OP_RE = re.compile(r"(movs|cmps|scas|lods|stos|ins|outs)[bwdq]", re.IGNORECASE)
_X86_VECTOR_WIDTHS = {"xmm": "v128", "ymm": "v256", "zmm": "v512"}
_X86_VECTOR_RE = re.compile(r"(?:[xyz]mm(?:[12]?[0-9]|3[01])|k[0-7]|mm[0-7])\Z")
_A64_ARRANGEMENT_RE = re.compile(r"[bdhqsvz](?:[0-9]|[12][0-9]|3[01])\.[0-9]*[bhsdq]\Z")
_A64_SCALAR_RE = re.compile(r"(?:x(?:[0-9]|[12][0-9]|30)|w(?:[0-9]|[12][0-9]|30)|xzr|wzr)\Z")
_A64_SCALAR_FP_RE = re.compile(r"[bhsdq](?:[0-9]|[12][0-9]|3[01])\Z")
# A bare v register is a vector spelling ARM and RISC-V share.
_A64_VECTOR_BARE_RE = re.compile(r"v(?:[0-9]|[12][0-9]|3[01])\Z")
# SVE predicate register with an optional /z or /m qualifier ("p0/z").
_A64_PRED_RE = re.compile(r"p(?:1[0-5]|[0-9])(?:/[zm])?\Z", re.IGNORECASE)
# A single SME ZA tile element ("za0h.b[w12, 0]"), never a multi-tile range
# ("za0h.b[w12, 0:1]") or a vector-length-grouped array ("za.b[w8, 0, vgx2]").
_A64_ZA_TILE_RE = re.compile(r"za\d+[vh]?\.[bhsd]\[\s*w\d+\s*,\s*#?\d+\s*\]", re.IGNORECASE)
# Operand spellings only AArch64 uses: a vector lane ("v1.d[1]"), a predicate
# with an arrangement ("p0.b"), a predicate counter ("pn8.b"), or a ZA tile.
_A64_ONLY_RE = re.compile(
    r"[zv](?:3[01]|[12]?\d)\.\d*[bhsdq]\[\d+\]|pn?(?:1[0-5]|[0-9])\.[bhsdq]|\{?\s*za",
    re.IGNORECASE,
)
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


# SME predicate-as-counter register with an optional qualifier ("pn8/z").
_A64_PNG_RE = re.compile(r"pn(?:1[0-5]|[0-9])(?:/[zm])?\Z", re.IGNORECASE)
# The register spelled at either end of a Z-range member ("z0.b-z3.b").
_A64_RANGE_END_RE = re.compile(r"z(?:3[01]|[12]?\d)\.\d*[bhsdq]", re.IGNORECASE)


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
            ends = _A64_RANGE_END_RE.findall(member)
            if len(ends) == 2:
                numbers = [int(re.match(r"z(\d+)", end, re.IGNORECASE).group(1)) for end in ends]
                count = numbers[1] - numbers[0] + 1
        elif _A64_ARRANGEMENT_RE.fullmatch(member):
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
    element = re.search(r"\.(\d*)([bhsdq])", arrangement, re.IGNORECASE)
    return f"av{element.group(2).casefold()}" if element else "av"


_ARM32_SCALAR_RE = re.compile(r"(?:r(?:[0-9]|1[0-5])|ip|pc)\Z")
# ARM has "sp" and "fp"; RISC-V shares both spellings and adds these scalar aliases.
_RISCV_SCALARS = frozenset(
    {f"a{i}" for i in range(8)}
    | {f"s{i}" for i in range(12)}
    | {f"t{i}" for i in range(7)}
    | {"zero", "ra", "gp", "tp"}
)
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
    for index, piece in enumerate(re.split(r",(?![^(\[{]*[)\]}])", operands)):
        token = piece.strip().split(None, 1)[0].casefold() if piece.strip() else ""
        if index and token in _ASM_SHIFT_KEYWORDS:
            if marks:
                marks[-1] = "e" if token in _EXTEND_SHIFT_KEYWORDS else "s"
            continue
        parts.append(piece.strip())
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


def _key_operand_bits(key: str) -> tuple:
    """The operand widths of a catalog key in order: R8l -> 8, M16 -> 16."""
    bits = []
    for op in _operands_of_key(key):
        match = re.search(r"\d+", op)
        bits.append(int(match.group(0)) if match else None)
    return tuple(bits)


def _lookup_mnemonic(conn, mnemonic: str, arch: frozenset = frozenset()):
    """Return (forms, width). The exact mnemonic wins, else an AT&T zero/sign-extend
    alias (movzbl -> MOVZX), else a dropped AT&T size suffix (addq -> add).

    The alias and the suffix strip run on an x86 or undecided line only: "addw"
    on a RISC-V or ARM32 line must not land on x86 "add". Rows of other
    architectures stay in the result; _match_form filters them by the line's
    candidates.
    """
    forms = load_instructions_by_mnemonic_from_db(conn, mnemonic)
    width = _ATT_SUFFIX_WIDTH.get(mnemonic[-1:].casefold())
    if forms:
        return forms, None
    if arch and "x86" not in arch:
        return [], None
    alias = _ATT_EXTEND_ALIAS.get(mnemonic.casefold())
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
            rows = [form for form in rows if _key_operand_bits(form.key) == (dst_bits, src_bits)]
        return rows, width
    # movsbl is a sign-extend, not the string op movsb.
    if not width or _STRING_OP_RE.fullmatch(mnemonic[:-1]):
        return [], None
    stripped = load_instructions_by_mnemonic_from_db(conn, mnemonic[:-1])
    return [form for form in stripped if form.architecture == "x86"], width


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
        reg = next((w for w, pattern in _GPR_PATTERNS if re.fullmatch(pattern, name)), None)
        if op.startswith("$"):
            strong.add("x86")
        elif _X86_VECTOR_RE.fullmatch(name):
            strong.add("x86")
        elif prefixed and reg:
            strong.add("x86")
        elif (bare := re.fullmatch(r"\{([^}]*)\}", op)) and (
            _A64_ARRANGEMENT_RE.fullmatch(
                re.split(r"[,\-]", bare.group(1).strip().casefold(), maxsplit=1)[0].strip()
            )
            or bare.group(1).strip().casefold().startswith("za")
        ):
            # An SVE register list ("{z0.b}"), range or ZA tile list votes AArch64.
            strong.add("arm")
        elif _A64_PNG_RE.fullmatch(name):
            strong.add("arm")
        elif not name or "(" in op or "[" in op or "{" in op:
            continue
        elif _A64_ARRANGEMENT_RE.fullmatch(name) or _A64_SCALAR_RE.fullmatch(name) or name == "wsp":
            strong.add("arm")
        elif _ARM32_SCALAR_RE.fullmatch(name):
            strong.add("arm")
        elif _A64_SCALAR_FP_RE.fullmatch(name):
            # s0-s11 are also RISC-V saved registers: a weak vote there, and
            # the mnemonic's catalog rows arbitrate between arm and riscv.
            if name in _RISCV_SCALARS:
                weak.update(("arm", "riscv"))
            else:
                strong.add("arm")
        elif name in _RISCV_SCALARS:
            strong.add("riscv")
        elif _A64_VECTOR_BARE_RE.fullmatch(name):
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
    if re.fullmatch(r"\$?0+", op):
        return "lit0"
    if re.fullmatch(r"\$?0*1", op):
        return "lit1"
    if op.startswith("$") or re.fullmatch(r"-?(0x[0-9a-f]+|\d+)", op, re.IGNORECASE):
        return "i"
    # AArch64 marks immediates with '#' ("#-16", "#0x10", "#:lo12:sym").
    if op.startswith("#"):
        return "i"
    if "{1to" in op:
        return "mbc"
    if "arm" in arch:
        list_match = re.fullmatch(r"\{([^}]*)\}", op.strip())
        if list_match:
            # The class of a brace list is decided by its shape.
            return _brace_list_class(list_match.group(1).strip())
        if _A64_PNG_RE.fullmatch(name):
            return "png"
        if _A64_PRED_RE.fullmatch(name):
            return "pred"
    if "{" in op:
        op = re.sub(r"\{[^}]*\}", "", op).strip()
        name = op.lstrip("*").lstrip("%").casefold()
    if "(" in op or "[" in op:
        sized = _INTEL_SIZE_RE.match(op)
        if sized:
            return f"m{_INTEL_SIZE_BITS[sized.group(1).casefold()]}"
        if "[" in op and "arm" in arch:
            return _a64_address_class(op)
        return "m"
    reg = next((w for w, pattern in _GPR_PATTERNS if re.fullmatch(pattern, name)), None)
    # In AT&T a bare "sp" is the ARM stack pointer or a symbol, never the r16.
    if reg and (intel or op.lstrip("*").startswith("%") or name not in ("sp", "bp", "si", "di")):
        if intel or arch.isdisjoint({"arm", "riscv"}):
            return f"r{reg}"
    if _X86_VECTOR_RE.fullmatch(name):
        if name[0] == "k":
            return "k"
        if name[0] == "m":
            return "mm"
        return _X86_VECTOR_WIDTHS[name[:3]]
    if not intel and op.startswith("*"):
        return "m"
    if "arm" in arch:
        if _A64_ARRANGEMENT_RE.fullmatch(name):
            return _av_class(name)
        if _A64_VECTOR_BARE_RE.fullmatch(name):
            return "av"
        if _A64_SCALAR_FP_RE.fullmatch(name):
            return name[0]
        if _A64_SCALAR_RE.fullmatch(name):
            return {"xzr": "xzr", "wzr": "wzr"}.get(name, name[0])
        if name in ("sp", "wsp", "fp", "lr"):
            return {"sp": "sp", "wsp": "wsp"}.get(name, "x")
        if _ARM32_SCALAR_RE.fullmatch(name):
            return "ar"
    if "riscv" in arch and (name in _RISCV_SCALARS or _A64_VECTOR_BARE_RE.fullmatch(name)):
        return "rv"
    if "x86" in arch or not arch:
        return "rel" if intel else "m"
    return None


_IMMEDIATE_ACCEPT = frozenset({"i", "lit0", "lit1"})
_LITERAL_ACCEPT = {frozenset({"lit0"}), frozenset({"lit1"})}
_CLASS_KIND = dict(
    [(c, "register") for c in ("r8", "r16", "r32", "r64")]
    + [(c, "vector") for c in ("v128", "v256", "v512", "k", "mm", "dfp", "pred", "png")]
    + [(c, "vector") for c in ("av", "avb", "avh", "avs", "avd", "avq", "av2", "av4")]
    + [(c, "memory") for c in ("m", "moffs", "mbc", "mo", "mpre", "mreg")]
    + [(f"m{bits}", "memory") for bits in set(_INTEL_SIZE_BITS.values())]
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


def _form_accept(op: str) -> frozenset:
    """The line classes a catalog-key operand accepts, as one frozenset.

    An immediate-width key (i8..i64) takes every immediate class. A literal 0/1
    takes only its own value: a fixed-1 shift is not the line's "2". An AArch64
    placeholder maps to the register family it encodes (<XNSP> = x or sp).
    """
    o = op.casefold()
    if o in ("0", "1"):
        return frozenset({"lit" + o})
    if re.fullmatch(r"i\d+", o):
        return _IMMEDIATE_ACCEPT
    if o in _ACCUMULATOR or o == "cl":
        return frozenset({_ACCUMULATOR.get(o, "r8")})
    m = re.fullmatch(r"r(8|16|32|64)[hl]?", o)
    if m:
        return frozenset({"r" + m.group(1)})
    if o in _X86_VECTOR_WIDTHS or o in ("k", "mm"):
        return frozenset({_X86_VECTOR_WIDTHS.get(o, o)})
    if re.fullmatch(r"m\d+_\d+to\d+", o):
        return frozenset({"mbc"})
    if re.fullmatch(r"m\d+", o):
        return frozenset({o, "m"})
    if re.fullmatch(r"moffs\d+", o):
        return frozenset({"moffs"})
    if re.fullmatch(r"rel\d*", o):
        return frozenset({"rel"})
    if op.startswith("["):
        # An AArch64 bracket key encodes the index mode: pre-index carries
        # '!' ("[<XNSP>, <HASH>#0]!"), a register offset holds a bare comma
        # ("[<XNSP>, <WORX>]"), an immediate offset keeps its '#' ("[#0]"),
        # and a plain base has neither ("[<XNSP>]"). "m" stays acceptable to
        # the unsigned-offset key, whose '#imm' defaults to 0 on the line.
        if "!" in op:
            return frozenset({"mpre"})
        body = op[1 : op.index("]")] if "]" in op else op[1:]
        if "," in body:
            return frozenset({"mreg"})
        if "#" in body:
            return frozenset({"mo", "m"})
        return frozenset({"m"})
    name = re.sub(r"(#\d+)+$", "", op.upper())
    if "<HASH>" in name:
        return _IMMEDIATE_ACCEPT
    # A ZA tile operand ("<ZAT...>[W1, #0]") is not a plain vector register.
    if "<ZAT" in name:
        return frozenset({"za"})
    if re.match(r"V\d+<|^[ZV]\d+\.|^\{", name):
        # A counted register list accepts only the same count.
        zts = re.findall(r"<ZT(\d)", name)
        if zts:
            return frozenset({f"av{max(int(number) for number in zts)}"})
        count = name.count(",") + 1
        if count in (2, 4):
            return frozenset({f"av{count}"})
        # A Z-register arrangement element ("Z0.S") accepts its class ("avs"),
        # so the same-lane row outranks the other Z-register rows.
        element = re.search(r"Z\d+\.\d*([BHSDQ])", name)
        return frozenset({"av", f"av{element.group(1).casefold()}"} if element else {"av"})
    # Numbered scalar FP placeholders: D0 or S1 style numbers, <BT> or <QT1> style aliases.
    if re.fullmatch(r"[BHSDQ]\d+", name):
        return frozenset({name[0].lower()})
    if re.fullmatch(r"<[BHSDQ]T\d*>", name):
        return frozenset({name[1].lower()})
    if re.match(r"^D<|^<FPFAR_DT>$", name):
        return frozenset({"dfp"})
    if "PNG" in name:
        return frozenset({"png"})
    if re.match(r"^P\d", name):
        return frozenset({"pred"})
    # An extend-flavored placeholder (a name with EXTEND) takes the extend
    # marker class; every register placeholder also takes the shift marker.
    extend = "EXTEND" in name
    if "WORX" in name or "R_OPTION" in name:
        return frozenset({"w", "x"} | ({"we", "xe"} if extend else {"ws", "xs"}))
    if "WZR" in name:
        return frozenset({"w", "wzr"} | ({"we"} if extend else {"ws"}))
    if "XZR" in name:
        return frozenset({"x", "xzr"} | ({"xe"} if extend else {"xs"}))
    if "WSP" in name:
        return frozenset({"w", "wsp"})
    if "SP" in name:
        return frozenset({"x", "sp"})
    return frozenset()


def _form_takes_line(form, classes: list) -> bool:
    """The form accepts the line's operand classes position by position.

    A bare key constrains nothing; neither does a placeholder _form_accept
    leaves empty. Rejecting on an empty set hid the AArch64 scalar saturating
    family (<V_OPTION><D> maps to no class). An operand the classifier gave no
    class (None) constrains nothing either.
    """
    ops = _operands_of_key(form.key)
    if not ops:
        return True
    if len(ops) != len(classes):
        return False
    for cls, op in zip(classes, ops):
        accept = _form_accept(op)
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
    ops = _operands_of_key(form.key)
    if not ops or len(ops) != len(classes):
        return False
    have = {cls for cls in classes if cls}
    for cls, op in zip(classes, ops):
        accept = _form_accept(op)
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


def _match_form(
    matches,
    operand_count: int | None,
    line: tuple,
    suffix_bits: int | None,
):
    """Pick the form the line names, under one total order.

    line is _classify_line's (arch candidates, ordered classes, reverse, weak
    votes). Rows of a non-candidate architecture drop out before ranking; an
    undecided line keeps every row. Rank: class-set score (accepted +1 / same kind 0 / else
    -1, a literal scores only on its own value), position score, iN over a
    literal 0/1, match with the AT&T suffix width, fewer operands, key name.
    The no-same-count fallback uses the same key.
    """
    arch, classes, reverse = line[0], line[1], line[2]
    if arch:
        matches = [m for m in matches if m.architecture in arch]
    # A size keyword may only add information: a sized memory class no row of
    # the mnemonic accepts ranks as a plain "m", so the pick equals the line
    # with the keyword removed (VMOVDQU has no M512 row, and "vmovdqu zmm0,
    # zmmword ptr [rdx]" keeps the unsized load form). The width reward in
    # _rank still prefers a form that matches the width exactly.
    if classes and any(c and re.fullmatch(r"m\d+", c) for c in classes):
        accepted_memory = {
            accept
            for form in matches
            for op in _operands_of_key(form.key)
            for accept in _form_accept(op)
        }
        classes = [
            "m" if c and re.fullmatch(r"m\d+", c) and c not in accepted_memory else c
            for c in classes
        ]
    # The line's classified parts are authoritative: a shift/extend suffix
    # folded into its register operand, so the parsed comma count can lag.
    count = len(classes) if classes else operand_count
    if count is None:
        return matches[0] if matches else None
    same = [m for m in matches if len(_operands_of_key(m.key)) == count]
    pool = same or matches
    if not pool:
        return None
    ordered = list(reversed(classes)) if reverse else classes
    line_set = {c for c in classes if c}
    line_kinds = {_CLASS_KIND[c] for c in line_set if c in _CLASS_KIND}

    def _rank(form):
        accepts = [_form_accept(op) for op in _operands_of_key(form.key)]
        set_score = position = in_over_lit = width = 0
        for acc in accepts:
            if acc in _LITERAL_ACCEPT:
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
                if re.fullmatch(r"[rv]\d+", c)
            ):
                width += 1
            # An Intel size keyword names the memory width; reward its match.
            if any(re.fullmatch(r"m\d+", c) and c in line_set for c in acc):
                width += 1
        for cls, acc in zip(ordered, accepts):
            if cls in acc:
                position += 1
            if cls in ("i", "lit0", "lit1"):
                if acc == _IMMEDIATE_ACCEPT:
                    in_over_lit += 1
                elif acc in _LITERAL_ACCEPT:
                    in_over_lit -= 1
        return (-set_score, -position, -in_over_lit, -width, len(accepts), form.key)

    return min(pool, key=_rank)


def _zreg(classes: list) -> str:
    """The width-apportioned zero register for the line's register family."""
    return "wzr" if classes and classes[0] in ("w", "wsp", "wzr") else "xzr"


def _a64_gpr_bits(name: str) -> int | None:
    """The width of a general-register spelling (sp, fp, lr included), or None."""
    bits = {"sp": 64, "fp": 64, "lr": 64, "xzr": 64, "wsp": 32, "wzr": 32}
    return bits.get(name) or (
        (64, 32)[name.startswith("w")] if _A64_SCALAR_RE.fullmatch(name) else None
    )


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
    if any(_ARM32_SCALAR_RE.fullmatch(name) for name in names):
        return None
    mnemonic = mnemonic.casefold()
    if mnemonic not in ("mov", "cmp", "cmn", "tst"):
        return None
    if "arm" not in arch and not any(_A64_ONLY_RE.match(name) for name in names):
        return None
    if mnemonic == "mov":
        # MOVA: one ZA tile element, a governing predicate, one Z register.
        if (
            len(parts) == 3
            and any(_A64_ZA_TILE_RE.fullmatch(name) for name in names)
            and any(_A64_PRED_RE.fullmatch(name) for name in names)
            and any(_A64_ARRANGEMENT_RE.fullmatch(name) for name in names)
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


def _hover_markdown(
    conn,
    word: str,
    allow_instruction: bool = True,
    operand_count: int | None = None,
    operand_text: str = "",
    intel: bool = False,
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
                best_cpi(item.arch_details) for item in linked if best_cpi(item.arch_details) != "-"
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
    line = _classify_line(operand_text, intel)
    # The folded shift/extend suffix makes the classified parts shorter than
    # the parsed comma count; the parts are authoritative for the two count
    # checks below.
    if line[1]:
        operand_count = len(line[1])
    # One alias step for hover and hint: an AArch64 alias line (mov, cmp,
    # cmn, tst) is judged on the alias mnemonic's rows before any weak-gate
    # or catalog-miss decision. An empty alias mnemonic means the operand
    # text does not decide the form: show nothing.
    alias = _a64_alias_for_line(word, line[0], operand_text)
    if alias:
        word_mn, transform = alias
    else:
        word_mn, transform = word, None
    if not word_mn:
        return None
    # Weak votes come from spellings two architectures share (a RISC-V `add
    # s0, s1, s2` vs the AArch64 scalar FP s-registers). Gate them once here
    # so the exact-name path and _match_form both honor the veto.
    matches = width = None
    if line[3]:
        matches, width = _lookup_mnemonic(conn, word_mn, line[0])
        line = _gate_weak_votes(matches, line, operand_text, intel, transform=transform)
        if line is None:
            return None
    instruction = load_instruction_from_db(conn, word)
    # An exact name can name a row with a different operand count (a bare
    # "movsd" is the string move), or a row from the wrong architecture
    # (a bare "nop" picks AArch64 NOP over x86 in an Intel file).
    # In both cases, fall back to _match_form so hover and hint agree.
    if instruction is not None:
        if line[0] and instruction.architecture not in line[0]:
            instruction = None
        elif operand_count is not None and len(_operands_of_key(instruction.key)) != operand_count:
            instruction = None
    if instruction is None:
        # Match the form on the source line like _hint_label does, so
        # "vaddps ymm0, ymm1, ymm2" gets the YMM form, not the first row.
        if matches is None:
            matches, width = _lookup_mnemonic(conn, word_mn, line[0])
        match_line = (line[0], transform(line[1]), line[2], line[3]) if transform else line
        instruction = _match_form(matches, operand_count, match_line, width)
    if instruction is not None:
        # The url and reference rows of the body already carry the source links.
        link = web_deep_link("instruction", instruction.key)
        return instruction_markdown(instruction, conn=conn) + f"\n\nFull page: {link}"
    return None


def _completion_candidates(conn, prefix: str, limit: int = 50) -> list[dict]:
    prefix_folded = prefix.casefold()
    emitted: set[tuple[str, str]] = set()
    items: list[dict] = []
    candidate_limit = max(limit * 3, 100)
    intrinsics = search_intrinsic_candidates_from_db(conn, prefix or "_mm", limit=candidate_limit)
    instructions = search_instruction_candidates_from_db(
        conn, prefix or "_mm", limit=candidate_limit
    )
    for result in search_records(intrinsics, instructions, prefix or "_mm", limit=candidate_limit):
        label = result.title
        if prefix_folded and not label.casefold().startswith(prefix_folded):
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


def _is_c_doc(language_id: str, uri: str) -> bool:
    return language_id in ("c", "cpp") or uri.lower().endswith(C_EXTENSIONS)


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
        for start, body in _asm_literals(spaced):
            if start <= offset <= start + len(body):
                cut = 0
                for split in ASM_SPLIT_RE.finditer(body):
                    if cut <= offset - start < split.start():
                        return body[cut : split.start()]
                    cut = split.end()
                return body[cut:]
        return None
    lines = text.split("\n")
    for line, source_line in enumerate(lines):
        end = offset - sum(len(item) + 1 for item in lines[:line])
        if 0 <= end <= len(source_line):
            return source_line
    return None


def _asm_literals(text: str) -> list[tuple[int, str]]:
    """Return (offset, body) for each string literal inside an asm(...) call in C source."""
    masked = C_TOKEN_RE.sub(
        lambda m: m.group(0) if m.group(0)[0] == '"' else re.sub(r"[^\n]", " ", m.group(0)),
        text,
    )
    return [
        (match.start(1) + part.start(1), part.group(1))
        for match in ASM_STRING_RE.finditer(masked)
        for part in STRING_PART_RE.finditer(match.group(1))
    ]


def _in_asm_string(text: str, line: int, character: int) -> bool:
    offset = sum(len(item) + 1 for item in text.split("\n")[:line]) + _code_point_index(
        text.split("\n")[line], character
    )
    return any(start <= offset <= start + len(body) for start, body in _asm_literals(text))


def _operand_count(operands: str) -> int:
    """Count top-level commas: "(%rax,%rbx,4)" and "{1to16}" count as one operand."""
    if not operands.strip():
        return 0
    depth, count = 0, 1
    for char in operands:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            count += 1
    return count


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


def _strip_asm_comment(line: str) -> str:
    """Strip one asm comment. A '#' that opens an immediate operand stays."""
    out, pos = [], 0
    for match in ASM_COMMENT_RE.finditer(line):
        token = match.group()
        if token.startswith("/*"):
            out.append(line[pos : match.start()])
            pos = match.end()
            continue
        if token == "#" and _hash_opens_operand(line[: match.start()], line[match.end() :]):
            continue
        out.append(line[pos : match.start()])
        return "".join(out)  # a '#' or '//' comment runs to end of line
    out.append(line[pos:])
    return "".join(out)


def _mnemonic_from_asm_line(line: str, semicolon_is_comment: bool = False):
    """Return (mnemonic, operand_count, operands) for an asm line, or None."""
    line = _strip_asm_comment(line)
    if semicolon_is_comment:
        line = ASM_SEMICOLON_COMMENT_RE.sub("", line)
    line = line.strip()
    if not line or line.startswith("."):
        return None
    parts = line.split(None, 1)
    if parts[0].endswith(":"):
        line = parts[1].strip() if len(parts) > 1 else ""
        if not line or line.startswith("."):
            return None
        parts = line.split(None, 1)
    if not MNEMONIC_RE.match(parts[0]):
        return None
    operands = parts[1] if len(parts) > 1 else ""
    return parts[0], _operand_count(operands), operands


def _operands_of_key(key: str) -> list[str]:
    start, end = key.find("("), key.rfind(")")
    inner = key[start + 1 : end].strip() if 0 <= start < end else ""
    if not inner:
        return []
    # AArch64 address operands hold their own commas ("[<XNSP>, <HASH>#0]!"),
    # so only a comma outside every bracket separates operands.
    ops, depth, cut = [], 0, 0
    for index, char in enumerate(inner):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            ops.append(inner[cut:index].strip())
            cut = index + 1
    ops.append(inner[cut:].strip())
    return ops


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


def _hint_label(
    db_path: str,
    mnemonic: str,
    operand_count: int,
    operands: str = "",
    intel: bool = False,
) -> str | None:
    line = _classify_line(operands, intel)
    # The same one alias step as hover, before the catalog-miss return, so
    # both entries judge the alias mnemonic's rows. An empty alias mnemonic
    # matches no rows and the hint is None, like hover.
    alias = _a64_alias_for_line(mnemonic, line[0], operands)
    lookup_mn, transform = alias if alias else (mnemonic, None)
    matches, width = _mnemonic_matches(db_path, lookup_mn, line[0])
    if not matches:
        return None
    # The same weak-vote gate as hover, so both entries pick the same form.
    line = _gate_weak_votes(matches, line, operands, intel, transform=transform)
    if line is None:
        return None
    if transform is not None:
        line = (line[0], transform(line[1]), line[2], line[3])
    chosen = _match_form(matches, operand_count, line, width)
    return _cut_label(chosen.summary or "") if chosen else None


def _inlay_hints(conn, text: str, language_id: str, uri: str, start: int, end: int) -> list[dict]:
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    source_lines = [line.rstrip("\r") for line in text.split("\n")]
    is_c_doc = _is_c_doc(language_id, uri)
    semicolon_is_comment = _semicolon_is_comment(language_id, uri)
    if is_c_doc:
        candidates = [
            (text.count("\n", 0, offset), segment)
            for offset, body in _asm_literals(text)
            for segment in ASM_SPLIT_RE.split(body.replace("\\t", " "))
        ]
    else:
        candidates = list(enumerate(source_lines))
    hints: dict[int, list[str]] = {}  # every instruction on the source line feeds its hint
    for line, segment in candidates:
        if not start <= line <= end:
            continue
        parsed = _mnemonic_from_asm_line(segment, semicolon_is_comment)
        label = _hint_label(db_path, *parsed, intel=semicolon_is_comment) if parsed else None
        if label is not None:
            hints.setdefault(line, []).append(label)
    return [
        {
            # LSP columns count UTF-16 code units.
            "position": {
                "line": line,
                "character": _utf16_column(source_lines[line], len(source_lines[line])),
            },
            "label": _cut_label("; ".join(hints[line])),
            "paddingLeft": True,
        }
        for line in sorted(hints)
    ]


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
            allow_instruction = not is_c_doc or (_in_asm_string(text, line, character))
            operand_count = None
            operand_text = ""
            if word and allow_instruction:
                offset = sum(len(item) + 1 for item in text.split("\n")[:line]) + _code_point_index(
                    text.split("\n")[line], character
                )
                segment = _asm_line_at(text, language_id, uri, offset)
                if segment:
                    # The C-escape pair "\t" starts a segment in asm("...\n\t...").
                    # Strip the backslash so the mnemonic survives.
                    parsed = _mnemonic_from_asm_line(
                        segment.lstrip("\\"), _semicolon_is_comment(language_id, uri)
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
            body = (
                _hover_markdown(
                    conn,
                    word,
                    allow_instruction,
                    operand_count=operand_count,
                    operand_text=operand_text,
                    intel=_semicolon_is_comment(language_id, uri),
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
            prefix = _line_prefix(text, params["position"]["line"], params["position"]["character"])
            items = _completion_candidates(conn, prefix) if conn else []
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
