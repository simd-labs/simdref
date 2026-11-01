"""Annotate assembly (``.s``) files with per-instruction summaries and perf.

Given GAS/AT&T-syntax assembly, emit an annotated ``.sa`` file where each
recognised instruction line carries a trailing ``# ...`` comment describing
what the instruction does plus latency / CPI figures pulled from the
simdref catalog.
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator

from simdref import perf
from simdref.lsp import NO_CATALOG_ROW, _CATALOG_ISA, _document_arch, identity_keys
from simdref.models import InstructionRecord
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
_INSTR_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<mnemonic>[A-Za-z][A-Za-z0-9_.]*)"
    r"(?:[ \t]+(?P<operands>.*))?$"
)
_LABEL_RE = re.compile(r"^[ \t]*[A-Za-z_.$][\w.$]*:")
# objdump -d line shape: optional whitespace, hex VA, ':', hex bytes, mnemonic ops.
# Byte tokens are 2-8 hex digits: x86 emits one whitespace-separated token per
# byte ("48 8b 45 f8"), ARM/RISC-V emit the whole encoding as one token
# ("a9bf7bfd" / "00050513").
_OBJDUMP_INSTR_RE = re.compile(
    r"^\s*(?P<addr>[0-9a-fA-F]+):\s+"
    r"(?:(?:[0-9a-fA-F]{2,8}\s+){1,10})?"
    r"(?P<rest>\S.*?)\s*$"
)
# objdump -S injects "file:line" comment lines before the instruction block.
_OBJDUMP_SRC_RE = re.compile(r"^\s*(?P<file>[^ \t/][^:]*):(?P<line>\d+)\s*$")

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


def _strip_prefix_tokens(head: str) -> str:
    """Drop leading x86 prefix tokens; return the text from the mnemonic on."""
    rest = head.lstrip()
    while True:
        parts = rest.split(None, 1)
        if len(parts) != 2 or parts[0].lower() not in _X86_PREFIX_TOKENS:
            return rest
        rest = parts[1]


def _parse_objdump_instr(line: str) -> tuple[int, str] | None:
    """Match an ``objdump -d`` instruction line: ``<hexaddr>: [bytes] mnemonic ...``.

    Returns ``(address, rest)`` where ``rest`` starts at the mnemonic, or None
    when the line is not objdump-shaped (e.g. a column-0 numeric local label
    ``1:`` does not match).
    """
    # Cheap str gates before the regex (HOWTO: "use string methods"): objdump
    # never starts an instruction line at column 0 (GAS local labels ``1:``
    # do), addresses start with a hex char, and the mandatory ':' sits within
    # the first 24 columns.
    if not line or line[0] not in " \t":
        return None
    bare = line.lstrip(" \t")
    if not bare or bare[0] not in "0123456789abcdefABCDEF" or ":" not in bare[:24]:
        return None
    obj_m = _OBJDUMP_INSTR_RE.match(line)
    if not obj_m:
        return None
    try:
        return int(obj_m.group("addr"), 16), obj_m.group("rest")
    except ValueError:
        return None


def _split_comment(text: str) -> tuple[str, str]:
    """Split AT&T ``#`` trailing comment off; returns (head, comment)."""
    head, sep, comment = text.partition("#")
    return head, (sep + comment if sep else "")


def parse_asm_line(line: str, *, track_positions: bool = False) -> AsmLine:
    stripped = line.rstrip("\n")
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

    if track_positions:
        obj = _parse_objdump_instr(stripped)
        if obj is not None:
            address, rest = obj
            head, comment = _split_comment(rest)
            m = _INSTR_RE.match(_strip_prefix_tokens(head))
            if m:
                return AsmLine(
                    kind=LineKind.INSTRUCTION,
                    raw=stripped,
                    indent=(stripped[: stripped.find(rest)] if rest in stripped else ""),
                    mnemonic=m.group("mnemonic") or "",
                    operands=(m.group("operands") or "").strip(),
                    trailing_comment=comment.strip(),
                    address=address,
                )
        # In objdump mode, any non-matching line is an objdump header,
        # `-S` source interleave, or an inline symbol annotation — never
        # an AT&T instruction. Bailing out here avoids classifying C++
        # source keywords ("return", "if", "for", ...) as mnemonics.
        return AsmLine(LineKind.COMMENT, stripped)

    if _LABEL_RE.match(stripped):
        return AsmLine(LineKind.LABEL, stripped)

    # objdump-style input (workflow §1d: `objdump -d ... > file.s`) is a
    # first-class input even without --track-positions: strip the leading
    # "  <hexaddr>:\t" (and any raw bytes column) so the mnemonic parses.
    obj = _parse_objdump_instr(stripped)
    if obj is not None:
        address, rest = obj
        head, comment = _split_comment(rest)
        m = _INSTR_RE.match(_strip_prefix_tokens(head))
        if m:
            return AsmLine(
                kind=LineKind.INSTRUCTION,
                raw=stripped,
                indent=(stripped[: stripped.find(rest)] if rest in stripped else ""),
                mnemonic=m.group("mnemonic") or "",
                operands=(m.group("operands") or "").strip(),
                trailing_comment=comment.strip(),
                address=address,
            )

    head, comment = _split_comment(stripped)
    m = _INSTR_RE.match(head)
    if not m:
        return AsmLine(LineKind.COMMENT, stripped)
    return AsmLine(
        kind=LineKind.INSTRUCTION,
        raw=stripped,
        indent=m.group("indent") or "",
        mnemonic=m.group("mnemonic") or "",
        operands=(m.group("operands") or "").strip(),
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


def _isa_family(architecture: str | None) -> str | None:
    """Map an ISA option or a record architecture to one catalog family."""
    name = (architecture or "").lower()
    if not name or name == "auto":
        return None
    if name.startswith(("arm", "aarch")):
        return "arm"
    if name.startswith("riscv"):
        return "riscv"
    return "x86"


_FILE_FORMAT_RE = re.compile(r"\bfile format\s+(\S+)")
# AArch64-only register spellings and branches: w0-w30, v0-v31, b.<cond>.
_A64_EXCL_RE = re.compile(
    r"(?<![%\w])(?:w(?:[0-9]|[12][0-9]|30)|v(?:[0-9]|[12][0-9]|3[01]))(?![\w])"
)
_A64_BCOND_RE = re.compile(
    r"(?<![\w])b\.(?:eq|ne|lt|le|gt|ge|hs|lo|hi|ls|mi|pl|vs|vc|al|nv)(?![\w])"
)
# x0-x30 and sp name both AArch64 and RISC-V registers.
_A64_SHARED_RE = re.compile(r"(?<![%\w])(?:x(?:[0-9]|[12][0-9]|30)|sp)(?![\w])")
# RISC-V register names: the ABI names and x31 (AArch64 stops at x30).
_RISCV_ISA_RE = re.compile(r"(?<![%\w])(?:zero|ra|gp|tp|fp|t[0-6]|a[0-7]|x31)(?![\w])")
# objdump symbol annotation ("jmp 50 <x0+0x50>"): a target label, never input.
_SYMBOL_ANNOTATION_RE = re.compile(r"<[^>]*>")


def detect_isa(lines: Iterable[str]) -> str | None:
    """Name the ISA the input speaks: ``"x86"``, ``"arm"`` or ``"riscv"``.

    An objdump ``file format`` line names the ISA outright. Otherwise the
    AArch64 register names x0-x30, w0-w30, v0-v31, sp, the ``b.<cond>``
    branches and the RISC-V register names vote. Returns None when the
    input carries no ISA signal.
    shortcut: a RISC-V listing that uses only xN names and sp votes arm,
    upgrade when a form pick on rv64 rows needs those listings split
    """
    excl_arm = excl_riscv = shared = 0
    for line in lines:
        fmt = _FILE_FORMAT_RE.search(line)
        if fmt:
            name = fmt.group(1).lower()
            if "riscv" in name:
                return "riscv"
            return "arm" if "aarch64" in name or "arm" in name else "x86"
        parsed = parse_asm_line(line)
        if parsed.kind != LineKind.INSTRUCTION:
            continue
        # Strip objdump symbol annotations: "<x0+0x50>" names a target, not
        # a register (review finding 1).
        text = _SYMBOL_ANNOTATION_RE.sub(" ", f"{parsed.mnemonic} {parsed.operands}")
        excl_arm += len(_A64_EXCL_RE.findall(text)) + len(_A64_BCOND_RE.findall(text))
        excl_riscv += len(_RISCV_ISA_RE.findall(text))
        shared += len(_A64_SHARED_RE.findall(text))
    if excl_arm or excl_riscv:
        return "arm" if excl_arm >= excl_riscv else "riscv"
    return "arm" if shared else None


def _line_isa(text: str) -> str | None:
    """Name the family one instruction line's own evidence shows.

    detect_isa's register and ``b.<cond>`` votes come first, then the
    strong operand votes ``_document_arch`` feeds byte identity. Returns
    None for a neutral line: its family is the stream's.
    """
    text = _SYMBOL_ANNOTATION_RE.sub(" ", text)
    vote = detect_isa([text])
    if vote:
        return vote
    picked = _document_arch((text,), False, False)
    return _CATALOG_ISA.get(picked[0]) if picked else None


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


_SIZE_SPECIFIER_RE = re.compile(
    r"\b(byte|word|dword|qword|xmmword|ymmword|zmmword)\s+ptr\b",
    re.IGNORECASE,
)
# Register classes ordered so longer/wider names match first.
_REG_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:r[abcd]x|r[sd]i|rbp|rsp|r(?:8|9|1[0-5]))\b"), "R64"),
    (re.compile(r"\b(?:e[abcd]x|e[sd]i|ebp|esp|r(?:8|9|1[0-5])d)\b"), "R32"),
    (re.compile(r"\b(?:[abcd]x|[sd]i|bp|sp|r(?:8|9|1[0-5])w)\b"), "R16"),
    (re.compile(r"\b(?:[abcd][lh]|[sd]il|bpl|spl|r(?:8|9|1[0-5])b)\b"), "R8"),
    (re.compile(r"\bxmm\d+\b"), "XMM"),
    (re.compile(r"\bymm\d+\b"), "YMM"),
    (re.compile(r"\bzmm\d+\b"), "ZMM"),
    (re.compile(r"\bk[0-7]\b"), "K"),
)


# ---- Operand-shape parser for variant selection ----------------------------
#
# We split an operand string like ``%eax, %eax`` or ``(%rdi), %xmm0`` into
# typed slots so we can reject variants whose shape doesn't match (e.g. a
# memory-destination form when there's no memory operand in the input) and
# recognise zeroing idioms (``xor %eax, %eax``) by operand aliasing.


_REG_TOKEN_RE = re.compile(
    r"%(?P<name>"
    r"[abcd][lh]|[sd]il|bpl|spl|r(?:8|9|1[0-5])b|"  # r8
    r"[abcd]x|[sd]i|bp|sp|r(?:8|9|1[0-5])w|"  # r16
    r"e[abcd]x|e[sd]i|ebp|esp|r(?:8|9|1[0-5])d|"  # r32
    r"r[abcd]x|r[sd]i|rbp|rsp|r(?:8|9|1[0-5])|"  # r64
    r"[xyz]mm\d+|k[0-7]"
    r")\b",
    re.IGNORECASE,
)


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
    if re.fullmatch(r"-?(?:0x[0-9a-fA-F]+|\d+)\s+<[^>]*>", s):
        return "REL", None
    # AT&T immediate: $... ; Intel immediate: bare number / 0x... / '$'-less literal
    if s.startswith("$") or re.fullmatch(r"-?0x[0-9a-fA-F]+|-?\d+", s):
        try:
            return _imm_kind(int(s.lstrip("$"), 0)), None
        except ValueError:
            return "I", None
    # AT&T memory: ``(%reg)``, ``disp(%reg,%idx,scale)``, or plain symbol
    # with a ``(...)`` section; Intel memory: ``[...]`` or explicit size ptr.
    if "(" in s or "[" in s or _SIZE_SPECIFIER_RE.search(s):
        return "M", None
    # Pure register?
    rm = _REG_TOKEN_RE.fullmatch(s) or _REG_TOKEN_RE.fullmatch("%" + s)
    if rm:
        name = rm.group("name").lower()
        for regex, tok in _REG_PATTERNS:
            if regex.fullmatch(name):
                return tok, name
        return "?", name
    # Plain symbol (branch target) or unknown.
    if re.fullmatch(r"[A-Za-z_.$][\w.$@]*", s):
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


_FORM_OPS_RE = re.compile(r"\(([^)]*)\)\s*$")


def _form_operand_tokens(form_or_key: str) -> list[str]:
    """Extract the comma-separated operand tokens from a catalog form/key.

    For ``"XOR (R32, 0)"`` this returns ``["R32", "0"]``; for
    ``"RET_NEAR"`` returns ``[]``. Tokens are upper-cased.
    """
    s = str(form_or_key or "")
    m = _FORM_OPS_RE.search(s)
    if not m:
        return []
    inner = m.group(1)
    return [t.strip().upper() for t in inner.split(",") if t.strip()]


_BROADCAST_TOK_RE = re.compile(r"_1TO\d+$")


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
        if sk == "M" and _BROADCAST_TOK_RE.search(ft):
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
            # Latency ranges render as "min-max" strings (simdref#50); the
            # float view takes the min so aggregates stay numeric.
            return float(str(v).split("-", 1)[0])
        except (TypeError, ValueError):
            continue
    return None


def _latency_for(details: dict[str, Any]) -> list[str]:
    return perf.latency_cycle_values(details.get("latencies") or [])


def _cpi_for(details: dict[str, Any], iclass: str | None = None) -> list[str]:
    return perf._cpi_values(details, iclass)


def aggregate_perf(
    record: InstructionRecord,
    *,
    mode: str = "avg",
    include_modeled: bool = False,
) -> PerfSummary:
    """Aggregate latency and CPI across ``record``'s measured microarches."""
    arch_details = record.arch_details or {}
    iclass = perf.record_iclass(record)

    def collect(kinds: tuple[str, ...]) -> tuple[list[float], list[float], list[str]]:
        lats: list[float] = []
        cpis: list[float] = []
        archs: list[str] = []
        for core, details in arch_details.items():
            kind = details.get("source_kind") or "measured"
            if kind not in kinds:
                continue
            lat = _per_arch_value(details, _latency_for)
            cpi = _per_arch_value(details, lambda d: _cpi_for(d, iclass))
            if lat is None and cpi is None:
                continue
            archs.append(core)
            if lat is not None:
                lats.append(lat)
            if cpi is not None:
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

    Missing values are ``None`` and ``source_kind`` falls back to
    ``"measured"`` when the row predates the provenance field.
    """
    details = (record.arch_details or {}).get(arch) or {}
    kind = details.get("source_kind") or "measured"
    iclass = perf.record_iclass(record)
    lat = _per_arch_value(details, _latency_for)
    cpi = _per_arch_value(details, lambda d: _cpi_for(d, iclass))
    return lat, cpi, kind


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


_SUMMARY_BITS_RE = re.compile(r"(\d+)\s*-?\s*bit", re.IGNORECASE)
_FORM_WIDTH_RE = re.compile(r"[MRI](\d+)")


def _summary_matches_form(summary: str, record: InstructionRecord) -> bool:
    """Drop obviously mislabeled SDM summaries (e.g. the generic MOV
    blurb "Move 32-bit integer operands." attached to MOV (M64, R64)).

    We only flag a summary as wrong when it declares an explicit bit-width
    that appears nowhere among the form's operand widths."""
    m = _SUMMARY_BITS_RE.search(summary or "")
    if not m:
        return True
    declared = m.group(1)
    widths: list[str] = []
    for op in record.operand_details or []:
        w = op.get("width")
        if w:
            widths.append(str(w))
    if not widths:
        key = str(getattr(record, "key", "") or "")
        widths = _FORM_WIDTH_RE.findall(key)
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
) -> str:
    """Compose the comment fragment (without the leading ``# `` marker)."""
    parts: list[str] = []
    if docs and record.summary and _summary_matches_form(record.summary, record):
        parts.append(record.summary.strip())
    if performance:
        uops_frag = ""
        if arch is not None:
            lat, cpi, kind = arch_perf(record, arch)
            tag = arch_perf_tag(arch, lat, cpi, kind)
            uops_val, microcoded_val = _uops_for(
                (record.arch_details or {}).get(arch) or {}, arch, perf.record_iclass(record)
            )
            if uops_val is not None:
                uops_frag = f" uops={uops_val}"
                if microcoded_val:
                    uops_frag += " microcoded"
        else:
            summary = aggregate_perf(record, mode=agg, include_modeled=include_modeled)
            lat, cpi = summary.latency, summary.cpi
            if summary.n_archs == 0:
                tag = "[no data]"
            elif arch is None and agg == "avg":
                tag = f"[avg of {summary.n_archs} archs, {summary.source_kind}]"
            else:
                tag = f"[{agg} of {summary.n_archs} archs, {summary.source_kind}]"
        perf_frag = f"lat={_fmt_num(lat)}c cpi={_fmt_num(cpi)}{uops_frag} {tag}"
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
) -> tuple[str, dict[str, Any] | None]:
    """Return the rendered output line and an optional JSON record.

    ``line_isa`` names the family the line's own evidence shows (a pinned
    ``--isa`` counts as one). None marks a neutral line in auto mode: the
    stream family answers first, then the family that holds rows.
    """
    if stats is not None:
        stats["parsed"] = stats.get("parsed", 0) + 1
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
        return f"{parsed.raw}   # {parsed.mnemonic}: no measured row", no_row
    # Byte identity wins over the form pick: the record is what the line
    # assembles to, and an empty row keeps the old pick unchanged.
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
            if parsed.trailing_comment:
                return parsed.raw, None
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
            return f"{parsed.raw}   {marker}", unknown_rec
        return parsed.raw, None

    if not (opts.performance or opts.docs):
        return parsed.raw, None

    annotation = format_annotation(
        record,
        performance=opts.performance,
        docs=opts.docs,
        arch=opts.arch,
        agg=opts.agg,
        include_modeled=opts.include_modeled,
    )
    if not annotation:
        return parsed.raw, None

    if opts.arch is not None:
        lat_val, cpi_val, _ = arch_perf(record, opts.arch)
        archs_used: list[str] = [opts.arch] if (opts.arch in (record.arch_details or {})) else []
    else:
        summary = aggregate_perf(record, mode=opts.agg, include_modeled=opts.include_modeled)
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
    arch_out: list[str] = []

    parsed_lines = [parse_asm_line(line, track_positions=opts.track_positions) for line in lines]
    # One llvm-mc run for the whole input: byte identity per instruction
    # line. An objdump line assembles as its mnemonic and operands only, so
    # the address and byte columns never reach the assembler.
    items = [
        (
            index,
            f"{parsed.mnemonic} {parsed.operands}".strip()
            if parsed.address is not None
            else parsed.raw,
        )
        for index, parsed in enumerate(parsed_lines)
        if parsed.kind == LineKind.INSTRUCTION
    ]
    identity = {
        index: key
        for (index, _text), key in zip(
            items,
            identity_keys([text for _index, text in items], None, conn, arch_out=arch_out),
        )
        if key
    }
    auto_isa = opts.isa not in ("x86", "arm", "riscv")
    if auto_isa:
        # auto: the input signals name the ISA, then the document arch from
        # _document_arch; x86 stays the guess when both stay silent.
        opts.isa = (
            detect_isa(parsed.raw for parsed in parsed_lines)
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
                # non-instruction lines reach the (pricier) marker regex.
                src_m = _OBJDUMP_SRC_RE.match(parsed.raw)
                if src_m:
                    pending_src_file = src_m.group("file").strip()
                    try:
                        pending_src_line = int(src_m.group("line"))
                    except ValueError:
                        pending_src_line = None
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
            # A pin forces one family; auto mode splits the batch per line.
            line_isa=opts.isa
            if not auto_isa
            else _line_isa(f"{parsed.mnemonic} {parsed.operands}"),
        )
        if collecting_json:
            if record is not None:
                json_records.append(record)
            continue
        yield out_line + "\n"

    if collecting_json:
        yield json.dumps(json_records, indent=2) + "\n"
