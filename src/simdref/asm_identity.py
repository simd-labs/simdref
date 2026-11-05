"""Byte-identity lookup: document text -> one catalog row per source line.

Runs ``llvm-mc -show-encoding`` once per document (triple and ``-mattr`` per
arch), maps each line to its encoded bytes, then maps the bytes to one
catalog row:

- x86: bytes -> XED iform (decoder-only CLI), then the uops.info variant
  attributes stored on the row narrow an iform pool (eosz, agen, high8,
  immzero, mask, zeroing).
- AArch64: 32-bit word -> rows whose encodeset (mask, match) accepts it,
  then the subset rule: drop a candidate strictly covered by another.
- RISC-V: 16/32-bit word -> rows whose YAML ``encoding.match`` pattern
  accepts it (scoped to the document base), then line-mnemonic equality
  resolves the documented ``c.*`` bit collisions, and a 16-bit tie the
  text does not name takes the xlen whose llvm-mc decode prints the
  line's own mnemonic (c.jal versus c.addiw).

No mnemonic or iform table and no fallback matcher (spec R2): a line ends
as one row, or with a reason why there is none.

Tool paths: ``SIMDREF_LLVM_MC`` / ``SIMDREF_XED`` override
``simdref_llvm_mc.path(...)``. That pinned dependency ships both tools on
the platforms it has wheels for; everywhere else point the env vars at a
23.1.4 ``llvm-mc`` and the XED decode CLI.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Measured invocation of the 144/401/703 corpora probes (asm-pip/probe).
_ARCH_CFG: dict[str, dict] = {
    "x86": {
        "triple": "x86_64-unknown-linux-gnu",
        "mattr": "+avx512f,+avx512bw,+avx512dq,+avx512vl,+avx512vbmi,+avx512vbmi2,"
        "+avx512vnni,+avx512bitalg,+avx512vpopcntdq,+avx512ifma,+vaes,+vpclmulqdq,"
        "+avx,+avx2,+fma,+f16c,+bmi,+bmi2,+lzcnt,+movbe",
    },
    "aarch64": {
        "triple": "aarch64-unknown-linux-gnu",
        "mattr": "+sve,+sve2,+sve2p1,+sme,+sme2,+sme2p1,+lse,+gcs,+aes",
    },
    "arm": {"triple": "armv7-unknown-linux-gnueabihf", "mattr": "+neon"},
    "riscv32": {
        "triple": "riscv32-unknown-linux-gnu",
        "mattr": "+m,+a,+f,+d,+c,+v,+zba,+zbb,+zbs",
    },
    "riscv64": {
        "triple": "riscv64-unknown-linux-gnu",
        "mattr": "+m,+a,+f,+d,+c,+v,+zba,+zbb,+zbs,+zvbb,+zvkb,+zvkned,+zfh",
    },
}
_X86_SYNTAX_FLAG = {"att": "-x86-asm-syntax=att", "intel": "-x86-asm-syntax=intel"}
# uops.info eosz attribute values per effective operand size.
_EOSZ_ATTR = {16: "1", 32: "2", 64: "3"}
# The four x86 registers uops.info tags as high-byte (its ``high8`` attr).
_HIGH8_REGS = frozenset({"AH", "BH", "CH", "DH"})

_COMMENT_STARTS = ("#", "//", "/*", "@", ";")
_ERR_RE = re.compile(r"^<stdin>:(\d+):(\d+): (?:error|warning): (.*)$", re.M)
# The encoding comment llvm-mc appends ends its echo line. Requiring the end
# keeps a string echo (".ascii \" # encoding: [0x90]\"") from faking bytes.
_ENC_RE = re.compile(r"encoding: \[([^\]]*)\]\s*$")


def _mcinst_opcode(line: str) -> str | None:
    """The MCInst opcode on an llvm-mc -show-inst comment line, else None.

    The stripped line must start with the marker, so a string echo cannot
    fake an opcode (RV tie-break input). The name stops at whitespace or the
    closing bracket: an operand-less MCInst closes on its opcode line
    ("# <MCInst #1 FENCE_I>").
    """
    s = line.strip()
    if not s.startswith("# <MCInst #"):
        return None
    fields = s[len("# <MCInst #") :].split()
    if len(fields) > 1 and fields[0].isdigit():
        return fields[1].rstrip(">") or None
    return None


_CONT_RE = re.compile(r"^\s+(#|//|@|;)")
_LABEL_RE = re.compile(r"^[\w.$]+:")

_ROW_COLUMNS = "db_key, key, architecture, mnemonic, form, enc_bits, enc_mask, enc_match, enc_extra, iform, split_attrs"


class ToolUnavailableError(RuntimeError):
    """A required external tool (llvm-mc or the XED decoder) is missing."""


@dataclass(frozen=True)
class Row:
    """One catalog row, as the lookup reads it (columns only, no payload)."""

    db_key: str
    key: str
    architecture: str
    mnemonic: str
    form: str
    enc_bits: int
    enc_mask: int
    enc_match: int
    enc_extra: tuple
    iform: str
    split_attrs: dict


@dataclass
class LineResult:
    """Identity of one source line: a row, or why there is none."""

    n: int
    text: str
    status: str  # "row" | "no-row" | "not-an-instruction"
    reason: str = ""
    row: Row | None = None
    candidates: list[str] = field(default_factory=list)


def _tool_path(env_var: str, tool: str) -> str | None:
    override = os.environ.get(env_var)
    if override:
        # A dead override answers like no tool: callers degrade to their
        # no-tool path instead of crashing (review REV3 finding 1).
        return override if os.access(override, os.X_OK) else None
    try:
        import simdref_llvm_mc
    except ImportError:
        return None
    try:
        return str(simdref_llvm_mc.path(tool))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# llvm-mc: one run per document, one result per source line
# ---------------------------------------------------------------------------


# A real llvm-mc diagnostic block ends in a caret line ("^", "^~~~", "  ^").
_CARET_RE = re.compile(r"^[ ~]*\^+[~]*$")


def _classify_line(text: str) -> str | None:
    """NOT-AN-INSTRUCTION kinds: blank, comment, label, directive."""
    stripped = text.strip()
    if not stripped:
        return "blank"
    if stripped.startswith(_COMMENT_STARTS):
        return "comment"
    label = _LABEL_RE.match(stripped)
    if label:
        stripped = stripped[label.end() :].strip()
        if not stripped:
            return "label"
    if stripped.startswith("."):
        return "directive"
    return None


# One x86 mode flag; the output file re-emits it when the assembler executes
# it. Only the bare echo counts: a label (".code16:") and a symbol assignment
# print (".code16 = 1", from ".set .code16, 1") echo user-chosen names at the
# line start in the same stream, and user text must never set the mode.
_MODE_DIR_RE = re.compile(r"^\.(code16gcc|code16|code32|code64)$")
_MODE_BITS = {"code16": "16", "code16gcc": "16", "code32": "32", "code64": "64"}


def _parse_echo_blocks(listing: str) -> list[tuple[str, str | None, str | None, str]]:
    """Output blocks, each with the mode llvm-mc assembled it in.

    The output file re-emits every active `.code` flag in execution order
    (probe-modeflags A-H: a flag in an `.if 0` body, a macro body or a
    comment does not emit), so the mode comes from what the assembler did,
    never from the source text."""
    blocks: list[list] = []
    mode = "64"  # the x86_64 triple starts in 64-bit mode
    for line in listing.splitlines():
        if _CONT_RE.match(line):
            if not blocks:
                blocks.append(["", None, None, mode])
        else:
            found = _MODE_DIR_RE.match(line.strip())
            if found:
                mode = _MODE_BITS[found.group(1)]
            blocks.append([line.rstrip(), None, None, mode])
        block = blocks[-1]
        if block[1] is None:
            opcode = _mcinst_opcode(line)
            if opcode is not None:
                block[1] = opcode
        if block[2] is None:
            match = _ENC_RE.search(line)
            if match:
                block[2] = match.group(1).strip()
    return [tuple(block) for block in blocks]


# Probe results are deterministic per (lines, arch, syntax, tool, mattr), so a
# document probed by both identify_lines and isa_by_acceptance runs llvm-mc
# once per triple, not twice (perf gate: 8 runs per document became 4).
_PROBE_CACHE: dict[tuple, list[dict]] = {}


def _probe_lines(
    lines: list[str], arch: str, syntax: str, llvm_mc: str, *, mattr: str | None = None
) -> list[dict]:
    key = (tuple(lines), arch, syntax, llvm_mc, mattr)
    cached = _PROBE_CACHE.get(key)
    if cached is None:
        if len(_PROBE_CACHE) >= 16:
            _PROBE_CACHE.clear()
        cached = _PROBE_CACHE[key] = _probe_lines_uncached(
            lines, arch, syntax, llvm_mc, mattr=mattr
        )
    return cached


def _probe_lines_uncached(
    lines: list[str], arch: str, syntax: str, llvm_mc: str, *, mattr: str | None = None
) -> list[dict]:
    """Run llvm-mc once per document; map each line to bytes or a reason.

    The map comes from llvm-mc's own output, never from counting: a unique
    label (the line marker) stands before every source line and echoes back
    among the output blocks, so the blocks after marker ``i`` and before the
    next forward marker belong to source line ``i``. One line can yield
    several blocks (``;`` statements, label + instruction, macro expansion)
    or none (comment, macro body, ``.endm``). Only a line with exactly one
    encoding block and its marker echoed exactly once has proven bytes;
    every other instruction line reports no row (review E1/E2)."""
    cfg = _ARCH_CFG[arch]
    cmd = [llvm_mc, f"--triple={cfg['triple']}", "-show-encoding", "-show-inst"]
    if mattr or cfg["mattr"]:
        cmd.append(f"-mattr={mattr or cfg['mattr']}")
    if arch == "x86":
        cmd.append(_X86_SYNTAX_FLAG[syntax])

    # The nonce keeps a user label from posing as a marker.
    nonce = os.urandom(4).hex()
    markers = [f".L__simdref_lm_{nonce}_{i}:" for i in range(1, len(lines) + 1)]
    work: list[str] = []
    for marker, text in zip(markers, lines):
        work.append(marker)
        work.append(text)
    # The trailing newline keeps trailing blank source lines in the input
    # (a bare "\n".join swallows them and the whole map fails, review E2).
    # Only the -o stream is parsed: .print writes user text to stdout, so
    # stdout is never read (Sol r3 P2). The descriptor opened before the run
    # still reads the listing after a failed run, whose output name llvm-mc
    # deletes: the inode stays readable (Sol r4 F3), so no hard link.
    with tempfile.TemporaryDirectory() as td:
        out_file = Path(td) / "out.s"
        with open(out_file, "w+") as out:
            run = subprocess.run(
                cmd + ["-o", str(out_file)],
                input="\n".join(work) + "\n",
                capture_output=True,
                text=True,
            )
            out.seek(0)
            listing = out.read()

    errors: dict[int, str] = {}
    stderr_lines = run.stderr.splitlines()
    for index, line in enumerate(stderr_lines):
        match = _ERR_RE.match(line)
        if match is None or index + 2 >= len(stderr_lines):
            continue
        claimed = int(match.group(1))
        # Error lines number the marked document: 2i-1 is the marker of
        # source line i, 2i is the source line itself. Only the exact llvm-mc
        # diagnostic shape counts: the diagnostic, the echo of the line it
        # names, then the caret line. A source line or string text can print
        # a diagnostic-shaped line of its own (Sol r4 F2) and that must never
        # decline another line.
        if not 0 < claimed <= len(work):
            continue
        if stderr_lines[index + 1].rstrip() != work[claimed - 1].rstrip():
            continue
        if not _CARET_RE.match(stderr_lines[index + 2]):
            continue
        errors.setdefault(claimed, match.group(3))
    blocks = _parse_echo_blocks(listing)

    index_of = {marker.strip(): i for i, marker in enumerate(markers, 1)}
    seen: dict[int, int] = {}
    buckets: dict[int, list] = {}
    current = 0
    for block in blocks:
        marker = index_of.get(block[0].strip())
        if marker is None:
            buckets.setdefault(current, []).append(block)
            continue
        seen[marker] = seen.get(marker, 0) + 1
        # Only a forward marker moves the attribution: a marker echoed again
        # comes from a macro or .rept body expansion, and those blocks belong
        # to the line that triggered the expansion, never to the body line.
        if marker > current:
            current = marker

    results: list[dict] = []
    kind_of: dict[str, str | None] = {}
    for i, text in enumerate(lines, 1):
        if text not in kind_of:
            kind_of[text] = _classify_line(text)
        kind = kind_of[text]
        error = errors.get(2 * i)
        if error is not None:
            results.append({"n": i, "text": text, "status": "REJECTED", "error": error})
            continue
        bucket = buckets.get(i, [])
        enc = [block for block in bucket if block[2] is not None]
        proven = seen.get(i) == 1 and errors.get(2 * i - 1) is None
        if kind in ("blank", "label", "directive"):
            # These kinds are certain from the text; a leaked block from a
            # line with a swallowed marker never becomes this line's row.
            results.append({"n": i, "text": text, "status": "NOT-AN-INSTRUCTION", "kind": kind})
            continue
        if len(enc) == 1 and proven:
            _echo, opcode, hexbytes, mode = enc[0]
            results.append(
                {
                    "n": i,
                    "text": text,
                    "status": "OK",
                    "bytes": hexbytes,
                    "opcode": opcode,
                    "mode": mode,
                }
            )
            continue
        if not enc and kind is not None:
            results.append({"n": i, "text": text, "status": "NOT-AN-INSTRUCTION", "kind": kind})
            continue
        if not enc and len(bucket) == 1:
            results.append(
                {"n": i, "text": text, "status": "NOT-AN-INSTRUCTION", "kind": "echo-no-inst"}
            )
            continue
        if not proven:
            reason = f"line mapping unverified (marker echoes {seen.get(i, 0)})"
        elif len(enc) > 1:
            reason = f"line mapping unverified ({len(enc)} instruction blocks on one line)"
        else:
            reason = "line mapping unverified (no instruction echo for the line)"
        results.append({"n": i, "text": text, "status": "REJECTED", "error": reason})
    return results


def _probe_lines_split(lines: list[str], arch: str, syntax: str, llvm_mc: str) -> list[dict]:
    """Probe each line with the encoding class its spelling names.

    A RISC-V line whose mnemonic does not start with ``c.`` must identify by
    its uncompressed encoding, so it is probed without ``+c``: with ``+c``
    llvm-mc compresses e.g. ``addi s0, sp, 16`` to the C.ADDI4SPN bytes and
    the bytes name the wrong row. ``+c`` stays on only for ``c.`` lines.
    """
    if not arch.startswith("riscv"):
        return _probe_lines(lines, arch, syntax, llvm_mc)
    out: list[dict | None] = [None] * len(lines)
    groups: dict[bool, list[int]] = {True: [], False: []}
    for index, text in enumerate(lines):
        parts = text.strip().split(None, 1)
        head = parts[0].casefold() if parts else ""
        groups[head.startswith("c.")].append(index)
    for is_compressed, indexes in groups.items():
        if not indexes:
            continue
        mattr = None
        if not is_compressed:
            mattr = ",".join(t for t in _ARCH_CFG[arch]["mattr"].split(",") if t != "+c")
        sub = _probe_lines([lines[i] for i in indexes], arch, syntax, llvm_mc, mattr=mattr)
        for index, entry in zip(indexes, sub):
            merged = dict(entry)
            merged["n"] = index + 1
            out[index] = merged
    return out


_BYTE_RE = re.compile(r"0x([0-9a-fA-F]{2})(?:'A')?$")
_BITS_RE = re.compile(r"0b([01Aa]{8})$")


def _covered_mask(hexbytes: str | None, *, marked_byte_covers: bool = True) -> int:
    """Bit mask of fixup-covered bits (bit i of the little-endian word).

    Mirrors the ``_bytes_to_words`` token shapes. llvm-mc names bits only in
    the binary shape; a whole byte, a marked byte and an unknown shape name
    none of their bits, so all 8 count as covered. With
    ``marked_byte_covers=False`` a marked hex byte keeps its printed digits
    as values, and only the bare placeholder tokens cover: that is the
    known-bit view the opcode decider compares against (review F1)."""
    if not hexbytes:
        return 0
    mask = 0
    for index, token in enumerate(hexbytes.split(",")):
        token = token.strip()
        bits = 0xFF
        match = _BITS_RE.match(token)
        if match:
            bits = sum(1 << (7 - pos) for pos, ch in enumerate(match.group(1)) if ch in "Aa")
        elif _BYTE_RE.match(token) and (not marked_byte_covers or not token.endswith("'A'")):
            bits = 0
        mask |= bits << (8 * index)
    return mask


def _bytes_to_words(hexbytes: str | None) -> bytes:
    """llvm-mc prints fixup-covered bits as 'A' (whole byte, nibble, or bit);
    zero-fill them. The ISA patterns never fix those bits (they are operand
    fields), so the zeroed bits cannot break a (mask, match) test."""
    if not hexbytes:
        return b""
    out = bytearray()
    for token in hexbytes.split(","):
        token = token.strip()
        if token in ("A", "'A'"):
            out.append(0)
            continue
        match = _BYTE_RE.match(token) or _BITS_RE.match(token)
        if not match:
            # shortcut: unknown llvm-mc byte format zero-fills the whole byte,
            # upgrade the shapes if llvm-mc prints new fixup forms
            out.append(0)
            continue
        digits = match.group(1)
        if token.startswith("0b"):
            out.append(int(digits.replace("A", "0").replace("a", "0"), 2))
        else:
            out.append(int(digits, 16))
    return bytes(out)


# ---------------------------------------------------------------------------
# catalog side: rows, pattern buckets, iform pools
# ---------------------------------------------------------------------------


class _Catalog:
    def __init__(self, path: Path, arch: str):
        self.rows: list[Row] = []
        self.a64_patterns: list[tuple[int, int, Row]] = []
        self.rv_patterns: dict[str, list[tuple[int, int, int, Row]]] = {}
        self.iform_pools: dict[str, list[Row]] = {}
        self.keyless = False
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            columns = {item[1] for item in conn.execute("PRAGMA table_info(instructions_data)")}
            if "enc_mask" not in columns:
                self.keyless = True
                return
            want = {
                "x86": "x86",
                "aarch64": "arm",
                "arm": "arm",
                "riscv32": "riscv",
                "riscv64": "riscv",
            }[arch]
            for raw in conn.execute(
                f"SELECT {_ROW_COLUMNS} FROM instructions_data WHERE architecture = ?", (want,)
            ):
                extra = raw[8]
                splits = raw[10]
                row = Row(
                    db_key=raw[0],
                    key=raw[1],
                    architecture=raw[2],
                    mnemonic=raw[3],
                    form=raw[4],
                    enc_bits=raw[5] or 0,
                    enc_mask=raw[6] or 0,
                    enc_match=raw[7] or 0,
                    # Most rows carry no alternate patterns and no split
                    # attributes: skip the JSON parse for the empty columns
                    # (json.loads("[]") == (), json.loads("{}") == {}).
                    enc_extra=(
                        ()
                        if not extra or extra == "[]"
                        else tuple(tuple(item) for item in json.loads(extra))
                    ),
                    iform=raw[9] or "",
                    split_attrs=({} if not splits or splits == "{}" else json.loads(splits)),
                )
                self.rows.append(row)
        finally:
            conn.close()
        if arch == "aarch64":
            self.a64_patterns = [
                (row.enc_mask, row.enc_match, row) for row in self.rows if row.enc_bits == 32
            ]
        elif arch.startswith("riscv"):
            for row in self.rows:
                for entry in row.enc_extra or ((None, row.enc_bits, row.enc_mask, row.enc_match),):
                    base, bits, mask, match = entry[0], entry[1], entry[2], entry[3]
                    self.rv_patterns.setdefault(str(base or ""), []).append(
                        (bits, mask, match, row)
                    )
        elif arch == "x86":
            for row in self.rows:
                if row.iform:
                    self.iform_pools.setdefault(row.iform, []).append(row)
        self.a64_buckets = _bucketize(self.a64_patterns, shift=24) if arch == "aarch64" else {}

    def rv_patterns_for(self, base: str) -> list[tuple[int, int, int, Row]]:
        out: list[tuple[int, int, int, Row]] = []
        for key, entries in self.rv_patterns.items():
            # RV32E/RV64E keys serve their whole base family; a bare key is
            # base-agnostic (flat encoding block).
            if not key or key[:4] == base[:4]:
                out.extend(entries)
        return out


def _bucketize(
    patterns: list[tuple[int, int, Row]], shift: int
) -> dict[int, list[tuple[int, int, Row]]]:
    """8-bit opcode-window buckets; each pattern lands in every window value
    its own bits allow, so a query is one dict lookup plus a linear scan."""
    buckets: dict[int, list[tuple[int, int, Row]]] = {}
    for mask, match, row in patterns:
        window_mask = (mask >> shift) & 0xFF
        window_match = (match >> shift) & 0xFF
        free = (~window_mask) & 0xFF
        subset = free
        while True:
            buckets.setdefault(window_match | subset, []).append((mask, match, row))
            if subset == 0:
                break
            subset = (subset - 1) & free
    return buckets


def _refine_subset(hits: list[tuple[int, int, Row]]) -> list[tuple[int, int, Row]]:
    """Drop a candidate strictly covered by another candidate's pattern.

    Derived from the encodesets themselves: a covered pattern can be a true
    decode only where the finer encoding is invalid for that word, and the
    corpora show no such case (DESIGN.md).
    shortcut: sibling forms of one encoding that a mask cannot separate report
    one covered row (e.g. PTRUE arrangement forms); upgrade when a finer key
    appears in the upstream data.
    """
    keep = []
    for mask_b, match_b, _row_b in hits:
        covered = False
        for mask_a, match_a, _row_a in hits:
            if mask_a == mask_b:
                continue
            if (mask_b & mask_a) == mask_b and (match_b & mask_b) == (match_a & mask_b):
                covered = True
                break
        if not covered:
            keep.append((mask_b, match_b, _row_b))
    return keep


# ---------------------------------------------------------------------------
# x86: XED decode + the uops.info attribute splits
# ---------------------------------------------------------------------------


def _xed_facts(hexes: list[str], xed: str) -> list[dict]:
    run = subprocess.run([xed], input="\n".join(hexes).encode(), capture_output=True)
    if run.returncode != 0:
        raise ToolUnavailableError(f"{xed} exited {run.returncode}: {run.stderr.decode()[:200]}")
    lines = run.stdout.decode().splitlines()
    if len(lines) != len(hexes):
        raise ToolUnavailableError(f"{xed} returned {len(lines)} lines for {len(hexes)} inputs")
    facts = []
    for line in lines:
        parts = line.split()
        if not parts:
            facts.append({"error": "EMPTY_OUTPUT"})
            continue
        if parts[0] == "DECODE_ERROR":
            facts.append({"error": parts[1] if len(parts) > 1 else "DECODE_ERROR"})
            continue
        kv = dict(part.split("=", 1) for part in parts[3:] if "=" in part)
        facts.append(
            {
                "iclass": parts[1],
                "iform": parts[2],
                "eosz": int(kv["eosz"]),
                "regs": kv["regs"].split(","),
                "imm": int(kv["imm"]),
                "base": kv["base"],
                "idx": kv["idx"],
                "scale": int(kv["scale"]),
                "displen": int(kv["displen"]),
                "zero": int(kv.get("zero", "0")),
            }
        )
    return facts


def _agen_shape(value: str) -> tuple[str, str, int]:
    """Parse one uops.info ``agen`` value into (base, index, disp-bits).

    The value strings come from the catalog (XML attribute values stored at
    ingest). Review F1: the old builder emitted "_IS_D32" where the
    vocabulary says "IS_D32"."""
    base = idx = ""
    disp = 0
    for part in value.split("_"):
        if part in ("B", "R"):
            base = part
        elif part in ("I", "IS"):
            idx = part
        elif part in ("D8", "D32"):
            disp = int(part[1:])
        else:
            raise ValueError(f"unknown agen token {part!r} in {value!r}")
    return base, idx, disp


def _agen_facts(d: dict) -> tuple[str, str, int]:
    """XED-decoded base/index/scale/disp facts in the same shape space."""
    base = "R" if d["base"] == "RIP" else "B" if d["base"] != "INVALID" else ""
    idx = "" if d["idx"] == "INVALID" else ("IS" if d["scale"] != 1 else "I")
    return base, idx, d["displen"] * 8


def _x86_pick(pool: list[Row], d: dict, imm_unknown: bool = False, text: str = "") -> list[Row]:
    """Narrow an iform pool with the uops.info variant attributes stored on
    each row against the XED-decoded facts. Data tables on both sides.

    ``imm_unknown``: a fixup placeholder covers the immediate (audit A1), so
    the immzero split cannot narrow and the caller treats its rows as a tie.
    ``text`` is the source line: the iform and the attributes leave M32_1toN
    and plain memory rows tied, so the ``{1toN}`` spelling splits them. An
    empty text cannot split them."""
    cur = list(pool)

    def filt(stage_ok, pred):
        nonlocal cur
        if len(cur) <= 1 or not stage_ok:
            return
        picked = [row for row in cur if pred(row.split_attrs)]
        if 0 < len(picked) < len(cur):
            cur = picked

    filt(
        any("eosz" in row.split_attrs for row in cur),
        lambda a: "eosz" not in a or a["eosz"] == _EOSZ_ATTR.get(d["eosz"]),
    )
    want_agen = _agen_facts(d)
    filt(
        any("agen" in row.split_attrs for row in cur),
        lambda a: "agen" not in a or _agen_shape(a["agen"]) == want_agen,
    )
    want_high8 = frozenset(f"REG{i}" for i, reg in enumerate(d["regs"]) if reg in _HIGH8_REGS)
    filt(
        any("high8" in row.split_attrs for row in cur),
        lambda a: frozenset(x for x in a.get("high8", "").split(",") if x) == want_high8,
    )
    imm_zero = d["imm"] == 0
    filt(
        not imm_unknown and any("immzero" in row.split_attrs for row in cur),
        lambda a: a.get("immzero", "1" if imm_zero else "0") == ("1" if imm_zero else "0"),
    )
    masked = any(reg.startswith("K") and reg != "K0" for reg in d["regs"])
    filt(
        any("mask" in row.split_attrs for row in cur),
        lambda a: a.get("mask", "1" if masked else "0") == ("1" if masked else "0"),
    )
    filt(
        any("zeroing" in row.split_attrs for row in cur),
        lambda a: a.get("zeroing", "1" if d["zero"] else "0") == ("1" if d["zero"] else "0"),
    )
    if text and len(cur) > 1:
        picked = [row for row in cur if ("_1to" in row.key) == ("{1to" in text)]
        if picked:
            cur = picked
    return cur


# ---------------------------------------------------------------------------
# document lookup
# ---------------------------------------------------------------------------

_CATALOG_CACHE: dict[tuple[str, str, int], _Catalog] = {}

# The span verdict is lexical, so one llvm-mc run over one .include per
# quoted line answers every line at once. Trust the first error of a run
# only: no line before it errored (and a line whose lexer state crosses
# buffers always errors itself), while lines after it ride the error
# recovery (an open string crosses include buffers and swallows following
# lines, probe B4), so the batch restarts after the erroring line.
_QUOTE_SPAN_CACHE: dict[tuple[str, str], bool] = {}
_UNTERM = "unterminated string constant"
_ERR_LINE_RE = re.compile(r"^(\S+):(\d+):(\d+): error: (.*)$")


def _string_spans_lines(lines: list[str], arch: str, syntax: str, llvm_mc: str | None) -> bool:
    """True when a quoted line leaves a string open at its end.

    Each quoted line is assembled in its own one-line buffer and llvm-mc's
    own lexer decides (Sol P1): a quote in a comment or a character constant
    balances a string delimiter for any counter. 'unterminated string
    constant' at the end of the one-line buffer means the string crosses the
    line. Any other error on the line alone keeps the document; a spoofed
    diagnostic that quotes the message can only decline (the safe direction).
    """
    quoted = []
    seen = set()
    for text in lines:
        if '"' in text and text not in seen:
            seen.add(text)
            quoted.append(text)
    if not quoted:
        return False
    if llvm_mc is None:
        llvm_mc = _tool_path("SIMDREF_LLVM_MC", "llvm-mc")
    if llvm_mc is None:
        raise ToolUnavailableError("llvm-mc not found (set SIMDREF_LLVM_MC)")
    cfg = _ARCH_CFG[arch]
    cmd = [llvm_mc, f"--triple={cfg['triple']}"]
    if cfg["mattr"]:
        cmd.append(f"-mattr={cfg['mattr']}")
    if arch == "x86":
        cmd.append(_X86_SYNTAX_FLAG[syntax])
    pending = []
    for text in quoted:
        spans = _QUOTE_SPAN_CACHE.get((arch, text))
        if spans:
            return True
        if spans is None:
            pending.append(text)
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for index, text in enumerate(pending):
            path = Path(td) / f"q{index}.s"
            path.write_text(text + "\n")
            paths.append(path)
        owner = {str(path): index for index, path in enumerate(paths)}
        start = 0
        while start < len(pending):
            top = Path(td) / "top.s"
            top.write_text("".join(f'.include "{path}"\n' for path in paths[start:]))
            run = subprocess.run(cmd + [str(top)], capture_output=True, text=True)
            errors = []
            for line in run.stderr.splitlines():
                match = _ERR_LINE_RE.match(line)
                if match:
                    errors.append((owner.get(match.group(1)), match.group(4)))
            if not errors:
                for text in pending[start:]:
                    _QUOTE_SPAN_CACHE[(arch, text)] = False
                return False
            anchor = errors[0][0]
            if anchor is None:
                # shortcut: a diagnostic outside our include files (a quoted
                # .include of a real file that errors there), decline like a
                # spoof; upgrade when that shows up in real documents
                return True
            # The anchor line parsed with no error before it, so its error
            # set is its lone run's set (probe B6: a lone-line error before
            # the string must not hide it). 'unterminated string constant'
            # anywhere in that set is the lone verdict for the line.
            spans = any(index == anchor and message == _UNTERM for index, message in errors)
            _QUOTE_SPAN_CACHE[(arch, pending[anchor])] = spans
            if spans:
                return True
            for text in pending[start:anchor]:
                _QUOTE_SPAN_CACHE[(arch, text)] = False
            start = anchor + 1
    return False


def identify(
    document: str,
    arch: str,
    syntax: str = "att",
    *,
    catalog_path: str | os.PathLike | None = None,
    llvm_mc: str | None = None,
    xed: str | None = None,
) -> list[LineResult]:
    """Map every line of ``document`` to one catalog row, or a reason.

    ``arch`` comes from the file type: ``x86``, ``aarch64``, ``arm``,
    ``riscv32`` or ``riscv64``. ``syntax`` is ``att`` or ``intel`` (x86 only).
    """
    if arch not in _ARCH_CFG:
        raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(_ARCH_CFG)}")
    if arch == "x86" and syntax not in _X86_SYNTAX_FLAG:
        raise ValueError(f"unknown x86 syntax {syntax!r}; expected att or intel")

    if catalog_path is None:
        from simdref.storage import SQLITE_PATH

        catalog_path = SQLITE_PATH
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)
    cache_key = (str(catalog_path), arch, catalog_path.stat().st_mtime_ns)
    catalog = _CATALOG_CACHE.get(cache_key)
    if catalog is None:
        catalog = _CATALOG_CACHE[cache_key] = _Catalog(catalog_path, arch)

    lines = document.splitlines()
    if _string_spans_lines(lines, arch, syntax, llvm_mc):
        # One open string poisons the listing and the diagnostics of the
        # whole document (Sol r4 F1/F2, r5 P2): no line gets byte identity
        # and the older matcher answers every line.
        return [
            LineResult(n=n, text=text, status="no-row", reason="quoted string spans lines")
            for n, text in enumerate(lines, 1)
        ]
    if catalog.keyless:
        # A schema-13 catalog carries no encoding keys (C4): every
        # instruction line reports that one reason, no tool run needed.
        results = []
        for i, text in enumerate(lines, 1):
            kind = _classify_line(text)
            results.append(
                LineResult(
                    n=i,
                    text=text,
                    status="not-an-instruction" if kind else "no-row",
                    reason=kind or "catalog lacks encoding keys (schema 13)",
                )
            )
        return results

    if llvm_mc is None:
        llvm_mc = _tool_path("SIMDREF_LLVM_MC", "llvm-mc")
    if llvm_mc is None:
        raise ToolUnavailableError("llvm-mc not found (set SIMDREF_LLVM_MC)")
    if arch == "x86" and xed is None:
        xed = _tool_path("SIMDREF_XED", "xed")
    if arch == "x86" and xed is None:
        raise ToolUnavailableError("xed decoder not found (set SIMDREF_XED)")

    # One shared no-compression rule for every path (identify, identify_lines,
    # isa_by_acceptance): a RISC-V source line whose mnemonic does not start
    # with "c." probes without +c, so the bytes name the uncompressed row.
    probed = _probe_lines_split(lines, arch, syntax, llvm_mc)
    results = [
        LineResult(
            n=entry["n"],
            text=entry["text"],
            status="not-an-instruction" if entry["status"] == "NOT-AN-INSTRUCTION" else "no-row",
            reason=(
                entry.get("kind", "")
                if entry["status"] == "NOT-AN-INSTRUCTION"
                else entry.get("error", "rejected by llvm-mc")
                if entry["status"] == "REJECTED"
                else ""
            ),
        )
        for entry in probed
    ]

    if arch == "arm":
        # No arm32 rows carry a byte key (spec K4: "arm32 (none usable
        # today)"), and A32/T32 patterns overlap, so no lookup runs.
        for entry, result in zip(probed, results):
            if entry["status"] == "OK":
                result.status = "no-row"
                result.reason = "no arm32 rows carry a byte key"
        return results

    if arch == "x86":
        _identify_x86(probed, results, catalog, xed)
    else:
        _identify_words(probed, results, catalog, arch)
    return results


# The catalog has no xlen field (the unified-db definedBy.xlen drops at
# ingest), so a 16-bit word like 0x2001 ties c.jal (xlen 32) against c.addiw
# (xlen 64). The xlen comes from the decode itself, never from a row list:
# decode the bytes with llvm-mc for riscv32 and riscv64 and keep the decode
# whose printed mnemonic equals the line's own text mnemonic (objdump prints
# the expanded or alias name too: jal, flw, sext.w). Both or neither match
# keeps the rv64 result (the current default).
def _rv64_only_ext(line: str) -> str | None:
    """The extension name in an "'X' is only supported for 'rv64'" error line.

    Returns None for any other line. The name sits after the last single quote
    before the fixed tail, so a prefix like "error: ..." is ignored.
    """
    head, sep, _tail = line.partition("' is only supported for 'rv64'")
    if not sep:
        return None
    return head.rpartition("'")[2] or None


_RV32_MATTR_CACHE: dict[tuple[str, str], str] = {}


def _rv32_decode_mattr(llvm_mc: str) -> str:
    """The riscv32 decode -mattr: the riscv64 one minus rv64-only extensions.

    Derived from llvm-mc's own errors (never hand-listed): the tool rejects
    an rv64-only extension on the rv32 triple by name. shortcut: an extension
    named differently in the error keeps its token, upgrade if a tool build
    rejects it silently."""
    base = _ARCH_CFG["riscv64"]["mattr"]
    cached = _RV32_MATTR_CACHE.get((llvm_mc, base))
    if cached is None:
        tokens = base.split(",")
        while True:
            run = subprocess.run(
                [
                    llvm_mc,
                    f"--triple={_ARCH_CFG['riscv32']['triple']}",
                    f"-mattr={','.join(tokens)}",
                    "-show-encoding",
                ],
                input="",
                capture_output=True,
                text=True,
            )
            drop = {
                name.casefold()
                for line in run.stderr.splitlines()
                for name in [_rv64_only_ext(line)]
                if name
            }
            if not drop:
                break
            remaining = [token for token in tokens if token.lstrip("+").casefold() not in drop]
            # guard: the tool names an extension absent from the token list,
            # so nothing shrank; break here or the loop spins forever
            if len(remaining) == len(tokens):
                break
            tokens = remaining
        cached = _RV32_MATTR_CACHE[(llvm_mc, base)] = ",".join(tokens)
    return cached


def _parse_disasm(listing: str) -> dict[bytes, tuple[str, str]]:
    """One (printed mnemonic, MCInst opcode) per decoded word.

    The encoding comment keys every block to its input word, so an invalid
    word (no block) never shifts the map (spike tie P5)."""
    found: dict[bytes, tuple[str, str]] = {}
    raw = None
    for line in listing.splitlines():
        opcode = _mcinst_opcode(line)
        if opcode is not None and raw is not None:
            found[raw] = (found[raw][0], opcode)
            continue
        if not line.strip() or _CONT_RE.match(line):
            continue
        enc = _ENC_RE.search(line)
        raw = _bytes_to_words(enc.group(1)) if enc is not None else None
        if raw is not None:
            found.setdefault(raw, (line.split()[0], ""))
    return found


_RV_DECODE_CACHE: dict[tuple[str, int, bytes], tuple[str, str] | None] = {}


def _rv_xlen_decodes(
    raws: list[bytes], llvm_mc: str, xlen: int
) -> dict[bytes, tuple[str, str] | None]:
    """Disassemble each 16-bit word at one xlen: (mnemonic, opcode) or None.

    One llvm-mc run covers every word the caller could not decide (the cost
    gate: only an undecided RVC line pays the decode)."""
    missing = [raw for raw in dict.fromkeys(raws) if (llvm_mc, xlen, raw) not in _RV_DECODE_CACHE]
    if missing:
        mattr = _ARCH_CFG["riscv64"]["mattr"] if xlen == 64 else _rv32_decode_mattr(llvm_mc)
        hexed = [" ".join(f"0x{byte:02x}" for byte in raw) for raw in missing]
        run = subprocess.run(
            [
                llvm_mc,
                f"--triple={_ARCH_CFG[f'riscv{xlen}']['triple']}",
                f"-mattr={mattr}",
                "-disassemble",
                "-show-inst",
                "-show-encoding",
            ],
            input="\n".join(hexed) + "\n",
            capture_output=True,
            text=True,
        )
        found = _parse_disasm(run.stdout)
        for raw in missing:
            _RV_DECODE_CACHE[(llvm_mc, xlen, raw)] = found.get(raw)
    return {raw: _RV_DECODE_CACHE[(llvm_mc, xlen, raw)] for raw in raws}


def _rv_named(rows: list[Row], decode: tuple[str, str] | None) -> list[Row]:
    """The rows the decode's own MCInst opcode names (LR_W_AQ to lr.w.aq)."""
    if decode is None:
        return []
    named = decode[1].casefold().replace("_", ".")
    return [row for row in rows if row.key.casefold() == named]


def _rv_xlen_pick(
    rows: list[Row], text: str, decodes: tuple[tuple[str, str] | None, ...]
) -> list[Row]:
    """Keep the xlen decode whose printed mnemonic is the line's own text.

    ``decodes`` is (riscv32, riscv64). A tie both xlens decode the same way
    is not an xlen tie and keeps its rows. When both or neither printed
    mnemonic matches the text the rv64 result stays (the current default,
    noted in the round-2 evidence); an rv64-invalid word then keeps the twin
    the rv32 decode does not name."""
    d32, d64 = decodes
    if d32 is not None and d64 is not None and d32[0].casefold() == d64[0].casefold():
        return rows
    spelling = text.split(None, 1)[0].casefold().rstrip(":")
    matched = [decode is not None and decode[0].casefold() == spelling for decode in (d32, d64)]
    if matched[0] != matched[1]:
        return _rv_named(rows, d32 if matched[0] else d64) or rows
    picked = _rv_named(rows, d64)
    named32 = _rv_named(rows, d32)
    if not picked and named32:
        picked = [row for row in rows if row not in named32]
    return picked or rows


def _rv_xlen_resolve(probed, results, catalog: _Catalog) -> None:
    """Settle every undecided 16-bit tie by the xlen rule above (issue #65).

    The decode runs only on the lines whose xlen is undecided: one llvm-mc
    run per xlen covers the whole document."""
    pending = [
        (result, _bytes_to_words(entry.get("bytes")))
        for entry, result in zip(probed, results)
        if result.status == "no-row"
        and result.reason.startswith("ambiguous:")
        and result.candidates
    ]
    pending = [(result, raw) for result, raw in pending if len(raw) == 2]
    if not pending:
        return
    llvm_mc = _tool_path("SIMDREF_LLVM_MC", "llvm-mc")
    if llvm_mc is None:
        return
    raws = [raw for _result, raw in pending]
    decodes_of = {xlen: _rv_xlen_decodes(raws, llvm_mc, xlen) for xlen in (32, 64)}
    by_key = {row.db_key: row for row in catalog.rows}
    for result, raw in pending:
        rows = [by_key[key] for key in result.candidates if key in by_key]
        picked = _rv_xlen_pick(rows, result.text, (decodes_of[32][raw], decodes_of[64][raw]))
        if len(picked) == 1:
            result.status = "row"
            result.row = picked[0]
            result.reason = ""
            result.candidates = []


def _rv_tiebreak(rows: list[Row], opcode: str) -> list[Row]:
    """Name the row by llvm-mc's own MCInst opcode (upstream data, review F3).

    The opcode comes from the same run as the bytes; LR_W_AQ names lr.w.aq
    with '_' for '.', and a j line on RV64 can never reach the RV32-only
    c.jal row because the tool rejects the spelling there."""
    named = opcode.casefold().replace("_", ".")
    picked = [row for row in rows if row.key.casefold() == named]
    if picked:
        return picked
    # shortcut: a tie the opcode does not name stays for the precision step
    # or ends ambiguous, upgrade if the catalog grows opcode-named alias rows
    refined = _refine_subset([(row.enc_mask, row.enc_match, row) for row in rows])
    return [row for _mask, _match, row in refined]


def _identify_words(probed, results, catalog: _Catalog, arch: str) -> None:
    doc_base = "RV32" if arch == "riscv32" else "RV64"
    for entry, result in zip(probed, results):
        if entry["status"] != "OK":
            continue
        raw = _bytes_to_words(entry.get("bytes"))
        covered = _covered_mask(entry.get("bytes"))
        if arch == "aarch64":
            if len(raw) != 4:
                result.reason = "no catalog row (not one 32-bit word)"
                continue
            word = int.from_bytes(raw, "little")
            if covered:
                # A fixup placeholder covers pattern bits (audit A4): match
                # the known bits only, never the zero fill.
                matched = [
                    (mask, match, row)
                    for mask, match, row in catalog.a64_patterns
                    if ((word ^ match) & mask & ~covered) == 0
                ]
            else:
                bucket = catalog.a64_buckets.get((word >> 24) & 0xFF, ())
                matched = [
                    (mask, match, row) for mask, match, row in bucket if (word & mask) == match
                ]
            refined = _refine_subset(matched)
            rows = [row for _mask, _match, row in refined]
            checks = [(mask, row) for mask, _match, row in matched]
        else:
            if len(raw) not in (2, 4):
                result.reason = "no catalog row (not a 16/32-bit word)"
                continue
            word = int.from_bytes(raw, "little")
            patterns = catalog.rv_patterns_for(doc_base)
            entries = [
                entry4
                for entry4 in patterns
                if entry4[0] == len(raw) * 8 and ((word ^ entry4[2]) & entry4[1] & ~covered) == 0
            ]
            rows = [row for _bits, _mask, _match, row in entries]
            if len(rows) > 1:
                rows = _rv_tiebreak(rows, entry.get("opcode") or "")
            checks = [(mask, row) for _bits, mask, _match, row in entries]
        if len(rows) > 1:
            # Byte-identical rows are true spellings of one encoding (B.CC
            # and B.LO share the cond value 3): the source spelling names the
            # row (issue #60). A spelling no row carries keeps every row.
            spelling = entry["text"].split(None, 1)[0].casefold().rstrip(":")
            named_rows = [row for row in rows if row.mnemonic.casefold() == spelling]
            if named_rows:
                rows = named_rows
        if covered:
            # The row is decided without the placeholder only when one row
            # survives on the known bits and one of its matching patterns
            # reads no covered bit (audit A4): then the pattern holds for
            # every value of the placeholder and the row is not a guess.
            picked = {id(row): row for row in rows}
            # llvm-mc's own MCInst opcode names the row too (review F3). The
            # RISC-V branch fixups print whole covered bytes, which hides the
            # opcode bits from the mask test, so the opcode name is the second
            # decider (issue #60). The name alone is no byte proof (review F1):
            # a row may win only when its (mask, match) agrees with the
            # printed bytes on every bit that is no fixup placeholder. A
            # contradiction on a known bit means no row, or the pick would
            # deny the operand bits llvm-mc printed.
            named = (entry.get("opcode") or "").casefold().replace("_", ".")
            placeholder = _covered_mask(entry.get("bytes"), marked_byte_covers=False)
            by_opcode = [
                row
                for row in picked.values()
                if row.key.casefold() == named
                and ((word ^ row.enc_match) & row.enc_mask & ~placeholder) == 0
            ]
            decided = len(picked) == 1 and (
                any((mask & covered) == 0 and id(row) in picked for mask, row in checks)
                or len(by_opcode) == 1
            )
            if decided:
                result.status = "row"
                result.row = next(iter(picked.values()))
            elif not picked:
                result.reason = "no catalog row"
            elif len(picked) > 1:
                result.reason = (
                    f"no catalog row (fixup placeholder decides between {len(picked)} rows)"
                )
                result.candidates = sorted(row.db_key for row in picked.values())
            else:
                result.reason = "no catalog row (fixup placeholder covers a pattern field)"
            continue
        if len(rows) == 1:
            result.status = "row"
            result.row = rows[0]
        elif not rows:
            result.reason = "no catalog row"
        else:
            result.reason = f"ambiguous: {len(rows)} rows match the encoding"
            result.candidates = [row.db_key for row in rows]
    if arch.startswith("riscv"):
        # The 16-bit xlen ties are the only ones left ambiguous here.
        _rv_xlen_resolve(probed, results, catalog)


_ENDBR_BYTES = {"f30f1efa": "ENDBR64", "f30f1efb": "ENDBR32"}


def _identify_x86(probed, results, catalog: _Catalog, xed: str) -> None:
    pending = []
    for entry, result in zip(probed, results):
        if entry["status"] != "OK":
            continue
        mode = entry["mode"]
        if mode != "64":
            # The decoder runs in 64-bit mode (audit B1); a line another mode
            # encodes decodes to the wrong facts, so it reports no row.
            result.reason = f"no catalog row (assembled in {mode}-bit mode)"
        else:
            pending.append((entry, result))
    if not pending:
        return
    hexes = [_bytes_to_words(entry.get("bytes")).hex() for entry, _result in pending]
    for (entry, result), facts, hexed in zip(pending, _xed_facts(hexes, xed), hexes):
        # A fixup placeholder covers the immediate value (audit A1): the
        # pick must not read it, and a row carrying an immzero claim about
        # it would be a guess either way.
        covered = bool(_covered_mask(entry.get("bytes")))
        if "error" in facts:
            result.reason = f"no catalog row (XED {facts['error']})"
            continue
        # XED decodes ENDBR32/64 as the NOP iform NOP_GPRv_GPRv_0F1E, which no uops.info row carries
        pool = catalog.iform_pools.get(_ENDBR_BYTES.get(hexed, facts["iform"]), [])
        if not pool:
            result.reason = f"no catalog row (iform {facts['iform']} absent)"
            continue
        try:
            rows = (
                _x86_pick(pool, facts, imm_unknown=covered, text=entry.get("text") or "")
                if len(pool) > 1
                else pool
            )
        except ValueError as err:
            result.reason = f"no catalog row ({err})"
            continue
        if covered:
            guessed = [row for row in rows if "immzero" in row.split_attrs]
            if len(rows) == 1 and not guessed:
                result.status = "row"
                result.row = rows[0]
            elif len(rows) > 1:
                result.reason = (
                    f"no catalog row (fixup placeholder decides between {len(rows)} rows)"
                )
                result.candidates = [row.db_key for row in rows]
            else:
                result.reason = "no catalog row (fixup placeholder covers an encoded field)"
            continue
        if len(rows) == 1:
            result.status = "row"
            result.row = rows[0]
        elif not rows:
            result.reason = f"no catalog row (splits emptied iform {facts['iform']})"
        else:
            result.reason = f"ambiguous: {len(rows)} rows share iform {facts['iform']}"
            result.candidates = [row.db_key for row in rows]


# ---------------------------------------------------------------------------
# ISA detection by assembler acceptance
# ---------------------------------------------------------------------------

# Candidate triples for family detection, mapped to the catalog families.
_ISA_CANDIDATES = (
    ("x86", "att", "x86"),
    ("x86", "intel", "x86"),
    ("aarch64", "att", "arm"),
    ("riscv64", "att", "riscv"),
)


def _probe_all_triples(lines: list[str], llvm_mc: str) -> dict[tuple[str, str], list[dict]]:
    """Probe every candidate triple, one llvm-mc run per triple.

    The runs are independent, so they run concurrently: on a 2000-line
    document the rejecting triples cost more than the accepting one and the
    wall time then falls to the slowest run, not the sum.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(_ISA_CANDIDATES)) as pool:
        futures = {
            (arch, syntax): pool.submit(_probe_lines_split, lines, arch, syntax, llvm_mc)
            for arch, syntax, _family in _ISA_CANDIDATES
        }
        return {key: future.result() for key, future in futures.items()}


def isa_by_acceptance(lines: list[str]) -> tuple[list[str | None], str | None]:
    """Name each line's ISA family by the triples whose assembler encodes it.

    One llvm-mc run per candidate triple (x86 AT&T and Intel, aarch64,
    riscv64). A line one triple alone accepts takes that family. A line
    several triples accept takes the document majority of the singly
    accepted lines. An entry stays None when no strict majority exists or
    llvm-mc is missing, and the caller keeps its old vote then. ``lines``
    holds instruction text (mnemonic and operands) with objdump columns
    and symbol annotations already stripped.
    """
    if not lines:
        return [], None
    llvm_mc = _tool_path("SIMDREF_LLVM_MC", "llvm-mc")
    if llvm_mc is None:
        return [None] * len(lines), None
    accepted: list[set[str]] = [set() for _ in lines]
    try:
        runs = _probe_all_triples(list(lines), llvm_mc)
        for arch, syntax, family in _ISA_CANDIDATES:
            for entry, families in zip(runs[(arch, syntax)], accepted):
                if entry["status"] == "OK":
                    families.add(family)
    except OSError:
        # A missing or failing llvm-mc answers like no tool: all None and
        # the caller keeps its register vote.
        return [None] * len(lines), None
    counts: dict[str, int] = {}
    for families in accepted:
        if len(families) == 1:
            name = next(iter(families))
            counts[name] = counts.get(name, 0) + 1
    majority = None
    if counts:
        top = max(counts.values())
        winners = [name for name, count in counts.items() if count == top]
        if len(winners) == 1:
            majority = winners[0]
    out: list[str | None] = []
    for families in accepted:
        if len(families) == 1:
            out.append(next(iter(families)))
        elif families:
            # the batch majority names the line's family only when the line's own assemblers accept it
            out.append(majority if majority in families else None)
        else:
            out.append(None)
    return out, majority


def identify_x86_instructions(
    texts: list[str],
    rows: "list[Row]",
    *,
    llvm_mc: str | None = None,
    xed: str | None = None,
) -> list[LineResult]:
    """Map concrete Intel-syntax instruction lines to one catalog row each.

    The x86 path of :func:`identify` for generated instruction lines (the
    AMD workbook rows at ingest): one llvm-mc run for the whole batch, one
    XED decode pass, and the iform pools built from ``rows`` decide each
    line exactly like a document line.
    """
    from types import SimpleNamespace

    if not texts:
        return []
    if llvm_mc is None:
        llvm_mc = _tool_path("SIMDREF_LLVM_MC", "llvm-mc")
    if llvm_mc is None:
        raise ToolUnavailableError("llvm-mc not found (set SIMDREF_LLVM_MC)")
    if xed is None:
        xed = _tool_path("SIMDREF_XED", "xed")
    if xed is None:
        raise ToolUnavailableError("xed decoder not found (set SIMDREF_XED)")
    probed = _probe_lines(texts, "x86", "intel", llvm_mc)
    results = [
        LineResult(
            n=entry["n"],
            text=entry["text"],
            status="not-an-instruction" if entry["status"] == "NOT-AN-INSTRUCTION" else "no-row",
            reason=(
                entry.get("kind", "")
                if entry["status"] == "NOT-AN-INSTRUCTION"
                else entry.get("error", "rejected by llvm-mc")
                if entry["status"] == "REJECTED"
                else ""
            ),
        )
        for entry in probed
    ]
    iform_pools: dict[str, list[Row]] = {}
    for row in rows:
        if row.iform:
            iform_pools.setdefault(row.iform, []).append(row)
    # _identify_x86 reads only iform_pools from its catalog argument
    _identify_x86(probed, results, SimpleNamespace(iform_pools=iform_pools), xed)
    return results


# ---------------------------------------------------------------------------
# per-line identity: one llvm-mc run per family, no document-wide vote
# ---------------------------------------------------------------------------


def _catalog_for(catalog_path: str | os.PathLike | None, arch: str) -> _Catalog:
    if catalog_path is None:
        from simdref.storage import SQLITE_PATH

        catalog_path = SQLITE_PATH
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(path)
    cache_key = (str(path), arch, path.stat().st_mtime_ns)
    catalog = _CATALOG_CACHE.get(cache_key)
    if catalog is None:
        catalog = _CATALOG_CACHE[cache_key] = _Catalog(path, arch)
    return catalog


def _bytes_entry(byte_hex: str, *, word: bool = False) -> dict:
    """Build one probe-shaped entry from an objdump byte column.

    ``word`` (AArch64 and RISC-V lines): objdump prints one hex token per
    instruction word, and the token is the word value, not memory order
    (``a021`` is the word 0xa021, bytes ``21 a0``). A column with more than
    one 2-digit token is memory order (x86 bytes, GNU per-byte columns).
    """
    tokens = byte_hex.split()
    if word and len(tokens) == 1 and len(tokens[0]) in (4, 8):
        raw = int(tokens[0], 16).to_bytes(len(tokens[0]) // 2, "little")
        hexbytes = ",".join(f"0x{byte:02x}" for byte in raw)
    else:
        joined = "".join(tokens)
        hexbytes = "0x" + ",0x".join(joined[pos : pos + 2] for pos in range(0, len(joined), 2))
    return {"status": "OK", "bytes": hexbytes, "mode": "64", "text": "", "n": 0}


def identify_lines(
    items: list[tuple[str, str | None]],
    *,
    catalog_path: str | os.PathLike | None = None,
    llvm_mc: str | None = None,
    xed: str | None = None,
    fallback_family: str | None = None,
) -> list[LineResult]:
    """Map each instruction text to one catalog row, decided per line.

    ``items`` holds one ``(text, byte_column_hex)`` pair per source line: the
    raw instruction text (objdump address and byte columns and any ``<sym>``
    suffix already stripped, the trailing comment kept so each triple applies
    its own comment rule) and the objdump byte column hex, or None. The
    column keeps its printed tokens: on AArch64 and RISC-V a single 4- or
    8-digit token is the instruction word value, and a multi-token column is
    memory order.

    One llvm-mc run per candidate triple covers the batch, and each line takes
    the family whose assembler accepts it and the x86 syntax its own text
    speaks. A line never depends on its neighbours: no document-wide family
    vote and no dropped lines. An Intel line that llvm-mc's Intel parser
    rejects takes the objdump byte column directly, or retries once through
    the AT&T parser.
    """
    if not items:
        return []
    texts = [text for text, _hex in items]
    results = [
        LineResult(n=index + 1, text=text, status="no-row") for index, text in enumerate(texts)
    ]
    catalog = _catalog_for(catalog_path, "x86")
    if catalog.keyless:
        for result in results:
            result.reason = "catalog lacks encoding keys (schema 13)"
        return results
    if llvm_mc is None:
        llvm_mc = _tool_path("SIMDREF_LLVM_MC", "llvm-mc")
    if not llvm_mc:
        raise ToolUnavailableError("llvm-mc not found (set SIMDREF_LLVM_MC)")
    for arch, syntax, _family in _ISA_CANDIDATES:
        if _string_spans_lines(texts, arch, syntax, llvm_mc):
            # One open string poisons the run (Sol r4 F1/F2, r5 P2): the
            # older matcher answers every line.
            for result in results:
                result.reason = "quoted string spans lines"
            return results
    probed: dict[tuple[str, str], list[dict]] = {}
    try:
        probed = _probe_all_triples(texts, llvm_mc)
    except OSError as exc:
        raise ToolUnavailableError(f"llvm-mc failed to run: {exc}") from exc
    # Family per line = isa_by_acceptance's rule: the line's own accepting
    # family, else the batch majority of the singly accepted lines, else the
    # line keeps the old matcher (no byte identity).
    families_of: list[set[str]] = []
    counts: dict[str, int] = {}
    for index in range(len(texts)):
        families = {
            family
            for arch, syntax, family in _ISA_CANDIDATES
            if probed[(arch, syntax)][index].get("status") == "OK"
        }
        families_of.append(families)
        if len(families) == 1:
            name = next(iter(families))
            counts[name] = counts.get(name, 0) + 1
    majority = None
    if counts:
        top = max(counts.values())
        winners = [name for name, count in counts.items() if count == top]
        if len(winners) == 1:
            majority = winners[0]
    groups: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    for index, text in enumerate(texts):
        if probed[("x86", "intel")][index].get("status") == "NOT-AN-INSTRUCTION":
            results[index].status = "not-an-instruction"
            results[index].reason = str(probed[("x86", "intel")][index].get("kind") or "")
            continue
        accepted = [
            (arch, syntax, family)
            for arch, syntax, family in _ISA_CANDIDATES
            if probed[(arch, syntax)][index].get("status") == "OK"
        ]
        chosen: tuple[str, str] | None = None
        entry: dict | None = None
        families = families_of[index]
        family = None
        if len(families) == 1:
            family = next(iter(families))
        elif families:
            family = majority if majority in families else None
        if family is None:
            # no per-line answer: the caller's old document vote says which
            # family's assembler to try (no keep mask; every line is tried)
            family = fallback_family
        if family is not None:
            runs = [(arch, syntax) for arch, syntax, name in accepted if name == family]
            if family == "x86":
                # The syntax the text speaks is tried first; a parser that
                # rejects the line falls through to the byte column or the
                # other syntax below (the Intel vs AT&T retry).
                primary = (
                    "x86",
                    "intel" if ("%" not in text and "$" not in text) else "att",
                )
                chosen = primary
            elif runs:
                chosen = runs[0]
            if chosen is not None:
                entry = probed[chosen][index]
        byte_hex = items[index][1]
        if chosen is not None and chosen[0] == "x86" and entry.get("status") != "OK":
            other = ("x86", "intel" if chosen[1] == "att" else "att")
            if byte_hex:
                # the objdump byte column is the real encoding
                chosen, entry = ("x86", "att"), _bytes_entry(byte_hex)
            elif probed[other][index].get("status") == "OK":
                chosen, entry = other, probed[other][index]
        # A line no assembler accepts gets no byte identity: its byte column is
        # no evidence for x86 (an ARM32 "bx lr" bytes decode as x86 LOOP).
        if chosen is None or entry is None or entry.get("status") != "OK":
            error = (
                probed[("x86", "intel")][index].get("error")
                if probed[("x86", "intel")][index].get("status") == "REJECTED"
                else "rejected by llvm-mc"
            )
            results[index].reason = str(error or "rejected by llvm-mc")
            continue
        if byte_hex:
            # The objdump byte column is the real encoding: raw bytes name
            # what they encode (a 2-byte c.j stays c.j). Only source text
            # follows the no-compression rule; the probe opcode then names
            # the text's instruction, never the bytes', so it is dropped.
            entry = {
                **entry,
                "bytes": _bytes_entry(byte_hex, word=chosen[0] != "x86")["bytes"],
                "opcode": "",
            }
        groups.setdefault(chosen, []).append((index, entry))
    for (arch, _syntax), members in groups.items():
        idxs = [index for index, _entry in members]
        entries = [entry for _index, entry in members]
        sub = [results[index] for index in idxs]
        if arch == "x86":
            if xed is None:
                xed = _tool_path("SIMDREF_XED", "xed")
            if not xed:
                raise ToolUnavailableError("xed decoder not found (set SIMDREF_XED)")
            _identify_x86(entries, sub, _catalog_for(catalog_path, "x86"), xed)
        else:
            _identify_words(entries, sub, _catalog_for(catalog_path, arch), arch)
    # REP/REPE/REPNE MOVSB, CMPSB, ... share their iform; the prefix byte is
    # the only byte-level difference and picks the sibling row.
    catalog_x86 = None
    hexbytes_of = {
        index: member_entry.get("bytes") or ""
        for members in groups.values()
        for index, member_entry in members
    }
    for index in range(len(texts)):
        result = results[index]
        if result.status != "no-row" or not result.candidates:
            continue
        hexbytes = hexbytes_of.get(index, "")
        words = _bytes_to_words(hexbytes) if hexbytes else []
        if not words or words[0] not in (0xF2, 0xF3):
            continue
        want = {"REPE", "REP"} if words[0] == 0xF3 else {"REPNE"}
        if catalog_x86 is None:
            catalog_x86 = _catalog_for(catalog_path, "x86")
        named = [
            row
            for row in catalog_x86.rows
            if row.db_key in result.candidates and row.key.split(" ", 1)[0] in want
        ]
        if len(named) == 1:
            result.status = "row"
            result.reason = ""
            result.row = named[0]
            result.candidates = []
    return results
