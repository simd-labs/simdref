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
  resolves the documented ``c.*`` bit collisions.

No mnemonic or iform table and no fallback matcher (spec R2): a line ends
as one row, or with a reason why there is none.

Tool paths: ``SIMDREF_LLVM_MC`` / ``SIMDREF_XED`` override
``simdref_llvm_mc.path(...)`` when that package is importable (it ships both
tools later). Until then point the env vars at a 23.1.4 ``llvm-mc`` and the
XED decode CLI.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
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
        "mattr": "+sve,+sve2,+sve2p1,+sme,+sme2,+sme2p1",
    },
    "arm": {"triple": "armv7-unknown-linux-gnueabihf", "mattr": "+neon"},
    "riscv32": {
        "triple": "riscv32-unknown-linux-gnu",
        "mattr": "+m,+a,+f,+d,+c,+v,+zba,+zbb,+zbs",
    },
    "riscv64": {
        "triple": "riscv64-unknown-linux-gnu",
        "mattr": "+m,+a,+f,+d,+c,+v,+zba,+zbb,+zbs",
    },
}
_X86_SYNTAX_FLAG = {"att": "-x86-asm-syntax=att", "intel": "-x86-asm-syntax=intel"}
# uops.info eosz attribute values per effective operand size.
_EOSZ_ATTR = {16: "1", 32: "2", 64: "3"}
# The four x86 registers uops.info tags as high-byte (its ``high8`` attr).
_HIGH8_REGS = frozenset({"AH", "BH", "CH", "DH"})

_COMMENT_STARTS = ("#", "//", "/*", "@", ";")
_ERR_RE = re.compile(r"^<stdin>:(\d+):(\d+): (?:error|warning): (.*)$", re.M)
_ENC_RE = re.compile(r"encoding: \[([^\]]*)\]")
_INST_RE = re.compile(r"<MCInst #\d+ (\S+)")
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
        return override
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


def _parse_echo_blocks(stdout: str) -> list[tuple[str, str | None, str | None]]:
    blocks: list[list] = []
    for line in stdout.splitlines():
        if _CONT_RE.match(line):
            if not blocks:
                blocks.append(["", None, None])
        else:
            blocks.append([line.rstrip(), None, None])
        block = blocks[-1]
        if block[1] is None:
            match = _INST_RE.search(line)
            if match:
                block[1] = match.group(1)
        if block[2] is None:
            match = _ENC_RE.search(line)
            if match:
                block[2] = match.group(1).strip()
    return [tuple(block) for block in blocks]


def _probe_lines(lines: list[str], arch: str, syntax: str, llvm_mc: str) -> list[dict]:
    """Run llvm-mc once per document; map each line to bytes or a reason."""
    cfg = _ARCH_CFG[arch]
    cmd = [llvm_mc, f"--triple={cfg['triple']}", "-show-encoding", "-show-inst"]
    if cfg["mattr"]:
        cmd.append(f"-mattr={cfg['mattr']}")
    if arch == "x86":
        cmd.append(_X86_SYNTAX_FLAG[syntax])

    work = list(lines)
    errors: dict[int, str] = {}
    stdout = ""
    last_errs: dict[int, str] = {}
    for _ in range(3):  # blank failing lines and re-run, at most 3 passes
        run = subprocess.run(cmd, input="\n".join(work), capture_output=True, text=True)
        stdout = run.stdout
        last_errs = {}
        for match in _ERR_RE.finditer(run.stderr):
            last_errs.setdefault(int(match.group(1)), match.group(3))
        for line_no, text in last_errs.items():
            errors.setdefault(line_no, text)
        if not last_errs:
            break
        for line_no in last_errs:
            work[line_no - 1] = ""

    results: list[dict] = []
    for i, text in enumerate(lines, 1):
        kind = _classify_line(text)
        if i in errors:
            results.append({"n": i, "text": text, "status": "REJECTED", "error": errors[i]})
        elif kind is not None:
            results.append({"n": i, "text": text, "status": "NOT-AN-INSTRUCTION", "kind": kind})
        else:
            results.append({"n": i, "text": text, "status": "PENDING"})

    # Lockstep: source lines map one-to-one to echo blocks, except comments
    # (dropped) and lines that failed in the final pass (no echo).
    echoed = [
        i
        for i, text in enumerate(lines, 1)
        if _classify_line(text) != "comment" and i not in last_errs
    ]
    blocks = _parse_echo_blocks(stdout)
    pending = [r for r in results if r["status"] == "PENDING"]
    if len(blocks) == len(echoed):
        for index, block in zip(echoed, blocks):
            result = results[index - 1]
            if result["status"] != "PENDING":
                continue
            _echo, opcode, hexbytes = block
            if opcode is not None:
                result["status"] = "OK"
                result["bytes"] = hexbytes
                result["opcode"] = opcode
            else:
                result["status"] = "NOT-AN-INSTRUCTION"
                result["kind"] = "echo-no-inst"
    else:
        # An unverified line-to-bytes mapping is not byte identity (review F4):
        # every unmapped line reports the mismatch instead of guessing order.
        for result in pending:
            result["status"] = "REJECTED"
            result["error"] = (
                f"line mapping unverified (echo {len(blocks)} vs source {len(echoed)})"
            )
    return results


_BYTE_RE = re.compile(r"0x([0-9a-fA-F]{2})(?:'A')?$")
_BITS_RE = re.compile(r"0b([01Aa]{8})$")


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
                row = Row(
                    db_key=raw[0],
                    key=raw[1],
                    architecture=raw[2],
                    mnemonic=raw[3],
                    form=raw[4],
                    enc_bits=raw[5] or 0,
                    enc_mask=raw[6] or 0,
                    enc_match=raw[7] or 0,
                    enc_extra=tuple(tuple(item) for item in json.loads(raw[8] or "[]")),
                    iform=raw[9] or "",
                    split_attrs=json.loads(raw[10] or "{}"),
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


def _x86_pick(pool: list[Row], d: dict) -> list[Row]:
    """Narrow an iform pool with the uops.info variant attributes stored on
    each row against the XED-decoded facts. Data tables on both sides."""
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
        any("immzero" in row.split_attrs for row in cur),
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
    return cur


# ---------------------------------------------------------------------------
# document lookup
# ---------------------------------------------------------------------------

_CATALOG_CACHE: dict[tuple[str, str, int], _Catalog] = {}


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

    probed = _probe_lines(lines, arch, syntax, llvm_mc)
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
        if arch == "aarch64":
            if len(raw) != 4:
                result.reason = "no catalog row (not one 32-bit word)"
                continue
            word = int.from_bytes(raw, "little")
            bucket = catalog.a64_buckets.get((word >> 24) & 0xFF, ())
            hits = [(mask, match, row) for mask, match, row in bucket if (word & mask) == match]
            hits = _refine_subset(hits)
            rows = [row for _mask, _match, row in hits]
        else:
            if len(raw) not in (2, 4):
                result.reason = "no catalog row (not a 16/32-bit word)"
                continue
            word = int.from_bytes(raw, "little")
            patterns = catalog.rv_patterns_for(doc_base)
            rows = [
                row
                for bits, mask, match, row in patterns
                if bits == len(raw) * 8 and (word & mask) == match
            ]
            if len(rows) > 1:
                rows = _rv_tiebreak(rows, entry.get("opcode") or "")
        if len(rows) == 1:
            result.status = "row"
            result.row = rows[0]
        elif not rows:
            result.reason = "no catalog row"
        else:
            result.reason = f"ambiguous: {len(rows)} rows match the encoding"
            result.candidates = [row.db_key for row in rows]


def _identify_x86(probed, results, catalog: _Catalog, xed: str) -> None:
    pending = [(entry, result) for entry, result in zip(probed, results) if entry["status"] == "OK"]
    if not pending:
        return
    hexes = [_bytes_to_words(entry.get("bytes")).hex() for entry, _result in pending]
    for (entry, result), facts in zip(pending, _xed_facts(hexes, xed)):
        if "error" in facts:
            result.reason = f"no catalog row (XED {facts['error']})"
            continue
        pool = catalog.iform_pools.get(facts["iform"], [])
        if not pool:
            result.reason = f"no catalog row (iform {facts['iform']} absent)"
            continue
        try:
            rows = _x86_pick(pool, facts) if len(pool) > 1 else pool
        except ValueError as err:
            result.reason = f"no catalog row ({err})"
            continue
        if len(rows) == 1:
            result.status = "row"
            result.row = rows[0]
        elif not rows:
            result.reason = f"no catalog row (splits emptied iform {facts['iform']})"
        else:
            result.reason = f"ambiguous: {len(rows)} rows share iform {facts['iform']}"
            result.candidates = [row.db_key for row in rows]
