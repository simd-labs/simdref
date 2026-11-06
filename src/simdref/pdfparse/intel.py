"""Intel SDM PDF parser.

Extracts per-instruction description sections from the Intel 64 and IA-32
Architectures Software Developer's Manual (combined volumes PDF).

The parser identifies instruction pages by their size-12 title font with
an all-caps mnemonic before an em-dash, then extracts size-10 section
headings and size-9 body text within each instruction's page range.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

try:
    from pdfminer.pdftypes import resolve1
except ImportError:  # pragma: no cover - exercised in minimal test envs

    def resolve1(value):
        return value


from simdref.pdfparse.base import chars_to_lines, extract_sections_from_lines
from simdref.pdfparse.registry import register_pdf_source
from simdref.pdfparse.types import PdfDescriptionPayload, PdfEnrichmentResult, PdfSourceSpec
from simdref.storage import DATA_DIR
from simdref.tablecols import split_cells

log = logging.getLogger(__name__)

INTEL_SDM_URL = "https://cdrdv2.intel.com/v1/dl/getContent/671200"
_REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_INTEL_SDM_PDFS = [
    _REPO_ROOT / "vendor" / "intel" / "intel-sdm.pdf",
]
INTEL_SDM_CACHE_PATH = DATA_DIR / "intel-sdm-descriptions.msgpack"
INTEL_SDM_CACHE_VERSION = 1
INTEL_SDM_SIGNATURE_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "base.py",
)

# Title pattern: mnemonic (with optional / or , separators) before em-dash.
# Lowercase is allowed only at the end of the mnemonic (CMOVcc, PREFETCHh).

# Words that indicate a chapter/section heading, not an instruction.
_SKIP_WORDS = frozenset({"CHAPTER", "CONTENTS", "APPENDIX", "VOLUME", "INSTRUCTION SET REFERENCE"})

# Font size thresholds (from empirical analysis of Intel SDM).
_TITLE_MIN_SIZE = 11.5
_HEADING_MIN_SIZE = 9.8
_BODY_MAX_SIZE = 9.5
_BODY_MIN_SIZE = 8.0  # Filter superscripts (®, footnote markers)

# Canonical section names and aliases.
KNOWN_SECTIONS: set[str] = {
    "Description",
    "Operation",
    "Intrinsic Equivalents",
    "Flags Affected",
    "FPU Flags Affected",
    "Exceptions",
    "Numeric Exceptions",
    "SIMD Floating-Point Exceptions",
    "Floating-Point Exceptions",
    "Other Exceptions",
    "Other Mode Exceptions",
    "Protected Mode Exceptions",
    "Real-Address Mode Exceptions",
    "Real Address Mode Exceptions",
    "Virtual-8086 Mode Exceptions",
    "Virtual-8086 Exceptions",
    "Virtual 8086 Mode Exceptions",
    "Compatibility Mode Exceptions",
    "64-Bit Mode Exceptions",
    "x87 FPU and SIMD Floating-Point Exceptions",
}

_SECTION_ALIASES: dict[str, str] = {
    "intel c/c++ compiler intrinsic equivalent": "Intrinsic Equivalents",
    "intel c/c++ compiler intrinsic equivalents": "Intrinsic Equivalents",
    "intel c/c++compiler intrinsic equivalent": "Intrinsic Equivalents",
    "intel c/c++compiler intrinsic equivalents": "Intrinsic Equivalents",
    "c/c++ compiler intrinsic equivalent": "Intrinsic Equivalents",
    "c/c++ compiler intrinsic equivalents": "Intrinsic Equivalents",
    "intrinsic equivalent": "Intrinsic Equivalents",
    "intrinsic equivalents": "Intrinsic Equivalents",
    "instruction operand encoding": "Operand Encoding",
    "fpu flags affected": "FPU Flags Affected",
    "floating-point exceptions": "Floating-Point Exceptions",
}


# Footer pattern: "MNEMONIC—... Vol. 2X N-NN" (space between volume and page optional)
def _is_footer_line(text: str) -> bool:
    head, sep, tail = text.rpartition("Vol.")
    while sep:
        if head and _is_footer_tail(tail):
            return True
        head, sep, tail = head.rpartition("Vol.")
    return False


def _is_footer_tail(tail: str) -> bool:
    rest = tail.lstrip()
    if rest[:1] != "2":
        return False
    rest = rest[1:]
    if rest[:1] in ("A", "B", "C", "D"):
        rest = rest[1:]
    digits, sep, pages = rest.lstrip().partition("-")
    return bool(sep) and digits.isdecimal() and pages.isdecimal()


# Sections to discard (tabular data that doesn't render well as text).
_DISCARD_SECTIONS: frozenset[str] = frozenset(
    {
        "Instruction Operand Encoding",
        "Operand Encoding",
    }
)

# Sections whose text is pseudocode and should preserve indentation from x0.
_CODE_SECTIONS: frozenset[str] = frozenset(
    {
        "Operation",
        "Intrinsic Equivalents",
    }
)

# ---------------------------------------------------------------------------
# Opcode table and Instruction Operand Encoding table
#
# The tables sit between the title and the first section heading. Section
# extraction cannot keep them (they have no heading), so they get their own
# parser over the title page lines. Column layout text has two or more spaces
# between cells; PyMuPDF page prep joins cells with single spaces, which the
# opcode-row scan accepts. Operand encoding rows need the wide gaps, so the
# PyMuPDF fast path may skip that table.
OPCODE_MODES_SECTION = "Opcode & Modes"
OPERAND_ENCODING_SECTION = "Operand Encoding"

# Longest alternative first, so RM does not shadow RMI etc.
_OP_EN_TOKENS = tuple(
    "M1T MIT RMI MRI RVM MVR VRM RM M1 MI MR AM MA "
    "ZO XO OI IO TT TI NP A B C D E I M O P S T".split()
)
_MODE_TOKENS = ("V", "I", "N.E.", "N.N.", "N.P.", "N.S.", "N.")
_OPCODE_TOKENS = tuple("NP NFx RNE REX REX. REXW REXw REX.W REX.w F2 F3 +".split())
_IMM_SUFFIXES = ("cb", "cw", "cd", "cp", "co", "ct", "ib", "iw", "id", "io", "is4")
_HEX_CHARS = "0123456789abcdefABCDEF"
_SLASH_TAIL_CHARS = "01234567abcdefghijklmnopqrstuvwxyz"


def _skip_ws(line: str, pos: int) -> int:
    while pos < len(line) and line[pos].isspace():
        pos += 1
    return pos


def _is_hex_byte(word: str) -> bool:
    return len(word) == 2 and all(char in _HEX_CHARS for char in word)


def _opcode_token_lengths(line: str, start: int) -> list[int]:
    """Opcode token match lengths at `start`, longest alternative first."""
    lengths: list[int] = []
    if line.startswith("NP", start):
        lengths.append(2)
    if line.startswith("NFx", start):
        lengths.append(3)
    if line.startswith("RNE", start):
        lengths.append(3)
    if line.startswith("REX", start):
        if line[start + 3 : start + 4] == ".":
            if line[start + 4 : start + 5] in ("W", "w"):
                lengths.append(5)
            lengths.append(4)
        if line[start + 3 : start + 4] in ("W", "w"):
            lengths.append(4)
        lengths.append(3)
    if line.startswith("F2", start) or line.startswith("F3", start):
        lengths.append(2)
    if line.startswith("+", start):
        lengths.append(1)
    if _is_hex_byte(line[start : start + 2]):
        lengths.append(2)
    return lengths


def _imm_ends(line: str, start: int) -> list[int]:
    """Row ends after the imm-suffix loop, longest parse first."""
    ends = [start]
    pos = start
    while True:
        nxt = _skip_ws(line, pos)
        if nxt == pos:
            break
        for suffix in _IMM_SUFFIXES:
            if line.startswith(suffix, nxt):
                pos = nxt + len(suffix)
                ends.append(pos)
                break
        else:
            break
    return ends[::-1]


def _tail_ends(line: str, start: int) -> list[int]:
    """Row ends after the optional /digit tail and the imm suffixes."""
    slash = _skip_ws(line, start)
    if line[slash : slash + 1] == "/":
        end = slash + 1
        while end < len(line) and line[end] in _SLASH_TAIL_CHARS:
            end += 1
        if end > slash + 1:
            return _imm_ends(line, end) + _imm_ends(line, start)
    return _imm_ends(line, start)


def _legacy_opcode_ends(line: str, start: int) -> list[int]:
    """Row ends for the legacy opcode column (NP 0F 58 /r ...)."""
    tokens: list[tuple[int, int]] = []
    pos = start
    while pos < len(line) and not line[pos].isspace():
        end = pos
        while end < len(line) and not line[end].isspace():
            end += 1
        tokens.append((pos, end))
        pos = _skip_ws(line, end)
    ends: list[int] = []
    for count in range(len(tokens), 0, -1):
        head = [line[a:b] for a, b in tokens[: count - 1]]
        if not all(word in _OPCODE_TOKENS or _is_hex_byte(word) for word in head):
            continue
        tok_start, tok_end = tokens[count - 1]
        for stop in _opcode_token_lengths(line, tok_start):
            stop += tok_start
            if stop == tok_end or line[stop : stop + 1] == "/":
                ends.extend(_tail_ends(line, stop))
    return ends


def _prefix_opcode_ends(line: str, start: int) -> list[int]:
    """Row ends for the VEX./EVEX./XOP. opcode column."""
    for head in ("VEX.", "EVEX.", "XOP."):
        if line.startswith(head, start):
            break
    else:
        return []
    pos = start + len(head)
    while pos < len(line) and (line[pos] == "." or line[pos].isascii() and line[pos].isalnum()):
        pos += 1
    if pos == start + len(head):
        return []
    ends: list[int] = []
    while True:
        nxt = _skip_ws(line, pos)
        if nxt == pos or not _is_hex_byte(line[nxt : nxt + 2]):
            break
        pos = nxt + 2
        ends.append(pos)
    out: list[int] = []
    for end in ends[::-1]:
        out.extend(_tail_ends(line, end))
    return out


def _row_tail(line: str, pos: int) -> tuple[str, str, str, str | None] | None:
    """Op/En, the V / V mode pair and the CPUID tail of an opcode row."""
    pos = _skip_ws(line, pos)
    for op_en in _OP_EN_TOKENS:
        if not line.startswith(op_en, pos):
            continue
        mode_at = _skip_ws(line, pos + len(op_en))
        if mode_at == pos + len(op_en):
            continue
        for mode_64 in _MODE_TOKENS:
            if not line.startswith(mode_64, mode_at):
                continue
            mid = mode_at + len(mode_64)
            if line[mid : mid + 1].isspace():
                mid += 1
            if line[mid : mid + 1] != "/":
                continue
            stop = mid + 1
            if line[stop : stop + 1].isspace():
                stop += 1
            for mode_32 in _MODE_TOKENS:
                if not line.startswith(mode_32, stop):
                    continue
                cpuid_at = stop + len(mode_32)
                rest = _skip_ws(line, cpuid_at)
                if rest > cpuid_at:
                    return op_en, mode_64, mode_32, line[rest:]
                if cpuid_at == len(line):
                    return op_en, mode_64, mode_32, None
    return None


def _match_opcode_row(line: str) -> tuple[str, str, str, str, str | None] | None:
    """Scan one opcode table row: opcode, Op/En, modes and CPUID."""
    start = _skip_ws(line, 0)
    for ends in (_prefix_opcode_ends, _legacy_opcode_ends):
        for end in ends(line, start):
            tail = _row_tail(line, end)
            if tail is not None:
                return (line[start:end],) + tail
    return None


def _oe_header_shape(cells: list[str]) -> tuple[bool, bool]:
    """Recognise an Operand Encoding header. Returns (is_header, has_tuple).

    PyMuPDF drops the caption and the Op/En cell, so the header may read
    bare "Tuple Type / Operand 1 .." or only "Operand 1 ..".
    """
    texts = [cell.strip() for cell in cells]
    if not texts:
        return False, False
    if texts[0].startswith("Op/En"):
        return True, any("Tuple Type" in cell for cell in texts)
    has_operand_columns = all(
        text.startswith("Operand ") and text[8:].isdecimal() for text in texts
    )
    if has_operand_columns:
        return True, False
    if "Tuple Type" in texts and any(text.startswith("Operand") for text in texts):
        return True, True
    return False, False


def _token_is_isa(token: str) -> bool:
    """A CPUID flag token: uppercase letters, digits and _ . - only."""
    bare = token.strip("()")
    if not bare:
        return token in ("(", ")")
    if bare in ("AND", "OR"):
        return True
    return len(bare) >= 2 and all(c.isupper() or c.isdigit() or c in "_.-" for c in bare)


def _isa_prefix(text: str) -> str:
    """Keep the CPUID flag tokens and cut the glued description column."""
    out: list[str] = []
    for token in text.split():
        if _token_is_isa(token):
            out.append(token)
        else:
            break
    return " ".join(out)


def _isa_cell(cell: str) -> bool:
    tokens = cell.split()
    return bool(tokens) and all(_token_is_isa(token) for token in tokens)


def parse_opcode_table(lines: list[str]):
    """Parse the SDM opcode table into mode and CPUID rows per form.

    Returns a list of dicts with opcode, op_en, mode_64, mode_32 and cpuid.
    Lines without a table give an empty list.
    """
    rows: list[dict[str, str]] = []
    stop_captions = ("Description", "Operation")
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip().rstrip(":")
        if "Instruction Operand Encoding" in line or stripped in stop_captions:
            break
        match = _match_opcode_row(line)
        if match is None:
            idx += 1
            continue
        opcode, op_en, mode_64, mode_32, cpuid = match
        row = {
            "opcode": " ".join(opcode.split()),
            "op_en": op_en,
            "mode_64": mode_64,
            "mode_32": mode_32,
            "cpuid": _isa_prefix(cpuid or ""),
        }
        # Follow-on lines carry the wrapped CPUID cells (and the operand
        # form, kept in uops, not here).
        idx += 1
        while idx < len(lines):
            follow = lines[idx]
            if not follow.strip() or _match_opcode_row(follow) is not None:
                break
            follow_stripped = follow.strip()
            if "Instruction Operand Encoding" in follow or follow_stripped in stop_captions:
                break
            follow_cells = split_cells(follow_stripped)
            if _oe_header_shape(follow_cells)[0]:
                break
            for cell in follow_cells:
                if _isa_cell(cell):
                    row["cpuid"] = " ".join(part for part in (row["cpuid"], cell) if part)
            idx += 1
        rows.append(row)
    for row in rows:
        if not row["cpuid"]:
            del row["cpuid"]
    return rows


def parse_operand_encoding_table(lines: list[str]):
    """Parse the Instruction Operand Encoding table.

    Returns a list of dicts with op_en, tuple_type and operands. N/A cells
    are dropped. Lines without the table give an empty list.
    """
    rows: list[dict[str, object]] = []
    active = False
    has_tuple = False
    seen_header = False
    for idx, line in enumerate(lines):
        cells = split_cells(line.strip())
        is_header, header_has_tuple = _oe_header_shape(cells)
        if not active:
            if is_header or "Instruction Operand Encoding" in line:
                active = True
            else:
                continue
        elif not is_header:
            if not line.strip():
                if seen_header or rows:
                    break
                continue
        if is_header:
            has_tuple = header_has_tuple
            seen_header = True
            continue
        if not line.strip():
            continue
        first = cells[0].strip()
        if not seen_header:
            continue
        if line.strip().lower() in _ALL_HEADING_NAMES:
            break
        if first not in _OP_EN_TOKENS:
            break
        offset = 2 if has_tuple else 1
        operands = [
            cell.strip() for cell in cells[offset:] if cell.strip() and cell.strip() != "N/A"
        ]
        tuple_type = cells[1].strip() if has_tuple and len(cells) > 1 else ""
        if tuple_type == "N/A":
            tuple_type = ""
        if not operands and not tuple_type:
            continue
        rows.append({"op_en": first, "tuple_type": tuple_type, "operands": operands})
    return rows


def _mode_row_text(row: dict[str, str]) -> str:
    text = (
        f"{row['opcode']} ({row['op_en']}): 64-bit {row['mode_64']}, compat/legacy {row['mode_32']}"
    )
    if row.get("cpuid"):
        text += f"; CPUID {row['cpuid']}"
    return text


def _operand_encoding_row_text(row: dict[str, object]) -> str:
    parts = "; ".join(row["operands"])
    if row["tuple_type"]:
        parts = f"{parts}; tuple: {row['tuple_type']}" if parts else f"tuple: {row['tuple_type']}"
    return f"{row['op_en']}: {parts}"


def form_table_sections(lines: list[str]) -> dict[str, str]:
    """Section text for the SDM opcode table and Operand Encoding table."""
    sections: dict[str, str] = {}
    mode_rows = parse_opcode_table(lines)
    if mode_rows:
        sections[OPCODE_MODES_SECTION] = "\n".join(_mode_row_text(row) for row in mode_rows)
    encoding_rows = parse_operand_encoding_table(lines)
    if encoding_rows:
        sections[OPERAND_ENCODING_SECTION] = "\n".join(
            _operand_encoding_row_text(row) for row in encoding_rows
        )
    return sections


def _cell_joined_lines(lines: list[tuple[float, float, float, str]]) -> list[str]:
    """Join same-row table cells with a wide gap (PyMuPDF gives one line
    per cell). Prose lines stay single-cell and pass through unchanged."""
    groups: list[list[tuple[float, str]]] = []
    group_tops: list[float] = []
    for top, _size, x0, text in lines:
        if groups and abs(top - group_tops[-1]) <= 2.0:
            groups[-1].append((x0, text))
        else:
            groups.append([(x0, text)])
            group_tops.append(top)
    out: list[str] = []
    for group in groups:
        group.sort()
        joined = "  ".join(text for _x0, text in group if text).strip()
        if joined:
            out.append(joined)
    return out


# All heading names (lowercase) for content-based heading detection.
# Some Intel SDM pages format section headings at body text size.
_ALL_HEADING_NAMES: frozenset[str] = frozenset(
    {s.lower() for s in KNOWN_SECTIONS}
    | set(_SECTION_ALIASES.keys())
    | {s.lower() for s in _DISCARD_SECTIONS}
)

# Headings the tabular-noise filter must not drop: kept sections only.
# Discard headings (Instruction Operand Encoding) still count as table noise.
_NOISE_EXEMPT_HEADINGS: frozenset[str] = _ALL_HEADING_NAMES - {s.lower() for s in _DISCARD_SECTIONS}


# Minimal safety-net for junk lines that appear outside table bounding boxes.
def _is_junk_line(text: str) -> bool:
    """Footnote references ('1. See note ...') and the trailing 'NOTES:' line."""
    if text.startswith("NOTES:"):
        return not text[6:].strip()
    cut = 0
    while cut < len(text) and text[cut].isdecimal():
        cut += 1
    return (
        cut > 0
        and text[cut : cut + 1] == "."
        and text[cut + 1 : cut + 2].isspace()
        and text[cut + 1 :].lstrip().startswith("See note ")
    )


def _is_numbered_item(line: str) -> bool:
    cut = 0
    while cut < len(line) and line[cut].isdecimal():
        cut += 1
    return cut > 0 and line[cut : cut + 1] == "." and line[cut + 1 : cut + 2].isspace()


# Maximum fraction of page area a single table can cover before we
# consider it a false-positive detection by pdfplumber.
_TABLE_MAX_PAGE_FRACTION = 0.85
_TABLE_GRAPHIC_PRIMITIVE_THRESHOLD = 6


@dataclass(slots=True)
class _PreparedPage:
    title: tuple[str, str] | None
    body_lines: list[tuple[float, float, float, str]]
    backend: str
    fallback_reason: str | None = None


_TABULAR_LINE_HEADS = (
    "Opcode",
    "Op/En",
    "64/32-Bit Mode",
    "64-Bit Mode",
    "32-Bit Mode",
    "CPUID Feature Flag",
)


def _has_tabular_head(text: str, heads: tuple[str, ...]) -> bool:
    lowered = text.lower()
    for head in heads:
        after = text[len(head) : len(head) + 1]
        if lowered.startswith(head.lower()) and not (after.isalnum() or after == "_"):
            return True
    return False


def _is_tabular_caption(text: str) -> bool:
    if _has_tabular_head(text, ("Instruction Operand Encoding",)):
        return True
    if text[:5].lower() != "table":
        return False
    rest = text[5:].lstrip()
    if rest == text[5:]:
        return False
    digits, sep, pages = rest.partition("-")
    cut = 0
    while cut < len(pages) and pages[cut].isdecimal():
        cut += 1
    return bool(sep) and digits.isdecimal() and cut > 0 and pages[cut : cut + 1] == "."


def _resolve_outline_page_number(pdf, dest) -> int | None:
    """Resolve a pdfminer outline destination to a 1-based page number."""
    if dest is None:
        return None
    try:
        resolved = resolve1(pdf.doc.get_dest(dest) if isinstance(dest, bytes) else dest)
        if isinstance(resolved, dict):
            resolved = resolve1(resolved.get("D"))
        if not isinstance(resolved, list) or not resolved:
            return None
        objid = getattr(resolved[0], "objid", None)
        if objid is None:
            return None
        if not hasattr(pdf, "_simdref_page_map"):
            pdf._simdref_page_map = {
                page.page_obj.pageid: i for i, page in enumerate(pdf.pages, start=1)
            }
        return pdf._simdref_page_map.get(objid)
    except Exception:
        return None


def _outline_starts_instruction_range(level: int, title: str) -> bool:
    lowered = title.casefold()
    if title.startswith("Chapter ") and "instruction" in lowered and "reference" in lowered:
        return True
    return "seam instruction reference" in lowered


def _instruction_page_ranges(pdf) -> list[tuple[int, int]]:
    """Return likely 0-based [start, end) ranges that contain instruction text."""
    total = len(pdf.pages)
    try:
        outlines: list[tuple[int, int, str]] = []
        for level, title, dest, _action, _se in pdf.doc.get_outlines():
            page_number = _resolve_outline_page_number(pdf, dest)
            if page_number is None:
                continue
            outlines.append((page_number, level, title))
        outlines.sort()

        ranges: list[tuple[int, int]] = []
        for idx, (page_number, level, title) in enumerate(outlines):
            if not _outline_starts_instruction_range(level, title):
                continue
            next_page = total + 1
            for later_page, later_level, _later_title in outlines[idx + 1 :]:
                if later_page > page_number and later_level <= level:
                    next_page = later_page
                    break
            start_idx = max(0, page_number - 1)
            end_idx = max(start_idx + 1, min(total, next_page - 1))
            ranges.append((start_idx, end_idx))

        if not ranges:
            return [(0, total)]

        merged: list[tuple[int, int]] = []
        for start_idx, end_idx in sorted(ranges):
            if not merged or start_idx > merged[-1][1]:
                merged.append((start_idx, end_idx))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end_idx))
        return merged
    except Exception:
        return [(0, total)]


def _page_might_have_tables(page) -> bool:
    """Cheap precheck before pdfplumber's expensive table finder."""
    primitive_count = len(page.rects) + len(page.lines) + len(page.curves)
    return primitive_count >= _TABLE_GRAPHIC_PRIMITIVE_THRESHOLD


def _table_bboxes_for_page(page) -> list[tuple[float, float, float, float]]:
    if not _page_might_have_tables(page):
        return []
    tables = page.find_tables()
    if not tables:
        return []
    page_area = page.width * page.height
    bboxes_list = []
    for table in tables:
        bx0, by0, bx1, by1 = table.bbox
        table_area = (bx1 - bx0) * (by1 - by0)
        if table_area / page_area < _TABLE_MAX_PAGE_FRACTION:
            bboxes_list.append((bx0, by0, bx1, by1))
    return bboxes_list


def _build_line_text(spans: list[tuple[float, str]]) -> str:
    parts: list[str] = []
    prev_right = -1.0
    for x0, text in spans:
        if prev_right >= 0 and x0 - prev_right > 10.0:
            if parts and not parts[-1].endswith(" "):
                parts.append(" ")
        parts.append(text)
        prev_right = x0 + max(len(text), 1) * 5.0
    return "".join(parts).strip()


def _line_is_tabular_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    # A kept section heading (for example 64-Bit Mode Exceptions) shares a
    # prefix with the opcode-table column headers. Keep the heading.
    if stripped.lower() in _NOISE_EXEMPT_HEADINGS:
        return False
    return _has_tabular_head(stripped, _TABULAR_LINE_HEADS) or _is_tabular_caption(stripped)


def _prepare_page_pdfplumber(page) -> _PreparedPage:
    page_chars = page.chars
    title_chars = [c for c in page_chars if c["size"] >= _TITLE_MIN_SIZE]
    parsed_title = None
    if title_chars:
        parsed_title = parse_instruction_title("".join(c["text"] for c in title_chars).strip())

    bboxes = _table_bboxes_for_page(page)
    body_chars: list[dict] = []
    for char in page_chars:
        if char["size"] >= _TITLE_MIN_SIZE or char["size"] < _BODY_MIN_SIZE:
            continue
        if bboxes and any(
            bbox[0] <= char["x0"] <= bbox[2] and bbox[1] <= char["top"] <= bbox[3]
            for bbox in bboxes
        ):
            continue
        body_chars.append(char)
    return _PreparedPage(
        title=parsed_title, body_lines=chars_to_lines(body_chars), backend="pdfplumber"
    )


def _prepare_page_from_pymupdf_dict(text_dict: dict[str, Any]) -> _PreparedPage:
    title_lines: list[str] = []
    body_lines: list[tuple[float, float, float, str]] = []

    for block_idx, block in enumerate(text_dict.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line_idx, line in enumerate(block.get("lines", [])):
            spans: list[tuple[float, str]] = []
            sizes: list[float] = []
            tops: list[float] = []
            left_edges: list[float] = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                bbox = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                x0 = float(bbox[0])
                top = float(bbox[1])
                size = float(span.get("size", 0.0))
                spans.append((x0, text))
                sizes.append(size)
                tops.append(top)
                left_edges.append(x0)
            if not spans:
                continue

            spans.sort(key=lambda item: item[0])
            text = _build_line_text(spans)
            if not text:
                continue
            top = min(tops)
            dominant_size = max(sizes)
            x0 = min(left_edges)

            if dominant_size >= _TITLE_MIN_SIZE:
                title_lines.append(text)
                continue
            if dominant_size < _BODY_MIN_SIZE:
                continue
            if _line_is_tabular_noise(text):
                continue
            body_lines.append((top, dominant_size, x0, text))

    title_text = " ".join(title_lines).strip()
    parsed_title = parse_instruction_title(title_text) if title_text else None
    body_lines.sort(key=lambda item: (item[0], item[2], item[1]))
    return _PreparedPage(title=parsed_title, body_lines=body_lines, backend="pymupdf")


def _prepare_page_pymupdf(page) -> _PreparedPage:
    return _prepare_page_from_pymupdf_dict(page.get_text("dict"))


def _prepared_page_needs_fallback(prepared: _PreparedPage) -> str | None:
    if prepared.title is None:
        return None
    if not prepared.body_lines:
        return "empty-fast-path"
    heading_lines = [
        text
        for _top, _size, _x0, text in prepared.body_lines
        if text.strip().lower() in _ALL_HEADING_NAMES
    ]
    if not heading_lines:
        return "missing-heading"
    return None


def normalize_section_name(raw: str) -> str:
    """Map a raw heading string to its canonical section name."""
    stripped = raw.strip()
    lowered = stripped.lower()
    if lowered in _SECTION_ALIASES:
        return _SECTION_ALIASES[lowered]
    for known in KNOWN_SECTIONS:
        if lowered == known.lower():
            return known
    return stripped


def _is_title_head_char(char: str) -> bool:
    return char.isascii() and char.isalnum() or char in "/,_" or char.isspace()


def _is_mnemonic_shape(mnemonic: str) -> bool:
    idx = 0
    while idx < len(mnemonic) and (
        "A" <= mnemonic[idx] <= "Z"
        or "0" <= mnemonic[idx] <= "9"
        or mnemonic[idx] in "/,_"
        or mnemonic[idx].isspace()
    ):
        idx += 1
    return idx >= 1 and all("a" <= char <= "z" for char in mnemonic[idx:])


def parse_instruction_title(text: str) -> tuple[str, str] | None:
    """Parse an instruction title into (mnemonic, summary).

    Returns None if the text is not an instruction title.
    """
    text = text.strip()
    if not ("A" <= text[:1] <= "Z" or "a" <= text[:1] <= "z"):
        return None
    head = 1
    while head < len(text) and head <= 80 and _is_title_head_char(text[head]):
        head += 1
    # The em-dash closes the head run; past 81 head chars only whitespace
    # may still reach it, and summary must keep at least one character.
    if head > 81 and text[81:head].strip():
        return None
    if text[head : head + 1] != "\u2014" or head + 1 == len(text):
        return None
    mnemonic = text[:head].strip()
    summary = text[head + 1 :].strip()
    if any(word in mnemonic for word in _SKIP_WORDS):
        return None
    # PyMuPDF splits tight letter groups into spans with artificial spaces
    # ("CMOV cc"). Mnemonics contain no spaces, so drop them.
    mnemonic = "".join(mnemonic.split())
    # A lowercase letter is valid only in a trailing run (CMOVcc, PREFETCHh).
    # The check rejects mixed-case prose at title size (chapter blurbs).
    if not _is_mnemonic_shape(mnemonic):
        return None
    return mnemonic, summary


def _mnemonic_variants(mnemonic: str) -> list[str]:
    """Expand a parsed title mnemonic into variant keys.

    Comma-separated names are independent titles on one page
    ("MOVDQU,VMOVDQU8/16/32/64"). A slash part that starts with a digit is
    a bare suffix of the first part ("VMOVDQU8/16/32/64").
    """
    variants: list[str] = []
    for name in mnemonic.split(","):
        parts = [part.strip() for part in name.split("/")]
        for idx, part in enumerate(parts):
            if not part:
                continue
            if idx and part[0].isdigit():
                part = parts[0].rstrip("0123456789") + part
            variants.append(part.upper())
    return variants


def parse_intel_sdm(
    pdf_path: Path,
    *,
    status: Callable[[str], None] | None = None,
) -> PdfEnrichmentResult:
    """Parse the Intel SDM PDF and return per-mnemonic description payloads.

    Returns a dict mapping uppercase mnemonic to a payload with:
    * ``sections``: dict of section name -> text
    * ``page_start``: 1-based first page in the PDF
    * ``page_end``: 1-based last page in the PDF

    Mnemonics with ``/`` separators are expanded so each variant maps to the
    same payload.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "PDF ingest needs the simdref[ingest] extra. Install it with: pip install 'simdref[ingest]'"
        ) from exc

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

    log.info("parsing Intel SDM: %s", pdf_path)
    try:
        import pymupdf as fitz
    except ImportError:
        fitz = None

    pdf = pdfplumber.open(pdf_path)
    fitz_doc = fitz.open(pdf_path) if fitz is not None else None
    total = len(pdf.pages)
    log.info("total pages: %d", total)
    page_ranges = _instruction_page_ranges(pdf)
    page_indices = [page_idx for start, end in page_ranges for page_idx in range(start, end)]
    parse_pages = len(page_indices)
    log.info("instruction page ranges: %s (%d pages)", page_ranges, parse_pages)

    interactive_progress = sys.stderr.isatty() and os.environ.get("GITHUB_ACTIONS") != "true"
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
    )
    if interactive_progress:
        progress.start()
    if status is not None:
        status(f"Opened Intel SDM PDF with {total} pages")
        if fitz_doc is not None:
            status("Using PyMuPDF fast path for page extraction with pdfplumber fallback")
        else:
            status("PyMuPDF unavailable; using pdfplumber page extraction")
        if parse_pages != total:
            status(
                f"Restricting SDM parse to {len(page_ranges)} outline-derived instruction ranges "
                f"covering {parse_pages} of {total} pages"
            )

    # Phase 1: preprocess each page once and collect instruction title pages.
    scan_task = (
        progress.add_task("Preprocessing pages", total=parse_pages)
        if interactive_progress
        else None
    )
    prepared_pages: dict[int, _PreparedPage] = {}
    title_pages: list[tuple[int, str, str]] = []
    fallback_pages = 0
    for offset, i in enumerate(page_indices):
        prepared = (
            _prepare_page_pymupdf(fitz_doc[i])
            if fitz_doc is not None
            else _prepare_page_pdfplumber(pdf.pages[i])
        )
        fallback_reason = _prepared_page_needs_fallback(prepared)
        if fallback_reason is not None:
            fallback_pages += 1
            prepared = _prepare_page_pdfplumber(pdf.pages[i])
            prepared.fallback_reason = fallback_reason
        prepared_pages[i] = prepared
        if prepared.title is not None:
            title_pages.append((i, prepared.title[0], prepared.title[1]))
        if interactive_progress and scan_task is not None:
            progress.advance(scan_task)
        elif status is not None and ((offset + 1) % 250 == 0 or offset + 1 == parse_pages):
            status(f"Preprocessing SDM pages: {offset + 1}/{parse_pages}")

    log.info("found %d instruction title pages", len(title_pages))
    if status is not None:
        status(f"Found {len(title_pages)} instruction title pages in Intel SDM")
        if fitz_doc is not None:
            status(f"Fell back to pdfplumber on {fallback_pages} pages")

    # Phase 2: assemble sections from cached per-page lines.
    extract_task = (
        progress.add_task("Extracting descriptions", total=len(title_pages))
        if interactive_progress
        else None
    )
    result: dict[str, PdfDescriptionPayload] = {}
    for idx, (page_start, mnemonic, _summary) in enumerate(title_pages):
        page_end = (
            title_pages[idx + 1][0] if idx + 1 < len(title_pages) else min(page_start + 10, total)
        )

        all_lines: list[tuple[float, float, float, str]] = []
        for page_idx in range(page_start, page_end):
            prepared = prepared_pages.get(page_idx)
            if prepared is not None:
                all_lines.extend(prepared.body_lines)

        raw_sections = extract_sections_from_lines(
            all_lines,
            heading_min_size=_HEADING_MIN_SIZE,
            body_max_size=_BODY_MAX_SIZE,
            known_headings=_ALL_HEADING_NAMES,
        )

        sections: dict[str, str] = {}
        for heading, line_tuples in raw_sections.items():
            canonical = normalize_section_name(heading)
            if canonical in _DISCARD_SECTIONS:
                continue
            # Filter footer and residual junk lines.
            filtered = [
                (x0, text)
                for x0, text in line_tuples
                if not _is_footer_line(text) and not _is_junk_line(text)
            ]
            if not filtered:
                continue
            if canonical in _CODE_SECTIONS:
                # Reconstruct indentation from x0 positions.
                min_x0 = min(x0 for x0, _ in filtered)
                indent_unit = 18.0  # ~18pt per indent level in Intel SDM
                out_lines = []
                for x0, text in filtered:
                    level = round((x0 - min_x0) / indent_unit)
                    out_lines.append("    " * level + text)
                cleaned = "\n".join(out_lines).strip()
            else:
                # Filter footnote lines between tables that have a
                # significantly different left-edge position (x0).
                if len(filtered) > 3:
                    x0_counts: Counter[int] = Counter(round(x0) for x0, _ in filtered)
                    dominant_x0 = x0_counts.most_common(1)[0][0]
                    filtered = [(x0, t) for x0, t in filtered if abs(round(x0) - dominant_x0) <= 3]
                prose_lines = [text for _, text in filtered]
                # Join PDF-wrapped lines into paragraphs.  A new paragraph
                # starts when the previous line ends with sentence-terminal
                # punctuation.  Bullet/numbered list items also start new
                # paragraphs.
                paragraphs: list[str] = []
                for line in prose_lines:
                    if not paragraphs:
                        paragraphs.append(line)
                    elif paragraphs[-1][-1:] in ".):;":
                        paragraphs.append(line)
                    elif line[:1] in ("\u2022", "\u2013") or _is_numbered_item(line):
                        # Bullet points or numbered list items.
                        paragraphs.append(line)
                    elif paragraphs[-1].endswith("-"):
                        # De-hyphenate word breaks (e.g. "indi-\ncate").
                        paragraphs[-1] = paragraphs[-1][:-1] + line
                    else:
                        paragraphs[-1] += " " + line
                cleaned = "\n".join(paragraphs).strip()
            if cleaned:
                sections[canonical] = cleaned

        # The opcode table and the Operand Encoding table sit above the
        # first heading, so heading extraction never keeps them. Parse them
        # from the title page lines.
        title_page = prepared_pages.get(page_start)
        if title_page is not None:
            page_texts = _cell_joined_lines(title_page.body_lines)
            for key, value in form_table_sections(page_texts).items():
                sections.setdefault(key, value)

        payload = PdfDescriptionPayload(
            sections=sections,
            source_url=INTEL_SDM_URL,
            page_start=page_start + 1,
            page_end=page_end,
        )
        for part in _mnemonic_variants(mnemonic):
            result[part] = payload
        if interactive_progress and extract_task is not None:
            progress.advance(extract_task)
        elif status is not None and ((idx + 1) % 50 == 0 or idx + 1 == len(title_pages)):
            status(f"Extracting SDM descriptions: {idx + 1}/{len(title_pages)} instructions")

    if interactive_progress:
        progress.stop()
    if fitz_doc is not None:
        fitz_doc.close()
    pdf.close()
    log.info("extracted descriptions for %d mnemonics", len(result))
    if status is not None:
        status(f"Extracted SDM descriptions for {len(result)} mnemonic variants")
    return PdfEnrichmentResult(
        descriptions=result,
        fallback_page_count=fallback_pages,
        stats={"mnemonic_variants": len(result)},
    )


def find_intel_sdm_pdf() -> Path | None:
    """Locate or download the Intel SDM PDF."""
    for pdf_path in LOCAL_INTEL_SDM_PDFS:
        if pdf_path.exists():
            return pdf_path
    try:
        from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

        dest = LOCAL_INTEL_SDM_PDFS[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            with client.stream("GET", INTEL_SDM_URL) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with Progress(
                    "[progress.description]{task.description}",
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                ) as progress:
                    task = progress.add_task("Downloading Intel SDM PDF", total=total or None)
                    with open(dest, "wb") as fh:
                        for chunk in resp.iter_bytes(65536):
                            fh.write(chunk)
                            progress.advance(task, len(chunk))
        return dest
    except Exception:
        return None


INTEL_PDF_SOURCE = PdfSourceSpec(
    source_id="intel-sdm",
    display_name="Intel SDM",
    source_url=INTEL_SDM_URL,
    local_candidates=tuple(LOCAL_INTEL_SDM_PDFS),
    cache_path=INTEL_SDM_CACHE_PATH,
    cache_version=INTEL_SDM_CACHE_VERSION,
    signature_paths=INTEL_SDM_SIGNATURE_PATHS,
    parser=parse_intel_sdm,
    find_source=find_intel_sdm_pdf,
)

register_pdf_source(INTEL_PDF_SOURCE)
