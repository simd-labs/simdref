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
import re
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
_TITLE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9/,_\s]{0,80})\s*—\s*(.+)")

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
_FOOTER_RE = re.compile(r"^.+Vol\.\s*2[A-D]?\s*\d+-\d+$")

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
_JUNK_LINE_RE = re.compile(
    r"^\d+\.\s+See note "  # footnote references (1. See note ...)
    r"|^NOTES:\s*$"  # trailing NOTES: line
)


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


_FASTPATH_TABULAR_LINE_RE = re.compile(
    r"^(Opcode|Op/En|64/32-Bit Mode|64-Bit Mode|32-Bit Mode|CPUID Feature Flag)\b",
    re.IGNORECASE,
)
_FASTPATH_TABULAR_CAPTION_RE = re.compile(
    r"^(Table\s+\d+-\d+\.|Instruction Operand Encoding\b)",
    re.IGNORECASE,
)


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
    return bool(
        _FASTPATH_TABULAR_LINE_RE.match(stripped) or _FASTPATH_TABULAR_CAPTION_RE.match(stripped)
    )


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


def parse_instruction_title(text: str) -> tuple[str, str] | None:
    """Parse an instruction title into (mnemonic, summary).

    Returns None if the text is not an instruction title.
    """
    text = text.strip()
    m = _TITLE_RE.match(text)
    if m is None:
        return None
    mnemonic = m.group(1).strip()
    summary = m.group(2).strip()
    if any(word in mnemonic for word in _SKIP_WORDS):
        return None
    # PyMuPDF splits tight letter groups into spans with artificial spaces
    # ("CMOV cc"). Mnemonics contain no spaces, so drop them.
    mnemonic = re.sub(r"\s+", "", mnemonic)
    # A lowercase letter is valid only in a trailing run (CMOVcc, PREFETCHh).
    # The check rejects mixed-case prose at title size (chapter blurbs).
    if not re.fullmatch(r"[A-Z0-9/,_\s]+[a-z]*", mnemonic):
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
                part = re.sub(r"[0-9]+$", "", parts[0]) + part
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
    import pdfplumber

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

    log.info("parsing Intel SDM: %s", pdf_path)
    try:
        import fitz
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
                if not _FOOTER_RE.match(text) and not _JUNK_LINE_RE.match(text)
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
                    elif line[:1] in ("\u2022", "\u2013") or re.match(r"^\d+\.\s", line):
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
