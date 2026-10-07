# PDF description parsing implementation plan

**Status:** historical implementation plan. The shipped code has moved; `src/simdref/pdfparse/registry.py` and `ingest_pdf.py` supersede the inline merge sketched here. Kept for history.

**Goal:** parse rich instruction descriptions from the Intel SDM PDF and show them as expandable sections in CLI (with pager) and web UI.

**Architecture:** add a `pdfparse` package that extracts per-instruction sections from the Intel SDM PDF using pdfplumber's font-based heading detection. Merge extracted descriptions into existing `InstructionRecord`s during `simdref update`. CLI wraps detail views in a pager; web UI uses collapsible `<details>` elements.

**Tech stack:** pdfplumber, Rich console.pager(), HTML `<details>`

______________________________________________________________________

### Task 1: Add pdfplumber dependency and gitignore PDF

**Files:**

- Modify: `pyproject.toml:15-19`

- Modify: `.gitignore:1-12`

- [ ] **Step 1: Add pdfplumber to dependencies**

In `pyproject.toml`, add pdfplumber to the dependencies list:

```python
dependencies = [
    "httpx>=0.28,<1",
    "pdfplumber>=0.11,<1",
    "rapidfuzz>=3.9,<4",
    "rich>=13,<15",
    "typer>=0.19,<1",
]
```

- [ ] **Step 2: Add PDF to gitignore**

Append to `.gitignore`:

```
# Intel SDM PDF (downloaded during update, not redistributed)
vendor/intel/*.pdf
```

- [ ] **Step 3: Verify install**

Run: `.venv/bin/python -c "import pdfplumber; print(pdfplumber.__version__)"`
Expected: version number printed

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore
git commit -m "chore: add pdfplumber dependency and gitignore vendor PDFs"
```

______________________________________________________________________

### Task 2: Add description field to InstructionRecord

**Files:**

- Modify: `src/simdref/models.py:53-89`

- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_models.py`:

```python
import json
from simdref.models import Catalog, InstructionRecord


def test_instruction_record_has_description_field():
    record = InstructionRecord(
        mnemonic="ADDPS",
        form="ADDPS (XMM, XMM)",
        summary="Add packed single precision floating-point values.",
        description={"Description": "Adds four packed...", "Operation": "DEST[31:0] := ..."},
    )
    assert record.description == {
        "Description": "Adds four packed...",
        "Operation": "DEST[31:0] := ...",
    }


def test_instruction_record_description_defaults_empty():
    record = InstructionRecord(mnemonic="NOP", form="NOP", summary="No operation.")
    assert record.description == {}


def test_catalog_roundtrip_with_description():
    record = InstructionRecord(
        mnemonic="ADDPS",
        form="ADDPS (XMM, XMM)",
        summary="Add packed single precision floating-point values.",
        description={"Description": "Adds four packed...", "Operation": "DEST[31:0] := ..."},
    )
    catalog = Catalog(
        intrinsics=[], instructions=[record], sources=[], generated_at="2026-01-01T00:00:00Z"
    )
    payload = catalog.to_dict()
    roundtripped = Catalog.from_dict(payload)
    assert roundtripped.instructions[0].description == {
        "Description": "Adds four packed...",
        "Operation": "DEST[31:0] := ...",
    }


def test_catalog_from_dict_without_description():
    """Old catalogs without description field should still load."""
    payload = {
        "intrinsics": [],
        "instructions": [
            {
                "mnemonic": "NOP",
                "form": "NOP",
                "summary": "No operation.",
                "isa": [],
                "operands": [],
                "operand_details": [],
                "metadata": {},
                "arch_details": {},
                "linked_intrinsics": [],
                "metrics": {},
                "aliases": [],
                "source": "uops.info",
            }
        ],
        "sources": [],
        "generated_at": "2026-01-01T00:00:00Z",
    }
    catalog = Catalog.from_dict(payload)
    assert catalog.instructions[0].description == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: FAIL on `test_instruction_record_has_description_field` (unexpected keyword argument 'description')

- [ ] **Step 3: Add description field to InstructionRecord**

In `src/simdref/models.py`, add after the `aliases` field (line ~65):

```python
    aliases: list[str] = field(default_factory=list)
    description: dict[str, str] = field(default_factory=dict)
    source: str = "uops.info"
```

Note: `description` must come before `source` since `source` has a default value and dataclass fields with defaults must come after fields without defaults. Both have defaults so ordering among them is flexible.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/simdref/models.py tests/test_models.py
git commit -m "feat: add description field to InstructionRecord"
```

______________________________________________________________________

### Task 3: Create PDF parsing package - base extractor

**Files:**

- Create: `src/simdref/pdfparse/__init__.py`

- Create: `src/simdref/pdfparse/base.py`

- Create: `tests/test_pdfparse.py`

- [ ] **Step 1: Write failing test for base section extractor**

Create `tests/test_pdfparse.py`:

```python
import pytest
from simdref.pdfparse.base import extract_sections_from_chars


def _make_char(text, fontname, size, x0=0, top=0):
    """Create a minimal char dict matching pdfplumber's char format."""
    return {"text": text, "fontname": fontname, "size": size, "x0": x0, "top": top}


def test_extract_sections_basic():
    """Section headers at size 10, body text at size 9."""
    chars = [
        _make_char("D", "NeoSansMedium", 10.0, x0=0, top=100),
        _make_char("e", "NeoSansMedium", 10.0, x0=8, top=100),
        _make_char("s", "NeoSansMedium", 10.0, x0=16, top=100),
        _make_char("c", "NeoSansMedium", 10.0, x0=24, top=100),
        _make_char("r", "NeoSansMedium", 10.0, x0=32, top=100),
        _make_char("i", "NeoSansMedium", 10.0, x0=40, top=100),
        _make_char("p", "NeoSansMedium", 10.0, x0=48, top=100),
        _make_char("t", "NeoSansMedium", 10.0, x0=56, top=100),
        _make_char("i", "NeoSansMedium", 10.0, x0=64, top=100),
        _make_char("o", "NeoSansMedium", 10.0, x0=72, top=100),
        _make_char("n", "NeoSansMedium", 10.0, x0=80, top=100),
        # Body text on next line
        _make_char("B", "Verdana", 9.0, x0=0, top=120),
        _make_char("o", "Verdana", 9.0, x0=8, top=120),
        _make_char("d", "Verdana", 9.0, x0=16, top=120),
        _make_char("y", "Verdana", 9.0, x0=24, top=120),
        _make_char(".", "Verdana", 9.0, x0=32, top=120),
    ]
    sections = extract_sections_from_chars(
        chars,
        heading_min_size=10.0,
        body_max_size=9.5,
    )
    assert "Description" in sections
    assert "Body." in sections["Description"]


def test_extract_sections_multiple():
    """Multiple sections detected in sequence."""
    chars = []
    y = 100
    for ch in "Description":
        chars.append(_make_char(ch, "NeoSansMedium", 10.0, top=y))
    y += 20
    for ch in "First body.":
        chars.append(_make_char(ch, "Verdana", 9.0, top=y))
    y += 40
    for ch in "Operation":
        chars.append(_make_char(ch, "NeoSansMedium", 10.0, top=y))
    y += 20
    for ch in "DEST := SRC":
        chars.append(_make_char(ch, "Verdana", 9.0, top=y))

    sections = extract_sections_from_chars(chars, heading_min_size=10.0, body_max_size=9.5)
    assert "Description" in sections
    assert "First body." in sections["Description"]
    assert "Operation" in sections
    assert "DEST := SRC" in sections["Operation"]


def test_extract_sections_empty_chars():
    sections = extract_sections_from_chars([], heading_min_size=10.0, body_max_size=9.5)
    assert sections == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pdfparse.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'simdref.pdfparse')

- [ ] **Step 3: Implement base extractor**

Create `src/simdref/pdfparse/__init__.py`:

```python
"""PDF parsing pipeline for extracting instruction descriptions.

Provides font-based section extraction from architecture reference PDFs.
ISA-specific modules (e.g., intel.py) define title patterns and section
names for each vendor's manual format.
"""
```

Create `src/simdref/pdfparse/base.py`:

```python
"""Base PDF section extractor using pdfplumber character-level font metadata.

Detects section headings by font size and accumulates body text under each
heading. ISA-specific modules configure the size thresholds and heading
patterns.
"""

from __future__ import annotations


def _chars_to_lines(chars: list[dict]) -> list[tuple[float, float, str]]:
    """Group characters into lines by vertical position.

    Returns a list of ``(top, size, text)`` tuples sorted by vertical
    position. Characters on the same line (within 2pt vertical tolerance)
    are concatenated.
    """
    if not chars:
        return []
    lines: list[tuple[float, list[dict]]] = []
    current_top = chars[0]["top"]
    current_chars: list[dict] = [chars[0]]

    for c in chars[1:]:
        if abs(c["top"] - current_top) > 2.0:
            lines.append((current_top, current_chars))
            current_top = c["top"]
            current_chars = [c]
        else:
            current_chars.append(c)

    lines.append((current_top, current_chars))

    result: list[tuple[float, float, str]] = []
    for top, line_chars in lines:
        text = "".join(c["text"] for c in line_chars).strip()
        max_size = max(c["size"] for c in line_chars)
        if text:
            result.append((top, max_size, text))
    return result


def extract_sections_from_chars(
    chars: list[dict],
    heading_min_size: float,
    body_max_size: float,
) -> dict[str, str]:
    """Extract named sections from a list of pdfplumber character dicts.

    Characters with font size >= *heading_min_size* are treated as section
    headings. Characters with font size <= *body_max_size* are accumulated
    as body text under the current heading.

    Returns a dict mapping heading text to accumulated body text.
    """
    lines = _chars_to_lines(chars)
    sections: dict[str, str] = {}
    current_heading: str | None = None
    body_parts: list[str] = []

    for _top, size, text in lines:
        if size >= heading_min_size:
            if current_heading is not None and body_parts:
                sections[current_heading] = "\n".join(body_parts).strip()
            current_heading = text
            body_parts = []
        elif size <= body_max_size and current_heading is not None:
            body_parts.append(text)

    if current_heading is not None and body_parts:
        sections[current_heading] = "\n".join(body_parts).strip()

    return sections
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pdfparse.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/simdref/pdfparse/__init__.py src/simdref/pdfparse/base.py tests/test_pdfparse.py
git commit -m "feat: add base PDF section extractor with font-based heading detection"
```

______________________________________________________________________

### Task 4: Create Intel SDM parser

**Files:**

- Create: `src/simdref/pdfparse/intel.py`

- Modify: `tests/test_pdfparse.py`

- [ ] **Step 1: Write failing test for Intel parser**

Append to `tests/test_pdfparse.py`:

```python
from simdref.pdfparse.intel import (
    parse_instruction_title,
    KNOWN_SECTIONS,
    normalize_section_name,
)


def test_parse_instruction_title_basic():
    assert parse_instruction_title("ADDPS—Add Packed Single Precision Floating-Point Values") == (
        "ADDPS",
        "Add Packed Single Precision Floating-Point Values",
    )


def test_parse_instruction_title_with_slash():
    result = parse_instruction_title(
        "MOVDQA/VMOVDQA32/VMOVDQA64—Move Aligned Packed Integer Values"
    )
    assert result == ("MOVDQA/VMOVDQA32/VMOVDQA64", "Move Aligned Packed Integer Values")


def test_parse_instruction_title_no_emdash():
    assert parse_instruction_title("CHAPTER 3") is None


def test_parse_instruction_title_lowercase_rejected():
    assert parse_instruction_title("The Intel® Pentium® Processor (1995—") is None


def test_normalize_section_name():
    assert normalize_section_name("Description") == "Description"
    assert (
        normalize_section_name("Intel C/C++ Compiler Intrinsic Equivalent")
        == "Intrinsic Equivalents"
    )
    assert (
        normalize_section_name("Intel C/C++Compiler Intrinsic Equivalent")
        == "Intrinsic Equivalents"
    )
    assert (
        normalize_section_name("SIMD Floating-Point Exceptions") == "SIMD Floating-Point Exceptions"
    )
    assert normalize_section_name("Numeric Exceptions") == "Numeric Exceptions"
    assert normalize_section_name("Other Exceptions") == "Other Exceptions"
    assert normalize_section_name("Flags Affected") == "Flags Affected"


def test_known_sections():
    assert "Description" in KNOWN_SECTIONS
    assert "Operation" in KNOWN_SECTIONS
    assert "Intrinsic Equivalents" in KNOWN_SECTIONS
    assert "Flags Affected" in KNOWN_SECTIONS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pdfparse.py::test_parse_instruction_title_basic -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement Intel SDM parser**

Create `src/simdref/pdfparse/intel.py`:

```python
"""Intel SDM PDF parser.

Extracts per-instruction description sections from the Intel 64 and IA-32
Architectures Software Developer's Manual (combined volumes PDF).

The parser identifies instruction pages by their size-12 title font with
an all-caps mnemonic before an em-dash, then extracts size-10 section
headings and size-9 body text within each instruction's page range.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from simdref.pdfparse.base import extract_sections_from_chars

log = logging.getLogger(__name__)

INTEL_SDM_URL = "https://cdrdv2.intel.com/v1/dl/getContent/671200"

# Title pattern: ALL-CAPS mnemonic (with optional / separators) before em-dash.
_TITLE_RE = re.compile(r"^([A-Z][A-Z0-9/_\s]{0,80})\s*—\s*(.+)")

# Words that indicate a chapter/section heading, not an instruction.
_SKIP_WORDS = frozenset({"CHAPTER", "CONTENTS", "APPENDIX", "VOLUME", "INSTRUCTION SET REFERENCE"})

# Font size thresholds (from empirical analysis of Intel SDM).
_TITLE_MIN_SIZE = 11.5
_HEADING_MIN_SIZE = 9.8
_BODY_MAX_SIZE = 9.5

# Canonical section names and aliases.
KNOWN_SECTIONS: set[str] = {
    "Description",
    "Operation",
    "Intrinsic Equivalents",
    "Flags Affected",
    "Exceptions",
    "Numeric Exceptions",
    "SIMD Floating-Point Exceptions",
    "Other Exceptions",
    "Protected Mode Exceptions",
    "Real-Address Mode Exceptions",
    "Virtual-8086 Mode Exceptions",
    "Compatibility Mode Exceptions",
    "64-Bit Mode Exceptions",
}

_SECTION_ALIASES: dict[str, str] = {
    "intel c/c++ compiler intrinsic equivalent": "Intrinsic Equivalents",
    "intel c/c++compiler intrinsic equivalent": "Intrinsic Equivalents",
    "c/c++ compiler intrinsic equivalent": "Intrinsic Equivalents",
    "intrinsic equivalent": "Intrinsic Equivalents",
    "instruction operand encoding": "Operand Encoding",
}

# Footer pattern: "MNEMONIC—... Vol. 2X N-NN"
_FOOTER_RE = re.compile(r"^.+Vol\.\s*2[A-D]?\s+\d+-\d+$")


def normalize_section_name(raw: str) -> str:
    """Map a raw heading string to its canonical section name."""
    stripped = raw.strip()
    lowered = stripped.lower()
    if lowered in _SECTION_ALIASES:
        return _SECTION_ALIASES[lowered]
    # Check if it's already a known section
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
    # Reject chapter headings and non-mnemonic text
    if any(word in mnemonic for word in _SKIP_WORDS):
        return None
    # Mnemonic must be predominantly uppercase letters/digits
    alpha = [c for c in mnemonic if c.isalpha()]
    if not alpha:
        return None
    if sum(1 for c in alpha if c.isupper()) / len(alpha) < 0.9:
        return None
    return mnemonic, summary


def parse_intel_sdm(pdf_path: Path) -> dict[str, dict[str, str]]:
    """Parse the Intel SDM PDF and return per-mnemonic description sections.

    Returns a dict mapping uppercase mnemonic (e.g., ``"ADDPS"``) to a dict
    of section name -> section text. Mnemonics with ``/`` separators (e.g.,
    ``"MOVDQA/VMOVDQA32/VMOVDQA64"``) are expanded so each variant maps to
    the same sections dict.

    Requires pdfplumber to be installed.
    """
    import pdfplumber

    log.info("parsing Intel SDM: %s", pdf_path)
    pdf = pdfplumber.open(pdf_path)
    total = len(pdf.pages)
    log.info("total pages: %d", total)

    # Phase 1: identify instruction title pages
    title_pages: list[tuple[int, str, str]] = []  # (page_idx, mnemonic, summary)
    for i in range(total):
        chars = pdf.pages[i].chars
        title_chars = [c for c in chars if c["size"] >= _TITLE_MIN_SIZE]
        if not title_chars:
            continue
        title_text = "".join(c["text"] for c in title_chars).strip()
        parsed = parse_instruction_title(title_text)
        if parsed is not None:
            title_pages.append((i, parsed[0], parsed[1]))

    log.info("found %d instruction title pages", len(title_pages))

    # Phase 2: extract sections for each instruction
    result: dict[str, dict[str, str]] = {}
    for idx, (page_start, mnemonic, _summary) in enumerate(title_pages):
        page_end = (
            title_pages[idx + 1][0] if idx + 1 < len(title_pages) else min(page_start + 10, total)
        )

        # Collect all chars across the instruction's pages, excluding title font
        all_chars: list[dict] = []
        for page_idx in range(page_start, page_end):
            page_chars = pdf.pages[page_idx].chars
            # Skip title-sized chars and footer lines
            for c in page_chars:
                if c["size"] < _TITLE_MIN_SIZE:
                    all_chars.append(c)

        raw_sections = extract_sections_from_chars(
            all_chars,
            heading_min_size=_HEADING_MIN_SIZE,
            body_max_size=_BODY_MAX_SIZE,
        )

        # Normalize section names and filter footers from body text
        sections: dict[str, str] = {}
        for heading, body in raw_sections.items():
            canonical = normalize_section_name(heading)
            # Strip footer lines
            lines = [line for line in body.split("\n") if not _FOOTER_RE.match(line)]
            cleaned = "\n".join(lines).strip()
            if cleaned:
                sections[canonical] = cleaned

        # Expand slash-separated mnemonics so each maps to the same sections
        for part in mnemonic.split("/"):
            part = part.strip()
            if part:
                result[part.upper()] = sections

    pdf.close()
    log.info("extracted descriptions for %d mnemonics", len(result))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pdfparse.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/simdref/pdfparse/intel.py tests/test_pdfparse.py
git commit -m "feat: add Intel SDM PDF parser with title detection and section extraction"
```

______________________________________________________________________

### Task 5: Integrate PDF parsing into ingest pipeline

**Files:**

- Modify: `src/simdref/ingest.py`

- [ ] **Step 1: Add Intel SDM PDF fetch and description merge to ingest.py**

Add these constants near the top of `ingest.py` (after existing URL constants):

```python
from simdref.pdfparse.intel import INTEL_SDM_URL, parse_intel_sdm

LOCAL_INTEL_SDM_PDFS = [
    _REPO_ROOT / "vendor" / "intel" / "intel-sdm.pdf",
]
```

Add this function after `fetch_intel_data`:

```python
def _find_intel_sdm_pdf(offline: bool = False) -> Path | None:
    """Locate or download the Intel SDM PDF. Returns path or None."""
    if offline:
        return None
    for pdf_path in LOCAL_INTEL_SDM_PDFS:
        if pdf_path.exists():
            return pdf_path
    # Try downloading
    try:
        dest = LOCAL_INTEL_SDM_PDFS[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            with client.stream("GET", INTEL_SDM_URL) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes(65536):
                        f.write(chunk)
        return dest
    except Exception:
        return None


def _merge_descriptions(
    instructions: list[InstructionRecord],
    descriptions: dict[str, dict[str, str]],
) -> None:
    """Merge parsed PDF descriptions into instruction records in-place."""
    for record in instructions:
        mnemonic = record.mnemonic.upper()
        if mnemonic in descriptions:
            record.description = descriptions[mnemonic]
```

Modify `build_catalog` to call the PDF parser:

```python
def build_catalog(offline: bool = False) -> Catalog:
    intel_text, intel_source = fetch_intel_data(offline=offline)
    uops_text, uops_source = fetch_uops_xml(offline=offline)
    intrinsics = parse_intel_payload(intel_text)
    instructions = parse_uops_xml(uops_text)
    link_records(intrinsics, instructions)

    # Parse Intel SDM PDF for rich descriptions
    sdm_path = _find_intel_sdm_pdf(offline=offline)
    if sdm_path is not None:
        try:
            descriptions = parse_intel_sdm(sdm_path)
            _merge_descriptions(instructions, descriptions)
        except Exception:
            pass  # PDF parsing failure is non-fatal

    return Catalog(
        intrinsics=sorted(intrinsics, key=lambda item: item.name),
        instructions=sorted(instructions, key=lambda item: (item.mnemonic, item.form)),
        sources=[intel_source, uops_source],
        generated_at=now_iso(),
    )
```

- [ ] **Step 2: Verify import works**

Run: `.venv/bin/python -c "from simdref.ingest import build_catalog; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/simdref/ingest.py
git commit -m "feat: integrate Intel SDM PDF parsing into catalog build pipeline"
```

______________________________________________________________________

### Task 6: Bump SQLite schema version

**Files:**

- Modify: `src/simdref/storage.py:58`

- [ ] **Step 1: Bump schema version**

In `src/simdref/storage.py`, change:

```python
SQLITE_SCHEMA_VERSION = "4"
```

The `description` dict is already serialized inside the JSON `payload` column via `asdict(record)`, so no SQL schema changes are needed. The version bump forces a rebuild so existing DBs get the new data.

- [ ] **Step 2: Verify schema check detects stale DB**

Run: `.venv/bin/python -c "from simdref.storage import sqlite_schema_is_current; print('current:', sqlite_schema_is_current())"`
Expected: `current: False` (because existing DB has version 3)

- [ ] **Step 3: Commit**

```bash
git add src/simdref/storage.py
git commit -m "chore: bump SQLite schema version to 4 for description field"
```

______________________________________________________________________

### Task 7: Add --short flag and pager to CLI

**Files:**

- Modify: `src/simdref/cli.py`

- Modify: `src/simdref/display.py`

- [ ] **Step 1: Add short flag to CLI global state**

In `src/simdref/cli.py`, add after `SHOW_FP16_ISAS = False`:

```python
SHORT_MODE = False
```

In the `main()` function, add `--short`/`-s` handling alongside the existing `--fp16` parsing:

```python
def main() -> int:
    """CLI entry point — dispatches to subcommand or smart lookup."""
    global SHOW_FP16_ISAS, SHORT_MODE
    argv = sys.argv[1:]
    if "--fp16" in argv:
        SHOW_FP16_ISAS = True
        argv = [arg for arg in argv if arg != "--fp16"]
    if "--short" in argv or "-s" in argv:
        SHORT_MODE = True
        argv = [arg for arg in argv if arg not in ("--short", "-s")]
        sys.argv = [sys.argv[0], *argv]
    else:
        sys.argv = [sys.argv[0], *argv]
    commands = {
        "update",
        "search",
        "show",
        "man",
        "doctor",
        "tui",
        "export-web",
        "llm",
        "complete",
        "shell-init",
        "--help",
        "-h",
    }
    if argv and argv[0] not in commands and not argv[0].startswith("-"):
        return _smart_lookup(" ".join(argv))
    app()
    return 0
```

- [ ] **Step 2: Add description rendering to display.py**

In `src/simdref/display.py`, add a new function after `print_instruction_metadata`:

```python
def print_description_sections(description: dict[str, str]) -> None:
    """Render instruction description sections as Rich panels."""
    if not description:
        return
    # Preferred display order
    order = [
        "Description",
        "Operation",
        "Intrinsic Equivalents",
        "Flags Affected",
        "Exceptions",
        "SIMD Floating-Point Exceptions",
        "Numeric Exceptions",
        "Other Exceptions",
        "Protected Mode Exceptions",
        "Real-Address Mode Exceptions",
        "Virtual-8086 Mode Exceptions",
        "Compatibility Mode Exceptions",
        "64-Bit Mode Exceptions",
    ]
    shown = set()
    for key in order:
        if key in description:
            console.print(Panel(description[key], title=key, border_style="dim"))
            shown.add(key)
    for key, value in description.items():
        if key not in shown:
            console.print(Panel(value, title=key, border_style="dim"))
```

- [ ] **Step 3: Integrate description and pager into render functions**

Modify `render_intrinsic` in `display.py` to accept `short` parameter and show descriptions:

```python
def render_intrinsic(catalog, item, conn=None, short: bool = False) -> None:
    """Render full intrinsic detail view to the terminal."""
    table = Table(show_header=False, box=None)
    table.add_row("signature", item.signature or "-")
    table.add_row("header", item.header or "-")
    table.add_row("isa", ", ".join(item.isa) or "-")
    table.add_row("category", item.category or "-")
    table.add_row("notes", "; ".join(item.notes) or "-")
    linked = linked_instruction_records(catalog, item, conn=conn)
    primary = linked[0] if linked else None
    if primary:
        if primary.metadata.get("url"):
            table.add_row("url", canonical_url(primary.metadata["url"]))
        if primary.metadata.get("url-ref"):
            table.add_row("reference", canonical_url(primary.metadata["url-ref"]))
    console.print(Panel(table, title=f"intrinsic: {item.name}", border_style="cyan"))
    if not short and primary and primary.description:
        print_description_sections(primary.description)
    if linked:
        console.print(Rule("intrinsic to instruction mapping", style="cyan"))
        print_instruction_mapping(catalog, item, conn=conn)
        console.print(
            Rule(f"instruction details: {display_instruction_title(primary)}", style="magenta")
        )
        print_operand_block(primary)
        print_generic_table(
            measurement_rows(primary),
            "measurements",
            preferred_order=_MEASUREMENT_PREFERRED_ORDER,
            border_style="green",
            exclude_keys=_MEASUREMENT_EXCLUDE_KEYS,
            include_extras=False,
        )
```

Modify `render_instruction_sections` similarly:

```python
def render_instruction_sections(
    catalog, item, include_title: bool = True, conn=None, short: bool = False
) -> None:
    """Render instruction detail with optional title panel."""
    if include_title:
        table = Table(show_header=False, box=None)
        table.add_row("mnemonic", item.mnemonic)
        table.add_row("form", display_instruction_form(item.form))
        table.add_row("isa", display_isa(item.isa))
        table.add_row("summary", item.summary or "-")
        url = item.metadata.get("url", "")
        if url:
            table.add_row("url", canonical_url(url))
        if item.metadata.get("url-ref"):
            table.add_row("reference", canonical_url(item.metadata["url-ref"]))
        if item.metadata.get("category"):
            table.add_row("category", item.metadata["category"])
        if item.metadata.get("cpl"):
            table.add_row("cpl", item.metadata["cpl"])
        console.print(
            Panel(
                table,
                title=f"instruction: {display_instruction_title(item)}",
                border_style="magenta",
            )
        )
    else:
        print_instruction_metadata(item)
    if not short and item.description:
        print_description_sections(item.description)
    console.print(Rule("instruction to intrinsic mapping", style="cyan"))
    print_intrinsic_mapping(catalog, item, conn=conn)
    print_operand_block(item)
    print_generic_table(
        measurement_rows(item),
        "measurements",
        preferred_order=_MEASUREMENT_PREFERRED_ORDER,
        border_style="green",
        exclude_keys=_MEASUREMENT_EXCLUDE_KEYS,
        include_extras=False,
    )
```

Update `render_instruction` to pass through `short`:

```python
def render_instruction(catalog, item, conn=None, short: bool = False) -> None:
    """Render full instruction detail view."""
    render_instruction_sections(catalog, item, include_title=True, conn=conn, short=short)
```

- [ ] **Step 4: Wire pager and short flag in cli.py**

In `cli.py`, update `_smart_lookup` to use pager when not in short mode. Replace the function:

```python
def _smart_lookup(query: str) -> int:
    ensure_runtime()
    short = SHORT_MODE

    family_query = (
        " ".join(query.split()[:-1]).strip()
        if query.split() and query.split()[-1].isdigit()
        else query
    )
    family_items = _find_instruction_family_fast(family_query)
    if family_items:
        indexed_family_variant = _select_instruction_variant(None, query, family_items)
        if indexed_family_variant is not None:
            with open_db() as conn:
                with console.pager(styles=not short) if not short else _nullcontext():
                    render_instruction(None, indexed_family_variant, conn=conn, short=short)
            return 0
        if family_query.casefold() == query.casefold():
            render_instruction_variants(query, family_items, show_fp16=SHOW_FP16_ISAS)
            return 0
    with open_db() as conn:
        intrinsic = load_intrinsic_from_db(conn, query)
        if intrinsic is not None:
            with console.pager(styles=not short) if not short else _nullcontext():
                render_intrinsic(None, intrinsic, conn=conn, short=short)
            return 0
        indexed_variant = _select_instruction_variant(
            None,
            query,
            _find_instructions_fast(" ".join(query.split()[:-1]))
            if query.split() and query.split()[-1].isdigit()
            else [],
        )
        if indexed_variant is not None:
            with console.pager(styles=not short) if not short else _nullcontext():
                render_instruction(None, indexed_variant, conn=conn, short=short)
            return 0
        instructions = _find_instructions_fast(query)
        if instructions:
            exact_form = next(
                (item for item in instructions if item.key.casefold() == query.casefold()), None
            )
            if exact_form is not None:
                with console.pager(styles=not short) if not short else _nullcontext():
                    render_instruction(None, exact_form, conn=conn, short=short)
            elif len(instructions) == 1:
                with console.pager(styles=not short) if not short else _nullcontext():
                    render_instruction(None, instructions[0], conn=conn, short=short)
            else:
                render_instruction_variants(query, instructions, show_fp16=SHOW_FP16_ISAS)
            return 0
    with open_db() as conn:
        _print_search_results_runtime(conn, query)
    return 0
```

Add a null context manager helper at the top of `cli.py` (after imports):

```python
from contextlib import nullcontext as _nullcontext
```

- [ ] **Step 5: Verify it compiles**

Run: `.venv/bin/python -c "from simdref.cli import main; print('OK')"`
Expected: OK

- [ ] **Step 6: Commit**

```bash
git add src/simdref/cli.py src/simdref/display.py
git commit -m "feat: add --short flag, pager output, and description section rendering"
```

______________________________________________________________________

### Task 8: Update web UI with expandable description sections

**Files:**

- Modify: `src/simdref/web.py`

- Modify: `src/simdref/templates/app.js`

- Modify: `src/simdref/templates/style.css`

- [ ] **Step 1: Include description in web detail chunks**

In `src/simdref/web.py`, modify `_detail_chunks` to include `description`:

In the dict comprehension inside `_detail_chunks`, add after `"summary": item.summary`:

```python
            "description": item.description,
```

So the chunk entry becomes:

```python
        chunks[prefix][item.key] = {
            "mnemonic": item.mnemonic,
            "form": item.form,
            "summary": item.summary,
            "description": item.description,
            "isa": item.isa,
            ...
        }
```

- [ ] **Step 2: Add CSS for description sections**

Append to `src/simdref/templates/style.css`:

```css
/* Description sections */
.desc-section { margin-bottom: 0.5rem; }
.desc-section summary {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-secondary);
  cursor: pointer;
  padding: 0.3rem 0;
}
.desc-section summary:hover { color: var(--text); }
.desc-body {
  font-size: 0.82rem;
  line-height: 1.6;
  padding: 0.5rem 0;
  white-space: pre-wrap;
}
.desc-code {
  font-family: var(--font-mono);
  font-size: 0.78rem;
  line-height: 1.5;
  padding: 0.5rem 0.75rem;
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow-x: auto;
  white-space: pre;
}
```

- [ ] **Step 3: Add description rendering to app.js**

Add a helper function in `app.js` after the `renderMeasurements` function:

```javascript
/* ── Description sections ────────────────────────────────────────── */
const descSectionOrder = [
  "Description", "Operation", "Intrinsic Equivalents",
  "Flags Affected", "Exceptions", "SIMD Floating-Point Exceptions",
  "Numeric Exceptions", "Other Exceptions",
  "Protected Mode Exceptions", "Real-Address Mode Exceptions",
  "Virtual-8086 Mode Exceptions", "Compatibility Mode Exceptions",
  "64-Bit Mode Exceptions",
];
const codeSections = new Set(["Operation", "Intrinsic Equivalents"]);

function renderDescriptionSections(description) {
  if (!description || !Object.keys(description).length) return "";
  const shown = new Set();
  let html = "";
  for (const key of descSectionOrder) {
    if (description[key]) {
      const isCode = codeSections.has(key);
      const isFirst = key === "Description";
      html += `<details class="desc-section"${isFirst ? " open" : ""}>
        <summary>${esc(key)}</summary>
        ${isCode ? `<pre class="desc-code">${esc(description[key])}</pre>` : `<div class="desc-body">${esc(description[key])}</div>`}
      </details>`;
      shown.add(key);
    }
  }
  for (const [key, value] of Object.entries(description)) {
    if (!shown.has(key)) {
      html += `<details class="desc-section">
        <summary>${esc(key)}</summary>
        <div class="desc-body">${esc(value)}</div>
      </details>`;
    }
  }
  return html;
}
```

- [ ] **Step 4: Update renderInstructionDetail to use description sections**

In `renderInstructionDetail`, replace the existing Description section:

```javascript
    <section class="section">
      <h3>Description</h3>
      <div>${esc(d.summary || item.summary || "-")}</div>
    </section>
```

With:

```javascript
    <section class="section">
      <h3>Summary</h3>
      <div>${esc(d.summary || item.summary || "-")}</div>
    </section>
    ${d.description && Object.keys(d.description).length ? `<section class="section">
      ${renderDescriptionSections(d.description)}
    </section>` : ""}
```

- [ ] **Step 5: Update renderIntrinsicDetail similarly**

In `renderIntrinsicDetail`, after the existing Description section, add instruction description sections. Replace:

```javascript
    <section class="section">
      <h3>Description</h3>
      <div>${esc(detail ? detail.description : item.description)}</div>
    </section>
```

With:

```javascript
    <section class="section">
      <h3>Description</h3>
      <div>${esc(detail ? detail.description : item.description)}</div>
    </section>
    ${detail && detail._instrDescription && Object.keys(detail._instrDescription).length ? `<section class="section">
      ${renderDescriptionSections(detail._instrDescription)}
    </section>` : ""}
```

Then in the `renderDetail` function where intrinsic details are assembled, after setting `detail._operands`, add:

```javascript
        detail._instrDescription = instrDetail.description || {};
```

- [ ] **Step 6: Verify web export works**

Run: `.venv/bin/python -c "from simdref.web import export_web; print('OK')"`
Expected: OK

- [ ] **Step 7: Commit**

```bash
git add src/simdref/web.py src/simdref/templates/app.js src/simdref/templates/style.css
git commit -m "feat: add expandable description sections to web UI"
```

______________________________________________________________________

### Task 9: Update man pages with description sections

**Files:**

- Modify: `src/simdref/manpages.py`

- [ ] **Step 1: Add description sections to instruction man page**

In `src/simdref/manpages.py`, modify `instruction_page` to include description sections:

```python
def instruction_page(record: InstructionRecord) -> str:
    parts = [f'.TH "{record.mnemonic}" "7" "simdref" "simdref" "SIMD Instruction Reference"\n']
    parts.append(_section("NAME", f"{_roff_escape(record.key)} \\- {_roff_escape(record.summary)}"))
    if record.description.get("Description"):
        parts.append(_section("DESCRIPTION", _roff_escape(record.description["Description"])))
    else:
        parts.append(_section("DESCRIPTION", _roff_escape(record.summary)))
    if record.description.get("Operation"):
        parts.append(
            _section("OPERATION", f".nf\n{_roff_escape(record.description['Operation'])}\n.fi")
        )
    parts.append(_section("ISA", _roff_escape(", ".join(record.isa) or "Unknown")))
    parts.append(
        _section(
            "OPERANDS", _roff_escape("\n".join(record.operands) or "No operand details available.")
        )
    )
    parts.append(
        _section("INTRINSICS", _roff_escape(", ".join(record.linked_intrinsics) or "None linked"))
    )
    parts.append(
        _section(
            "PERFORMANCE DETAILS",
            _roff_escape("\n".join(_metric_lines(record)) or "No performance metrics available."),
        )
    )
    if record.description.get("Flags Affected"):
        parts.append(_section("FLAGS AFFECTED", _roff_escape(record.description["Flags Affected"])))
    for exc_key in (
        "Exceptions",
        "SIMD Floating-Point Exceptions",
        "Numeric Exceptions",
        "Other Exceptions",
    ):
        if record.description.get(exc_key):
            parts.append(_section(exc_key.upper(), _roff_escape(record.description[exc_key])))
    return "".join(parts)
```

- [ ] **Step 2: Verify import works**

Run: `.venv/bin/python -c "from simdref.manpages import instruction_page; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/simdref/manpages.py
git commit -m "feat: include description sections in generated man pages"
```

______________________________________________________________________

### Task 10: Run full test suite and verify

**Files:** (none modified)

- [ ] **Step 1: Run all existing tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests PASS (existing tests should not break since `description` defaults to `{}`)

- [ ] **Step 2: Run new tests**

Run: `.venv/bin/python -m pytest tests/test_models.py tests/test_pdfparse.py -v`
Expected: all tests PASS

- [ ] **Step 3: Verify CLI help works**

Run: `.venv/bin/python -m simdref --help`
Expected: help output without errors

- [ ] **Step 4: Quick smoke test with offline catalog**

Run: `.venv/bin/python -c "from simdref.ingest import build_catalog; c = build_catalog(offline=True); print(f'{len(c.instructions)} instructions, description fields: {sum(1 for i in c.instructions if i.description)}')"`
Expected: instructions count > 0, description fields = 0 (since offline mode has no PDF)

- [ ] **Step 5: Commit any fixes if needed**

If any tests failed, fix and commit:

```bash
git add -u
git commit -m "fix: address test failures from description integration"
```
