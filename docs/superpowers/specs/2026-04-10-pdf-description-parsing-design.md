# PDF description parsing design

**Date:** 2026-04-10

## Goal

Add rich instruction descriptions parsed from architecture PDFs to simdref without baking any one vendor into the generic ingest flow. Intel SDM is the first implementation. New sources add one parser module plus a registry entry.

## Architecture

`simdref.pdfparse` exposes a source-neutral registry:

- `PdfSourceSpec`: source id, display name, canonical URL, local candidates, cache metadata, parser callback, source locator
- `PdfDescriptionPayload`: section text plus source URL and page range
- `PdfEnrichmentResult`: mnemonic payload map plus parser stats

The Intel implementation lives behind this interface. Generic ingest calls `ingest_pdf.load_or_parse_pdf_source()` and never reaches into Intel-specific parsing logic.

## Data model

`InstructionRecord` keeps merged section text in `description` and carries normalized PDF references in `pdf_refs`:

```python
@dataclass(slots=True)
class InstructionRecord:
    description: dict[str, str] = field(default_factory=dict)
    pdf_refs: list[dict[str, str]] = field(default_factory=list)
```

Each `pdf_refs` entry includes `source_id`, `label`, `url`, `page_start`, `page_end`.

Legacy `intel-sdm-*` metadata keys remain readable and are still emitted for compatibility during migration.

## Intel parsing pipeline

`pdfparse/intel.py`:

1. Use PDF outlines to narrow parsing to instruction-reference ranges when possible.
1. Run a PyMuPDF fast path to extract title and body lines.
1. Fall back to pdfplumber on pages where the fast path finds no useful headings or no body text.
1. Normalize section headings and preserve indentation for pseudocode sections.
1. Return `PdfEnrichmentResult` for generic merge into instruction records.

## Cache invalidation

`ingest_pdf.load_or_parse_pdf_source()` invalidates cached parse results when any of these change: cache version, parser signature derived from source files, canonical source URL, source PDF SHA-256.

This keeps source-specific cache behavior out of generic catalog assembly.

## Rendering and export

- CLI, TUI, and web export consume normalized `pdf_refs`.
- Shared helpers format PDF references instead of rebuilding Intel-specific strings inline.
- Search behavior is unchanged. PDF-derived text stays out of ranking.

## File map

| File                               | Purpose                                              |
| ---------------------------------- | ---------------------------------------------------- |
| `src/simdref/pdfparse/types.py`    | Source-neutral PDF enrichment dataclasses            |
| `src/simdref/pdfparse/registry.py` | PDF source registration and lookup                   |
| `src/simdref/pdfparse/intel.py`    | Intel SDM parser implementation and registration     |
| `src/simdref/ingest_pdf.py`        | Generic PDF cache/load/merge dispatch                |
| `src/simdref/ingest_sources.py`    | Upstream/local/fixture acquisition                   |
| `src/simdref/ingest_catalog.py`    | Parse/link/assemble catalog records                  |
| `src/simdref/pdfrefs.py`           | Shared normalized PDF ref helpers                    |
| `src/simdref/ingest.py`            | Stable public entrypoints and compatibility wrappers |

## Testing strategy

- registry lookup and cache dispatch tests
- Intel parser tests covering fast path and page-local fallback routing
- metadata normalization tests for `pdf_refs` plus legacy Intel metadata loading
- CLI/TUI/web export tests for normalized PDF ref rendering
- integration coverage for full local builds with Intel SDM enabled

## Out of scope

- ARM or RISC-V parser implementations
- search/ranking changes based on PDF text
- figure or table extraction beyond current plain-text handling
