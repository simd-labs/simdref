# Adding a PDF source

PDF enrichment plugs into `simdref.pdfparse`. Add one module that registers a `PdfSourceSpec`.

## Fields of `PdfSourceSpec`

- `source_id`: stable internal id, used in cache keys and `InstructionRecord.pdf_refs`
- `display_name`: human-facing label
- `source_url`: canonical upstream PDF URL
- `local_candidates`: local/vendor cache path order
- `cache_path`: cache file for parsed descriptions
- `cache_version`: increment when the shape changes
- `signature_paths`: source files whose contents invalidate the cache
- `parser`: returns `PdfEnrichmentResult`
- `find_source`: finds or downloads the PDF, gives a local path

## Parser

Return `PdfEnrichmentResult` with:

- `descriptions`: mnemonic → `PdfDescriptionPayload`
- `fallback_page_count`: pages needing a slower fallback, if any
- `stats`: optional counters

Each `PdfDescriptionPayload` has `sections` (merged text keyed by canonical name), `source_url`, `page_start`, `page_end`.

The parser module owns source-specific constants, heuristics, fallback logic. Generic ingest stays free of section aliases and parser internals.

## Cache invalidation

`ingest_pdf.load_or_parse_pdf_source()` invalidates when one of these changes: `cache_version`, parser signature from `signature_paths`, `source_url`, PDF SHA-256.

Use `cache_version` for shape changes. Use `signature_paths` for behavior changes.

## Data model contract

- Attach references through `InstructionRecord.pdf_refs`, not source-specific metadata keys.
- Keep parsed section text in `InstructionRecord.description`.
- Migration-time compatibility metadata goes in a shared helper, not UI code.

## Tests to write

- registry lookup returns the registered spec
- cache hit/miss on parser signature or PDF checksum changes
- parser unit tests for source-specific extraction rules
- metadata normalization tests for `pdf_refs`
- CLI/TUI/web export tests show normalized refs drawn without source-specific logic
- an integration path proving the source joins a local build

## CI

- Keep GitHub Actions workflow logic generic.
- Add source-specific smoke checks through a shared validation script or Python entrypoint.
