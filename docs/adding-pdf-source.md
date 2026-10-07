# Adding a PDF source

PDF enrichment is source-pluggable under `simdref.pdfparse`. Add one module that defines and registers a `PdfSourceSpec`.

## Required `PdfSourceSpec` fields

- `source_id`: stable internal id, also used in cache keys and `InstructionRecord.pdf_refs`
- `display_name`: human-facing label
- `source_url`: canonical upstream PDF URL
- `local_candidates`: preferred local/vendor cache paths
- `cache_path`: cache file for parsed descriptions
- `cache_version`: bump when the serialized shape changes
- `signature_paths`: source files whose contents invalidate the cache
- `parser`: returns `PdfEnrichmentResult`
- `find_source`: locates or downloads the PDF and returns a local path

## Parser responsibilities

Return `PdfEnrichmentResult` with:

- `descriptions`: mnemonic → `PdfDescriptionPayload`
- `fallback_page_count`: pages that needed a slower fallback, if relevant
- `stats`: optional counters

Each `PdfDescriptionPayload` has `sections` (merged section text keyed by canonical name), `source_url`, `page_start`, `page_end`.

The parser module owns all source-specific constants, heuristics, and fallback logic. Generic ingest stays free of page-title patterns, section aliases, and parser internals.

## Cache invalidation

`ingest_pdf.load_or_parse_pdf_source()` invalidates when any of these change: `cache_version`, parser signature from `signature_paths`, canonical `source_url`, PDF SHA-256.

Use `cache_version` for serialized payload shape changes. Use `signature_paths` for parser behavior changes.

## Data model expectations

- Attach references through `InstructionRecord.pdf_refs`, not source-specific metadata keys.
- Keep parsed section text in `InstructionRecord.description`.
- If a source needs compatibility metadata during migration, put it in a shared helper, not in UI code.

## Expected tests

- registry lookup returns the registered spec
- cache hit/miss behavior for parser signature or PDF checksum changes
- parser unit tests for source-specific extraction rules
- metadata normalization tests for `pdf_refs`
- CLI/TUI/web export tests showing normalized refs render without source-specific logic
- an integration path proving the source can join a local build

## CI

- Keep GitHub Actions workflow logic generic.
- Add source-specific smoke checks through a shared validation script or shared Python entrypoint.
