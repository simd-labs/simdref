# Architecture

## Module layout

```
src/simdref/
  models.py        IntrinsicRecord, InstructionRecord, Catalog
  ingest.py        Public ingest entrypoints and compatibility wrappers
  ingest_sources.py  Source fetch for Intel and Arm bundles
  ingest_catalog.py  Parse, link, assemble Catalog records
  ingest_pdf.py    PDF enrichment cache/load/merge dispatch
  storage.py       JSON and SQLite persistence, FTS5 search
  search.py        Fuzzy ranking with intent detection
  perf.py          Latency/throughput extraction helpers
  queries.py       Record-linking and lookup helpers
  display.py       Rich terminal formatting
  pdfrefs.py       PDF reference helpers shared by CLI/TUI/export
  cli.py           Typer commands and smart lookup dispatch
  lsp.py           JSON-RPC language server (hover, completion)
  tui.py           Curses-based interactive search
  manpages.py      Roff manpage generation
  export.py        Site-data JSON export for the simdref-web static app
  pdfparse/        Source-pluggable PDF enrichment + registry
```

## Data flow

```
Intel CDN / uops.info / Arm ACLE / Arm A64 docs / vendor archives / fixtures
        |
        v
ingest_sources.py     fetch + source versioning
        |
        v
ingest_catalog.py     parse + link + assemble
        |
optional ingest_pdf.py  cache + dispatch to pdfparse registry
        |
        v
     Catalog
        |
  +-----+-----+
  v           v
catalog.    catalog.db
msgpack     (SQLite + FTS5)
  |           |
  v           v
export.py   cli.py / lsp.py
```

## PDF enrichment

- `pdfparse.types.PdfSourceSpec` gives one source entry: id, display name, source URL, local candidates, cache metadata, parser callback.
- `pdfparse.registry` is the only registration point. New PDF sources add one parser module and one `PdfSourceSpec` registration.
- `ingest_pdf.load_or_parse_pdf_source()` owns cache invalidation. Cache keys are source id, parser signature, source URL, PDF SHA-256.
- `InstructionRecord.pdf_refs` is the normalized public shape for CLI, TUI, and web export: `source_id`, `label`, `url`, `page_start`, `page_end`.
- Legacy Intel metadata keys stay readable and writable during migration.

## Storage strategy

- `catalog.msgpack` holds the full catalog snapshot for portability and offline use. `_write_atomic` writes a random sibling temp file, copies the file mode to it, then replaces the published file with `os.replace`.
- `catalog.db` holds FTS5 search with BM25 ranking for CLI `search`, `show`, `complete`, `llm`. simdref rebuilds the database when `schema_version` is stale. `build_sqlite` writes a sibling `.tmp` and replaces the published file atomically.

## Search algorithm

`search.py` scores candidates in this sequence:

1. Intent detection. `_mm`-prefixed queries bias to intrinsics. Mnemonic-like queries (`add`, `vmov`) bias to instructions.
1. Equal, prefix, substring hits add 220/175/135 points.
1. Normalized token overlap on splits of `_`, `,`, `{}`.
1. Fuzzy match uses rapidfuzz `token_set_ratio`, `partial_ratio`, `ratio`.
1. Width family bonus: +22 for the same SIMD width (`mm256`, `ymm`), −22 for a mismatch.
1. Results below 35 points drop out.

In the CLI path, FTS5 gives the candidate set (`max(limit * 6, 60)` rows per table) and the scoring pipeline re-ranks them.

## Multi-architecture ingest

- Architecture-specific bundles assemble the catalog, not one implicit x86 path.
- The x86 bundle is Intel intrinsics + uops.info.
- The Arm bundle covers the `arm` family with `AArch64` documentation in v1 (ACLE intrinsics, A64 instruction docs).
- `IntrinsicRecord.architecture` and `InstructionRecord.architecture` use `x86` or `arm`.
- Instruction storage keys are architecture-aware where display mnemonics or forms collide.

## x86 support matrix

Status meanings:

- `complete`: the pytest path plus the upstream rebuild path cover it, with blocking validation.
- `strong`: good coverage, representative not complete for some subfamilies.
- `partial`: ingest or presentation support. Validation or semantic coverage is not full.

| Area          | Source authority                         | Status   | Validation gate                                                 |
| ------------- | ---------------------------------------- | -------- | --------------------------------------------------------------- |
| Ingestion     | Intel Intrinsics Guide + uops.info       | complete | `tools/validate_sources.py` + `tests/test_source_validation.py` |
| Linking       | Intrinsics Guide refs to uops.info forms | strong   | full link validation plus ambiguity tests                       |
| SDM semantics | Intel SDM PDF/cache                      | strong   | section/anchor validation in the validator path                 |
| Perf          | uops.info measurements                   | complete | parse validation and render checks                              |
| Search        | catalog search + web ranking             | strong   | intrinsic- and instruction-first regressions                    |
| Rendering     | CLI/TUI/web/manpage detail views         | strong   | description, equivalents, and perf-table checks                 |

### x86 family status

| Family                        | Status  |
| ----------------------------- | ------- |
| SSE / SSE2 / SSE4             | strong  |
| AVX / AVX2                    | strong  |
| AVX-512                       | strong  |
| AMX                           | partial |
| APX / ADX-style new GPR forms | partial |
| BMI / BMI2                    | strong  |
| AES / SHA / CRC               | strong  |
| REP / string / system-control | partial |

## ISA filtering

Instruction variants sort by ISA generation in a shared cross-architecture taxonomy. APX and FP16/BF16 variants hide by default. Use `--fp16` to show them.

- Top-level families are the x86 groupings plus `Arm`.
- Arm sub-ISAs shown in CLI, TUI, web: `NEON`, `SVE`, `SVE2`.
- `AArch64` is execution-state scope and source context, not the primary UI filter label.
