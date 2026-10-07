# Architecture

## Module layout

```
src/simdref/
  models.py        IntrinsicRecord, InstructionRecord, Catalog
  ingest.py        Public ingest entrypoints and compatibility wrappers
  ingest_sources.py  Source acquisition for Intel and Arm bundles
  ingest_catalog.py  Parse, link, and assemble Catalog records
  ingest_pdf.py    PDF enrichment cache/load/merge dispatch
  storage.py       JSON and SQLite persistence, FTS5 search
  search.py        Fuzzy ranking with intent detection
  perf.py          Latency/throughput extraction helpers
  queries.py       Record-linking and lookup helpers
  display.py       Rich terminal formatting
  pdfrefs.py       PDF reference helpers shared by CLI/TUI/export
  cli.py           Typer commands and smart lookup dispatch
  lsp.py           JSON-RPC language server (hover + completion)
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

- `pdfparse.types.PdfSourceSpec` defines one source entry: id, display name, source URL, local candidates, cache metadata, parser callback.
- `pdfparse.registry` is the only registration point. A new PDF source adds one parser module and one `PdfSourceSpec` registration.
- `ingest_pdf.load_or_parse_pdf_source()` owns cache invalidation. Cache keys are source id, parser signature, source URL, PDF SHA-256.
- `InstructionRecord.pdf_refs` is the normalized public shape consumed by CLI, TUI, and web export. Each ref has `source_id`, `label`, `url`, `page_start`, `page_end`.
- Legacy Intel metadata keys stay readable and writable for compatibility during migration.

## Storage strategy

- `catalog.msgpack` holds the full catalog snapshot for portability and offline use. `_write_atomic` writes a random sibling temp file, copies the existing file's mode to it, then replaces the published file with `os.replace`.
- `catalog.db` holds FTS5 search with BM25 ranking for CLI `search`, `show`, `complete`, and `llm`. The schema is versioned and rebuilt automatically when stale. `build_sqlite` writes a sibling `.tmp` and replaces the published file atomically.

## Search algorithm

`search.py` scores candidates in order:

1. Intent detection. Queries starting with `_mm` bias to intrinsics. Mnemonic-like queries (`add`, `vmov`) bias to instructions.
1. Exact, prefix, substring matches add 220/175/135 points.
1. Normalized token matching splits on `_`, `,`, `{}`.
1. Fuzzy matching uses rapidfuzz `token_set_ratio`, `partial_ratio`, `ratio`.
1. Width family bonus: +22 for a matching SIMD width (`mm256`, `ymm`), −22 for a mismatch.
1. Results below 35 points are dropped.

In the CLI path, FTS5 gives the candidate set (`max(limit * 6, 60)` rows per table) and the scoring pipeline re-ranks them.

## Multi-architecture ingest

- The catalog is assembled from architecture-specific bundles, not one implicit x86 path.
- The x86 bundle is Intel intrinsics + uops.info.
- The Arm bundle is scoped to the `arm` family with `AArch64` documentation coverage in v1 (ACLE intrinsics, A64 instruction docs).
- `IntrinsicRecord.architecture` and `InstructionRecord.architecture` use `x86` or `arm`.
- Instruction storage keys are architecture-aware even when display mnemonics or forms collide.

## x86 support matrix

Status meanings:

- `complete`: covered by the pytest path plus the upstream rebuild path, with blocking validation.
- `strong`: good coverage, but representative rather than exhaustive for some subfamilies.
- `partial`: supported in ingest or presentation, but validation or semantic coverage is incomplete.

| Area          | Source authority                         | Status   | Validation gate                                                 |
| ------------- | ---------------------------------------- | -------- | --------------------------------------------------------------- |
| Ingestion     | Intel Intrinsics Guide + uops.info       | complete | `tools/validate_sources.py` + `tests/test_source_validation.py` |
| Linking       | Intrinsics Guide refs to uops.info forms | strong   | exhaustive link validation plus ambiguity tests                 |
| SDM semantics | Intel SDM PDF/cache                      | strong   | section/anchor validation in the validator path                 |
| Perf          | uops.info measurements                   | complete | parse validation and render checks                              |
| Search        | catalog search + web ranking             | strong   | intrinsic- and instruction-first regressions                    |
| Rendering     | CLI/TUI/web/manpage detail surfaces      | strong   | description, equivalents, and perf-table checks                 |

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

Instruction variants are sorted chronologically by ISA generation in a shared cross-architecture taxonomy. APX and FP16/BF16 variants are hidden by default; pass `--fp16` to show them.

- Top-level families are the x86 groupings plus `Arm`.
- Arm sub-ISAs exposed in CLI, TUI, and web are `NEON`, `SVE`, and `SVE2`.
- `AArch64` is execution-state scope and source context, not the primary UI filter label.
