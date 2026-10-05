# Architecture

## Module layout

```
src/simdref/
  models.py      Data classes: IntrinsicRecord, InstructionRecord, Catalog
  ingest.py      Stable public ingest entrypoints / compatibility wrappers
  ingest_sources.py  Source acquisition for Intel and Arm source bundles
  ingest_catalog.py  Parse, link, and assemble Catalog records
  ingest_pdf.py      PDF enrichment cache/load/merge dispatch
  storage.py     JSON and SQLite persistence, FTS5 search
  search.py      Fuzzy ranking with intent detection
  perf.py        Shared latency/throughput extraction helpers
  queries.py     Shared record-linking and lookup helpers
  display.py     Rich terminal formatting and rendering
  pdfrefs.py     Normalized PDF reference helpers shared by CLI/TUI/export
  cli.py         Typer commands and smart lookup dispatch
  lsp.py         JSON-RPC language server (hover + completion)
  tui.py         Curses-based interactive search
  manpages.py    Roff manpage generation
  export.py      Site-data JSON export for the simdref-web static app
  pdfparse/      Source-pluggable PDF enrichment implementations + registry
```

## Data flow

```
Intel CDN / uops.info / Arm ACLE / Arm A64 docs / vendor archives / fixtures
                    |
                    v
          ingest_sources.py
         fetch + source versioning
                    |
                    v
          ingest_catalog.py
          parse + link + assemble
                    |
          optional ingest_pdf.py
      cache + dispatch to pdfparse registry
                    |
                    v
              Catalog
                    |
        +-----------+-----------+
        v                       v
  catalog.msgpack          catalog.db
  (portable snapshot)   (SQLite + FTS5)
        |                       |
        v                       v
    export.py            cli.py / lsp.py
  (site-data JSON      (runtime queries)
   for simdref-web)
```

## PDF enrichment

- `pdfparse.types.PdfSourceSpec` defines one source entry: id, display name, source URL, local candidates, cache metadata, and parser callback.
- `pdfparse.registry` is the only registration point. Adding a new PDF source should be limited to implementing one parser module and registering its `PdfSourceSpec`.
- `ingest_pdf.load_or_parse_pdf_source()` owns cache invalidation. Cache keys include source id, parser signature, source URL, and PDF SHA-256.
- `InstructionRecord.pdf_refs` is the normalized public shape consumed by CLI/TUI/web export. Each ref includes `source_id`, `label`, `url`, `page_start`, and `page_end`.
- Legacy Intel metadata keys remain readable and are still written for compatibility during the migration window.

## Storage strategy

- **Msgpack** (`catalog.msgpack`): complete serialised catalog snapshot for
  portability and offline use. `_write_atomic` (storage.py) writes a random
  sibling temp file, sets the existing file's mode on it before the first
  byte, and replaces the published file atomically with `os.replace`.
- **SQLite** (`catalog.db`): FTS5 full-text search with BM25 ranking for fast
  CLI `search`, `show`, `complete`, and `llm` queries. Schema is versioned;
  rebuilt automatically when stale. `build_sqlite` builds into a sibling
  `.tmp` and replaces the published file atomically.

## Search algorithm

The search pipeline in `search.py` scores candidates through multiple factors:

1. **Intent detection** -- queries starting with `_mm` bias towards intrinsics;
   mnemonic-like queries (`add`, `vmov`) bias towards instructions.
1. **Exact/prefix/substring matching** -- strong bonuses (220/175/135 points).
1. **Normalised token matching** -- splits on `_`, `,`, `{}` and compares tokens.
1. **Fuzzy matching** -- rapidfuzz `token_set_ratio`, `partial_ratio`, `ratio`.
1. **Width family bonus** -- +22 for matching SIMD width (`mm256`, `ymm`).
1. **Score threshold** -- results below 35 points are discarded.

For runtime search, FTS5 provides the candidate set (up to 12x the limit) and
the scoring pipeline re-ranks them.

## Multi-architecture ingest

- The catalog is now assembled from architecture-specific bundles instead of one implicit x86 path.
- The current x86 bundle remains Intel intrinsics + uops.info.
- The Arm bundle is scoped to the `arm` family with `AArch64` documentation coverage in v1:
  - Arm ACLE intrinsics sources
  - Arm A64 instruction documentation sources
- `IntrinsicRecord.architecture` and `InstructionRecord.architecture` are explicit and currently use `x86` or `arm`.
- Instruction storage keys are architecture-aware even when display mnemonics/forms collide across families.

## Source notes

- Intel intrinsics: Intel Intrinsics Guide data.
- Intel instructions/perf: uops.info plus optional Intel SDM PDF enrichment.
- Arm intrinsics: official ACLE repository and published docs:
  - https://github.com/ARM-software/acle
  - https://arm-software.github.io/acle/main/
  - https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
- Arm instructions: official Arm A64/AArch64 instruction docs:
  - https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions
  - vendored local instruction imports can also consume the Arm A-profile
    machine-readable BSD archive when placed in `vendor/arm/`

## x86 support matrix

Status meanings:

- `complete`: covered by the standard pytest path plus the upstream rebuild path, with blocking validation for ingest, linkage, and user-visible regressions.
- `strong`: good source authority and regression coverage exist, but coverage is still representative rather than exhaustive for some subfamilies.
- `partial`: supported in ingest or presentation, but validation or semantic coverage is still incomplete.

| Area          | Source authority                                      | Status   | Validation gate                                                                           |
| ------------- | ----------------------------------------------------- | -------- | ----------------------------------------------------------------------------------------- |
| Ingestion     | Intel Intrinsics Guide + uops.info                    | complete | `tools/validate_sources.py` raw-source checks under `tests/test_source_validation.py`     |
| Linking       | Intel Intrinsics Guide refs -> parsed uops.info forms | strong   | exhaustive x86 intrinsic-link validation plus targeted ambiguity tests                    |
| SDM semantics | Intel SDM PDF/cache                                   | strong   | section/anchor validation and coverage thresholds in the same validator path              |
| Perf          | uops.info measurements                                | complete | parse validation plus runtime rendering checks                                            |
| Search        | catalog search + web ranking heuristics               | strong   | pytest regressions for intrinsic-first and instruction-first behavior                     |
| Rendering     | CLI/TUI/web/manpage detail surfaces                   | strong   | pytest regression checks for description sections, intrinsic equivalents, and perf tables |

### x86 family status

| Family                        | Status  | Notes                                                                           |
| ----------------------------- | ------- | ------------------------------------------------------------------------------- |
| SSE / SSE2 / SSE4             | strong  | ingest and linking are solid; SDM semantic checks are representative            |
| AVX / AVX2                    | strong  | search and width-sensitive ranking are regression-tested                        |
| AVX-512                       | strong  | masked and maskz forms are explicitly covered in linking/search/render tests    |
| AMX                           | partial | semantic coverage hooks exist, but fixture coverage is still limited            |
| APX / ADX-style new GPR forms | partial | ranking and linkage scaffolding exist; broader source fixtures are still needed |
| BMI / BMI2                    | strong  | alias-sensitive semantic checks are present                                     |
| AES / SHA / CRC               | strong  | semantic validation includes representative crypto/system families              |
| REP / string / system-control | partial | semantic checks are representative; render/search coverage is lighter           |

## ISA filtering

Instruction variants are sorted chronologically by ISA generation within a
shared cross-architecture taxonomy. APX and FP16/BF16 variants are hidden by
default (pass `--fp16` to show them).

- Top-level families include x86 groupings plus `Arm`.
- Arm sub-ISAs currently exposed in CLI/TUI/web are `NEON`, `SVE`, and `SVE2`.
- `AArch64` is treated as execution-state scope and source context, not as the
  primary UI filter label.
