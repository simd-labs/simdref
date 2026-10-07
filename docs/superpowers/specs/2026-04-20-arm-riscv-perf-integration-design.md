# ARM + RISC-V performance data integration

**Date:** 2026-04-20
**Status:** historical design. The shipped pipeline differs: `simdref/perf_sources/llvm_scheduling.py` drives a three-stage `llvm-exegesis` → `llvm-mc` → `llvm-mca` pipeline per canonical core (see `docs/SOURCES.md`). OSACA and rvv-bench ingestion did not land. This file is kept for design history.

## Goal

Extend simdref's per-microarchitecture performance data (latency, throughput,
ports, pipeline) from x86-only (uops.info) to cover AArch64 and RISC-V, without
misleading users about measurement provenance.

## Non-goals

- Running the measurement harness ourselves.
- Redistributing vendor-licensed material (Arm SWOG PDFs, unlicensed tables).
- Full scalar RISC-V coverage beyond what LLVM sched models already provide.
- Maintaining a fixture corpus that simulates a catalog (see fixture retirement
  below).

## Data sources

All sources are fetched at build time by `simdref build`. Only the derived catalog ships in wheels and GitHub Release artifacts. License compatibility is enforced at the ingest boundary: no AGPL or redistribution-restricted bytes enter the release.

Shipped:

| Source                                  | Kind    | ISA             | Coverage                                          | License                        | Boundary                          |
| --------------------------------------- | ------- | --------------- | ------------------------------------------------- | ------------------------------ | --------------------------------- |
| LLVM sched models via `llvm-mca --json` | modeled | AArch64, RISC-V | 13 AArch64 cores, 7 RISC-V cores (see `cores.py`) | Apache-2.0 with LLVM exception | system `llvm-mca` on builder PATH |

Deferred (not shipped):

- OSACA YAML (`github.com/RRZE-HPC/OSACA`). Measured rows for AArch64. License AGPL-3.0 upstream, parsed rows only.
- `camel-cdr/rvv-bench-results` JSON. Kernel-level, not per-mnemonic.
- dougallj/applecpu. No LICENSE file; human-facing citation link only until licensing is clarified.
- Arm SWOGs. Redistribution-restricted PDFs; cite via `pdf_refs`, do not ingest.
- ARM uops.info equivalent. Does not exist publicly as of 2026-04.

## Data model

Each `InstructionRecord` gains a `perf[]` list. Each entry preserves its
source's natural granularity and carries provenance:

```python
@dataclass(frozen=True)
class PerfEntry:
    source: str  # "llvm-mca", "osaca", "rvv-bench"
    source_kind: str  # "measured" | "modeled"
    source_version: str  # e.g. "llvm-21.1.0", "osaca@<sha>"
    core: str  # canonical core id: "neoverse-v2", "cortex-a72", "sifive-p670", ...
    applies_to: str  # "encoding" | "form" | "class" | "lmul"
    latency: dict[str, str]  # {"cycles": "3", "cycles_mem": "5", ...}
    measurement: dict[str, str]  # {"TP": "0.5", "TP_ports": "1*p01", ...}
    ports: list[str] | None  # canonicalized port labels
    citation_url: str | None  # back-link to authoritative source
```

Existing x86 rows (from uops.info) are migrated into the same `perf[]` shape
during the same build step, so all ISAs use one codepath.

### Core-id canonicalization

Source-specific names (`"NeoverseV2"` in LLVM, `"V2"` in OSACA, `"c910"` in
rvv-bench) are mapped to stable canonical ids (`"neoverse-v2"`, `"cortex-a72"`,
`"thead-c910"`, …) via a table in `simdref/perf/cores.py`. Unknown ids from
upstream log a warning and pass through verbatim; never silently dropped.

## Ranking and rendering

`best_latency` / `best_cpi` become **prefer-measured**:

- If any `perf[]` entry has `source_kind == "measured"`, the summary min is
  taken over measured entries only.
- Otherwise the summary min is taken over modeled entries and the rendered
  value is suffixed with `(modeled)`.
- Every rendered value carries a source-kind label in CLI, TUI, LSP hover, and
  web UI — e.g. `lat=3 (measured, cortex-a72)` or `cpi=0.5 (modeled, neoverse-v2)`.
- The detailed per-core table always shows every `perf[]` entry regardless of
  kind; hiding modeled data is only for the headline "best" number.

LLM JSON output exposes the full `perf[]` list plus pre-computed
`best_latency_measured`, `best_latency_modeled`, `best_cpi_measured`,
`best_cpi_modeled` so consumers can choose their policy.

## Build pipeline

The build adds three ingesters, composed into the existing catalog build:

1. LLVM sched ingester (`simdref/perf_sources/llvm_scheduling.py`).

   - Reads a pinned list of `(triple, cpu)` pairs covering all sched-modeled AArch64 and RISC-V cores.
   - For each instruction mnemonic, invokes `llvm-mca --json --mcpu=<cpu> --mtriple=<triple>` and parses the JSON.
   - Emits `PerfEntry(source="llvm-mca", source_kind="modeled", applies_to="class")`.
   - Records the actual LLVM version in `source_version`.
   - Fails the build if `llvm-mca` is not on PATH, with an install hint.

1. OSACA ingester. Not shipped.

1. rvv-bench ingester. Not shipped.

Each ingester is independent and can fail gracefully (warn, continue). The catalog build never hard-fails because one source is down. Missing sources are logged in `docs/coverage/summary.json` so users can see why a core's data is absent.

## Fixture retirement

The bundled fixture corpus and `simdref update --offline` are gone:

- Users who cannot build locally use `simdref update` to fetch the pre-built release artifact.
- `simdref build` needs network, `llvm-mca`, and upstream sources. It errors when any is missing.
- Tests use inline per-test data (short strings, `tmp_path` files for I/O). Each parser gets direct unit tests against a representative row of its real source format.
- CI runs a smoke test that downloads the current release artifact and exercises the CLI, plus a full `simdref build` job on a runner with `llvm-mca` installed.

This removes the "tiny fake catalog" failure mode where users searched the
offline build and got near-empty results.

## Interfaces affected

- `simdref/models.py`: add `PerfEntry`, migrate `arch_details` to `perf[]`.
- `simdref/perf.py`: `best_latency`/`best_cpi` become prefer-measured; add `best_*_measured` and `best_*_modeled` variants.
- `simdref/perf_sources/`: `llvm_scheduling.py`, `cores.py`, `merge.py`. (The original `simdref/perf/` package idea became `perf_sources/`.)
- `simdref/ingest_sources.py`: register the new ingesters.
- `simdref/cli.py`, `simdref/tui.py`, `simdref/lsp.py`, `simdref/display.py`, `simdref/manpages.py`: render source-kind labels and per-core tables.
- `simdref/queries.py`: allow `--core` filtering across ISAs; expose `--source-kind measured|modeled|any`.
- Drop `src/simdref/fixtures/` and the `--offline` branch in `simdref/cli.py`.
- `docs/SOURCES.md`: document new sources, license boundaries, and the fact that SWOGs and dougallj are intentionally not ingested.

## Testing

- Unit tests per ingester against inline representative rows of each source
  format (no shared fixture corpus).
- Schema round-trip tests for `PerfEntry`.
- Ranking tests covering prefer-measured behavior: measured-present,
  modeled-only, mixed-with-ties.
- Core-id canonicalization tests for known and unknown inputs.
- CLI/TUI/LSP rendering tests asserting every perf value carries a source-kind
  label.
- Integration smoke test on CI: run `--build-local` with `llvm-mca` installed,
  assert at least one measured + modeled row lands for each target ISA.

## Open items deferred to implementation plan

- Exact set of instruction mnemonics to probe with `llvm-mca` per triple.
- Canonical core-id table contents.
- Release artifact layout (whether perf is a separate JSON from the intrinsic
  catalog or merged).
- Web UI filter controls for source-kind.
