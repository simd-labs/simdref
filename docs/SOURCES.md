# Upstream sources

simdref builds its catalog from upstream feeds. Each entry gives URL, license, refresh cadence, and gaps measured by `tools/audit_coverage.py` against a local catalog build.

Run `python tools/audit_coverage.py report` for live coverage. Snapshot at `docs/coverage/summary.json`.

## x86

### Intel Intrinsics Guide

- URL: <https://cdrdv2.intel.com/v1/dl/getContent/764289> (offline zip), <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/> (index HTML).
- Format: XML in a JavaScript wrapper (`data.js`) in a versioned zip.
- License: Intel permits redistribution with attribution. See `LICENSE.TXT` in the zip.
- Refresh: a few times per year with new extensions. Pin the local `vendor/intel/` copy, update on releases.
- Gaps: none in the ~7k published intrinsics.

### uops.info

- URL: <https://uops.info/instructions.xml>.
- Format: XML, one `<instruction>` per entry, timings per CPU generation.
- License: public research data, citation requested (<https://www.uops.info/about.html>).
- Refresh: multiple times per year.
- Gaps: some new AVX-512 refinements can lag.

## Arm

### Arm ACLE intrinsics

- URL: <https://developer.arm.com/architectures/instruction-sets/intrinsics/data/intrinsics.json> plus the ACLE spec at <https://arm-software.github.io/acle/>.
- Format: JSON, the canonical compiler-consumed version.
- License: Arm Developer site terms. ACLE spec: Apache-2.0.
- Refresh: aligned with ACLE releases.
- Gaps: about 30% of upstream entries missing. The audit normalizes upstream names by stripping bracketed alternatives (`[__arm_]vddupq[_n]_u8` → `vddupq_u8`), so this is a real ingestion shortfall, probably on the SVE or MVE side. Repair: extend `parse_arm_intrinsics_payload` in `ingest_catalog.py`.

### Arm AARCHMRS (A64 instructions)

- URL: AARCHMRS tarball on the Arm developer site.
- Format: tar.gz of JSON machine-readable spec files.
- License: Arm EULA for the machine-readable spec.
- Refresh: follows the Arm architecture revision (yearly).
- Gaps: offline snapshots use the fixture sample. `SIMDREF_LIVE=1` covers the full spec.

## RISC-V

### RVV intrinsics

- URL: <https://github.com/riscv-non-isa/riscv-rvv-intrinsic-doc>, `auto-generated/intrinsics.json` and fallbacks.
- Format: JSON, ~75k entries per release.
- License: Apache-2.0.
- Refresh: RVV spec revisions drive it.
- Gaps: none against the vendored snapshot.

### RISC-V unified DB (instructions)

- URL: <https://github.com/riscv-software-src/riscv-unified-db>, `generated/instructions.json` and fallbacks.
- Format: JSON, ~700 instruction records, plus HTML pages from <https://docs.riscv.org/>.
- License: Apache-2.0.
- Refresh: continuous.
- Gaps: none against the vendored snapshot.

## Microarchitectural perf data

Each perf row carries a `source_kind` so users do not confuse modeled numbers with measurements.

### uops.info (x86, measured)

See above. All rows carry `source_kind="measured"`.

### LLVM scheduling models: llvm-exegesis, llvm-mc, llvm-mca (AArch64, RISC-V, modeled)

- Binaries: `llvm-exegesis`, `llvm-mc`, `llvm-mca`, LLVM 18+.
- Driver: `src/simdref/perf_sources/llvm_scheduling.py`, three steps per canonical core:
  1. `llvm-exegesis --benchmark-phase=prepare-and-assemble-snippet` walks LLVM's target-instruction table, gives one YAML document per schedulable opcode with an `assembled_snippet` hex stream.
  1. Frequency-counting fixed-width chunks at natural ISA alignment recovers the repeated instruction bytes. No regex, no asm synthesis.
  1. `llvm-mc --disassemble` turns bytes into canonical assembly. `llvm-mca --instruction-tables=full --json` measures `Latency`, `RThroughput` per line. Join key: the assembly mnemonic.
- Runtime: ~3 subprocess calls per core (60 total) instead of tens of thousands of one-snippet calls.
- Cache: intermediate artifacts in `vendor/perf-cache/<triple>/<cpu>/{exegesis.yaml, disassembly.s, mca.json}`. Same-host reruns short-circuit.
- Coverage: 13 AArch64 cores (Cortex-A72/76/78, Cortex-X1/X2, Neoverse-N1/N2/V1/V2, Apple M1/M2, A64FX, ThunderX2), 7 RISC-V cores (SiFive U74/X280/P400/P600, XiangShan C908/C910, SpacemiT X60).
- License: Apache-2.0 with LLVM exception.
- `simdref build` aborts with an install hint when an LLVM binary is missing. `simdref update` fetches the pre-built release catalog, needs no LLVM.

### RISC-V measured per-instruction perf (not available)

No public upstream publishes per-mnemonic measured RVV latency or throughput. `camel-cdr/rvv-bench-results`, the only candidate examined, publishes kernel-level benchmarks (memcpy, chacha20, mandelbrot) with cycle counts per kernel variant, not per-mnemonic tables. RISC-V per-core rows come only from llvm-mca.

## How refresh works

1. Edit `src/simdref/ingest_sources.py` candidate-URL lists when upstreams move.
1. `python -m simdref update` downloads the pre-built release catalog.
1. `python -m simdref build` rebuilds from live sources, needs `llvm-exegesis`, `llvm-mc`, `llvm-mca` on PATH.
1. `python tools/audit_coverage.py fetch` re-runs extraction, compares against the fresh catalog, rewrites `docs/coverage/summary.json`.
1. Commit the updated summary. `tests/test_coverage_parity.py` obeys the floors in `docs/coverage/thresholds.toml` on each CI run.
