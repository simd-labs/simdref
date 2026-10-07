# Upstream sources

simdref builds its catalog from upstream feeds. This page records each feed's URL, license, refresh cadence, and current gaps measured by `tools/audit_coverage.py` against a local catalog build.

Run `python tools/audit_coverage.py report` for the live coverage summary. Snapshot at `docs/coverage/summary.json`.

## x86

### Intel Intrinsics Guide

- URL: <https://cdrdv2.intel.com/v1/dl/getContent/764289> (offline zip) and <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/> (index HTML).
- Format: XML embedded in a JavaScript wrapper (`data.js`) inside a versioned zip.
- License: Intel permits redistribution with attribution. See the zip's `LICENSE.TXT`.
- Refresh: a few times a year with new extensions. Pin the local `vendor/intel/` copy and update on releases.
- Known gaps: none in the ~7k published intrinsics.

### uops.info

- URL: <https://uops.info/instructions.xml>.
- Format: XML, one `<instruction>` per entry with timings per CPU generation.
- License: public research data, citation requested (<https://www.uops.info/about.html>).
- Refresh: multiple times per year.
- Known gaps: some very new AVX-512 refinements can lag.

## Arm

### Arm ACLE intrinsics

- URL: <https://developer.arm.com/architectures/instruction-sets/intrinsics/data/intrinsics.json> plus the ACLE spec at <https://arm-software.github.io/acle/>.
- Format: JSON, the canonical compiler-consumed version.
- License: Arm Developer site terms. The ACLE spec is Apache-2.0.
- Refresh: aligned with ACLE releases.
- Known gaps: about 30% of upstream entries are missing. The audit normalizes upstream names by stripping bracketed alternatives (`[__arm_]vddupq[_n]_u8` → `vddupq_u8`), so this is a real ingestion shortfall, probably on the SVE or MVE side. To repair it, extend `parse_arm_intrinsics_payload` in `ingest_catalog.py`.

### Arm AARCHMRS (A64 instructions)

- URL: AARCHMRS tarball distributed on the Arm developer site.
- Format: tar.gz holding JSON machine-readable spec files.
- License: Arm EULA for the machine-readable spec.
- Refresh: follows the Arm architecture revision (yearly).
- Known gaps: offline snapshots use the fixture sample. Live fetch (`SIMDREF_LIVE=1`) covers the full spec.

## RISC-V

### RVV intrinsics

- URL: <https://github.com/riscv-non-isa/riscv-rvv-intrinsic-doc>, `auto-generated/intrinsics.json` and fallback locations.
- Format: JSON, about 75k entries per release.
- License: Apache-2.0.
- Refresh: driven by RVV spec revisions.
- Known gaps: none against the vendored snapshot.

### RISC-V unified DB (instructions)

- URL: <https://github.com/riscv-software-src/riscv-unified-db>, `generated/instructions.json` and fallback paths.
- Format: JSON, about 700 instruction records, plus HTML doc pages from <https://docs.riscv.org/>.
- License: Apache-2.0.
- Refresh: continuous.
- Known gaps: none against the vendored snapshot.

## Microarchitectural perf data

Every perf row carries a `source_kind` so users do not confuse modeled numbers for measurements.

### uops.info (x86, measured)

See above. All rows carry `source_kind="measured"`.

### LLVM scheduling models via llvm-exegesis, llvm-mc, llvm-mca (AArch64 and RISC-V, modeled)

- Binaries: `llvm-exegesis`, `llvm-mc`, `llvm-mca` from LLVM 18 or later.
- Driver: `src/simdref/perf_sources/llvm_scheduling.py` runs three stages per canonical core:
  1. `llvm-exegesis --benchmark-phase=prepare-and-assemble-snippet` walks LLVM's target-instruction table and emits a YAML document per schedulable opcode with an `assembled_snippet` hex stream.
  1. The repeated instruction bytes are recovered by frequency-counting fixed-width chunks at natural ISA alignment. No regex, no asm synthesis.
  1. `llvm-mc --disassemble` turns the bytes into canonical assembly. `llvm-mca --instruction-tables=full --json` measures `Latency` and `RThroughput` per line. The join key is the assembly mnemonic.
- Runtime: about 3 subprocess calls per core (about 60 in total) instead of tens of thousands of one-snippet calls.
- Cache: intermediate artifacts under `vendor/perf-cache/<triple>/<cpu>/{exegesis.yaml, disassembly.s, mca.json}`. Reruns on the same host short-circuit.
- Coverage: 13 AArch64 cores (Cortex-A72/76/78, Cortex-X1/X2, Neoverse-N1/N2/V1/V2, Apple M1/M2, A64FX, ThunderX2) and 7 RISC-V cores (SiFive U74/X280/P400/P600, XiangShan C908/C910, SpacemiT X60).
- License: Apache-2.0 with LLVM exception.
- Build-time requirement: `simdref build` aborts with an install hint when any LLVM binary is missing. `simdref update` fetches the pre-built release catalog and does not need LLVM on PATH.

### RISC-V measured per-instruction perf (not available)

No public upstream publishes per-mnemonic measured RVV latency or throughput. `camel-cdr/rvv-bench-results`, the only candidate evaluated, publishes kernel-level benchmarks (memcpy, chacha20, mandelbrot) with cycle counts per kernel variant, not per-mnemonic tables. RISC-V per-core rows come only from the llvm-mca modeled pipeline.

## How refresh works

1. Edit `src/simdref/ingest_sources.py` candidate-URL lists when upstreams move.
1. `python -m simdref update` downloads the pre-built release catalog. No llvm-mca required.
1. `python -m simdref build` rebuilds from live sources and needs `llvm-exegesis`, `llvm-mc`, `llvm-mca` on PATH.
1. `python tools/audit_coverage.py fetch` re-runs extraction, compares against the freshly-built catalog, and rewrites `docs/coverage/summary.json`.
1. Commit the updated summary. `tests/test_coverage_parity.py` enforces the floors in `docs/coverage/thresholds.toml` on every CI run.
