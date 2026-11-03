# Upstream sources

simdref builds its catalog from upstream sources. Each entry gives URL, license, refresh cadence, and gaps measured by `tools/audit_coverage.py` against a local catalog build.

Run `python tools/audit_coverage.py report` for live coverage. Snapshot at `docs/coverage/summary.json`.

## x86

### Intel Intrinsics Guide

- URL: <https://cdrdv2.intel.com/v1/dl/getContent/764289> (offline zip), <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/> (index HTML).
- Format: XML in a JavaScript wrapper (`data.js`) in a versioned zip.
- License: no licence grant found in the zip or on the Intel site. Used under the removal-request policy in the README.
- Refresh: a few times per year with new extensions. Pin the local `vendor/intel/` copy, update on releases.
- Gaps: none in the ~7k published intrinsics.

### uops.info

- URL: <https://uops.info/instructions.xml>.
- Format: XML, one `<instruction>` per entry, timings per CPU generation.
- License: no licence text found on <https://uops.info/>. Cite uops.info when you use the data. Used under the removal-request policy in the README.
- Refresh: multiple times per year.
- Gaps: some new AVX-512 refinements can lag.

## Arm

### Arm ACLE intrinsics

- URL: <https://developer.arm.com/architectures/instruction-sets/intrinsics/data/intrinsics.json> plus the ACLE spec at <https://arm-software.github.io/acle/>.
- Format: JSON, the canonical compiler-consumed version.
- License: Arm Developer site terms. ACLE spec: Apache-2.0.
- Refresh: follows ACLE releases.
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
- Refresh: follows RVV spec revisions.
- Gaps: none against the vendored snapshot.

### RISC-V unified DB (instructions)

- URL: <https://github.com/riscv-software-src/riscv-unified-db>, `generated/instructions.json` and fallbacks.
- Format: JSON, ~700 instruction records, plus HTML pages from <https://docs.riscv.org/>.
- License: Apache-2.0.
- Refresh: continuous.
- Gaps: none against the vendored snapshot.

## Microarchitectural perf data

Each perf row has a `source_kind` so users do not confuse modeled numbers with measurements.

### uops.info (x86, measured)

See above. All rows carry `source_kind="measured"`.

### Intel Intrinsics Guide perf tables (x86, modeled)

- URL: <https://cdrdv2.intel.com/v1/dl/getContent/764289?fileName=Intel-Intrinsics-Guide-Offline-3.6.9.zip> (the offline bundle is the only form that ships `perf2.js`).
- Format: `perf2.js` object literal plus `notes.json` in the bundle; derived per-core JSON under `vendor/intel_guide/`.
- Refresh: `python -m simdref.perf_sources.intel_guide refresh RAWDIR` with RAWDIR holding `perf2.js` and `notes.json`.
- License: no licence grant found in the zip or on the Intel site. Used under the removal-request policy in the README.

### Intel Optimization Reference Manual volume 2 (x86, measured)

- URL: <https://cdrdv2-public.intel.com/930022/356477-051-optimization-reference-manual-vol2.pdf>.
- Format: PDF latency and throughput tables (chapter 7 and table 6-19) parsed with `pdftotext -layout`; derived rows in `vendor/intel_optref/rows.json`.
- Refresh: `python -m simdref.perf_sources.intel_optref refresh RAWDIR` with RAWDIR holding the PDF.
- License: no licence grant found; used under the removal-request policy in the README. Notice text (Volume 1 Notices and Disclaimers, Volume 2 repeats it): "No license (express or implied, by estoppel or otherwise) to any intellectual property rights is granted by this document, with the sole exception that a) you may publish an unmodified copy and b) code included in this document is licensed subject to the Zero-Clause BSD open source license (0BSD), https://opensource.org/licenses/0BSD. You may create software implementations based on this document and in compliance with the foregoing that are intended to execute on the Intel product(s) referenced in this document. No rights are granted to create modifications or derivatives of this document."

### InstLatX64 dumps (x86, measured)

- URL: <https://github.com/InstLatx64/InstLatX64>; the per-dump file URL is recorded on each row.
- Format: fixed-column text dumps from AIDA64; derived per-core JSON under `vendor/instlatx64/`.
- Refresh: `python -m simdref.perf_sources.instlatx64 refresh RAWDIR` with RAWDIR holding the dump text files.
- License: no licence grant found; used under the removal-request policy in the README. The dumps carry a FinalWire copyright line and no licence text.

### AMD Software Optimization Guide latency workbooks (x86, modeled)

- URL: Zen 5 doc 58455 and Zen 4 doc 57647 on <https://docs.amd.com/v/u/en-US/58455_1.00> and <https://docs.amd.com/v/u/en-US/57647_1.00>. The Zen 1 to Zen 3 workbooks come from a GitHub mirror of AMD files, <https://github.com/bitRAKE/amd_zen_sogs>.
- Format: xlsx latency workbooks; derived per-core JSON under `vendor/amd_swopt/`.
- Refresh: `python -m simdref.perf_sources.amd_swopt refresh RAWDIR` with RAWDIR holding the xlsx files.
- License: no licence grant found; used under the removal-request policy in the README. Notice text: "Information in the spreadsheet is based on estimates and is subject to change."

### Arm Neoverse Software Optimization Guides (AArch64, modeled)

- URL: one guide PDF per CPU on `documentation-service.arm.com`; the per-CPU URL and issue number live in `src/simdref/perf_sources/arm_swopt.py`.
- Format: PDF instruction tables parsed from `pdftotext -layout` text; derived per-CPU JSON under `vendor/arm_swopt/`.
- Refresh: `python -m simdref.perf_sources.arm_swopt refresh RAWDIR` with RAWDIR holding the PDFs under the file names in the module's `CPUS` table.
- License: no licence grant found; used under the removal-request policy in the README. Notice text (Non-Confidential Proprietary Notice): "This document is protected by copyright and other related rights and the practice or implementation of the information contained in this document may be protected by one or more patents or pending patent applications. No part of this document may be reproduced in any form by any means without the express prior written permission of Arm. No license, express or implied, by estoppel or otherwise to any intellectual property rights is granted by this document unless specifically stated."

### LLVM scheduling models: llvm-exegesis, llvm-mc, llvm-mca (AArch64, RISC-V, modeled)

- Binaries: `llvm-exegesis`, `llvm-mc`, `llvm-mca`, LLVM 18+.
- Driver: `src/simdref/perf_sources/llvm_scheduling.py`, three steps per canonical core:
  1. `llvm-exegesis --benchmark-phase=prepare-and-assemble-snippet` walks LLVM's target-instruction table, gives one YAML document per schedulable opcode with an `assembled_snippet` hex stream.
  1. Frequency-counting fixed-width chunks at natural ISA alignment finds the instruction bytes. No regex, no asm synthesis.
  1. `llvm-mc --disassemble` turns bytes into canonical assembly. `llvm-mca --instruction-tables=full --json` measures `Latency`, `RThroughput` per line. Join key: the assembly mnemonic.
- Runtime: ~3 subprocess calls per core (60 total) instead of tens of thousands of one-snippet calls.
- Cache: intermediate artifacts in `vendor/perf-cache/<triple>/<cpu>/{exegesis.yaml, disassembly.s, mca.json}`. Same-host reruns short-circuit.
- Coverage: 17 AArch64 cores (Cortex-A72/76/78, Cortex-X1/X2, Neoverse-N1/N2/E1/V1/V2/V3/V3AE, Apple M1/M2, A64FX, ThunderX2), 7 RISC-V cores (SiFive U74/X280/P400/P600, XiangShan C908/C910, SpacemiT X60).
- License: Apache-2.0 with LLVM exception.
- `simdref build` uses the `simdref[ingest]` extra. It aborts with an install hint when an LLVM binary is missing. `simdref update` fetches the pre-built release catalog, needs no LLVM.

### RISC-V measured per-instruction perf (not available)

No public upstream publishes per-mnemonic measured RVV latency or throughput. `camel-cdr/rvv-bench-results`, the only candidate examined, publishes kernel-level benchmarks (memcpy, chacha20, mandelbrot) with cycle counts per kernel variant, not per-mnemonic tables. RISC-V per-core rows come only from llvm-mca.

## How refresh works

1. Edit `src/simdref/ingest_sources.py` candidate-URL lists when upstreams move.
1. `python -m simdref update` downloads the pre-built release catalog.
1. `python -m simdref build` rebuilds from live sources. It uses the `simdref[ingest]` extra, and `llvm-exegesis`, `llvm-mc`, `llvm-mca` on PATH.
1. `python tools/audit_coverage.py fetch` re-runs extraction, compares against the fresh catalog, rewrites `docs/coverage/summary.json`.
1. Commit the updated summary. `tests/test_coverage_parity.py` obeys the floors in `docs/coverage/thresholds.toml` on each CI run.
