# simdref

[![CI](https://github.com/simd-labs/simdref/actions/workflows/ci.yml/badge.svg)](https://github.com/simd-labs/simdref/actions/workflows/ci.yml)
[![TestPyPI](https://img.shields.io/pypi/v/simdref?pypiBaseUrl=https%3A%2F%2Ftest.pypi.org&label=TestPyPI)](https://test.pypi.org/project/simdref/)
[![Python](https://img.shields.io/pypi/pyversions/simdref?pypiBaseUrl=https%3A%2F%2Ftest.pypi.org)](https://pypi.org/project/simdref/)

A single searchable reference for SIMD intrinsics and instructions across
**x86 (Intel + uops.info)**, **Arm (ACLE / AARCHMRS)**, and **RISC-V
(RVV + unified-db)**. Runs as a CLI, a Textual TUI, an LSP server,
on-demand manpages (`simdref man` — or run `simdref install-manpages`
so plain `man vpaddd` works), a JSON site-data export for
[simdref-web](https://github.com/simd-labs/simdref-web), and a
structured JSON interface for LLM skills.

[Web App](https://simdref.diamondinoia.com/) ·
[TestPyPI](https://test.pypi.org/project/simdref/) ·
[GitHub](https://github.com/simd-labs/simdref) ·
[Contributing](CONTRIBUTING.md)

<!-- Screenshots are hosted on the `assets/docs` ref so the main
     branch stays lightweight to clone. -->

<p align="center">
  <img alt="simdref TUI" src="https://raw.githubusercontent.com/simd-labs/simdref/refs/assets/docs/img/tui.svg" width="720">
  <br><em>Interactive TUI with ISA filters, ranked results, and measured/modeled performance tables.</em>
</p>

## Install the Claude Code and Codex skill

The `asm-analysis` skill for Claude Code and OpenAI Codex lives in
[simd-labs/simdref-skill](https://github.com/simd-labs/simdref-skill).
Follow the install steps in that repo.

## Install

```bash
pip install simdref
isa update     # download the pre-built catalog
isa doctor     # confirm everything is wired up
isa            # open the TUI
```

The package installs two equivalent executables, **`isa`** (short) and
**`simdref`** (explicit). The rest of this README uses `isa`.

`isa update` pulls the combined catalog (x86 measured + Arm/RISC-V
measured & modeled) from the latest GitHub Release — **no `llvm-mca`
required**. Only contributors doing a full local rebuild with
`isa build` need `llvm-mca` 18+ on `PATH`.

Pre-release builds live on TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ simdref
```

## Quickstart

```bash
isa _mm_add_ps       # exact intrinsic  -> detailed view
isa VPADDD           # exact instruction -> detailed view
isa _mm_add          # fuzzy -> ranked search results
isa mm add           # tokenized query -> intrinsic-biased search
isa ADD              # mnemonic-like -> instruction-biased search
isa VADDPS 2         # pick variant #2 from the last result list
isa                  # open the interactive TUI
```

## Interfaces

**Web app** — a self-contained static SPA with filters and performance
tables, published to GitHub Pages at
[simdref.diamondinoia.com](https://simdref.diamondinoia.com/) and
built from the `simdref-web` repo. This repo exports the
JSON data it consumes:

```bash
isa export --out-dir ./site-data
```

The [live demo](https://simdref.diamondinoia.com/) hosts the same
build — search across ~122k entries with ISA filters and per-uarch perf
tables.

**LSP** — hover docs + completion for intrinsic names and instruction
mnemonics in any LSP-capable editor:

```bash
simdref-lsp                      # speaks JSON-RPC over stdio
```

```lua
-- Neovim
vim.lsp.start({ name = "simdref", cmd = { ".venv/bin/simdref-lsp" } })
```

**LLM interface** — stable JSON / NDJSON for agents and editor skills,
with meaningful exit codes so tools can distinguish *no match* (2),
*ambiguous* (3), and *bad flag* (1):

```bash
isa llm query _mm_add_ps --source-kind measured
echo -e "_mm_add_ps\nVPADDD" | isa llm batch
isa llm list --pattern "*gather*" --isa Intel
```

See [docs/LLM.md](docs/LLM.md) for the full payload shape and a
Claude-skill recipe.

**Assembly annotator** — turn compiler output into a self-documented
`.sa` file. Given `hello_simd.s`:

```asm
dot8:
    vmovups (%rdi), %ymm0
    vmovups (%rsi), %ymm1
    vmulps  %ymm1, %ymm0, %ymm0
    vaddps  %ymm0, %ymm0, %ymm0
    vhaddps %ymm0, %ymm0, %ymm0
    ret
```

```bash
isa annotate hello_simd.s           # writes hello_simd.sa
isa annotate hello_simd.s --arch skylake-x -o -   # to stdout, skylake-x only
```

produces:

```asm
dot8:
    vmovups (%rdi), %ymm0   # Move Unaligned Packed Single Precision FP Values. | lat=10.3c cpi=0.78 [avg of 25 archs, measured]
    vmovups (%rsi), %ymm1   # Move Unaligned Packed Single Precision FP Values. | lat=10.3c cpi=0.78 [avg of 25 archs, measured]
    vmulps  %ymm1, %ymm0, %ymm0   # Multiply Packed Single Precision FP Values. | lat=3.8c cpi=0.54 [avg of 25 archs, measured]
    vaddps  %ymm0, %ymm0, %ymm0   # Add Packed Single Precision FP Values.      | lat=3.1c cpi=0.58 [avg of 25 archs, measured]
    vhaddps %ymm0, %ymm0, %ymm0   # Horizontal Add Packed Single Precision FP.  | lat=5.6c cpi=2.22 [avg of 25 archs, measured]
    ret
```

The output is still valid assembly — comments start with `#`, so `as`
and `ld` still consume it.

## Commands

`isa --help` groups commands into **Commands** (day-to-day) and
**Dev commands** (rebuild / export / completion).

**Commands**

| Command                 | Description                                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `isa`                   | Open the TUI                                                                                                      |
| `isa <query>`           | Open the TUI pre-filled with the query in a TTY; print ranked records to stdout otherwise                         |
| `isa doctor`            | Check the installation — pass/fail per component, non-zero exit on failure                                        |
| `isa update`            | Download the pre-built release catalog (no `llvm-mca` required); `--from-release` for the GitHub Release artifact |
| `isa annotate <file.s>` | Annotate a `.s` assembly file with per-instruction summaries and latency/CPI — writes `<file>.sa`                 |
| `isa llm query <q>`     | Strict lookup → JSON/NDJSON/Markdown (see [docs/LLM.md](docs/LLM.md))                                             |
| `isa llm batch`         | Resolve many queries from stdin in one invocation (NDJSON out)                                                    |
| `isa llm list`          | Dump the `FilterSpec` or stream matching catalog entries                                                          |
| `isa llm schema`        | Print the JSON schema for `llm` payloads                                                                          |

**Dev commands**

| Command                          | Description                                                                                     |
| -------------------------------- | ----------------------------------------------------------------------------------------------- |
| `isa build`                      | Full local rebuild from upstream sources, including Intel SDM parsing (`llvm-mca` 18+ required) |
| `isa export`                     | Export site data (JSON only) — the release contract for the `simdref-web` repo                  |
| `isa completion install [SHELL]` | Install shell completion into the user's profile                                                |
| `isa completion show [SHELL]`    | Print the completion script for a shell                                                         |

## Data sources

| Source                  | What                                                   | Entries¹            |
| ----------------------- | ------------------------------------------------------ | ------------------- |
| Intel Intrinsics Guide  | Signatures, descriptions, ISA, categories              | 7,146 intrinsics    |
| uops.info               | Instructions, operands, latency, throughput, ports     | 22,276 instructions |
| Arm ACLE (NEON/SVE)     | Intrinsic signatures and descriptions                  | 10,791 intrinsics   |
| Arm AARCHMRS (A64)      | Base instruction forms and operand tables              | live-only²          |
| riscv-rvv-intrinsic-doc | RVV intrinsics with deterministic instruction refs     | 74,319 intrinsics   |
| RISC-V unified-db       | RVV instruction forms, ISA tags, Description/Operation | 2,868 instructions  |

¹ Counts from the current vendored snapshot. See
[`docs/coverage/summary.json`](docs/coverage/summary.json) for live
parity against upstream and [`docs/SOURCES.md`](docs/SOURCES.md) for
licenses and refresh cadence.

² The full AARCHMRS A64 spec is only available via live fetch or by
dropping the tarball under `vendor/arm/`.

Every rendered latency / CPI is tagged `(measured, <core>)` or
`(modeled, <core>)` so measured and modeled numbers never get silently
mixed.

### Scope caveats

- Performance data is x86-only in v1.
- RISC-V coverage is RVV-focused — not full scalar or privileged ISA.

## Development

```bash
git clone https://github.com/simd-labs/simdref.git
cd simdref
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/isa build          # requires llvm-mca 18+
.venv/bin/python -m pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full dev flow
(tests, adding a new source, build stages) and
[ARCHITECTURE.md](ARCHITECTURE.md) for module layout.

## License

[GNU General Public License v3.0](LICENSE).
