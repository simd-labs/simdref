# simdref

[![CI](https://github.com/simd-labs/simdref/actions/workflows/ci.yml/badge.svg)](https://github.com/simd-labs/simdref/actions/workflows/ci.yml)
[![TestPyPI](https://img.shields.io/pypi/v/simdref?pypiBaseUrl=https%3A%2F%2Ftest.pypi.org&label=TestPyPI)](https://test.pypi.org/project/simdref/)
[![Python](https://img.shields.io/pypi/pyversions/simdref?pypiBaseUrl=https%3A%2F%2Ftest.pypi.org)](https://pypi.org/project/simdref/)

simdref is a SIMD intrinsic and instruction reference for x86 (Intel +
uops.info), Arm (ACLE, AARCHMRS), and RISC-V (RVV + unified-db). It runs as
a CLI, a TUI, an LSP server, and an on-demand manpage renderer.

[Web app](https://simdref.diamondinoia.com/) ·
[TestPyPI](https://test.pypi.org/project/simdref/) ·
[GitHub](https://github.com/simd-labs/simdref) ·
[Contributing](CONTRIBUTING.md)

<p align="center">
  <img alt="simdref TUI" src="https://raw.githubusercontent.com/simd-labs/simdref/refs/assets/docs/img/tui.svg" width="720">
</p>

## Install the Claude Code and Codex skill

The `asm-analysis` skill lives in
[simd-labs/simdref-skill](https://github.com/simd-labs/simdref-skill). Follow
the install steps in that repo.

## Install

```bash
pip install simdref
isa update     # download the pre-built catalog
isa doctor     # check the install
isa            # open the TUI
```

The package installs two equivalent executables, `isa` and `simdref`. The
catalog download does not use `llvm-mca`. Python 3.11 or newer is
necessary. The plain `pip install simdref` covers queries, the LSP and
`isa update`. `isa build` uses the ingest extra:
`pip install 'simdref[ingest]'`, or `uv tool install 'simdref[ingest]'`
with uv. Only `isa build` uses `llvm-mca` 18+ on `PATH`.

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
isa                  # open the TUI
```

## Interfaces

The web app at [simdref.diamondinoia.com](https://simdref.diamondinoia.com/)
reads the JSON site data this repo exports:

```bash
isa export --out-dir ./site-data
```

The LSP server speaks JSON-RPC on stdio:

```bash
simdref-lsp
```

Editor clients, one for each editor:

- Zed: [zed-asm-intrinsics](https://github.com/simd-labs/zed-asm-intrinsics)
- VS Code: [vscode-simdref](https://github.com/simd-labs/vscode-simdref)
- JetBrains: [jetbrains-simdref](https://github.com/simd-labs/jetbrains-simdref)
- Neovim: [nvim-simdref](https://github.com/simd-labs/nvim-simdref)

Each client shows a one-line brief at the end of each instruction line and
the full manpage on hover, from the local catalog.

The LLM interface writes JSON and NDJSON with exit codes 0 (match), 1 (bad
flag), 2 (no match), 3 (ambiguous):

```bash
isa llm query _mm_add_ps --source-kind measured
echo -e "_mm_add_ps\nVPADDD" | isa llm batch
isa llm list --pattern "*gather*" --isa Intel
```

See [docs/LLM.md](docs/LLM.md) for the payload shape.

The annotator turns compiler `.s` output into an annotated `.sa` file.

```bash
isa annotate hello_simd.s                       # writes hello_simd.sa
isa annotate hello_simd.s --arch skylake-x -o - # stdout, skylake-x only
```

The annotator adds a trailing comment with the summary, latency, and CPI to
each instruction line. The output stays valid assembly.

## Commands

`isa --help` groups commands into Commands and Dev commands.

Commands

| Command                 | Description                                                               |
| ----------------------- | ------------------------------------------------------------------------- |
| `isa`                   | Open the TUI                                                              |
| `isa <query>`           | Open the TUI with the query in a TTY, else print ranked results to stdout |
| `isa doctor`            | Check the install, non-zero exit on failure                               |
| `isa update`            | Download the pre-built catalog, `--from-release` for the GitHub Release   |
| `isa annotate <file.s>` | Annotate a `.s` file with summaries and latency/CPI, writes `<file>.sa`   |
| `isa man <name>`        | Show a manpage, made on demand                                            |
| `isa install-manpages`  | Pre-generate man7 pages so plain `man vpaddd` works                       |
| `isa llm query <q>`     | Strict lookup to JSON, NDJSON, or Markdown                                |
| `isa llm batch`         | Resolve many queries from stdin in one invocation                         |
| `isa llm list`          | Write the FilterSpec or stream matching catalog entries                   |
| `isa llm schema`        | Print the JSON schema for `llm` payloads                                  |

Dev commands

| Command                          | Description                                                                   |
| -------------------------------- | ----------------------------------------------------------------------------- |
| `isa build`                      | Rebuild the catalog from upstream, needs `simdref[ingest]` and `llvm-mca` 18+ |
| `isa export`                     | Export the site-data JSON for the `simdref-web` repo                          |
| `isa completion install [SHELL]` | Install shell completion into the user profile                                |
| `isa completion show [SHELL]`    | Print the completion script                                                   |

## Data sources

| Source                  | What                                                    | Entries¹           |
| ----------------------- | ------------------------------------------------------- | ------------------ |
| Intel Intrinsics Guide  | Signatures, descriptions, ISA, categories               | 7,381 intrinsics   |
| uops.info               | Instructions, operands, latency, throughput, ports      | 2,558 instructions |
| Arm ACLE (NEON, SVE)    | Intrinsic signatures and descriptions                   | 10,791 intrinsics  |
| Arm AARCHMRS (A64)      | Base instruction forms and operand tables               | live-only²         |
| riscv-rvv-intrinsic-doc | RVV intrinsics with deterministic instruction refs      | 74,289 intrinsics  |
| RISC-V unified-db       | RVV instruction forms, ISA tags, description, operation | 672 instructions   |
| Intel Guide perf tables | Per-core latency and throughput                         | 9,134 rows         |
| Intel Optimization Ref. | Latency and throughput tables (volume 2)                | 3,155 rows         |
| InstLatX64              | Measured latency and throughput dumps                   | 16,235 rows        |
| AMD Software Opt. Guide | Latency workbooks, Zen 1 to Zen 5                       | 8,386 rows         |
| Arm Neoverse SWOPT      | Instruction tables for 8 CPUs                           | 10,213 rows        |

¹ Counts from the vendored snapshot. See
[`docs/coverage/summary.json`](docs/coverage/summary.json) for parity
against upstream and [`docs/SOURCES.md`](docs/SOURCES.md) for licenses.

### Removal requests

Some sources in [`docs/SOURCES.md`](docs/SOURCES.md) do not state a licence. If you own one of these sources and want simdref to stop using it, open an issue at <https://github.com/simd-labs/simdref/issues>. The maintainers then remove the data and the links from the next release.

² The full AARCHMRS A64 spec needs a live fetch or a tarball under
`vendor/arm/`.

Each perf row carries a `(measured, <core>)` or `(modeled, <core>)` tag.

### Scope

- Performance data covers x86, AArch64 and RISC-V: measured rows from uops.info and InstLatX64, vendor-guide rows from Intel, AMD and Arm, and modeled rows from the LLVM scheduling models.
- RISC-V coverage is RVV only, not scalar or privileged ISA.

## Development

```bash
git clone https://github.com/simd-labs/simdref.git
cd simdref
python3 -m venv .venv
.venv/bin/pip install -e '.[ingest]'
.venv/bin/isa build          # needs llvm-mca 18+
.venv/bin/python -m pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[ARCHITECTURE.md](ARCHITECTURE.md).

## License

[GNU General Public License v3.0](LICENSE).
