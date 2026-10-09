# Changelog

All notable changes to `simdref` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- A failed catalog auto-update (for example when offline) no longer stamps the
  new version. The next command retries the download instead of keeping the
  old catalog until the next release.

## [0.0.12], 2026-10-07

### Fixed

- Add tests for UTF-16 inlay hint columns.

## [0.0.11], 2026-10-07

### Fixed

- The LSP hover takes the mnemonic from the parsed assembler segment and
  converts the UTF-16 column to a code-point index. A second instruction in
  one `asm("...\n\t...")` literal and a mnemonic after an astral character now
  get their page.

## [0.0.10], 2026-10-07

### Fixed

- The LSP hover now parses the same assembler segment as the inlay hint. In
  C/C++ documents the hover reads the asm string literal under the cursor, and
  in .asm and .nasm documents the hover knows `;` starts a comment.

## [0.0.9], 2026-10-07

### Changed

- The LSP hover on an instruction now shows the full `isa` page for the
  operand-matched form: metadata, every description section, the intrinsic
  mapping and the per-microarchitecture performance table, plus the web deep
  link and the uops.info and felixcloutier.com reference URLs. The hover picks
  the same form the inlay hint names, so `vaddps ymm0, ymm1, ymm2` shows the
  YMM form instead of the first masked row.

## [0.0.6], 2026-09-29

Validation release for the 3-repo split (core/web/skill), cut on a temporary
test fork. Not for general use.

### Added

- **`simdref llm query` / `llm batch`** gained `--arch <core>`. It pins `lat`
  and `cpi` to one microarchitecture and drops the instruction forms that
  core cannot execute. Without it both scalars are the best value across
  every part in the catalog (issue #24).
- Every `llm` instruction and intrinsic record now carries `timing`, a map
  from canonical core id to `lat`, `cpi`, the upstream `ports` string and
  `uops`. Latency and throughput are properties of the part, not of the ISA,
  so a single scalar could not represent them (issue #24).
- `llm` payloads now carry `generated_at` and `source_versions`, so a
  consumer can tell which catalog build produced a number (issue #24).

### Fixed

- **`simdref show <mn> --arch <core>`** no longer stamps `[measured]` on rows
  that hold no measurement for that core; an empty row is tagged
  `[missing:<core>]`, matching `annotate`. Variants the core cannot execute
  are omitted with a count line instead of being listed with a blank perf
  row (issue #23).
- **`install.sh`** no longer passes `--no-build-isolation` to `uv pip install`. A fresh `uv venv` carries no setuptools, so the wheel build
  failed with `ModuleNotFoundError: No module named 'setuptools'` (issue
  #25).
- **`build_sqlite`** now checkpoints the WAL before the atomic publish rename.
  The rename moves only the main file, so on CPython 3.11 (which does not
  checkpoint on close while a cursor is alive) `catalog.db` was published as a
  bare 4096-byte header with all 16 tables orphaned in `catalog.db.tmp-wal`.
  Found by widening the CI matrix; 3.10, 3.12, 3.13 and 3.14 were unaffected.

### Changed

- **python support** now spans the whole non-EOL CPython range:
  `requires-python = ">=3.10"` (3.10 goes EOL 2026-10-31). CI runs the full
  test suite on every one of 3.10 through 3.14 instead of 3.12 alone.
  `datetime.UTC` is replaced by `timezone.utc`, and `tomllib` is imported with
  a `tomli` fallback in the five dev-only call sites that read a TOML file.
- **dependencies** upgraded to their latest releases and given explicit
  bounds: `click>=8.5`, `httpx>=0.28`, `msgpack>=1.2.1`, `pdfplumber>=0.11.10`,
  `PyMuPDF>=1.28`, `PyYAML>=6.0.3`, `rapidfuzz>=3.14`, `rich>=15`,
  `textual>=8.2`, `typer>=0.27`, each capped at the next major (next minor for
  0.x). The lower bound is the version CI resolves and tests; the `msgpack`
  floor sits above GHSA-6v7p-g79w-8964. This clears the 19 Dependabot
  advisories that were open against the lockfile.
- **ci:** Dependabot now opens one grouped weekly PR per ecosystem
  (`github-actions` and `uv`), and every workflow action is on its current
  major, so no job runs on the deprecated Node 20 runtime.

## [0.0.5], 2026-07-31

### Fixed

- **`simdref annotate`** now parses `objdump -d`-style lines (`addr: bytes mnemonic operands`) in its default pass-through mode instead of echoing
  them verbatim with exit 0; a warning is emitted on stderr when no
  instruction lines are recognized (issue #21).
- **`simdref profile run`** (issue #22):
  - Hot samples are no longer silently misattributed to `main` on C++
    binaries: `perf script` now runs with `--no-demangle` (demangled names
    contain spaces and could not match the sample regex) and duplicate
    local symbols from thin-LTO are disambiguated using `nm -S` sizes.
  - The pipeline annotates only the top-N hot loops (a `hot.disasm.s`
    excerpt) instead of the whole-binary disassembly, so it no longer
    appears to hang for tens of minutes on large binaries and no longer
    produces 100MB+ artifacts; stale `perf.data.old` backups are removed
    before recording.

### Changed

- **data footprint:** install/update now materializes only
  `catalog.db`; the ~120k-file manpage tree, the static web export, and
  the legacy `catalog.json` are no longer written by default (render on
  demand via `simdref man` / `simdref web`). Manpages are opt-in via
  `simdref install-manpages`, which targets the XDG data-root man dir
  (`~/.local/share/man`) so plain `man vpaddd` works out of the box on
  man-db systems.
- **storage:** SQLite payloads are msgpack+zlib compressed
  (schema v13; `catalog.db` ~487MB → ~233MB); `catalog.msgpack` is deleted
  after a successful install/update and transparently rebuilt from the
  database when needed.
- **ci:** `build-catalog` installs LLVM 22 (`llvm.sh 23` broke: trunk moved,
  apt finds no `clang-23`); the weekly RISC-V source-validation gate is
  deterministic again — upstream unified-db now ships `operation()` bodies
  empty for 847/1326 instructions (post-sail() removal) and the unversioned
  docs.riscv.org URLs serve 4xx-byte redirect stubs, so the fallback
  semantics source `vendor/riscv/docs_pages.json` is now committed
  (un-ignored) and guarded by a redirect-stub regression test.
- **perf:** the profile pipeline's speedup is algorithmic — annotate now runs
  on a hot-loops excerpt instead of the whole disassembly and caches mnemonic
  lookups; the string-method fast paths in the parsers add a further ~1.3×
  on a 935k-line objdump disassembly.

## [0.0.4], 2026-06-05

- **deps:** declare `click` as a direct dependency so a clean install can run
  the CLI without relying on it being pulled in transitively.
- **packaging:** derive `__version__` from installed package metadata, ending
  the drift between the module version and `pyproject.toml`.
- **release:** every release now publishes a versioned `data-v<version>` build
  automatically — the release workflow dispatches the data build on the freshly
  pushed tag, since release events created with `GITHUB_TOKEN` don't cascade.
- **asm-analysis skill:** added paired-interleaved benchmark, codegen-audit, and
  memory-traffic gates to the workflow; trimmed the skill description under the
  1024-char frontmatter cap without changing what triggers it.
- **web:** split the search index into three shards and load intrinsics lazily;
  extend kind/family buckets when intrinsics arrive in Phase 2.
- **catalog:** auto-update on version change; capture Intel Operation pseudocode
  and source URL.

## [0.0.3], 2026-04-29

- **`simdref annotate`** — annotate a `.s` assembly file with per-instruction
  summaries and latency/CPI figures, emitting a `.sa` file that still
  assembles. `--performance` and `--docs` are on by default; `--arch` pins
  output to a specific microarch, otherwise aggregates across measured
  archs (`--agg avg|median|best|worst`).
- **ci:** fix runtime-validation step that still checked for the removed
  monolithic `web/intrinsic-details.json`; it now validates the per-prefix
  chunks under `web/intrinsic-chunks/`.
- **TUI:** respect kind toggles inside `_fts_search`; show no-sub-family
  instructions and bias ranking by query kind.

## [0.0.1], 2026-04-28

- **`simdref profile`** documented alongside the bare-query CLI form in the
  README.
- **Codex plugin** marketplace and `asm-analysis` skill added, single-sourced
  from the canonical `skill/` tree via `scripts/build_skill_bundles.py`.
- **TUI:** ISA presets (`intel`/`arm32`/`arm64`/`riscv`) now include
  instructions, not just intrinsics.
- **CLI:** report progress during first-run catalog bootstrap; bare-query
  dispatch resolves `--arch` and skips the TUI on an exact match.
- **packaging:** drop the stray `simdref-build-skills` console script.

## [0.0.0], 2026-04-22 — initial public release

First tagged release. The baseline set of interfaces the project aims to
support is in place:

- **CLI** with smart bare-word lookup (`simdref _mm_add_ps`, `simdref VPADDD`,
  fuzzy multi-token queries), grouped `--help` output (Commands / Dev commands),
  and stable exit codes. Also installed under the `isa` alias — both
  executables accept every subcommand.
- **`simdref build`** — full local rebuild from upstream sources
  (including Intel SDM parsing); replaces the old `simdref update --build` flag.
- **`simdref completion install|show`** subcommands for shell completion,
  replacing Typer's auto-generated `--install-completion` / `--show-completion`
  top-level options.
- **TUI** (Textual-based) browser with ISA/kind filters, presets, detail
  pane, `/` `?` `j/k` `1-9` `c` keybindings, and a help modal.
- **LSP** (`simdref-lsp`) providing hover + completion over JSON-RPC/stdio.
- **Manpages** generated per intrinsic / instruction.
- **Static web app** (`simdref web` / `simdref serve`) — a gzip-aware,
  self-contained SPA publishable to GitHub Pages, with build-stamp
  metadata for staleness warnings.
- **`simdref llm` JSON interface** for LLM / tool consumption:
  `query`, `batch` (stdin-driven, amortized catalog load), `list` (with
  optional `--pattern GLOB --isa FAM`), `schema`. See `docs/LLM.md`.
- **Source coverage**: Intel Intrinsics Guide, uops.info, Arm ACLE
  intrinsics + AARCHMRS A64, RISC-V `riscv-rvv-intrinsic-doc` and
  `riscv-unified-db`, with `docs.riscv.org` HTML fallback. Every perf row
  is tagged `measured` or `modeled` so the two never mix.

[0.0.0]: https://github.com/simd-labs/simdref/releases/tag/v0.0.0
[0.0.1]: https://github.com/simd-labs/simdref/releases/tag/v0.0.1
[0.0.3]: https://github.com/simd-labs/simdref/releases/tag/v0.0.3
