# Contributing to simdref

Thanks for your interest. This document covers the short path from a fresh
clone to a local development install, the test workflow, and the minimum you
need to know to add a new upstream source.

## Dev install

```bash
git clone https://github.com/simd-labs/simdref.git
cd simdref
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Prime the runtime data (fast, no toolchain required):

```bash
isa update                # downloads the pre-built release catalog
isa doctor                # sanity-check
```

> The package installs as both `isa` and `simdref` — pick whichever
> name you prefer. The docs use `isa` for brevity.

## Running the tests

```bash
pytest                    # the full suite
pytest tests/test_cli_llm.py -v
pytest --cov=src --cov-report=term-missing
```

The TUI smoke tests (`tests/test_tui.py`) require `textual` (already a
runtime dep) and skip automatically if no catalog is present. Running
`simdref update` first is usually enough to unlock them.

## Full local rebuild (`isa build`)

The `build` command rebuilds the catalog from upstream sources. It needs
an external toolchain:

- **`llvm-mca` 18+** on `PATH` — used to model ARM/RISC-V latencies we can't
  measure directly. On Debian/Ubuntu: `sudo apt install llvm`.
- **Enough RAM** — catalog construction touches millions of AARCHMRS and
  uops.info records; budget ~4 GB headroom.
- Optional: a local `AARCHMRS_BSD*.tar.gz` archive under `vendor/arm/` to
  skip the one-time download.

```bash
isa build                 # download + parse, rebuild from scratch (includes Intel SDM PDF)
```

## Adding a new source

1. Read `docs/SOURCES.md` for the existing sources, their refresh cadence,
   and license notes.
1. Write an ingestor under `src/simdref/ingest_sources.py` (or a new module
   if the source is substantial) that returns typed records matching
   `simdref.models.IntrinsicRecord` / `InstructionRecord`. Every perf row
   must be tagged with a `source_kind` (`measured` or `modeled`) — this is
   load-bearing invariant the rest of the pipeline relies on.
1. Wire the new ingestor into `simdref.ingest.build_catalog`.
1. Add a small fixture to `tests/fixtures/` and extend `tests/conftest.py`
   so the offline test path carries at least one record from the new source.
1. Add a coverage row to `docs/coverage/summary.json` and run
   `python tools/audit_coverage.py fetch` to verify parity.
1. Update `docs/SOURCES.md` and the "Data sources" table in `README.md`.

## Commit hygiene

- Follow conventional-commits-style prefixes (`feat:`, `fix:`, `ci:`,
  `refactor!:`, …). The release notes are generated against them.
- Keep `ci:` commits scoped to CI; they get squashed in release prep.
- No unprompted `print()` in source code — prefer `logging` or the Rich
  console that `simdref.cli` already wires up.

## Releasing

Release is manual; there is no tag-triggered workflow. To cut a release:

1. Add the `CHANGELOG.md` entry for the new version on main.
1. Move every `## Unreleased` entry in `CHANGELOG.md` under a
   `## [<version>] — <date>` heading for the release.
1. Run the Bump Version workflow (`bump-version.yml`) with the new
   `version`, first with `dry_run: true` to check the diff, then with
   `dry_run: false`. The workflow commits the `pyproject.toml` bump to
   main and starts CI on the bump commit.
1. Wait for CI to go green on the bump commit.
1. Run the Release workflow (`release-candidate.yml`) with the same
   `version`, first with `dry_run: true` to prove every gate, then with
   `dry_run: false`. The workflow builds the wheel and sdist, publishes to
   PyPI via OIDC trusted publishing, pushes the `v<version>` tag, and
   creates the GitHub Release with the built artifacts.
