# Contributing to simdref

This file covers the dev install, the test flow, adding a source, and the
release steps.

## Dev install

```bash
git clone https://github.com/simd-labs/simdref.git
cd simdref
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
isa update     # download the pre-built catalog
isa doctor     # check the install
```

The package installs as `isa` and `simdref`. The docs use `isa`.

## Tests

```bash
pytest
pytest tests/test_cli_llm.py -v
pytest --cov=src --cov-report=term-missing
```

`tests/test_tui.py` needs `textual` (a runtime dep). The tests skip without
a catalog. Run `isa update` first.

## Full local rebuild

`isa build` rebuilds the catalog from upstream. It needs:

- `llvm-mca` 18+ on `PATH`. On Debian or Ubuntu: `sudo apt install llvm`.
- About 4 GB of free RAM.
- Optional: a local `AARCHMRS_BSD*.tar.gz` under `vendor/arm/` to skip the
  download.

```bash
isa build
```

## Add a new source

1. Read `docs/SOURCES.md` for the existing sources, refresh cadence, and
   licenses.
1. Write an ingestor in `src/simdref/ingest_sources.py` that returns
   `simdref.models.IntrinsicRecord` or `InstructionRecord`. Tag each perf
   row with `source_kind` (`measured` or `modeled`).
1. Wire the ingestor into `simdref.ingest.build_catalog`.
1. Add a fixture under `tests/fixtures/` and extend `tests/conftest.py` so
   the offline tests cover the new source.
1. Add a row to `docs/coverage/summary.json`. Run
   `python tools/audit_coverage.py fetch` to check parity.
1. Update `docs/SOURCES.md` and the Data sources table in `README.md`.

## Commit style

- Use conventional-commit prefixes (`feat:`, `fix:`, `ci:`, `refactor!:`).
  The release notes build on them.
- Keep `ci:` commits scoped to CI.
- Use `logging` or the Rich console in `simdref.cli`, not bare `print()`.

## Release

There is no tag-triggered workflow. To cut a release:

1. Move the `## Unreleased` entries in `CHANGELOG.md` under a
   `## [<version>] — <date>` heading on main.
1. Run the `bump-version.yml` workflow with the new `version`, first with
   `dry_run: true`, then `dry_run: false`. This commits the
   `pyproject.toml` bump to main and starts CI.
1. Wait for CI to pass on the bump commit.
1. Run the `release-candidate.yml` workflow with the same `version`, first
   with `dry_run: true`, then `dry_run: false`. This builds the wheel and
   sdist, publishes to PyPI through OIDC trusted publishing, pushes the
   `v<version>` tag, and creates the GitHub Release.
