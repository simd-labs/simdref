# Deployment & release topology

This file documents the simdref pipeline. When you change a workflow,
update this file.

## Pipeline (`.github/workflows/ci.yml`)

Single workflow, single DAG, explicit `needs:` for ordering. No
opportunistic skipping — every edge is a hard dependency.

```
                    ┌──────────── test ────────────┐
                    │                              │
 build-catalog ─────┤                              ├─── publish-data ─── validate-release
 (creates catalog   │                              │   (on main /          (on main /
  artifact)         └── package (parallel) ────────┘    release /           release /
                                                        schedule)           schedule)
```

### Phase 1 — data creation

- **`build-catalog`** — installs LLVM 22, vendors RISC-V sources, runs
  `simdref build` (SDM is always included), validates upstream
  ingestion, uploads the catalog bundle as a workflow artefact
  (`catalog`).

### Phase 2 — data usage (parallel)

- **`test`** *(needs: build-catalog)* — downloads the artefact,
  installs the package, asserts schema is current, runs the full
  pytest suite, syntax-checks Python + JS sources, runs
  `simdref doctor`, asserts catalog structural invariants.
- **`package`** *(needs: build-catalog)* — `uv build` → `twine check`,
  uploads the wheel/sdist artefact.

### Phase 3 — deploy (only on push to main, release, schedule, or manual dispatch)

- **`publish-data`** *(needs: build-catalog, test, package)* —
  downloads the catalog artefact, publishes `data-latest`. On
  `release: published` events (or `workflow_dispatch` with
  `publish_versioned=true`) it also publishes `data-v<version>`.
- **`validate-release`** *(needs: publish-data)* — on a fresh runner,
  `simdref update --from-release` pulls `data-latest` and runs the
  full test suite against it. Independent check that the published
  artefact is consumable end-to-end.

The static web app moved to the simdref-web repo; Pages deployment
lives there. `publish-data` ships `web-data.zip` (the `simdref export`
output) for that repo to consume.

### Trigger matrix

- **pull_request** → `build-catalog` → `test`, `package`. Phase 3 skips.
- **push to main** → full chain.
- **push to other branches** → phases 1–2 only.
- **release: published** → full chain + `data-v<version>` publication.
- **schedule (weekly Mon 00:00 UTC)** → full chain. Refreshes
  `data-latest` from upstream drift.
- **workflow_dispatch** → full chain.
- **workflow_call** (used by release-candidate.yml) → phases 1–2 only.

Caching: none. Every run rebuilds the catalog from upstream. Trades
runtime for guaranteed freshness; no chance of a stale cache masking a
schema/ingestion regression.

## Release flow (`.github/workflows/release-candidate.yml`)

Single atomic workflow — tag and PyPI publish share a success
boundary. If PyPI fails the tag is rolled back, so `v<version>`
existing on origin is a reliable signal that the release is on PyPI.

```
 human: gh workflow run release-candidate.yml -f version=X.Y.Z -f dry_run=true
                 │
                 ▼
   ┌──────────────────────┐
   │   preflight          │  — version match, tag absent,
   │                      │    PyPI absent, CI green on HEAD
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │   build-wheel        │  — uv build + twine check
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │   install-smoke      │  — pip install wheel, simdref --version
   └──────────┬───────────┘
              ▼       (only when dry_run=false)
   ┌──────────────────────────────────┐
   │   publish-and-tag                │
   │   1. git tag -a vX.Y.Z && push   │
   │   2. pypa/gh-action-pypi-publish │  — `pypi` environment approval
   │   3. on PyPI failure: delete tag │
   └──────────┬───────────────────────┘
              ▼
   ┌──────────────────────┐
   │   github-release     │  — gh release create vX.Y.Z dist/*
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │   trigger-data-build │  — dispatches ci.yml on the tag with
   │                      │    publish_versioned=true (GitHub's
   │                      │    anti-recursion rule blocks the
   │                      │    release: published cascade, so the
   │                      │    release workflow dispatches ci.yml
   │                      │    explicitly)
   └──────────────────────┘
```

## Cutting a release

Two clicks, no local checkout needed:

1. **Bump** — dispatch `bump-version.yml` from the Actions tab (or
   `gh workflow run bump-version.yml -f version=X.Y.Z -f dry_run=false`).
   This runs `scripts/bump-version.py X.Y.Z` on a fresh main checkout,
   rewrites `pyproject.toml`, and commits and pushes to main.
   Guard rails: refuses if the tag already exists or the version is
   already on PyPI.
1. Wait for CI on the bump commit to go green.
1. **Dry-run the release** —
   `gh workflow run release-candidate.yml -f version=X.Y.Z -f dry_run=true`
   proves every gate (green CI, matching versions, wheel builds, smoke
   install) without side effects.
1. If green, **re-dispatch with `dry_run=false`**. The `pypi`
   environment gates `publish-and-tag` on manual approval in the
   GitHub UI, then the workflow pushes the tag, publishes to PyPI, and
   cuts the GitHub Release.
1. `trigger-data-build` then dispatches `ci.yml` on the new tag with
   `publish_versioned=true`, which publishes `data-v<version>`
   alongside the refreshed `data-latest`.

Fully local alternative to step 1: `python scripts/bump-version.py X.Y.Z && git commit -am 'chore(release): bump to X.Y.Z' && git push`.

## Configured environments

- `pypi` — manual-approval gate for PyPI trusted-publisher OIDC.
- `testpypi` — the `testpypi` job in `ci.yml` publishes each tested main commit as a dev build.

## Recovery playbook

- **Catalog build fails** — upstream source drift. Check the
  validation steps in `build-catalog`; pin or patch the ingester.
- **`publish-data` fails** — GitHub Releases API flake. Re-run the job.
- **`validate-release` fails** — the published `data-latest` is
  broken. Investigate the catalog bundle in the previous
  `build-catalog` run's artefact. Do not tag a release until green.
- **`release-candidate / preflight` fails** — one of (a) pyproject
  mismatch, (b) tag already exists, (c) version already on PyPI. Fix
  upstream state; do not force a tag.
- **`release-candidate / publish-and-tag` stuck** — the `pypi`
  environment awaits manual approval.
