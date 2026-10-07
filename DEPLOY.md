# Deployment and release topology

This file documents the simdref pipeline. When you change a workflow, update this file.

## Pipeline (`.github/workflows/ci.yml`)

Single workflow, single DAG, hard `needs:` edges. No opportunistic skipping.

```
 build-catalog --- test ---+--- publish-data --- validate-release
                 \         |   (main/release/    (same events)
                  package -+    schedule)
                           |
                           testpypi (push to main only)
```

Edges: `test` and `package` need `build-catalog`. `publish-data` needs all three. `validate-release` needs `publish-data`. `testpypi` needs `test` and `package`.

### Phase 1: data creation

- `build-catalog` installs LLVM 22, vendors RISC-V sources, runs `simdref build` (SDM is always included), validates upstream ingestion, uploads the catalog bundle as the `catalog` artifact.

### Phase 2: data usage (parallel)

- `test` (`needs: build-catalog`). Matrix over Python 3.10 to 3.14. Downloads the artifact, installs the package, asserts schema is current, runs pytest, syntax-checks Python sources, runs `simdref doctor`, asserts catalog structural invariants.
- `package` (`needs: build-catalog`). `uv build`, `twine check`, uploads the wheel artifact.

### Phase 3: deploy (push to main, release, schedule, or manual dispatch)

- `publish-data` (`needs: build-catalog, test, package`). Downloads the catalog artifact, publishes `data-latest`. On `release: published` (or `workflow_dispatch` with `publish_versioned=true`) it also publishes `data-v<version>`.
- `validate-release` (`needs: publish-data`). On a fresh runner, `simdref update --from-release` pulls `data-latest` and runs the full test suite against it.

The static web app moved to the simdref-web repo; Pages deployment lives there. `publish-data` ships `web-data.zip` (the `simdref export` output) for that repo to consume.

### Trigger matrix

- `pull_request`: phases 1-2 only.
- push to main: full chain.
- push to other branches: phases 1-2 only.
- `release: published`: full chain plus `data-v<version>` publication.
- schedule (weekly Mon 00:00 UTC): full chain. Refreshes `data-latest` from upstream drift.
- `workflow_dispatch`: full chain.
- `workflow_call` (from release-candidate.yml): phases 1-2 only.

Caching: none. Every run rebuilds the catalog from upstream. Trades runtime for guaranteed freshness; a stale cache cannot mask an ingestion regression.

## Release flow (`.github/workflows/release-candidate.yml`)

Tag and PyPI publish share one success boundary. If PyPI fails, the tag is rolled back. `v<version>` on origin means the release is on PyPI.

Job order: `preflight` (version match, tag absent, PyPI absent, CI green on HEAD) → `build-wheel` (`uv build` + `twine check`) → `install-smoke` (`pip install` wheel, `simdref --version`) → `publish-and-tag` (only when `dry_run=false`): git tag and push, then `pypa/gh-action-pypi-publish` behind the `pypi` environment gate, with tag rollback on PyPI failure. Then `github-release` runs `gh release create vX.Y.Z dist/*`. Then `trigger-data-build` dispatches `ci.yml` on the tag with `publish_versioned=true` (GitHub anti-recursion blocks the `release: published` cascade).

## Cutting a release

1. Bump. Dispatch `bump-version.yml` from the Actions tab (`gh workflow run bump-version.yml -f version=X.Y.Z -f dry_run=false`). It runs `scripts/bump-version.py X.Y.Z` on a fresh main checkout and pushes the version commit to main. Guard rails: refuses if the tag exists or the version is already on PyPI.
1. Wait for CI on the bump commit to go green.
1. Dry-run the release. `gh workflow run release-candidate.yml -f version=X.Y.Z -f dry_run=true` proves every gate without side effects.
1. If green, re-dispatch with `dry_run=false`. The `pypi` environment gates `publish-and-tag` on manual approval in the GitHub UI. Then the workflow pushes the tag, publishes to PyPI, and cuts the GitHub Release.
1. `trigger-data-build` dispatches `ci.yml` on the new tag with `publish_versioned=true`, publishing `data-v<version>` alongside the refreshed `data-latest`.

Local alternative to step 1: `python scripts/bump-version.py X.Y.Z && git commit -am 'chore(release): bump to X.Y.Z' && git push`.

## Configured environments

- `pypi`: manual-approval gate for PyPI trusted-publisher OIDC.
- `testpypi`: the `testpypi` job in `ci.yml` publishes each tested main commit as a dev build.

## Recovery playbook

- `build-catalog` fails. Upstream source drift. Check the validation steps; pin or patch the ingester.
- `publish-data` fails. GitHub Releases API flake. Re-run the job.
- `validate-release` fails. The published `data-latest` is broken. Investigate the catalog bundle in the previous `build-catalog` artifact. Do not tag a release until green.
- `release-candidate / preflight` fails. One of: pyproject mismatch, tag already exists, version already on PyPI. Fix upstream state; do not force a tag.
- `release-candidate / publish-and-tag` stuck. The `pypi` environment awaits manual approval.
