# Deployment and release topology

Update this file when a workflow changes.

## Pipeline (`.github/workflows/ci.yml`)

Single workflow, single DAG, hard `needs:` edges.

```
 build-catalog --- test ---+--- publish-data --- validate-release
                 \         |   (main/release/    (same events)
                  package -+    schedule)
                           |
                           testpypi (push to main only)
```

Edges: `test` and `package` need `build-catalog`. `publish-data` needs all three. `validate-release` needs `publish-data`. `testpypi` needs `test`, `package`.

### Phase 1: data creation

- `build-catalog` installs LLVM 22, vendors RISC-V sources, runs `simdref build` (SDM always included), validates upstream ingestion, uploads the catalog bundle as the `catalog` artifact.

### Phase 2: data usage (parallel)

- `test` (`needs: build-catalog`). Matrix 3.10 to 3.14. Downloads the artifact, installs the package, asserts schema current, runs pytest, syntax-checks Python sources, runs `simdref doctor`, checks catalog structural invariants.
- `package` (`needs: build-catalog`). `uv build`, `twine check`, uploads the wheel artifact.

### Phase 3: deploy (push to main, release, schedule, manual dispatch)

- `publish-data` (`needs: build-catalog, test, package`). Downloads the catalog artifact, publishes `data-latest`. On `release: published` (or `workflow_dispatch` with `publish_versioned=true`) also publishes `data-v<version>`.
- `validate-release` (`needs: publish-data`). On a clean runner, `simdref update --from-release` pulls `data-latest` and runs the full test suite on it.

The static web app moved to the simdref-web repo. Pages deployment lives there. `publish-data` ships `web-data.zip` (`simdref export` output) for that repo.

### Trigger matrix

- `pull_request`: phases 1-2 only.
- push to main: full chain.
- push to other branches: phases 1-2 only.
- `release: published`: full chain plus `data-v<version>`.
- schedule (weekly Mon 00:00 UTC): full chain. Refreshes `data-latest` from upstream moves.
- `workflow_dispatch`: full chain.
- `workflow_call` (release-candidate.yml): phases 1-2 only.

Caching: none. Each run rebuilds the catalog from upstream, trading runtime for freshness. A stale cache cannot hide an ingestion regression.

## Release flow (`.github/workflows/release-candidate.yml`)

Tag and PyPI publish share one success boundary. On PyPI failure the tag rolls back. `v<version>` on origin implies release on PyPI.

Job sequence:

1. `preflight`: version agree, tag missing, PyPI missing, CI green on HEAD.
1. `build-wheel`: `uv build` + `twine check`.
1. `install-smoke`: `pip install` wheel, `simdref --version`.
1. `publish-and-tag` (when `dry_run=false`): git tag and push, then `pypa/gh-action-pypi-publish` behind the `pypi` environment gate, tag rollback on PyPI failure.
1. `github-release` runs `gh release create vX.Y.Z dist/*`.
1. `trigger-data-build` sends `ci.yml` on the tag with `publish_versioned=true` (GitHub anti-recursion blocks `release: published` cascade).

## Cutting a release

1. Bump. `gh workflow run bump-version.yml -f version=X.Y.Z -f dry_run=false` runs `scripts/bump-version.py X.Y.Z` on a clean main checkout and pushes the version commit to main. Refuses if the tag exists or the version is on PyPI.
1. Wait for CI on the bump commit to go green.
1. Dry-run. `gh workflow run release-candidate.yml -f version=X.Y.Z -f dry_run=true` proves each gate without side effects.
1. If green, re-run with `dry_run=false`. The `pypi` environment gates `publish-and-tag` on manual approval. The workflow pushes the tag, publishes to PyPI, cuts the GitHub Release.
1. `trigger-data-build` sends `ci.yml` on the new tag with `publish_versioned=true`, publishing `data-v<version>` and new `data-latest`.

Local step 1 alternative: `python scripts/bump-version.py X.Y.Z && git commit -am 'chore(release): bump to X.Y.Z' && git push`.

## Configured environments

- `pypi`: manual-approval gate for PyPI trusted-publisher OIDC.
- `testpypi`: publishes each tested main commit as a dev build.

## Recovery playbook

- `build-catalog` fails. Upstream source moved. Check the validation steps. Pin or patch the ingester.
- `publish-data` fails. GitHub Releases API flake. Re-run the job.
- `validate-release` fails. Published `data-latest` broken. Examine the catalog bundle in the previous `build-catalog` artifact. Do not tag a release until green.
- `release-candidate / preflight` fails. One of: pyproject mismatch, tag exists, version on PyPI. Repair upstream state. Do not force a tag.
- `release-candidate / publish-and-tag` stuck. The `pypi` environment waits on manual approval.
