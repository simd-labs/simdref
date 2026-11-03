"""Live-fetch parity drift detection — runs only when SIMDREF_LIVE=1.

Re-fetches every upstream feed, recomputes coverage against the local
catalog in-memory (no summary.json write), and asserts the same per-source
thresholds hold. Catches upstream drift between snapshot refreshes.

Gated behind an env var so offline CI stays green.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

if os.environ.get("SIMDREF_LIVE") != "1":
    pytest.skip("SIMDREF_LIVE=1 not set; skipping live-fetch parity", allow_module_level=True)

import tomllib  # noqa: E402

from simdref.storage import load_catalog  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = REPO_ROOT / "docs" / "coverage" / "thresholds.toml"
DEFAULT_THRESHOLD = 0.95


def _load_thresholds() -> dict[str, float]:
    if not THRESHOLDS_PATH.exists():
        return {}
    with THRESHOLDS_PATH.open("rb") as fh:
        payload = tomllib.load(fh)
    return {k: float(v) for k, v in (payload.get("thresholds") or {}).items()}


def test_live_parity_all_sources():
    """Refetch every upstream feed and assert coverage >= floor."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import audit_coverage  # type: ignore
    from conftest import load_any_catalog

    catalog = load_any_catalog()
    specs = audit_coverage._source_specs(catalog)
    thresholds = _load_thresholds()
    default_th = float(os.environ.get("SIMDREF_COVERAGE_THRESHOLD", DEFAULT_THRESHOLD))

    failures: list[str] = []
    for spec in specs:
        result = audit_coverage._diff_source(spec)
        cov = result.get("coverage")
        if not isinstance(cov, float):
            failures.append(f"{spec['id']}: live fetch failed ({result.get('error')})")
            continue
        th = thresholds.get(spec["id"], default_th)
        if cov < th:
            failures.append(f"{spec['id']}: live coverage {cov:.3f} < {th:.3f}")
    assert not failures, "\n".join(failures)
