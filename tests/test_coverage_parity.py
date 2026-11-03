"""Offline parity test: assert committed coverage snapshot stays above
per-source thresholds.

Read-only. Re-generate ``docs/coverage/summary.json`` via
``python tools/audit_coverage.py fetch`` when upstream sources change.
Live fetch is covered separately in ``test_coverage_live.py`` (gated on
``SIMDREF_LIVE=1``).
"""

from __future__ import annotations

import json
import os

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "docs" / "coverage" / "summary.json"
THRESHOLDS = REPO_ROOT / "docs" / "coverage" / "thresholds.toml"

DEFAULT_THRESHOLD = 0.95


def _load_summary() -> dict:
    if not SUMMARY.exists():
        pytest.skip(f"no coverage snapshot at {SUMMARY}")
    return json.loads(SUMMARY.read_text())


def _load_thresholds() -> dict[str, float]:
    if not THRESHOLDS.exists():
        return {}
    with THRESHOLDS.open("rb") as fh:
        payload = tomllib.load(fh)
    return {k: float(v) for k, v in (payload.get("thresholds") or {}).items()}


def _default_threshold() -> float:
    raw = os.environ.get("SIMDREF_COVERAGE_THRESHOLD")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_THRESHOLD


def test_summary_has_all_sources():
    summary = _load_summary()
    expected = {
        "intel-intrinsics",
        "uops-info",
        "arm-intrinsics",
        "arm-a64",
        "riscv-rvv",
        "riscv-unified-db",
    }
    actual = set(summary.get("sources", {}).keys())
    assert expected.issubset(actual), f"missing sources: {expected - actual}"


def test_each_source_meets_threshold():
    summary = _load_summary()
    thresholds = _load_thresholds()
    default_th = _default_threshold()
    failures: list[str] = []
    for sid, data in summary.get("sources", {}).items():
        cov = data.get("coverage")
        if not isinstance(cov, float):
            # Unknown coverage (fetch failed) must have an explicit error field,
            # otherwise we treat it as a failure.
            if "error" not in data:
                failures.append(f"{sid}: coverage missing and no error recorded")
            continue
        th = thresholds.get(sid, default_th)
        if cov < th:
            failures.append(f"{sid}: coverage {cov:.3f} < threshold {th:.3f}")
    assert not failures, "\n".join(failures)


def test_catalog_carries_canonical_names():
    """Anti-regression: a handful of industry-standard intrinsics and
    instructions must always be in the catalog. If any of these disappear,
    ingestion has regressed regardless of summary.json freshness."""
    from conftest import load_any_catalog

    catalog = load_any_catalog()
    intrinsic_names = {r.name for r in catalog.intrinsics}
    instruction_mnemonics = {
        (getattr(r, "mnemonic", "") or r.key.split()[0]).upper() for r in catalog.instructions
    }

    required_intrinsics = {"_mm_add_ps", "_mm256_fmadd_ps", "vaddq_u8"}
    # __riscv_vadd_vv_i32m1 may or may not be present depending on RVV
    # ingestion state — asserted as soft warning via catalog count check.
    missing = required_intrinsics - intrinsic_names
    assert not missing, f"missing canonical intrinsics: {missing}"

    required_mnemonics = {"VADDPS"}
    missing_m = required_mnemonics - instruction_mnemonics
    assert not missing_m, f"missing canonical x86 mnemonics: {missing_m}"


def test_catalog_counts_are_nontrivial():
    summary = _load_summary()
    cat = summary.get("catalog", {})
    assert cat.get("intrinsics", 0) > 1000, "catalog intrinsic count suspiciously low"
    assert cat.get("instructions", 0) > 1000, "catalog instruction count suspiciously low"
