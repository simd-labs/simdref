"""Visualizer export contract: per-row source, cores.json, identity keys.

The checks sample the real local catalog (``conftest.load_any_catalog``),
the same one ``test_export.py`` uses, because they need 200 real rows.
"""

from __future__ import annotations

import json
import random

import pytest

from simdref.export import export_site_data
from simdref.perf_sources.cores import CANONICAL_CORES

from conftest import load_any_catalog


@pytest.fixture(scope="module")
def viz_export(tmp_path_factory):
    catalog = load_any_catalog()
    out = tmp_path_factory.mktemp("viz")
    export_site_data(catalog, out)
    return catalog, out


def test_measurement_rows_carry_stored_source(viz_export):
    """Every measurement row has a non-empty source: the stored name."""
    catalog, out = viz_export
    stored = {}
    for item in catalog.instructions:
        for uarch, details in item.arch_details.items():
            if details.get("measurement"):
                stored[(item.db_key, uarch)] = details.get("source") or ""
    assert len(stored) >= 200, "catalog too small for a real check"
    seen = 0
    for chunk_path in sorted((out / "detail-chunks").glob("*.json")):
        for db_key, detail in json.loads(chunk_path.read_text()).items():
            for row in detail["measurements"]:
                assert row.get("source"), (db_key, row["uarch"])
                want = stored[(db_key, row["uarch"])]
                if want:
                    assert row["source"] == want, (db_key, row["uarch"])
                seen += 1
    assert seen == len(stored)


def test_cores_json_covers_cores_py(viz_export):
    """cores.json holds every cores.py entry with its stored fields."""
    _, out = viz_export
    cores = json.loads((out / "cores.json").read_text())
    assert len(cores) == len(CANONICAL_CORES)
    for core in CANONICAL_CORES:
        assert cores[core.canonical_id] == {
            "isa": core.architecture,
            "llvm_cpu": core.llvm_cpu,
            "aliases": sorted(core.aliases),
        }


def _identity_index(rows):
    """identity.js ``buildShard``: a64 patterns, riscv buckets, x86 iform pools."""
    a64, rv, iforms = [], {}, {}
    for row in rows:
        bits = row.get("enc_bits") or 0
        mask = (row.get("enc_mask") or 0) & 0xFFFFFFFF
        match = (row.get("enc_match") or 0) & 0xFFFFFFFF
        if row["arch"] == "aarch64" and bits == 32:
            a64.append((mask, match, row["key"]))
        elif row["arch"] == "riscv":
            extras = row.get("enc_extra") or [["", bits, mask, match]]
            for base, ebits, emask, ematch in extras:
                bucket = rv.setdefault(str(base or ""), [])
                bucket.append(
                    (
                        ebits or 0,
                        (emask or 0) & 0xFFFFFFFF,
                        (ematch or 0) & 0xFFFFFFFF,
                        row["key"],
                    )
                )
        elif row["arch"] == "x86" and row.get("iform"):
            iforms.setdefault(row["iform"], []).append(row["key"])
    return a64, rv, iforms


def test_sampled_encodings_map_back_to_db_key(viz_export):
    """200 sampled rows resolve to their own db_key through the identity lookup."""
    _, out = viz_export
    rows = json.loads((out / "search-index-encodings.json").read_text())
    a64, rv, iforms = _identity_index(rows)
    db_keys = set(json.loads((out / "latency-index.json").read_text()))
    indexable = [
        row
        for row in rows
        if (row["arch"] == "aarch64" and row.get("enc_bits") == 32)
        or row["arch"] == "riscv"
        or (row["arch"] == "x86" and row.get("iform"))
    ]
    assert len(indexable) >= 200, "catalog too small for a 200-entry sample"
    for row in random.Random(0).sample(indexable, 200):
        key = row["key"]
        assert key in db_keys, key
        if row["arch"] == "aarch64":
            match = (row.get("enc_match") or 0) & 0xFFFFFFFF
            candidates = [k for mask, m, k in a64 if ((match ^ m) & mask) == 0]
        elif row["arch"] == "riscv":
            extras = row.get("enc_extra") or [
                [
                    "",
                    row.get("enc_bits") or 0,
                    (row.get("enc_mask") or 0) & 0xFFFFFFFF,
                    (row.get("enc_match") or 0) & 0xFFFFFFFF,
                ]
            ]
            candidates = []
            for base, bits, mask, m in extras:
                for bucket, entries in rv.items():
                    if bucket and bucket[:4] != str(base)[:4]:
                        continue
                    for pbits, pmask, pm, pkey in entries:
                        if pbits == (bits or 0) and ((m ^ pm) & pmask) == 0:
                            candidates.append(pkey)
        else:
            candidates = list(iforms.get(row["iform"], []))
        assert key in candidates, f"{key} does not resolve through the identity index"
