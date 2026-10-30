"""Tests for simdref.export: schema_version and latency-index.json parity.

``test_latency_index_matches_catalog_db`` and its positive control load the
real local catalog (see ``conftest.load_any_catalog``) because the row
requires 200 sampled entries; the bundled test fixtures only carry 32
instructions.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from simdref.export import (
    _columnar_instructions,
    _columnar_intrinsics,
    _instr_perf_map,
    _search_instructions,
    _search_intrinsics,
    _truncate,
    export_site_data,
)
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.perf import variant_perf_summary
from simdref.storage import build_sqlite, load_instruction_from_db, open_db

from conftest import build_fixture_catalog


def _tiny_catalog() -> Catalog:
    return Catalog(
        intrinsics=[],
        instructions=[
            InstructionRecord(
                mnemonic="VADDPS",
                form="xmm, xmm, xmm",
                architecture="x86",
                summary="Add packed single-precision floats.",
                isa=["AVX"],
            )
        ],
        sources=[
            SourceVersion(
                source="test", version="t", fetched_at="2025-01-01T00:00:00+00:00", url="test://"
            )
        ],
        generated_at="2025-01-01T00:00:00+00:00",
    )


def test_schema_version_is_int(tmp_path: Path):
    export_site_data(_tiny_catalog(), tmp_path)
    stamp = json.loads((tmp_path / "build_stamp.json").read_text())
    assert isinstance(stamp["schema_version"], int)


@pytest.fixture(scope="module")
def latency_index_and_db(tmp_path_factory) -> tuple[dict, Path]:
    """Real-catalog latency-index.json plus the catalog.db exported alongside it.

    Module-scoped: both consuming tests query the same build instead of
    rebuilding the ~29k-instruction sqlite file twice.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from conftest import load_any_catalog

        catalog = load_any_catalog()
    finally:
        sys.path.pop(0)

    tmp_path = tmp_path_factory.mktemp("export")
    build_sqlite(catalog, tmp_path / "catalog.db")
    export_site_data(catalog, tmp_path / "web")
    latency_index = json.loads((tmp_path / "web" / "latency-index.json").read_text())
    assert len(latency_index) >= 200, "catalog too small for a 200-entry sample"
    return latency_index, tmp_path / "catalog.db"


def test_latency_index_matches_catalog_db(latency_index_and_db):
    """200 sampled latency-index.json entries match a direct catalog.db query."""
    latency_index, db_path = latency_index_and_db
    conn = open_db(db_path)
    try:
        sample = random.Random(0).sample(list(latency_index), 200)
        for key in sample:
            record = load_instruction_from_db(conn, key)
            assert record is not None, f"{key!r} missing from catalog.db"
            lat, cpi = variant_perf_summary(record.arch_details)
            assert latency_index[key] == {"lat": lat, "cpi": cpi}, key
    finally:
        conn.close()


def test_latency_index_corruption_is_caught(latency_index_and_db):
    """Positive control: a corrupted entry must fail the same comparison."""
    latency_index, db_path = latency_index_and_db
    conn = open_db(db_path)
    try:
        key = next(iter(latency_index))
        record = load_instruction_from_db(conn, key)
        lat, cpi = variant_perf_summary(record.arch_details)
        corrupted = dict(latency_index[key])
        corrupted["lat"] = corrupted["lat"] + "-corrupted"
        assert corrupted != {"lat": lat, "cpi": cpi}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Columnar search-index encoding: app.js's decodeInstructions/decodeIntrinsics
# invert ``_columnar_instructions``/``_columnar_intrinsics`` exactly. These
# Python mirrors of that decoder catch any drift in the encoding without
# needing a JS runtime.
# ---------------------------------------------------------------------------


def _decode_instructions(payload: dict) -> list[dict]:
    c = payload["cols"]
    out = []
    for i in range(payload["n"]):
        isa, display_isa, isa_families, isa_subs = payload["isa"][c["isa"][i]]
        architecture, display_architecture = payload["arch"][c["arch"][i]]
        lat, cpi = payload["perf"][c["perf"][i]]
        display_key = c["dkey"][i]
        display_form = c["dform"][i] or display_key
        form = c["form"][i] or display_form
        display_mnemonic = c["dmn"][i]
        mnemonic = c["mn"][i] or display_mnemonic
        full = c["sum"][i]
        key = c["key"][i] or f"{architecture}:{form.lower()}"
        summary = payload["summaries"].get(i, _truncate(full, 80))
        search_fields = payload["fields"].get(
            i, [display_mnemonic, display_key, display_form, full, display_isa]
        )
        out.append(
            {
                "key": key,
                "mnemonic": mnemonic,
                "form": form,
                "architecture": architecture,
                "summary": summary,
                "isa": isa,
                "lat": lat,
                "cpi": cpi,
                "display_architecture": display_architecture,
                "display_key": display_key,
                "display_form": display_form,
                "display_mnemonic": display_mnemonic,
                "display_isa": display_isa,
                "isa_families": isa_families,
                "isa_subs": isa_subs,
                "search_fields": search_fields,
            }
        )
    return out


def _decode_intrinsics(payload: dict) -> list[dict]:
    c = payload["cols"]
    out = []
    for i in range(payload["n"]):
        isa, display_isa, isa_families, isa_subs = payload["isa"][c["isa"][i]]
        architecture, display_architecture = payload["arch"][c["arch"][i]]
        lat, cpi = payload["perf"][c["perf"][i]]
        name = c["name"][i]
        desc = payload["desc"][c["desc"][i]]
        entry = {
            "name": name,
            "subtitle": payload["summaries"].get(i, _truncate(desc, 80)),
            "architecture": architecture,
            "isa": isa,
            "lat": lat,
            "cpi": cpi,
            "display_architecture": display_architecture,
            "display_isa": display_isa,
            "isa_families": isa_families,
            "isa_subs": isa_subs,
            "search_fields": payload["fields"].get(
                i, [name, desc, display_isa, payload["ins"][c["ins"][i]]]
            ),
        }
        if c["prim"][i] >= 0:
            entry["primary_instr"] = payload["prim"][c["prim"][i]]
        if c["arm"][i] >= 0:
            entry["arm_arch"] = payload["arm"][c["arm"][i]]
        if c["cat"][i] >= 0:
            entry["category"] = payload["cat"][c["cat"][i]]
        out.append(entry)
    return out


def test_columnar_instructions_roundtrip():
    catalog = build_fixture_catalog()
    instr_perf = _instr_perf_map(catalog)
    entries = _search_instructions(catalog, instr_perf)
    assert entries  # positive control: an empty fixture would pass vacuously
    # linked_intrinsics is intentionally dropped from the search index; the
    # web client reads it from the detail chunk instead (also exported here).
    expected = [{k: v for k, v in e.items() if k != "linked_intrinsics"} for e in entries]
    assert _decode_instructions(_columnar_instructions(entries)) == expected


def test_columnar_intrinsics_roundtrip():
    catalog = build_fixture_catalog()
    instr_perf = _instr_perf_map(catalog)
    entries = _search_intrinsics(catalog, instr_perf)
    assert entries  # positive control: an empty fixture would pass vacuously
    assert _decode_intrinsics(_columnar_intrinsics(entries)) == entries


# ---------------------------------------------------------------------------
# search-index-encodings.json: the byte-identity shard the web repo uses to
# re-implement asm_identity from export output (schema 14 catalogs only).
# ---------------------------------------------------------------------------

_ENC_ALLOWED_KEYS = {
    "key",
    "arch",
    "enc_bits",
    "enc_mask",
    "enc_match",
    "enc_extra",
    "iform",
    "split_attrs",
}


def test_encodings_shard_matches_records(tmp_path: Path):
    catalog = build_fixture_catalog()
    # The fixture pipeline sets x86 iform metadata but no bit patterns; give
    # one arm row a pattern and one riscv row enc_extra so the schema-14 gate
    # opens, and tag one x86 row with split_attrs.
    arm = next(item for item in catalog.instructions if item.architecture == "arm")
    arm.enc_bits, arm.enc_mask, arm.enc_match = 32, 0xFFFF0000, 0x14000000
    riscv = next(item for item in catalog.instructions if item.architecture == "riscv")
    riscv.enc_bits, riscv.enc_mask, riscv.enc_match = 16, 0xE003, 0x9002
    riscv.enc_extra = [["", 16, 0xE003, 0x9002]]
    x86 = next(item for item in catalog.instructions if item.metadata.get("iform"))
    x86.metadata["split_attrs"] = '{"immzero": "1"}'
    export_site_data(catalog, tmp_path)
    enc_path = tmp_path / "search-index-encodings.json"
    assert enc_path.exists()
    entries = json.loads(enc_path.read_text())
    assert entries  # positive control: the injected rows must appear
    keys = [entry["key"] for entry in entries]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))
    for entry in entries:
        assert set(entry) <= _ENC_ALLOWED_KEYS, entry
        assert entry["arch"] and entry["key"].split(":", 1)[0] in ("x86", "arm", "riscv")
    # Every entry resolves in the instruction shard (0 misses).
    shard = json.loads((tmp_path / "search-index-instructions.json").read_text())
    resolved = {row["key"] for row in _decode_instructions(shard)}
    missing = [key for key in keys if key not in resolved]
    assert not missing, f"{len(missing)} encodings keys missing from the instruction shard"
    # The fixture pipeline tags x86 rows with an iform and no bit pattern.
    iform_rows = {entry["key"]: entry["iform"] for entry in entries if "iform" in entry}
    assert iform_rows["x86:addps (xmm, xmm)"] == "ADDPS_XMMps_XMMps"
    by_key = {entry["key"]: entry for entry in entries}
    assert by_key[arm.db_key] == {
        "key": arm.db_key,
        "arch": "aarch64",
        "enc_bits": 32,
        "enc_mask": 0xFFFF0000,
        "enc_match": 0x14000000,
    }
    # enc_extra keeps its parsed [base, bits, mask, match] shape.
    assert by_key[riscv.db_key]["enc_extra"] == [["", 16, 0xE003, 0x9002]]
    # split_attrs is a parsed object, not the raw metadata string.
    assert by_key[x86.db_key]["split_attrs"] == {"immzero": "1"}


def test_encodings_shard_absent_without_encoding_data(tmp_path: Path):
    # No bit patterns anywhere, as on a schema-13 catalog: no file, no error.
    catalog = build_fixture_catalog()
    for item in catalog.instructions:
        item.enc_bits = item.enc_mask = item.enc_match = 0
        item.enc_extra = []
    enc_path = tmp_path / "search-index-encodings.json"
    enc_path.write_bytes(b"[]")  # stale shard a re-export must remove
    export_site_data(catalog, tmp_path)
    assert not enc_path.exists()
    assert not Path(f"{enc_path}.gz").exists()
    assert (tmp_path / "search-index-instructions.json").exists()
