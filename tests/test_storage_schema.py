"""Schema v11 regression tests.

Builds a tiny in-memory catalog through the real ``build_sqlite`` code path
and asserts the indexed ``category`` column is present on
``instructions_data`` alongside the matching index, so that queries that
push category filtering down into SQL stay fast."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from simdref.models import Catalog, InstructionRecord, IntrinsicRecord, SourceVersion
from simdref.storage import (
    SQLITE_SCHEMA_VERSION,
    build_sqlite,
    sqlite_schema_is_current,
)


def _mini_catalog() -> Catalog:
    intr = IntrinsicRecord(
        name="_mm_add_ps",
        signature="__m128 _mm_add_ps(__m128 a, __m128 b)",
        description="Add packed single-precision floats.",
        header="xmmintrin.h",
        architecture="x86",
        isa=["SSE"],
        category="Arithmetic",
    )
    inst = InstructionRecord(
        mnemonic="VADDPS",
        form="VADDPS xmm, xmm, xmm",
        summary="Add packed single-precision floats.",
        architecture="x86",
        isa=["AVX"],
        metadata={"category": "Arithmetic"},
    )
    return Catalog(
        intrinsics=[intr],
        instructions=[inst],
        sources=[
            SourceVersion(
                source="test", version="t", fetched_at="2025-01-01T00:00:00+00:00", url="test://"
            )
        ],
        generated_at="2025-01-01T00:00:00+00:00",
    )


@pytest.fixture()
def built_db(tmp_path: Path) -> Path:
    db = tmp_path / "catalog.db"
    build_sqlite(_mini_catalog(), db)
    return db


def test_schema_version_is_14(built_db: Path):
    assert SQLITE_SCHEMA_VERSION == "14"
    conn = sqlite3.connect(built_db)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == "14"


def test_intrinsics_data_has_arm_arch_column(built_db: Path):
    conn = sqlite3.connect(built_db)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(intrinsics_data)")}
    finally:
        conn.close()
    assert "arm_arch" in cols


def test_intrinsics_data_has_arm_arch_index(built_db: Path):
    conn = sqlite3.connect(built_db)
    try:
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(intrinsics_data)")}
    finally:
        conn.close()
    assert "idx_intrinsic_arm_arch" in indexes


def test_instructions_data_has_category_column(built_db: Path):
    conn = sqlite3.connect(built_db)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(instructions_data)")}
    finally:
        conn.close()
    assert "category" in cols


def test_instructions_data_has_category_index(built_db: Path):
    conn = sqlite3.connect(built_db)
    try:
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(instructions_data)")}
    finally:
        conn.close()
    assert "idx_instruction_category" in indexes


def test_instruction_category_populated_from_metadata(built_db: Path):
    conn = sqlite3.connect(built_db)
    try:
        row = conn.execute(
            "SELECT category FROM instructions_data WHERE mnemonic = 'VADDPS'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == "Arithmetic"


def test_sqlite_schema_is_current_true(built_db: Path):
    assert sqlite_schema_is_current(built_db) is True


def test_sqlite_schema_is_current_false_for_older_schema(tmp_path: Path):
    # Build a db, then flip schema_version to an older value; helper must
    # reject it so the CLI triggers a rebuild.
    db = tmp_path / "catalog.db"
    build_sqlite(_mini_catalog(), db)
    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE meta SET value = '9' WHERE key = 'schema_version'")
        conn.commit()
    finally:
        conn.close()
    assert sqlite_schema_is_current(db) is False


def test_arm_arch_populated_from_supported_architectures(tmp_path: Path):
    intr_both = IntrinsicRecord(
        name="vaddq_u8",
        signature="x vaddq_u8()",
        description=".",
        header="arm_neon.h",
        architecture="arm",
        isa=["NEON"],
        category="Arithmetic",
        metadata={"supported_architectures": "v7/A32/A64"},
    )
    intr_a64 = IntrinsicRecord(
        name="svadd_s32_z",
        signature="x svadd_s32_z()",
        description=".",
        header="arm_sve.h",
        architecture="arm",
        isa=["SVE"],
        category="Arithmetic",
        metadata={"supported_architectures": "A64"},
    )
    intr_x86 = IntrinsicRecord(
        name="_mm_add_ps",
        signature="x _mm_add_ps()",
        description=".",
        header="xmmintrin.h",
        architecture="x86",
        isa=["SSE"],
        category="Arithmetic",
    )
    cat = Catalog(
        intrinsics=[intr_both, intr_a64, intr_x86],
        instructions=[],
        sources=[
            SourceVersion(
                source="t", version="t", fetched_at="2025-01-01T00:00:00+00:00", url="test://"
            )
        ],
        generated_at="2025-01-01T00:00:00+00:00",
    )
    db = tmp_path / "catalog.db"
    build_sqlite(cat, db)
    conn = sqlite3.connect(db)
    try:
        rows = {
            row[0]: row[1] for row in conn.execute("SELECT name, arm_arch FROM intrinsics_data")
        }
    finally:
        conn.close()
    assert rows["vaddq_u8"] == "BOTH"
    assert rows["svadd_s32_z"] == "A64"
    assert rows["_mm_add_ps"] is None


def test_build_sqlite_publishes_a_self_contained_file(tmp_path: Path):
    # `build_sqlite` renames only the main file, so an uncheckpointed WAL page is
    # lost at publish. Python 3.11 exhibits this: without the explicit
    # `wal_checkpoint(TRUNCATE)` it wrote a 4096-byte header and left the content
    # in `catalog.db.tmp-wal`, so every table was missing from the published db.
    db = tmp_path / "catalog.db"
    build_sqlite(_mini_catalog(), db)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["catalog.db"]

    # Copy the published file alone: any table still living in a sidecar is gone.
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    shutil.copy(db, isolated / "catalog.db")
    conn = sqlite3.connect(isolated / "catalog.db")
    try:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()
    assert {"meta", "sources", "intrinsics_data", "instructions_data"} <= tables
    assert sqlite_schema_is_current(isolated / "catalog.db")
