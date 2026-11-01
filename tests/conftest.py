"""Pytest fixtures and test-only catalog builders.

Tests build their working catalog directly from the small JSON/XML files
in ``tests/fixtures/`` — the production ingest pipeline is online-only
and has no fixture fallback. This module owns the parsing path so the
production code never needs to know fixtures exist.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from simdref.ingest_catalog import (
    link_records,
    parse_arm_intrinsics_payload,
    parse_intel_payload,
    parse_uops_xml,
)
from simdref.arm_instructions import parse_arm_instruction_payload
from simdref.models import Catalog, SourceVersion
from simdref.riscv import (
    parse_riscv_instruction_payload,
    parse_riscv_intrinsics_payload,
)

FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"

# A version-stamp mismatch must not trigger a catalog download during tests.
os.environ.setdefault("SIMDREF_SKIP_AUTOUPDATE", "1")


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fixture_text(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


def _fixture_source(source_id: str, fixture_name: str) -> SourceVersion:
    return SourceVersion(
        source=source_id,
        version="fixture",
        fetched_at=_iso_now(),
        url=f"fixture:{fixture_name}",
    )


def build_fixture_catalog(*, status: Callable[[str], None] | None = None) -> Catalog:
    """Build a :class:`Catalog` from the bundled test fixtures.

    Mirrors the production :func:`simdref.ingest_catalog.build_catalog`
    pipeline but reads every feed from ``tests/fixtures/`` so the whole
    build is deterministic and offline.
    """
    emit = status or (lambda _msg: None)

    emit("Loading Intel fixture")
    intrinsics = parse_intel_payload(_fixture_text("intel_intrinsics_sample.json"))
    emit("Loading Arm ACLE fixture")
    intrinsics.extend(
        parse_arm_intrinsics_payload(_fixture_text("arm_acle_intrinsics_sample.json"))
    )
    emit("Loading RISC-V RVV intrinsics fixture")
    intrinsics.extend(
        parse_riscv_intrinsics_payload(_fixture_text("riscv_rvv_intrinsics_sample.json"))
    )

    emit("Loading uops.info fixture")
    instructions = parse_uops_xml(_fixture_text("uops_sample.xml"))
    emit("Loading Arm A64 instruction fixture")
    instructions.extend(
        parse_arm_instruction_payload(_fixture_text("arm_a64_instructions_sample.json"))
    )
    emit("Loading RISC-V unified-db fixture")
    instructions.extend(
        parse_riscv_instruction_payload(_fixture_text("riscv_unified_db_sample.json"))
    )

    emit("Linking intrinsics to instructions")
    link_records(intrinsics, instructions)

    return Catalog(
        intrinsics=sorted(intrinsics, key=lambda item: item.name),
        instructions=sorted(
            instructions,
            key=lambda item: (item.architecture, item.mnemonic, item.form),
        ),
        sources=[
            _fixture_source("intel-intrinsics-guide", "intel_intrinsics_sample.json"),
            _fixture_source("uops.info", "uops_sample.xml"),
            _fixture_source("arm-acle", "arm_acle_intrinsics_sample.json"),
            _fixture_source("arm-a64", "arm_a64_instructions_sample.json"),
            _fixture_source("rvv-intrinsic-doc", "riscv_rvv_intrinsics_sample.json"),
            _fixture_source("riscv-unified-db", "riscv_unified_db_sample.json"),
        ],
        generated_at=_iso_now(),
    )


def load_any_catalog() -> Catalog:
    """Load the repo catalog from the msgpack snapshot, or rebuild it from
    the SQLite runtime when the snapshot has been pruned (post-v13
    install/update footprint policy)."""
    from simdref.storage import CATALOG_PATH, load_catalog, load_catalog_from_db

    if CATALOG_PATH.exists():
        return load_catalog()
    return load_catalog_from_db()


__all__ = ["FIXTURES_DIR", "build_fixture_catalog", "load_any_catalog"]

# Keep dataclass 'replace' importable by tests that want to tweak catalogs.
_ = replace
