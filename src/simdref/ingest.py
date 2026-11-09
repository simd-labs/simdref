"""Public ingest entrypoints and compatibility wrappers."""

from __future__ import annotations

from simdref.ingest_catalog import (
    _instruction_summary,
    _normalize_operand_xtype,
    build_catalog,
    link_records,
    parse_intel_payload,
    parse_riscv_instruction_payload,
    parse_riscv_intrinsics_payload,
    parse_uops_xml,
)
from simdref.ingest_sources import (
    fetch_intel_data,
    fetch_ptx_isa_data,
    fetch_riscv_rvv_intrinsics_data,
    fetch_riscv_unified_db_data,
    fetch_uops_xml,
    now_iso,
)

__all__ = [
    "build_catalog",
    "fetch_intel_data",
    "fetch_ptx_isa_data",
    "fetch_riscv_rvv_intrinsics_data",
    "fetch_riscv_unified_db_data",
    "fetch_uops_xml",
    "link_records",
    "_instruction_summary",
    "_normalize_operand_xtype",
    "now_iso",
    "parse_intel_payload",
    "parse_riscv_instruction_payload",
    "parse_riscv_intrinsics_payload",
    "parse_uops_xml",
]
