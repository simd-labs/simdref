"""Phase F — structural assertions across CLI / manpage / web surfaces.

No golden files — every assertion is on presence/shape, not on exact bytes.
Run against a small in-test catalog so the fixtures stay deterministic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdref.manpages import intrinsic_page, instruction_page, write_manpages
from simdref.models import Catalog, InstructionRecord, IntrinsicRecord, SourceVersion
from simdref.export import export_site_data

from test_export import _decode_instructions, _decode_intrinsics


@pytest.fixture()
def catalog() -> Catalog:
    intr = IntrinsicRecord(
        name="_mm_add_ps",
        signature="__m128 _mm_add_ps(__m128 a, __m128 b)",
        description="Add packed single-precision floats.",
        header="xmmintrin.h",
        architecture="x86",
        isa=["SSE"],
        category="Arithmetic",
        instructions=["ADDPS"],
    )
    intr_arm = IntrinsicRecord(
        name="vaddq_u8",
        signature="uint8x16_t vaddq_u8(uint8x16_t a, uint8x16_t b)",
        description="Add packed u8 lanes.",
        header="arm_neon.h",
        architecture="arm",
        isa=["NEON"],
        category="Arithmetic",
    )
    inst = InstructionRecord(
        mnemonic="VADDPS",
        form="VADDPS xmm, xmm, xmm",
        summary="Add packed single-precision floats.",
        architecture="x86",
        isa=["AVX"],
        metadata={"category": "Arithmetic"},
        description={"Description": "Adds packed single-precision lanes."},
    )
    return Catalog(
        intrinsics=[intr, intr_arm],
        instructions=[inst],
        sources=[
            SourceVersion(
                source="test", version="t", fetched_at="2025-01-01T00:00:00+00:00", url="test://"
            )
        ],
        generated_at="2025-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# F4 — manpage structure
# ---------------------------------------------------------------------------

REQUIRED_MAN_SECTIONS = [".TH", ".SH NAME", ".SH SYNOPSIS", ".SH DESCRIPTION"]


def test_intrinsic_manpage_has_required_sections(catalog: Catalog):
    page = intrinsic_page(catalog.intrinsics[0], catalog)
    for section in REQUIRED_MAN_SECTIONS:
        assert section in page, f"missing {section!r} in intrinsic manpage"
    assert "_mm_add_ps" in page
    assert ".SH ISA" in page
    assert ".SH CATEGORY" in page


def test_intrinsic_manpage_has_single_performance_section(catalog: Catalog):
    page = intrinsic_page(catalog.intrinsics[0], catalog)
    assert page.count(".SH PERFORMANCE") == 1


def test_instruction_manpage_has_required_sections(catalog: Catalog):
    page = instruction_page(catalog.instructions[0])
    assert ".TH" in page
    assert ".SH NAME" in page
    assert ".SH DESCRIPTION" in page
    assert "VADDPS" in page


def test_write_manpages_covers_every_intrinsic(catalog: Catalog, tmp_path: Path):
    write_manpages(catalog, tmp_path)
    written = {p.name for p in (tmp_path / "man7").iterdir()}
    assert "_mm_add_ps.7" in written
    assert "vaddq_u8.7" in written
    # Instruction gets a slugged page and an x86-shortcut mnemonic page.
    assert "VADDPS.7" in written


def test_write_manpages_bounds_long_unsafe_names(catalog: Catalog, tmp_path: Path):
    # A PTX key can carry "::" and run past the file-name limit: the page
    # names stay unique and inside the limit.
    catalog.instructions.append(
        InstructionRecord(
            mnemonic="fabric.try_put." + ".".join(f"seg{i}" for i in range(40)) + "::reads",
            form="d, a, b",
            summary="generated",
            architecture="ptx",
            isa=["PTX"],
            metadata={},
            description={},
            source="ptx-isa",
        )
    )
    write_manpages(catalog, tmp_path)
    written = {p.name for p in (tmp_path / "man7").iterdir()}
    ptx_pages = [name for name in written if name.startswith(("instruction-ptx-", "ptx-"))]
    assert len(ptx_pages) == 2
    for name in written:
        assert name.endswith(".7")
        assert len(name.encode("utf-8")) <= 255
        assert ":" not in name


# ---------------------------------------------------------------------------
# F3 — web export structure
# ---------------------------------------------------------------------------


def test_web_export_emits_expected_artifacts(catalog: Catalog, tmp_path: Path):
    export_site_data(catalog, tmp_path)
    for name in (
        "search-index-meta.json",
        "search-index-instructions.json",
        "search-index-intrinsics.json",
        "filter_spec.json",
        "build_stamp.json",
    ):
        assert (tmp_path / name).is_file(), f"web export missing {name}"
    assert not (tmp_path / "search-index.json").exists(), (
        "legacy monolithic search-index.json must not be emitted"
    )


def test_filter_spec_json_exposes_presets(catalog: Catalog, tmp_path: Path):
    export_site_data(catalog, tmp_path)
    spec = json.loads((tmp_path / "filter_spec.json").read_text())
    presets = spec.get("presets") or {}
    assert {"default", "intel", "arm32", "arm64", "riscv", "none", "all"} <= set(presets)
    # Arm32/Arm64 must carry arm_arch facet; Intel must not.
    assert presets["arm32"]["arm_arch"] == ["A32", "BOTH"]
    assert presets["arm64"]["arm_arch"] == ["A64", "BOTH"]
    assert presets["intel"]["arm_arch"] is None
    # The default preset stays intrinsic-only; ISA-targeted presets surface
    # both kinds so e.g. selecting "Intel" still finds VADDPS the instruction.
    assert presets["default"]["kind"] == ["intrinsic"]
    for name in ("intel", "arm32", "arm64", "riscv"):
        assert set(presets[name]["kind"]) == {"intrinsic", "instruction"}, (
            f"{name} must include both kinds"
        )
    assert set(presets["all"]["kind"]) == {"intrinsic", "instruction"}


def test_filter_spec_json_carries_chip_families(catalog: Catalog, tmp_path: Path):
    export_site_data(catalog, tmp_path)
    spec = json.loads((tmp_path / "filter_spec.json").read_text())
    chips = spec.get("chip_families") or []
    assert chips, "filter_spec.json has no chip_families"
    # The chips list every family the UI shows; CUDA and PTX are preset-only.
    assert "CUDA" not in chips and "PTX" not in chips
    assert set(chips) == set(spec["family_order"]) - {"PTX", "CUDA"}


def test_search_index_intrinsics_have_required_fields(catalog: Catalog, tmp_path: Path):
    export_site_data(catalog, tmp_path)
    payload = json.loads((tmp_path / "search-index-intrinsics.json").read_text())
    intrinsics = _decode_intrinsics(payload)
    assert intrinsics, "search-index-intrinsics.json exposes no intrinsics"
    # Slim shape: only what the client needs for search + result-card render.
    required = {"name", "subtitle", "isa", "display_isa", "isa_families", "search_fields"}
    forbidden = {
        "signature",
        "header",
        "url",
        "notes",
        "metadata",
        "search_tokens",
        "display_isa_tokens",
    }
    for item in intrinsics:
        missing = required - item.keys()
        assert not missing, f"intrinsic {item.get('name')!r} missing {missing}"
        extra = forbidden & item.keys()
        assert not extra, f"intrinsic {item.get('name')!r} has unexpected fat fields {extra}"
        assert item["name"], "empty name"


def test_search_index_and_details_are_gzipped(catalog: Catalog, tmp_path: Path):
    export_site_data(catalog, tmp_path)
    # Pre-compressed sidecars must be emitted for gzip-aware static serve.
    for name in (
        "search-index-meta.json",
        "search-index-instructions.json",
        "search-index-intrinsics.json",
        "filter_spec.json",
    ):
        assert (tmp_path / f"{name}.gz").is_file(), f"missing {name}.gz sidecar"
    # Intrinsic details are sharded — every emitted chunk carries a .gz.
    intr_chunks = list((tmp_path / "intrinsic-chunks").glob("*.json"))
    assert intr_chunks, "intrinsic-chunks directory is empty"
    for chunk in intr_chunks:
        assert (chunk.with_suffix(".json.gz")).is_file(), f"missing {chunk.name}.gz sidecar"


def test_search_index_instructions_have_required_fields(catalog: Catalog, tmp_path: Path):
    export_site_data(catalog, tmp_path)
    payload = json.loads((tmp_path / "search-index-instructions.json").read_text())
    instructions = _decode_instructions(payload)
    assert instructions, "search-index-instructions.json exposes no instructions"
    required = {"key", "mnemonic", "summary", "isa"}
    for item in instructions:
        missing = required - item.keys()
        assert not missing, f"instruction {item.get('key')!r} missing {missing}"


def test_filter_spec_json_categories_are_populated(catalog: Catalog, tmp_path: Path):
    export_site_data(catalog, tmp_path)
    spec = json.loads((tmp_path / "filter_spec.json").read_text())
    cats = spec.get("categories") or []
    assert cats, "filter_spec.json has no categories"
    for cat in cats:
        assert cat.get("count", 0) > 0
        assert cat.get("family")
        assert cat.get("category")


def test_build_stamp_has_generated_at(catalog: Catalog, tmp_path: Path):
    export_site_data(catalog, tmp_path)
    stamp = json.loads((tmp_path / "build_stamp.json").read_text())
    assert stamp.get("built_at") and stamp.get("catalog_generated_at")


# ---------------------------------------------------------------------------
# F1 — CLI render (structural, captured via rich.console)
# ---------------------------------------------------------------------------


def test_cli_renders_intrinsic_detail_with_signature_and_isa(catalog: Catalog):
    from simdref import display

    with display.console.capture() as capture:
        display.render_intrinsic(catalog, catalog.intrinsics[0])
    output = capture.get()
    assert "_mm_add_ps" in output
    assert "__m128" in output
    assert "SSE" in output
    assert "xmmintrin.h" in output


def test_cli_renders_instruction_detail_with_mnemonic(catalog: Catalog):
    from simdref import display

    with display.console.capture() as capture:
        display.render_instruction(catalog, catalog.instructions[0])
    output = capture.get()
    assert "VADDPS" in output
    assert "AVX" in output
