"""Node 2 check (design E): the export keeps the C.1 shape byte-for-byte.

``doc_sources`` serializes in source order, the metadata carries no
``result`` key, and two exports of one catalog are byte-identical under
``sort_keys=True``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from simdref.export import export_site_data
from simdref.models import Catalog, IntrinsicRecord, SourceVersion

SHUFFLE_NAMES = (
    "__zyx_shfl_v1",
    "__zyx_shfl_up_v2",
    "__zyx_shfl_down_v3",
    "__zyx_shfl_xor_v4",
)
SOURCE_ORDER = ["cuda-guide", "cuda-math-api", "ptx-isa"]


def _entry(source: str, name: str) -> dict:
    return {
        "source": source,
        "page": f"https://example.invalid/{source}.html",
        "url": f"https://example.invalid/{source}.html#zyx-{name}",
        "title": "Zyx",
        "heading": name,
        "sections": [
            {
                "heading": "",
                "shared": False,
                "blocks": [{"kind": "paragraph", "owner": name, "text": f"{name} in {source}."}],
            }
        ],
    }


def _row(name: str) -> IntrinsicRecord:
    return IntrinsicRecord(
        name=name,
        signature=f"int {name}(int)",
        description="CUDA device intrinsic from __clang_cuda_math.h.",
        header="__clang_cuda_math.h",
        url=f"https://example.invalid/zyx_guide.html#zyx-{name}",
        architecture="cuda",
        isa=["CUDA"],
        source="clang-cuda-headers",
        metadata={"doc_source": "cuda-math-api"},
        doc_sources=[_entry(source, name) for source in SOURCE_ORDER],
    )


def _catalog() -> Catalog:
    rows = [_row(name) for name in (*SHUFFLE_NAMES, "__zyx_sin_v2")]
    return Catalog(
        intrinsics=rows,
        instructions=[],
        sources=[
            SourceVersion(
                source="test",
                version="t",
                fetched_at="2025-01-01T00:00:00+00:00",
                url="test://",
            )
        ],
        generated_at="2025-01-01T00:00:00+00:00",
    )


def _exported_rows(out_dir: Path) -> dict[str, dict]:
    chunks = {}
    for path in (out_dir / "intrinsic-chunks").glob("*.json"):
        chunks.update(json.loads(path.read_text()))
    return chunks


def test_doc_sources_keep_source_order_and_drop_result(
    tmp_path: Path,
) -> None:
    export_site_data(_catalog(), tmp_path)
    rows = _exported_rows(tmp_path)
    for name in (*SHUFFLE_NAMES, "__zyx_sin_v2"):
        entry = rows[name]
        assert [e["source"] for e in entry["doc_sources"]] == SOURCE_ORDER, name
        assert "result" not in entry["metadata"], name


def test_two_exports_are_byte_identical(tmp_path: Path, monkeypatch) -> None:
    # built_at would differ legitimately between two runs; freeze it.
    import simdref.export as export_module

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(export_module, "datetime", _FrozenDateTime)
    first, second = tmp_path / "first", tmp_path / "second"
    export_site_data(_catalog(), first)
    export_site_data(_catalog(), second)
    names = sorted(item.relative_to(first) for item in first.rglob("*") if item.is_file())
    assert names  # positive control: an empty tree would pass vacuously
    for relative in names:
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), relative
