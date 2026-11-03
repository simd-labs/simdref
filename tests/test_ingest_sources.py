"""Unit tests for simdref.ingest_sources.

Covers URL candidate-fallback loops, zip/tar archive extraction helpers,
and malformed-input handling. Network is never exercised — every
outbound call is monkeypatched. The fetchers now raise
:class:`~simdref.ingest_sources.SourceUnavailableError` when every
candidate source fails (no bundled-fixture fallback).
"""

from __future__ import annotations

import io
import json
import tarfile
import zipfile

import pytest

from simdref import ingest_sources as src
from simdref.storage import derive_arm_arch


@pytest.mark.parametrize(
    "supported, isa, expected",
    [
        ("v7/A32/A64", ["NEON"], "BOTH"),
        ("A32/A64", ["NEON"], "BOTH"),
        ("A64", ["SVE"], "A64"),
        ("A32", ["NEON"], "A32"),
        ("v7", ["NEON"], "A32"),
        ("", ["MVE"], "A32"),
        ("", ["NEON"], None),
        ("unknown", ["NEON"], None),
    ],
)
def test_derive_arm_arch_classifies_supported_architectures(supported, isa, expected):
    metadata = {"supported_architectures": supported} if supported else {}
    assert derive_arm_arch(isa, metadata) == expected


def test_derive_arm_arch_none_for_non_arm():
    assert derive_arm_arch(["SSE"], {}) is None
    assert derive_arm_arch(["AVX512F"], None) is None


# ---------------------------------------------------------------------------
# Candidate-URL fallback loops
# ---------------------------------------------------------------------------


def _isolate_network(monkeypatch, *, raise_on_urls=(), responses=None):
    """Disable all local archive/vendor paths and stub _fetch_text."""
    monkeypatch.setattr(src, "LOCAL_INTEL_ARCHIVES", [])
    monkeypatch.setattr(src, "LOCAL_INTEL_CACHE_ARCHIVES", [])
    monkeypatch.setattr(src, "LOCAL_UOPS_XMLS", [])
    monkeypatch.setattr(src, "LOCAL_ARM_ACLE_JSONS", [])
    monkeypatch.setattr(src, "LOCAL_ARM_INTRINSICS_JSONS", [])
    monkeypatch.setattr(src, "LOCAL_ARM_OPERATIONS_JSONS", [])
    monkeypatch.setattr(src, "LOCAL_ARM_EXAMPLES_JSONS", [])
    monkeypatch.setattr(src, "LOCAL_ARM_ACLE_ARCHIVES", [])
    monkeypatch.setattr(src, "LOCAL_ARM_A64_JSONS", [])
    monkeypatch.setattr(src, "LOCAL_ARM_A64_ARCHIVES", [])
    monkeypatch.setattr(src, "LOCAL_RISCV_UNIFIED_DB_JSONS", [])
    monkeypatch.setattr(src, "LOCAL_RISCV_RVV_INTRINSICS_JSONS", [])
    monkeypatch.setattr(src, "LOCAL_RISCV_DOCS_JSONS", [])

    responses = dict(responses or {})

    def fake_fetch(url: str) -> str:
        if url in raise_on_urls:
            raise RuntimeError(f"stubbed failure for {url}")
        if url in responses:
            return responses[url]
        raise RuntimeError(f"unexpected url {url}")

    monkeypatch.setattr(src, "_fetch_text", fake_fetch)

    # Also neutralise httpx.Client so any remaining direct use raises.
    class _BlockedClient:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("httpx.Client blocked in test")

    monkeypatch.setattr(src.httpx, "Client", _BlockedClient)


def test_riscv_rvv_candidate_fallback(monkeypatch):
    urls = src.RISCV_RVV_INTRINSICS_CANDIDATE_URLS
    assert len(urls) >= 2
    payload = '{"intrinsics": []}'
    _isolate_network(
        monkeypatch,
        raise_on_urls=(urls[0],),
        responses={urls[1]: payload},
    )
    text, version = src.fetch_riscv_rvv_intrinsics_data()
    assert text == payload
    assert version.url == urls[1]
    assert version.version == "live"


def test_riscv_unified_db_candidate_fallback(monkeypatch):
    urls = src.RISCV_UNIFIED_DB_CANDIDATE_URLS
    assert len(urls) >= 2
    payload = '{"instructions": []}'
    _isolate_network(
        monkeypatch,
        raise_on_urls=tuple(urls[:-1]),
        responses={urls[-1]: payload},
    )
    text, version = src.fetch_riscv_unified_db_data()
    assert json.loads(text) == {"instructions": []}
    assert version.url == urls[-1]


def test_intel_data_candidate_fallback(monkeypatch):
    urls = src.INTEL_CANDIDATE_DATA_URLS
    assert urls
    payload = "intel_payload"
    _isolate_network(
        monkeypatch,
        raise_on_urls=(urls[0],),
        responses={urls[1]: payload} if len(urls) > 1 else {urls[0]: payload},
    )
    # Intel has zip-download fallback; blocked client makes it fail cleanly.
    if len(urls) > 1:
        text, version = src.fetch_intel_data()
        assert text == payload
        assert version.url in urls


def _write_intel_cache_zip(
    tmp_path, data_js="cached intel data", fetched_at="2026-11-03T00:00:00+00:00"
):
    archive = tmp_path / "intel-intrinsics.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("files/data.js", data_js)
        zf.writestr(
            "SOURCE.json",
            json.dumps({"url": "u", "fetched_at": fetched_at, "sha256": "s"}),
        )
    return archive


def test_intel_cache_fallback_used_when_network_fails(tmp_path, monkeypatch):
    cache = _write_intel_cache_zip(tmp_path)
    _isolate_network(monkeypatch, raise_on_urls=tuple(src.INTEL_CANDIDATE_DATA_URLS))
    monkeypatch.setattr(src, "LOCAL_INTEL_CACHE_ARCHIVES", [cache])
    text, version = src.fetch_intel_data()
    assert text == "cached intel data"
    assert version.version == "upstream-cache:2026-11-03T00:00:00+00:00"


def test_intel_cache_ignored_when_network_works(tmp_path, monkeypatch):
    cache = _write_intel_cache_zip(tmp_path)
    url = src.INTEL_CANDIDATE_DATA_URLS[0]
    _isolate_network(monkeypatch, responses={url: "live payload"})
    monkeypatch.setattr(src, "LOCAL_INTEL_CACHE_ARCHIVES", [cache])
    text, version = src.fetch_intel_data()
    assert text == "live payload"
    assert version.version == "live"


# ---------------------------------------------------------------------------
# Full-unavailable path raises SourceUnavailableError (no fixture fallback)
# ---------------------------------------------------------------------------


def test_rvv_all_sources_unreachable_raises(monkeypatch):
    _isolate_network(
        monkeypatch,
        raise_on_urls=tuple(src.RISCV_RVV_INTRINSICS_CANDIDATE_URLS),
    )
    with pytest.raises(src.SourceUnavailableError):
        src.fetch_riscv_rvv_intrinsics_data()


def test_unified_db_all_sources_unreachable_raises(monkeypatch):
    _isolate_network(
        monkeypatch,
        raise_on_urls=tuple(src.RISCV_UNIFIED_DB_CANDIDATE_URLS),
    )
    with pytest.raises(src.SourceUnavailableError):
        src.fetch_riscv_unified_db_data()


def test_uops_all_sources_unreachable_raises(monkeypatch):
    _isolate_network(monkeypatch, raise_on_urls=(src.UOPS_XML_URL,))
    with pytest.raises(src.SourceUnavailableError):
        src.fetch_uops_xml()


# ---------------------------------------------------------------------------
# Malformed input handling
# ---------------------------------------------------------------------------


def test_augment_riscv_unified_db_payload_handles_malformed_json():
    # Invalid JSON → helper returns the input string unchanged.
    bad = "{not json"
    assert src._augment_riscv_unified_db_payload_with_docs(bad) == bad


def test_augment_riscv_unified_db_payload_passes_through_non_dict():
    # Valid JSON but not a dict → unchanged.
    text = "[1, 2, 3]"
    assert src._augment_riscv_unified_db_payload_with_docs(text) == text


def test_augment_riscv_unified_db_payload_skips_if_docs_already_present():
    text = json.dumps({"instructions": [], "docs_pages": {"x": "y"}})
    assert src._augment_riscv_unified_db_payload_with_docs(text) == text


def test_riscv_missing_semantics_urls_filters_non_docs_urls():
    payload = {
        "instructions": [
            {
                "description": {"Description": "ok", "Operation": "ok"},
                "url": "https://docs.riscv.org/foo",
            },
            {
                "description": {"Description": "", "Operation": ""},
                "url": "https://docs.riscv.org/bar",
            },
            {
                "description": {"Description": "", "Operation": ""},
                "url": "https://example.com/other",
            },
        ]
    }
    urls = src._riscv_missing_semantics_urls(payload)
    assert urls == ["https://docs.riscv.org/bar"]


# ---------------------------------------------------------------------------
# Zip / tar extraction helpers
# ---------------------------------------------------------------------------


def _build_zip(entries: dict[str, str]) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in entries.items():
            zf.writestr(name, text)
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_extract_zip_text_by_suffix():
    zf = _build_zip({"a/b/file.csv": "hello"})
    assert src._extract_zip_text(zf, "file.csv") == "hello"


def test_extract_zip_text_missing_raises():
    zf = _build_zip({"other.txt": "nope"})
    with pytest.raises(KeyError):
        src._extract_zip_text(zf, "missing.csv")


def test_extract_zip_text_by_match_returns_name():
    zf = _build_zip({"x/y/instructions_a64.json": '{"ok": 1}', "readme.txt": "r"})
    text, name = src._extract_zip_text_by_match(zf, src._looks_like_arm_instruction_json)
    assert json.loads(text) == {"ok": 1}
    assert name.endswith("instructions_a64.json")


def test_extract_zip_text_by_match_missing_raises():
    zf = _build_zip({"readme.txt": "nothing to see"})
    with pytest.raises(KeyError):
        src._extract_zip_text_by_match(zf, src._looks_like_arm_instruction_json)


def _build_targz(entries: dict[str, str]) -> tarfile.TarFile:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in entries.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return tarfile.open(fileobj=buf, mode="r:*")


def test_extract_tar_text_by_match_returns_name():
    tf = _build_targz({"pkg/a64_instructions.json": '{"a": 1}', "pkg/readme.md": "r"})
    text, name = src._extract_tar_text_by_match(tf, src._looks_like_arm_instruction_json)
    assert json.loads(text) == {"a": 1}
    assert "a64_instructions" in name


def test_extract_tar_text_by_match_missing_raises():
    tf = _build_targz({"pkg/readme.md": "r"})
    with pytest.raises(KeyError):
        src._extract_tar_text_by_match(tf, src._looks_like_arm_instruction_json)


# ---------------------------------------------------------------------------
# ARM instruction-JSON name predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, expected",
    [
        ("pkg/a64_instructions.json", True),
        ("pkg/A64_Instructions.JSON", True),
        ("pkg/a64_register_map.json", False),
        ("pkg/a64_system_regs.json", False),
        ("pkg/other.json", False),
        ("pkg/a64_instructions.xml", False),
    ],
)
def test_looks_like_arm_instruction_json(path, expected):
    assert src._looks_like_arm_instruction_json(path) is expected


# ---------------------------------------------------------------------------
# Bundle payload helpers
# ---------------------------------------------------------------------------


def test_arm_acle_bundle_payload_shape():
    payload = src._arm_acle_bundle_payload("csv1", "csv2", "acle_md", "neon_md")
    parsed = json.loads(payload)
    assert parsed["format"] == "acle-neon-csv-v1"
    assert parsed["intrinsics_csv"] == "csv1"
    assert parsed["classification_csv"] == "csv2"
    assert parsed["acle_markdown"] == "acle_md"
    assert parsed["neon_markdown"] == "neon_md"


def test_arm_intrinsics_bundle_payload_defaults_examples():
    payload = src._arm_intrinsics_bundle_payload("{}", "{}")
    parsed = json.loads(payload)
    assert parsed["format"] == "arm-intrinsics-json-v1"
    assert parsed["examples_json"] == "[]"


def test_arm_instruction_bundle_payload_shape():
    payload = src._arm_instruction_bundle_payload('{"x":1}')
    parsed = json.loads(payload)
    assert parsed["format"] == "arm-aarchmrs-instructions-v1"
    assert parsed["instructions_json"] == '{"x":1}'


# ---------------------------------------------------------------------------
# now_iso sanity
# ---------------------------------------------------------------------------


def test_read_local_text_found(tmp_path, monkeypatch):
    file_path = tmp_path / "local.json"
    file_path.write_text('{"hello": 1}')
    result = src._read_local_text([file_path], "test-src", "local")
    assert result is not None
    text, version = result
    assert text == '{"hello": 1}'
    assert version.source == "test-src"
    assert version.version.startswith("local:")


def test_read_local_text_not_found(tmp_path):
    result = src._read_local_text([tmp_path / "missing.json"], "test-src", "local")
    assert result is None


def test_read_local_intel_archive_reads_zip(tmp_path, monkeypatch):
    archive = tmp_path / "intel.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Intel Intrinsics Guide/files/data.js", "// intel data")
    monkeypatch.setattr(src, "LOCAL_INTEL_ARCHIVES", [archive])
    result = src._read_local_intel_archive()
    assert result is not None
    text, version = result
    assert "intel data" in text
    assert version.source == "intel-intrinsics-guide"


def test_read_local_intel_archive_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(src, "LOCAL_INTEL_ARCHIVES", [tmp_path / "missing.zip"])
    assert src._read_local_intel_archive() is None


def test_read_local_arm_instruction_archive_zip(tmp_path, monkeypatch):
    archive = tmp_path / "arm.zip"
    payload = '{"instructions": [{"mnemonic": "ADD"}]}'
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("pkg/a64_instructions.json", payload)
    monkeypatch.setattr(src, "LOCAL_ARM_A64_ARCHIVES", [archive])
    result = src._read_local_arm_instruction_archive()
    assert result is not None
    text, version = result
    parsed = json.loads(text)
    assert parsed["format"] == "arm-aarchmrs-instructions-v1"
    assert "ADD" in parsed["instructions_json"]
    assert version.source == "arm-a64"


def test_read_local_arm_instruction_archive_targz(tmp_path, monkeypatch):
    archive = tmp_path / "arm.tar.gz"
    payload = b'{"instructions": [{"mnemonic": "SUB"}]}'
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo(name="pkg/a64_instructions.json")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr(src, "LOCAL_ARM_A64_ARCHIVES", [archive])
    result = src._read_local_arm_instruction_archive()
    assert result is not None
    text, version = result
    assert "SUB" in text
    assert version.source == "arm-a64"


def test_fetch_uops_xml_uses_local_file(tmp_path, monkeypatch):
    xml_path = tmp_path / "instructions.xml"
    xml_path.write_text("<root/>")
    monkeypatch.setattr(src, "LOCAL_UOPS_XMLS", [xml_path])
    path_or_text, version = src.fetch_uops_xml()
    assert path_or_text == xml_path
    assert version.source == "uops.info"
    assert version.version.startswith("local-xml:")


def test_read_local_arm_intrinsics_bundle_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(src, "LOCAL_ARM_INTRINSICS_JSONS", [tmp_path / "nope.json"])
    monkeypatch.setattr(src, "LOCAL_ARM_OPERATIONS_JSONS", [tmp_path / "nope2.json"])
    monkeypatch.setattr(src, "LOCAL_ARM_EXAMPLES_JSONS", [tmp_path / "nope3.json"])
    assert src._read_local_arm_intrinsics_bundle() is None


def test_read_local_arm_intrinsics_bundle_present(monkeypatch, tmp_path):
    intr = tmp_path / "intr.json"
    ops = tmp_path / "ops.json"
    ex = tmp_path / "ex.json"
    intr.write_text("[]")
    ops.write_text("[]")
    ex.write_text("[]")
    monkeypatch.setattr(src, "LOCAL_ARM_INTRINSICS_JSONS", [intr])
    monkeypatch.setattr(src, "LOCAL_ARM_OPERATIONS_JSONS", [ops])
    monkeypatch.setattr(src, "LOCAL_ARM_EXAMPLES_JSONS", [ex])
    result = src._read_local_arm_intrinsics_bundle()
    assert result is not None
    text, version = result
    parsed = json.loads(text)
    assert parsed["format"] == "arm-intrinsics-json-v1"
    assert version.source == "arm-intrinsics-site"


def test_now_iso_is_utc_iso_format():
    stamp = src.now_iso()
    # Parses without error and ends with +00:00 (UTC offset).
    from datetime import datetime

    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0
