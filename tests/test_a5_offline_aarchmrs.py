"""Offline discovery of the vendored AARCHMRS 2025-09 A-profile archive.

The vendor tree ships ``AARCHMRS_OPENSOURCE_A_profile_FAT-2025-09_ASL0.tar.gz``.
``fetch_arm_a64_data`` must find it offline, without any network fetch.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from simdref import ingest_sources as src

ARCHIVE_NAME = "AARCHMRS_OPENSOURCE_A_profile_FAT-2025-09_ASL0.tar.gz"
MAIN_CHECKOUT_ARCHIVE = Path("/home/marco/repos/simd-labs/simdref/vendor/arm") / ARCHIVE_NAME


class _BlockedClient:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError("httpx.Client blocked in test")


@pytest.mark.skipif(not MAIN_CHECKOUT_ARCHIVE.is_file(), reason="vendored archive not present")
def test_offline_lookup_finds_vendored_2025_09_archive(monkeypatch, tmp_path):
    vendor_dir = tmp_path / "vendor" / "arm"
    vendor_dir.mkdir(parents=True)
    shutil.copy(MAIN_CHECKOUT_ARCHIVE, vendor_dir / ARCHIVE_NAME)
    # Mirror the module list in tmp so the module's own names decide the hit.
    monkeypatch.setattr(
        src,
        "LOCAL_ARM_A64_ARCHIVES",
        [vendor_dir / entry.name for entry in src.LOCAL_ARM_A64_ARCHIVES],
    )
    monkeypatch.setattr(src, "LOCAL_ARM_A64_JSONS", [])
    monkeypatch.setattr(src.httpx, "Client", _BlockedClient)

    text, version = src.fetch_arm_a64_data()

    assert ARCHIVE_NAME in version.url
    payload = json.loads(text)
    assert payload["format"] == "arm-aarchmrs-instructions-v1"
    assert json.loads(payload["instructions_json"])["instructions"]
