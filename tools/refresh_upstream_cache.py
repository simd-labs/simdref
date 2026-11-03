#!/usr/bin/env python3
"""Refresh the ``upstream-cache`` release asset from the live upstream feeds.

Runs daily from .github/workflows/upstream-cache.yml. Each row of
``SOURCES`` is one cached feed. The script uploads the zip only when the
feed content changed. A fetch that fails or looks insane exits nonzero
and never replaces the asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from simdref import ingest_sources
from simdref.ingest_sources import SourceVersion, fetch_intel_data

RELEASE_TAG = "upstream-cache"


def _fetch_intel_live() -> tuple[str, SourceVersion]:
    # A vendor or cache archive would defeat the refresh; force the network.
    ingest_sources.LOCAL_INTEL_ARCHIVES = []
    ingest_sources.LOCAL_INTEL_CACHE_ARCHIVES = []
    return fetch_intel_data()


# One row per cached feed; a later source is one more row here.
SOURCES = [
    {
        "asset": "intel-intrinsics.zip",
        "member": "files/data.js",
        "fetch": _fetch_intel_live,
        "count_token": "<intrinsic ",
        # 7757 "<intrinsic " entries in the 2026-11 live fetch. The floor
        # is about 90% of that count, so a truncated or wrong payload
        # never replaces a good asset.
        "min_count": 6980,
        "version_re": re.compile(r"intrinsics_list[^>]*version=\\?\"([^\"\\]+)"),
    },
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_feed(spec: dict) -> tuple[str, dict]:
    text, source = spec["fetch"]()
    count = text.count(spec["count_token"])
    if count < spec["min_count"]:
        raise SystemExit(
            f"{spec['asset']}: only {count} {spec['count_token']!r} entries, "
            f"want >= {spec['min_count']}"
        )
    match = spec["version_re"].search(text)
    meta = {
        "url": source.url,
        "fetched_at": source.fetched_at,
        "sha256": _sha256(text.encode("utf-8")),
        "intrinsics_list_version": match.group(1) if match else None,
    }
    return text, meta


def build_zip(spec: dict, text: str, meta: dict, out_path: Path) -> None:
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(spec["member"], text)
        zf.writestr("SOURCE.json", json.dumps(meta, indent=2) + "\n")


def current_member_sha(spec: dict, work_dir: Path) -> str | None:
    # Missing release or asset means the first run; then nothing to compare.
    proc = subprocess.run(
        ["gh", "release", "download", RELEASE_TAG, "-p", spec["asset"], "-D", str(work_dir)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    with zipfile.ZipFile(work_dir / spec["asset"]) as zf:
        return _sha256(zf.read(spec["member"]))


def ensure_release() -> None:
    view = subprocess.run(["gh", "release", "view", RELEASE_TAG], capture_output=True)
    if view.returncode == 0:
        return
    subprocess.run(
        [
            "gh",
            "release",
            "create",
            RELEASE_TAG,
            "--prerelease",
            "--latest=false",
            "--title",
            "Upstream source cache",
            "--notes",
            "Cached upstream source snapshots. Refreshed by upstream-cache.yml.",
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        help="write the zip into DIR and skip every gh call (local testing)",
    )
    args = parser.parse_args()
    out_dir = args.out or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in SOURCES:
        text, meta = fetch_feed(spec)
        out_path = out_dir / spec["asset"]
        build_zip(spec, text, meta, out_path)
        if args.out:
            print(f"wrote {out_path}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            old_sha = current_member_sha(spec, Path(tmp))
        if old_sha == meta["sha256"]:
            print(f"{spec['asset']}: content unchanged, keeping the current asset")
            continue
        ensure_release()
        subprocess.run(
            ["gh", "release", "upload", RELEASE_TAG, str(out_path), "--clobber"],
            check=True,
        )
        print(f"{spec['asset']}: uploaded (sha256 {meta['sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
