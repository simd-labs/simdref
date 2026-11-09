#!/usr/bin/env python3
"""Fetch the NVIDIA PTX ISA page and write the derived instruction bundle.

Downloads the single-page Sphinx HTML into vendor/ptx (a gitignored
cache) and derives the rows with simdref.ptx.derive_rows. The bundle at
vendor/ptx/ptx_instructions.json is the JSON input
parse_ptx_instructions_payload accepts.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from simdref.ptx import BUNDLE_FORMAT, PTX_ISA_DOC_URL, derive_rows  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "vendor" / "ptx"
CACHE_NAME = "parallel-thread-execution.html"
BUNDLE_NAME = "ptx_instructions.json"
USER_AGENT = "simdref"


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def write_bundle(out_dir: Path, rows: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format": BUNDLE_FORMAT,
        "upstream_url": PTX_ISA_DOC_URL,
        "instructions": rows,
    }
    path = out_dir / BUNDLE_NAME
    path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the PTX ISA page and write the derived bundle."
    )
    parser.add_argument("--html", type=Path, help="parse a local page instead of downloading")
    parser.add_argument("--force", action="store_true", help="re-download when the cache exists")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.html is not None:
        page = args.html.read_text(encoding="utf-8", errors="replace")
        source = f"{args.html} (local)"
    else:
        cache = args.out_dir / CACHE_NAME
        if cache.exists() and not args.force:
            page = cache.read_text(encoding="utf-8", errors="replace")
            source = f"{cache} (cache)"
        else:
            page = fetch(PTX_ISA_DOC_URL)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(page, encoding="utf-8")
            source = PTX_ISA_DOC_URL

    rows = derive_rows(page)
    path = write_bundle(args.out_dir, rows)
    print(f"source: {source}")
    print(f"rows: {len(rows)}")
    print(f"wrote {path} instructions={len(rows)}")


if __name__ == "__main__":
    main()
