"""CUDA C++ device intrinsics parsing and derived record rows.

Parses the JSON bundle produced by tools/vendor_cuda_sources.py, which
extracts CUDA API names from four clang headers and resolves PTX
instruction links by parsing LLVM TableGen .td files.
"""

from __future__ import annotations

import json
from typing import Any

from simdref.models import IntrinsicRecord

BUNDLE_FORMAT = "clang-cuda-headers-v2"
CUDA_INTRINSICS_DOC_URL = "https://docs.nvidia.com/cuda/cuda-c-programming-guide/"


def parse_cuda_intrinsics_payload(text: str) -> list[IntrinsicRecord]:
    """Parse the CUDA intrinsics JSON bundle into IntrinsicRecord rows."""
    data = json.loads(text)
    if data.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"Unexpected bundle format: {data.get('format')!r}")

    rows: list[IntrinsicRecord] = []
    for item in data.get("intrinsics", []):
        metadata = item.get("metadata", {})
        classification = metadata.get("classification", "")

        row = IntrinsicRecord(
            name=item["name"],
            signature=item.get("signature", ""),
            description=item.get("description", ""),
            header=item.get("header", ""),
            url=item.get("url", ""),
            architecture="cuda",
            isa=["CUDA"],
            category=item.get("category", ""),
            subcategory=item.get("subcategory", ""),
            instructions=item.get("instructions", []),
            instruction_refs=item.get("instruction_refs", []),
            metadata={"classification": classification},
            doc_sections=item.get("doc_sections", {}),
            notes=item.get("notes", []),
            aliases=item.get("aliases", []),
            source="clang-cuda-headers",
        )
        rows.append(row)

    return rows
