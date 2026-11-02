"""Issue #60 class 5: annotate JSON records carry ``line`` (the 1-based
input line number) and ``source`` (the input line text without the newline)
on every record. A known record also carries the matched catalog row's
``architecture`` and ``isa``; an unknown record carries ``line`` and
``source`` only.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import simdref.annotate
from simdref.annotate import AnnotateOptions, annotate_stream
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.storage import build_sqlite, open_db

ROWS = [
    InstructionRecord(
        mnemonic="ADD",
        form="ADD (R64, R64)",
        summary="Add 64-bit operands.",
        architecture="x86",
        isa=["sse2"],
    ),
    InstructionRecord(
        mnemonic="ADD",
        form="Add registers",
        summary="Add operands.",
        architecture="arm",
        isa=["aarch64"],
    ),
]

X86_LINE = "add %rax,%rdx"
ARM_LINE = "add x0, x1, x2"
UNKNOWN_LINE = "notanopcode %rax"


def _annotate(lines: list[str], isa: str) -> list[dict]:
    tmp = tempfile.TemporaryDirectory(prefix="simdref-issue60-")
    db = Path(tmp.name) / "catalog.db"
    build_sqlite(
        Catalog(
            intrinsics=[],
            instructions=list(ROWS),
            sources=[
                SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")
            ],
            generated_at="2025-01-01",
        ),
        db,
    )
    conn = open_db(path=db)
    try:
        # Stub byte identity: the fixture rows carry no enc data and the
        # contract under test is the JSON record, not the assembler match.
        # This also keeps the test independent of llvm-mc.
        with patch.object(
            simdref.annotate, "identity_keys", lambda texts, *_args, **_kwargs: [None] * len(texts)
        ):
            out = "".join(
                annotate_stream(
                    [line + "\n" for line in lines],
                    opts=AnnotateOptions(fmt="json", isa=isa),
                    conn=conn,
                )
            )
    finally:
        conn.close()
        tmp.cleanup()
    return json.loads(out)


class Issue60ContractTests(unittest.TestCase):
    def test_known_record_carries_line_source_arch_and_isa(self):
        records = _annotate([X86_LINE], "x86")
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(
            list(record),
            [
                "mnemonic",
                "known",
                "summary",
                "annotation",
                "latency",
                "cpi",
                "ports",
                "line",
                "source",
                "architecture",
                "isa",
            ],
        )
        self.assertEqual(record["line"], 1)
        self.assertEqual(record["source"], X86_LINE)
        self.assertEqual(record["architecture"], "x86")
        self.assertEqual(record["isa"], ["sse2"])

    def test_line_numbers_match_input_positions(self):
        records = _annotate([X86_LINE, UNKNOWN_LINE, X86_LINE], "x86")
        self.assertEqual([record["line"] for record in records], [1, 2, 3])
        self.assertEqual(
            [record["source"] for record in records],
            [X86_LINE, UNKNOWN_LINE, X86_LINE],
        )

    def test_unknown_record_carries_line_and_source_only(self):
        records = _annotate([UNKNOWN_LINE], "x86")
        self.assertEqual(
            records,
            [
                {
                    "mnemonic": "notanopcode",
                    "known": False,
                    "line": 1,
                    "source": UNKNOWN_LINE,
                }
            ],
        )

    def test_arm_row_labels_its_architecture_and_isa(self):
        records = _annotate([ARM_LINE], "arm")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["known"], True)
        self.assertEqual(records[0]["architecture"], "arm")
        self.assertEqual(records[0]["isa"], ["aarch64"])


if __name__ == "__main__":
    unittest.main()
