"""Cross-checks for the join of issues #51 to #57.

The three groups of checks prove the merged lanes keep their fixes when
the other lanes' changes run next to them:

1. The #57 b.eq/b.ne listing lines and the #51 AArch64 lines annotate
   together: b.eq is known, ``lsl x9, x9, #2`` gets an Arm meaning and
   the auto ISA detection names arm.
2. The #52 objdump prefix lines on an x86 listing still parse and
   annotate with #51 auto ISA detection naming x86.
3. The #53 add/sub immediate row picks and the #57 ZEN4 VPCOMPRESSD
   numbers (cpi 72.5, uops 144, microcoded) still hold.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from simdref import annotate as annotate_mod
from simdref.annotate import (
    AnnotateOptions,
    annotate_stream,
    detect_isa,
    lookup,
    pick_record,
)
from simdref.ingest_catalog import parse_uops_xml
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.storage import SQLITE_PATH, build_sqlite, open_db

FIXTURES = Path(__file__).parent / "fixtures"

_USER_CATALOG = Path.home() / ".local" / "share" / "simdref" / "catalog.db"
_catalog_env = os.environ.get("SIMDREF_CATALOG", "")
_missing = Path(tempfile.gettempdir()) / "simdref-test-no-catalog.db"
CATALOG = (
    Path(_catalog_env)
    if _catalog_env and Path(_catalog_env).is_file()
    else (_missing if _catalog_env else (SQLITE_PATH if SQLITE_PATH.is_file() else _USER_CATALOG))
)

ISSUE53_ADD_SUB_ROWS = [
    "ADD (R64, I8)",
    "ADD (R32, I8)",
    "SUB (R32, I8)",
]


def _db(tmp_path, rows) -> Path:
    db = tmp_path / "join.db"
    build_sqlite(
        Catalog(
            intrinsics=[],
            instructions=rows,
            sources=[
                SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")
            ],
            generated_at="2025-01-01",
        ),
        db,
    )
    return db


def _no_identity(monkeypatch) -> None:
    # Keep byte identity off: the legacy matcher path is the joint surface.
    monkeypatch.setattr("simdref.asm_identity._tool_path", lambda *args: None)


def _annotate_json(tmp_path, text: str, rows, monkeypatch, **opts) -> list[dict]:
    _no_identity(monkeypatch)
    conn = open_db(path=_db(tmp_path, rows))
    try:
        out = "".join(
            annotate_stream(
                text.splitlines(keepends=True),
                opts=AnnotateOptions(fmt="json", **opts),
                conn=conn,
            )
        )
    finally:
        conn.close()
    return json.loads(out)


def test_arm_bcond_and_lsl_annotate_together(tmp_path, monkeypatch):
    rows = [
        InstructionRecord(
            mnemonic="LSL", form="LSL (R16, R16)", summary="Load Segment Limit.", architecture="x86"
        ),
        InstructionRecord(
            mnemonic="LSL", form="LSL (X0, X0, I12)", summary="Lsl instruction.", architecture="arm"
        ),
        InstructionRecord(
            mnemonic="b.eq", form="B.EQ", summary="Branch conditionally.", architecture="arm"
        ),
    ]
    text = "b.eq 0x118 <dotn+0x118>\nlsl x9, x9, #2\n"
    assert detect_isa(text.splitlines()) == "arm"
    records = _annotate_json(tmp_path, text, rows, monkeypatch)
    assert records[0]["known"] is True
    assert records[0]["summary"] == "Branch conditionally."
    assert records[1]["known"] is True
    assert records[1]["summary"] == "Lsl instruction."


def test_mixed_batch_serves_each_line_its_own_family(tmp_path, monkeypatch):
    # Review finding 3 repro (the issue #57 batch): the x86 lines keep x86
    # rows and the b.* lines keep Arm rows. A --isa pin still forces one family.
    rows = [
        InstructionRecord(
            mnemonic="VPCMPD",
            form="VPCMPD (K, ZMM, ZMM, I8)",
            summary="Compare Packed Integer Values Into Mask.",
            architecture="x86",
        ),
        InstructionRecord(
            mnemonic="JNLE", form="JNLE (Rel32)", summary="Jump If Not Less.", architecture="x86"
        ),
        InstructionRecord(
            mnemonic="VPCOMPRESSD",
            form="VPCOMPRESSD (M512, K, ZMM)",
            summary="Store Sparse Packed Doubleword.",
            architecture="x86",
        ),
        InstructionRecord(
            mnemonic="b.eq", form="B.EQ", summary="Branch conditionally.", architecture="arm"
        ),
        InstructionRecord(
            mnemonic="b.ne", form="B.NE", summary="Branch conditionally.", architecture="arm"
        ),
        InstructionRecord(mnemonic="B", form="B (#0)", summary="B instruction.", architecture="arm"),
    ]
    text = (
        "vpcmpd k1, zmm0, zmm1, 1\n"
        "jnle 20 <x>\n"
        "vpcompressd ZMMWORD PTR [rsi+rcx*4]{k1}, zmm0\n"
        "b.eq 0x118 <dotn+0x118>\n"
        "b.ne 0x4c <dotn+0x4c>\n"
        "b 0xd4 <dotn+0xd4>\n"
    )
    records = _annotate_json(tmp_path, text, rows, monkeypatch)
    assert [(record["known"], record["summary"]) for record in records] == [
        (True, "Compare Packed Integer Values Into Mask."),
        (True, "Jump If Not Less."),
        (True, "Store Sparse Packed Doubleword."),
        (True, "Branch conditionally."),
        (True, "Branch conditionally."),
        (True, "B instruction."),
    ]
    forced_x86 = _annotate_json(tmp_path, text, rows, monkeypatch, isa="x86")
    assert [record["known"] for record in forced_x86] == [True, True, True, False, False, False]
    forced_arm = _annotate_json(tmp_path, text, rows, monkeypatch, isa="arm")
    assert [record["known"] for record in forced_arm] == [False, False, False, True, True, True]


def test_x86_prefix_lines_annotate_with_auto_detection(tmp_path, monkeypatch):
    rows = [
        InstructionRecord(
            mnemonic="NOP", form="NOP (R16)", summary="No Operation.", architecture="x86"
        ),
    ]
    prefix_lines = (FIXTURES / "issue_52" / "prefix_padding.s").read_text()
    text = "drv:     file format elf64-x86-64\n" + prefix_lines
    assert detect_isa(text.splitlines()) == "x86"
    records = _annotate_json(tmp_path, text, rows, monkeypatch)
    assert [record["mnemonic"] for record in records] == ["nopw", "nopw"]
    assert all(record["known"] for record in records)
    assert all(record["summary"] == "No Operation." for record in records)


class Issue53And57Rows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CATALOG.is_file():
            raise unittest.SkipTest(f"simdref catalog missing at {CATALOG}")
        cls._conn = open_db(path=CATALOG)
        cls._conn.execute("PRAGMA query_only=1")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_conn", None) is not None:
            cls._conn.close()

    def test_issue53_add_sub_rows_still_pick(self):
        lines = (FIXTURES / "issue_53" / "issue53_att.s").read_text().splitlines()[:3]
        rows = []
        for line in lines:
            parsed = annotate_mod.parse_asm_line(line)
            record = pick_record(lookup(parsed.mnemonic, self._conn), operands=parsed.operands)
            self.assertIsNotNone(record, line)
            rows.append(record.key)
        self.assertEqual(rows, ISSUE53_ADD_SUB_ROWS)


def test_zen4_vpcompressd_row_still_reports(monkeypatch):
    fixture = FIXTURES / "issue_57" / "uops_measurements.xml"
    rec = {r.form: r for r in parse_uops_xml(fixture.read_text())}["VPCOMPRESSD (M512, K, ZMM)"]

    def _fake(mnemonic, conn, isa=None):
        return [rec] if mnemonic.lower() == rec.mnemonic.lower() else []

    monkeypatch.setattr(annotate_mod, "lookup", _fake)

    class _FakeConn:
        def close(self):
            pass

    opts = AnnotateOptions(fmt="json", arch="ZEN4")
    out = "".join(
        annotate_stream(
            ["vpcompressd ZMMWORD PTR [rsi+rcx*4]{k1},zmm0\n"], opts=opts, conn=_FakeConn()
        )
    )
    row = json.loads(out)[0]
    assert row["cpi"] == 72.5
    assert row["uops"] == 144
    assert row["microcoded"] is True


if __name__ == "__main__":
    unittest.main()
