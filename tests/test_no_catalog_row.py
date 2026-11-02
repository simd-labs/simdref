"""Issue #49: an identified form the catalog lacks reports no row.

uops.info has no AVX register-register VMOVAPD/VMOVUPS rows, so byte identity
correctly finds no row for ``vmovapd %xmm1, %xmm0``. Annotate, hover, hints
and the llm query must say "no measured row" by name and never fall back to
the legacy nearest-form pick (which joined the line to the store row, lat
12c). The legacy pick stays only where byte identity could not run.

Tool-dependent tests skip cleanly when llvm-mc or the XED decoder is absent;
nothing fakes a pass.
"""

from __future__ import annotations

import json

import pytest
from test_asm_wire import _fail

from simdref import annotate as annotate_mod
from simdref import cli as cli_mod
from simdref import lsp
from simdref.asm_identity import _tool_path
from simdref.lsp import NO_CATALOG_ROW, identity_keys
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.storage import build_sqlite, open_db

LLVM_MC = _tool_path("SIMDREF_LLVM_MC", "llvm-mc") or ""
XED = _tool_path("SIMDREF_XED", "xed") or ""
needs_xed = pytest.mark.skipif(not (LLVM_MC and XED), reason="llvm-mc or xed absent")


# A fixture catalog with only the store rows: the reg-reg iforms
# (VMOVAPD_XMMdq_XMMdq_28, VMOVUPS_XMMdq_XMMdq_10) are absent, like the real
# catalog. Measured SKL latency 12c mirrors the issue's wrong join.
def _store_row(mnemonic: str, iform: str) -> InstructionRecord:
    return InstructionRecord(
        mnemonic=mnemonic,
        form=f"{mnemonic} (M128, XMM)",
        summary=f"Store via {mnemonic}.",
        architecture="x86",
        arch_details={
            "SKL": {
                "latencies": [{"cycles": "12"}],
                "measurement": {"TP": "1.0"},
                "source_kind": "measured",
            }
        },
        metadata={"iform": iform, "split_attrs": "{}"},
    )


def _store_db(tmp_path):
    rows = [
        _store_row("VMOVAPD", "VMOVAPD_MEMdq_XMMdq"),
        _store_row("VMOVUPS", "VMOVUPS_MEMdq_XMMdq"),
    ]
    db = tmp_path / "store-only.db"
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


REG_REG = ["vmovapd %xmm1, %xmm0", "vmovups %xmm3, %xmm2"]
STORE = ["vmovapd %xmm0, (%rax)", "vmovups %xmm2, (%rsi)"]


@needs_xed
class TestNoCatalogRow:
    def test_identity_marks_the_miss(self, tmp_path):
        conn = open_db(path=_store_db(tmp_path))
        try:
            keys = identity_keys(REG_REG + STORE, False, conn)
            store = [open_key for open_key in keys[2:]]
        finally:
            conn.close()
        assert keys[:2] == [NO_CATALOG_ROW, NO_CATALOG_ROW]
        assert all(key and key != NO_CATALOG_ROW for key in store)

    def test_annotate_reports_no_row(self, tmp_path, monkeypatch):
        conn = open_db(path=_store_db(tmp_path))
        monkeypatch.setattr(annotate_mod, "lookup", _fail("the old lookup ran"))
        try:
            out = "".join(
                annotate_mod.annotate_stream(
                    [line + "\n" for line in REG_REG + STORE],
                    opts=annotate_mod.AnnotateOptions(),
                    conn=conn,
                )
            ).splitlines()
        finally:
            conn.close()
        assert "# vmovapd: no measured row" in out[0]
        assert "# vmovups: no measured row" in out[1]
        for line in (out[0], out[1]):
            assert "Store via" not in line
            assert "lat=" not in line
        # A store line still gets the store row.
        assert "Store via VMOVAPD." in out[2]
        assert "lat=12.0c" in out[2]
        assert "Store via VMOVUPS." in out[3]

    def test_annotate_json_marks_no_row(self, tmp_path, monkeypatch):
        conn = open_db(path=_store_db(tmp_path))
        monkeypatch.setattr(annotate_mod, "lookup", _fail("the old lookup ran"))
        try:
            out = "".join(
                annotate_mod.annotate_stream(
                    [REG_REG[0] + "\n", STORE[0] + "\n"],
                    opts=annotate_mod.AnnotateOptions(fmt="json"),
                    conn=conn,
                )
            )
        finally:
            conn.close()
        records = json.loads(out)
        assert records[0] == {
            "mnemonic": "vmovapd",
            "known": False,
            "no_measured_row": True,
            "line": 1,
            "source": REG_REG[0],
        }
        assert records[1]["known"] is True

    def test_hover_reports_no_row(self, tmp_path, monkeypatch):
        conn = open_db(path=_store_db(tmp_path))
        monkeypatch.setattr(lsp, "_match_form", _fail("the old form matcher ran"))
        monkeypatch.setattr(lsp, "_lookup_mnemonic", _fail("the old matcher ran"))
        try:
            key = identity_keys(REG_REG, False, conn)[0]
            body = lsp._hover_markdown(conn, "vmovapd", identity_key=key)
        finally:
            conn.close()
        assert body == "`vmovapd`: no measured row"

    def test_hint_reports_no_row(self, tmp_path, monkeypatch):
        conn = open_db(path=_store_db(tmp_path))
        monkeypatch.setattr(lsp, "_hint_label", _fail("the old hint matcher ran"))
        try:
            hints = lsp._inlay_hints(
                conn, REG_REG[0] + "\n" + STORE[0] + "\n", "asm", "file:///x.s", 0, 1
            )
        finally:
            conn.close()
        assert [hint["label"] for hint in hints] == [
            "vmovapd: no measured row",
            "Store via VMOVAPD.",
        ]

    def test_llm_reports_no_row(self, tmp_path, monkeypatch):
        conn = open_db(path=_store_db(tmp_path))
        monkeypatch.setattr(cli_mod, "_find_instructions_fast", _fail("the old name lookup ran"))
        monkeypatch.setattr(cli_mod, "_search_runtime", _fail("the old search ran"))
        try:
            payload = cli_mod._build_llm_payload(conn, REG_REG[0], 8, None, None, None)
        finally:
            conn.close()
        assert payload["mode"] == "exact"
        assert payload["match_kind"] == "instruction"
        assert payload["detail"] == "vmovapd: no measured row"
        assert payload["results"] == []


class TestMarkerPathNeedsNoTools:
    def test_marker_never_calls_the_legacy_pick(self, tmp_path, monkeypatch):
        # The marker reaches annotate even with tools absent; the old lookup
        # must stay silent either way.
        conn = open_db(path=_store_db(tmp_path))
        monkeypatch.setattr(lsp, "_identity_batch", lambda *a, **k: (NO_CATALOG_ROW,))
        monkeypatch.setattr(annotate_mod, "lookup", _fail("the old lookup ran"))
        try:
            out = "".join(
                annotate_mod.annotate_stream(
                    [REG_REG[0] + "\n"],
                    opts=annotate_mod.AnnotateOptions(),
                    conn=conn,
                    stats={},
                )
            )
        finally:
            conn.close()
        assert "# vmovapd: no measured row" in out

    def test_no_marker_keeps_the_legacy_pick(self, tmp_path, monkeypatch):
        # Byte identity could not run (None): the old matcher answers exactly
        # as before the fix — here the issue's line with only the store row.
        conn = open_db(path=_store_db(tmp_path))
        monkeypatch.setattr(lsp, "_identity_batch", lambda *a, **k: (None,))
        try:
            out = "".join(
                annotate_mod.annotate_stream(
                    [REG_REG[0] + "\n"],
                    opts=annotate_mod.AnnotateOptions(),
                    conn=conn,
                )
            )
        finally:
            conn.close()
        assert "no measured row" not in out
        assert out.strip().startswith("vmovapd")
