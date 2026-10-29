"""Tests for the shared byte-identity entry wired into the three consumers.

Tool-dependent tests skip cleanly when llvm-mc is absent; nothing fakes a pass.
Each consumer test raises from its old matcher, so the test fails when the
wiring is reverted and the old path answers instead.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from simdref import annotate as annotate_mod
from simdref import cli as cli_mod
from simdref import lsp
from simdref.asm_identity import _tool_path
from simdref.lsp import _document_arch, identity_keys
from simdref.storage import (
    load_instruction_from_db,
    load_instructions_by_mnemonic_from_db,
    open_db,
)
from test_asm_identity import _arm_catalog, _enc_node

LLVM_MC = _tool_path("SIMDREF_LLVM_MC", "llvm-mc") or ""
needs_tools = pytest.mark.skipif(not LLVM_MC, reason="llvm-mc absent (set SIMDREF_LLVM_MC)")

# add x0, x1, x2 encodes to 0x8b020020; ret and nop to the RET and HINT rows.
_ADD = _enc_node("ADD_32_addsub_shift", "ADD", [(0, 32, "10001011000000100000000000100000")])
_RET = _enc_node(
    "RET_64R_branch_reg",
    "RET",
    [(25, 7, "1101011"), (21, 4, "0010"), (16, 5, "11111"), (10, 6, "000000")],
)
_HINT = _enc_node("HINT_hints", "HINT", [(22, 10, "1101010100")])


def _wire_db(tmp_path):
    return _arm_catalog([_ADD, _RET, _HINT], tmp_path)


def _fail(message):
    def fail(*args, **kwargs):
        raise AssertionError(message)

    return fail


class TestDocumentArch:
    def test_strong_votes_name_the_arch(self):
        assert _document_arch(("addq %rax, %rbx",), False, False)[0] == "x86"
        assert _document_arch(("add x0, x1, x2",), False, False)[0] == "aarch64"
        assert _document_arch(("add a0, a1, a2",), False, False)[0] == "riscv64"

    def test_no_votes_means_no_identity(self):
        assert _document_arch(("nop", "ret"), False, False) is None

    def test_weak_votes_alone_mean_no_identity(self):
        # fneg s0, s1 reads as AArch64 scalar FP and as RISC-V saved registers.
        assert _document_arch(("fneg s0, s1",), False, False) is None

    def test_arm32_means_no_identity(self):
        assert _document_arch(("mov r0, r1",), False, False) is None

    def test_cross_arch_line_is_not_kept(self):
        picked = _document_arch(
            ("addq %rax, %rbx", "subq %rax, %rbx", "add x0, x1, x2"), False, False
        )
        assert picked is not None
        assert picked[2] == (True, True, False)


class TestIdentityKeys:
    def test_keyless_catalog_falls_back(self, tmp_path):
        db = tmp_path / "keyless.db"
        raw = sqlite3.connect(db)
        raw.execute("CREATE TABLE instructions_data (db_key TEXT PRIMARY KEY, key TEXT)")
        raw.commit()
        raw.close()
        conn = open_db(path=db)
        try:
            assert identity_keys(["add x0, x1, x2", "ret"], False, conn) == [None, None]
        finally:
            conn.close()

    def test_no_tools_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.setattr("simdref.asm_identity._tool_path", lambda *args: None)
        conn = open_db(path=_wire_db(tmp_path))
        try:
            assert identity_keys(["add x0, x1, x2", "ret"], False, conn) == [None, None]
        finally:
            conn.close()

    @needs_tools
    def test_bytes_name_the_row(self, tmp_path):
        conn = open_db(path=_wire_db(tmp_path))
        try:
            keys = identity_keys(["add x0, x1, x2", "ret", "nop", "# comment"], False, conn)
            mnemonics = [
                (load_instruction_from_db(conn, key).mnemonic if key else None) for key in keys
            ]
        finally:
            conn.close()
        assert mnemonics == ["ADD", "RET", "HINT", None]


@needs_tools
class TestConsumers:
    def test_inlay_hint_reads_the_byte_identity_row(self, tmp_path, monkeypatch):
        conn = open_db(path=_wire_db(tmp_path))
        monkeypatch.setattr(lsp, "_hint_label", _fail("the old hint matcher ran"))
        try:
            summaries = [
                load_instructions_by_mnemonic_from_db(conn, "ADD")[0].summary,
                load_instructions_by_mnemonic_from_db(conn, "RET")[0].summary,
            ]
            hints = lsp._inlay_hints(conn, "add x0, x1, x2\nret\n", "asm", "file:///x.s", 0, 2)
        finally:
            conn.close()
        assert [hint["label"] for hint in hints] == [
            lsp._cut_label(text or "") for text in summaries
        ]
        assert [hint["position"]["line"] for hint in hints] == [0, 1]

    def test_hover_markdown_reads_the_byte_identity_row(self, tmp_path, monkeypatch):
        conn = open_db(path=_wire_db(tmp_path))
        monkeypatch.setattr(lsp, "_match_form", _fail("the old form matcher ran"))
        try:
            keys = identity_keys(["add x0, x1, x2", "ret"], False, conn)
            ret = load_instructions_by_mnemonic_from_db(conn, "RET")[0]
            body = lsp._hover_markdown(conn, "ret", identity_key=keys[1])
            expect = (
                lsp.instruction_markdown(ret, conn=conn)
                + f"\n\nFull page: {lsp.web_deep_link('instruction', ret.key)}"
            )
        finally:
            conn.close()
        assert body == expect

    def test_annotate_stream_reads_the_byte_identity_record(self, tmp_path, monkeypatch):
        conn = open_db(path=_wire_db(tmp_path))
        monkeypatch.setattr(annotate_mod, "lookup", _fail("the old lookup ran"))
        try:
            out = "".join(
                annotate_mod.annotate_stream(
                    ["add x0, x1, x2\n", "ret\n"],
                    opts=annotate_mod.AnnotateOptions(fmt="json"),
                    conn=conn,
                )
            )
            expect = [
                load_instructions_by_mnemonic_from_db(conn, "ADD")[0].summary,
                load_instructions_by_mnemonic_from_db(conn, "RET")[0].summary,
            ]
        finally:
            conn.close()
        records = json.loads(out)
        assert [record["mnemonic"] for record in records] == ["add", "ret"]
        assert [record["summary"] for record in records] == expect

    def test_llm_query_resolves_a_line_by_bytes(self, tmp_path, monkeypatch):
        conn = open_db(path=_wire_db(tmp_path))
        monkeypatch.setattr(cli_mod, "_find_instructions_fast", _fail("the old name lookup ran"))
        monkeypatch.setattr(cli_mod, "_search_runtime", _fail("the old search ran"))
        try:
            payload = cli_mod._build_llm_payload(conn, "add x0, x1, x2", 8, None, None, None)
        finally:
            conn.close()
        assert payload["mode"] == "exact"
        assert payload["match_kind"] == "instruction"
        assert len(payload["results"]) == 1
