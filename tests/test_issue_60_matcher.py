"""Issue #60 lane C237: alias misses, batch-context identity, silent record
drops and Intel vs AT&T spellings. Every test uses a fixture catalog built
from llvm-mc/xed ground truth, or runs without tools on an empty catalog.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from simdref.annotate import (
    AnnotateOptions,
    _drop_symbol_target,
    annotate_stream,
)
from simdref.asm_identity import _bytes_to_words, _probe_lines, _tool_path, _xed_facts
from simdref.lsp import identity_keys
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.storage import build_sqlite, open_db


def _tool_paths():
    return _tool_path("SIMDREF_LLVM_MC", "llvm-mc"), _tool_path("SIMDREF_XED", "xed")


needs_tools = pytest.mark.skipif(not all(_tool_paths()), reason="llvm-mc/xed not found")

# Verbatim repro lines from issue #60.
X86_LINES = [
    "cltq",
    "je     401012 <printf@plt-0x1e>",
    "cmova  %rax,%rbp",
    "cmova  rbp,rax",
    "movabs $0x1fffffffffffffff,%rax",
    "movabs rax,0x1fffffffffffffff",
    "setg   al",
    "sete   %al",
    "movslq %ebp,%rax",
    "movzwl 0x3352(%rip),%r15d        # 4052e8 <_ZTISt4plusIdE+0x158>",
    "mul    r14",
    "rep movs BYTE PTR es:[rdi],BYTE PTR ds:[rsi]",
    "rep movsb %ds:(%rsi),%es:(%rdi)",
    "repz cmpsb %es:(%rdi), %ds:(%rsi)",
    "notq (%rdx)",
    "vpcmpltd k1,zmm1,zmm3",
    "vpcmpltd %zmm3,%zmm1,%k1",
    "movq %rax,%rbx",
    "vpclmulhqhqdq xmm0,xmm6,xmm20",
]
ARM_LINES = [
    "mov x1, #0x1234",
    "cmp x0, #16",
    "cmn x0, #8",
    "tst w0, #0xff",
]
PREFIXES = ["lock", "rex", "rex.W", "addr32", "data16", "cs", "addr16", "data32"]


def _catalog(records: list[InstructionRecord]) -> Catalog:
    return Catalog(
        intrinsics=[],
        instructions=records,
        sources=[SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")],
        generated_at="2025-01-01",
    )


# The catalog holds no arm alias rows (GAS mov/cmp/cmn/tst assemble as
# MOVZ/SUBS/ADDS/ANDS): the rows carry the target name, so only byte identity
# names them and the old matcher never covers a lost identity.
_ALIAS_TARGETS = {"mov": "movz", "cmp": "subs", "cmn": "adds", "tst": "ands"}


def _fixture_db(tmp_path: Path) -> Path:
    """One row per x86 iform and per arm encoding the repro lines hit."""
    llvm_mc, xed = _tool_paths()
    rows: list[InstructionRecord] = []
    seen: set[str] = set()
    for syntax in ("att", "intel"):
        cleaned = {line: _drop_symbol_target(line).strip() for line in X86_LINES}
        lines = [
            cleaned[line]
            for line in X86_LINES
            if _probe_lines([cleaned[line]], "x86", syntax, llvm_mc)[0].get("status") == "OK"
        ]
        for entry in _probe_lines(lines, "x86", syntax, llvm_mc):
            if entry.get("status") != "OK":
                continue
            parts = [
                part if len(part) == 2 else "00"
                for part in entry["bytes"].replace("0x", "").split(",")
            ]
            facts = _xed_facts(["".join(parts)], xed)
            iform = (facts[0].get("iform") or "") if facts else ""
            if not iform or iform in seen:
                continue
            seen.add(iform)
            if iform.startswith("REP"):
                # REP/REPE/REPNE siblings share the iform; the prefix byte
                # alone picks the row (issue #60 class 7).
                for mnemonic in ("REPE MOVSB", "REPNE MOVSB"):
                    rows.append(
                        InstructionRecord(
                            mnemonic=mnemonic,
                            form=f"{iform} {mnemonic.split()[0]}",
                            summary=f"{iform} row.",
                            architecture="x86",
                            metadata={"iform": iform, "split_attrs": "{}"},
                        )
                    )
            else:
                rows.append(
                    InstructionRecord(
                        mnemonic=iform.split("_")[0],
                        form=iform,
                        summary=f"{iform} row.",
                        architecture="x86",
                        metadata={"iform": iform, "split_attrs": "{}"},
                    )
                )
    for line in ARM_LINES:
        entry = _probe_lines([line], "aarch64", "att", llvm_mc)[0]
        if entry.get("status") != "OK":
            continue
        word = _bytes_to_words(entry["bytes"])
        match = sum(byte << (8 * index) for index, byte in enumerate(word))
        rows.append(
            InstructionRecord(
                mnemonic=_ALIAS_TARGETS.get(line.split()[0], line.split()[0]),
                form=line,
                summary=f"{line} row.",
                architecture="arm",
                enc_bits=32,
                enc_mask=0xFFFFFFFF,
                enc_match=match,
                enc_extra=[("", 32, 0xFFFFFFFF, match)],
            )
        )
    # The AARCHMRS B.cond encodeset (mask 0xFF00001F, cond LE) for the C6
    # symbol line. Its key is not "b.le", so only byte identity names it.
    rows.append(
        InstructionRecord(
            mnemonic="B",
            form="B (LE, <label>)",
            summary="Branch conditionally.",
            architecture="arm",
            isa=["A64"],
            enc_bits=32,
            enc_mask=0xFF00001F,
            enc_match=0x5400000D,
            enc_extra=[("", 32, 0xFF00001F, 0x5400000D)],
        )
    )
    db = tmp_path / "catalog.db"
    build_sqlite(_catalog(rows), db)
    return db


def _annotate(lines: list[str], conn) -> list[dict]:
    out = "".join(
        annotate_stream(
            [line + "\n" for line in lines],
            opts=AnnotateOptions(fmt="json"),
            conn=conn,
        )
    )
    import json

    return json.loads(out) if out.strip() else []


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    llvm_mc, xed = _tool_paths()
    if not (llvm_mc and xed):
        pytest.skip("llvm-mc/xed not found")
    db = _fixture_db(tmp_path_factory.mktemp("issue60"))
    connection = open_db(path=db)
    yield connection
    connection.close()


class TestSymbolExpression:
    def test_strip_keeps_the_symbol_expression(self):
        # Lane C6: "b.le 114 <my_memcmp+0x40>" must reach llvm-mc as
        # "b.le my_memcmp+0x40"; llvm-mc rejects the bare numeric offset.
        assert _drop_symbol_target("b.le\t114 <my_memcmp+0x40>").strip() == "b.le\tmy_memcmp+0x40"

    @needs_tools
    def test_b_le_symbol_line_resolves_to_the_b_cond_row(self, conn):
        # The stripped symbol assembles and its bytes name the B.cond row
        # (mask 0xFF00001F, cond LE); the bare "b.le 114" is rejected.
        records = _annotate(["b.le\t114 <my_memcmp+0x40>"], conn)
        assert records[0]["known"] is True
        assert records[0]["summary"] == "Branch conditionally."

    @needs_tools
    def test_aarch64_symbol_line_assembles(self):
        llvm_mc = _tool_path("SIMDREF_LLVM_MC", "llvm-mc")
        assert _probe_lines(["b.le my_memcmp+0x40"], "aarch64", "att", llvm_mc)[0]["status"] == "OK"
        assert _probe_lines(["b.le 114"], "aarch64", "att", llvm_mc)[0]["status"] == "REJECTED"


class TestSilentRecordDrops:
    @needs_tools
    def test_trailing_comment_lines_keep_their_record(self, conn):
        # issue #60 class 4 (b): two lines, two records, input order.
        lines = [
            "movzwl 0x3352(%rip),%r15d        # 4052e8 <_ZTISt4plusIdE+0x158>",
            "mul    r14",
        ]
        records = _annotate(lines, conn)
        assert [record["mnemonic"] for record in records] == ["movzwl", "mul"]
        assert [record["known"] for record in records] == [True, True]
        assert [record["mnemonic"] for record in _annotate(list(reversed(lines)), conn)] == [
            "mul",
            "movzwl",
        ]

    @needs_tools
    def test_every_line_yields_one_record(self, conn):
        records = _annotate(X86_LINES + ARM_LINES, conn)
        assert len(records) == len(X86_LINES) + len(ARM_LINES)


class TestCommentImmediates:
    @needs_tools
    def test_hash_immediate_lines_keep_their_identity(self, conn):
        # A "#" immediate (AArch64) is not a comment and must reach llvm-mc,
        # so the bytes name the row; a mixed batch keeps the x86 line too.
        records = _annotate(["mov x1, #0x1234", "cmp x0, #16", "cltq"], conn)
        assert [record["known"] for record in records] == [True, True, True]
        assert [record["summary"] for record in records] == [
            "mov x1, #0x1234 row.",
            "cmp x0, #16 row.",
            "CDQE row.",
        ]

    def test_unknown_lines_with_trailing_comments_keep_their_record(self, tmp_path):
        # issue #60 class 4 (b): a line no row answers still keeps its JSON
        # record, trailing comment or not. Two lines, two records, input
        # order, on a catalog whose rows name neither line.
        db = tmp_path / "empty.db"
        build_sqlite(_catalog([]), db)
        connection = open_db(path=db)
        try:
            records = _annotate(
                [
                    "movzwl 0x3352(%rip),%r15d        # 4052e8 <_ZTISt4plusIdE+0x158>",
                    "mul    r14",
                ],
                connection,
            )
        finally:
            connection.close()
        assert [record["mnemonic"] for record in records] == ["movzwl", "mul"]


class TestBatchContextIdentity:
    @needs_tools
    def test_context_flip_is_gone(self, conn):
        # issue #60 class 3: vpclmulhqhqdq answered only when alone.
        flip = "vpclmulhqhqdq xmm0,xmm6,xmm20"
        neighbour = "movq %rax,%rbx"
        alone = _annotate([flip], conn)[0]
        beside = _annotate([neighbour, flip], conn)[1]
        after = _annotate([flip, neighbour], conn)[0]
        for record in (beside, after):
            assert record["known"] is True
            assert (record["mnemonic"], record["summary"]) == (
                alone["mnemonic"],
                alone["summary"],
            )

    @needs_tools
    def test_each_line_matches_its_singleton(self, conn):
        batch = _annotate(X86_LINES + ARM_LINES, conn)
        for line, record in zip(X86_LINES + ARM_LINES, batch):
            alone = _annotate([line], conn)[0]
            assert (record["known"], record["summary"]) == (
                alone["known"],
                alone["summary"],
            ), line


class TestIntelVsAtt:
    @needs_tools
    def test_twins_agree(self, conn):
        pairs = [
            ("cmova  %rax,%rbp", "cmova  rbp,rax"),
            ("movabs $0x1fffffffffffffff,%rax", "movabs rax,0x1fffffffffffffff"),
            ("sete   %al", "sete al"),
            ("vpcmpltd %zmm3,%zmm1,%k1", "vpcmpltd k1,zmm1,zmm3"),
            ("rep movsb %ds:(%rsi),%es:(%rdi)", "rep movs BYTE PTR es:[rdi],BYTE PTR ds:[rsi]"),
        ]
        for att, intel in pairs:
            att_record = _annotate([att], conn)[0]
            intel_record = _annotate([intel], conn)[0]
            assert (att_record["known"], att_record["summary"]) == (
                intel_record["known"],
                intel_record["summary"],
            ), (att, intel)

    @needs_tools
    def test_alias_lines_answer(self, conn):
        records = _annotate(
            ["cltq", "je     401012 <printf@plt-0x1e>", "setg   al", "movslq %ebp,%rax"],
            conn,
        )
        assert [record["known"] for record in records] == [True, True, True, True]

    @needs_tools
    def test_intel_style_jump_line_resolves_through_the_att_retry(self, conn):
        # The Intel parser rejects objdump jump text; the AT&T retry assembles
        # it and the bytes name the JZ row (issue #60 class 7). An AT&T line in
        # an Intel-context document resolves through the same retry.
        records = _annotate(["je     401012 <printf@plt-0x1e>"], conn)
        assert records[0]["known"] is True
        assert records[0]["summary"] == "JZ_RELBRb row."
        # "cltq" reads as Intel text (no %/$) but only the AT&T parser takes
        # it: the retry must be what resolves the line to the CDQE row.
        keys = identity_keys(["cltq"], True, conn)
        assert keys == ["x86:cdqe"]


class TestLonePrefixLines:
    def test_prefix_lines_are_known(self, tmp_path):
        db = tmp_path / "empty.db"
        build_sqlite(_catalog([]), db)
        connection = open_db(path=db)
        try:
            records = _annotate(PREFIXES, connection)
        finally:
            connection.close()
        assert len(records) == len(PREFIXES)
        for token, record in zip(PREFIXES, records):
            assert record["known"] is True
            assert "prefix" in record["summary"].lower()
            assert record["mnemonic"] == token


class TestNoToolFallback:
    def test_alias_lines_stay_unknown_without_llvm_mc(self, conn, monkeypatch):
        # The delta rule: no alias map in the fallback; without llvm-mc the
        # alias lines stay known=false (their tests skip).
        monkeypatch.setattr("simdref.asm_identity._tool_path", lambda *args: None)
        from simdref import lsp

        lsp._identity_batch.cache_clear()
        records = _annotate(["cltq"], conn)
        assert records[0]["known"] is False
