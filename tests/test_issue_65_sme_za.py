"""Issue #65 M11: SME ZA-tile lines resolve to the ZA rows, not plain rows.

An SME line whose operands name a ZA tile must land on the tile form, which
carries no timing; the plain SVE/NEON forms of the same mnemonic share the
operand count, so operand shape alone picked a timed plain row. Both matcher
paths get one test here: annotate's pick_record and the hover/hint matcher.
"""

from __future__ import annotations

import sqlite3

import pytest

from simdref.annotate import pick_record
from simdref.catkey import key_names_za_tile, names_za_tile
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.storage import build_sqlite

FMLA_PLAIN = "FMLA (V0.4S, V1.4S, V2.4S)"
FMLA_SVE = "FMLA (<ZDA>.S, Z1.S, Z2.S[#0])"
FMLA_ZA = (
    "FMLA (Z3.4S[<WV>, #0<OPTIONAL_COMMA_VGX2>], {<OPT_SPACE><ZN1>.4S-<ZN2>.4S<OPT_SPACE>}, Z2.4S)"
)
LD1W_SVE = "LD1W ({<OPT_SPACE>Z0.S<OPT_SPACE>}, P0/Z, [<XNSP__3>#0])"
LD1W_ZA = (
    "LD1W ({<OPT_SPACE><ZAT__0TO3><HV>.S[W1, #0]<OPT_SPACE>}, P0/Z, [<XNSP__3><OPTIONAL_XM__7>])"
)

_TIMED = {
    "neoverse-v2": {
        "latencies": [{"cycles": "4", "start_op": "1", "target_op": "1"}],
        "measurement": {"TP_loop": "0.5", "ports": "1*p0", "uops": "1"},
    }
}


def _record(form: str, isa: tuple[str, ...], timed: int) -> InstructionRecord:
    # TIMED arch-detail counts mirror the shipped catalog: the NEON row leads
    # on measurement coverage over the SVE row; the ZA rows carry none.
    details = {f"core-{index}": dict(_TIMED["neoverse-v2"]) for index in range(timed)}
    return InstructionRecord(
        mnemonic=form.split(" ", 1)[0],
        form=form,
        summary="test",
        architecture="arm",
        arch_details=details,
        isa=list(isa),
    )


RECORDS = [
    _record(FMLA_PLAIN, ("NEON",), 2),
    _record(FMLA_SVE, ("SVE",), 1),
    _record(FMLA_ZA, ("SME",), 0),
    _record(LD1W_SVE, ("SVE",), 1),
    _record(LD1W_ZA, ("SME",), 0),
]


@pytest.mark.parametrize(
    ("mnemonic", "operands", "expected"),
    [
        ("fmla", "za.s[w8, 0, vgx2], {z0.s-z3.s}, z2.s", FMLA_ZA),
        ("ld1w", "{za0h.s[w8, 0], p0/z, [x0]}", LD1W_ZA),
        ("fmla", "z0.s, z0.s, z2.s", FMLA_PLAIN),
        ("ld1w", "{z0.s}, p0/z, [x0]", LD1W_SVE),
    ],
)
def test_pick_record_picks_the_za_row(mnemonic: str, operands: str, expected: str):
    records = [record for record in RECORDS if record.mnemonic.casefold() == mnemonic.casefold()]
    picked = pick_record(records, operands=operands)
    assert picked is not None and picked.key == expected


def _db(records: list[InstructionRecord], tmp_path) -> sqlite3.Connection:
    db = tmp_path / "catalog.db"
    catalog = Catalog(
        intrinsics=[],
        instructions=records,
        sources=[SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")],
        generated_at="2025-01-01",
    )
    build_sqlite(catalog, db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "fmla za.s[w8, 0, vgx2], {z0.s-z3.s}, z2.s",
            "FMLA (Z3.4S[<WV>, #0<OPTIONAL_COMMA_VGX2>], {<OPT_SPACE><ZN1>.4S-<ZN2>.4S<OPT_SPACE>}, Z2.4S)",
        ),
        (
            "ld1w {za0h.s[w8, 0], p0/z, [x0]}",
            "LD1W ({<OPT_SPACE><ZAT__0TO3><HV>.S[W1, #0]<OPT_SPACE>}, P0/Z, [<XNSP__3><OPTIONAL_XM__7>])",
        ),
        ("fmla z0.s, z0.s, z2.s", FMLA_SVE),
        ("ld1w {z0.s}, p0/z, [x0]", LD1W_SVE),
    ],
)
def test_hover_matcher_picks_the_za_row(text: str, expected: str, tmp_path):
    from simdref import lsp

    conn = _db(RECORDS, tmp_path)
    parsed = lsp._mnemonic_from_asm_line(text, False)
    assert parsed
    picked = lsp._resolve_line_form(conn, parsed[0], parsed[1], parsed[2], False, exact=None)
    assert picked is not None and picked.key == expected


def test_names_za_tile_boundary_scan():
    assert names_za_tile("za.s[w8, 0, vgx2], {z0.s-z3.s}, z2.s")
    assert names_za_tile("{za0h.s[w8, 0], p0/z, [x0]}")
    assert not names_za_tile("z0.s, z0.s, z2.s")
    assert not names_za_tile("{z0.s-z3.s}, z2.s, zt0")
    assert not names_za_tile("")
    assert key_names_za_tile(FMLA_ZA)
    assert key_names_za_tile(LD1W_ZA)
    assert not key_names_za_tile(FMLA_PLAIN)
    assert not key_names_za_tile(LD1W_SVE)
    # SVE counted-list keys reuse "<ZT..>" placeholder names for plain Z lists.
    assert not key_names_za_tile(
        "LD2B ({<OPT_SPACE><ZT1__5>.B, <ZT2__4>.B<OPT_SPACE>}, P0/Z, [<XNSP__3>#0])"
    )
