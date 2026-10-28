"""AArch64 index-mode and '#' immediate lines must pick their own form.

A post-index line ("ldr s1, [x11], #4") hovers the post-index row, not the
register-offset row; a '#' immediate holds class i, so the immediate row
outranks a same-shape register row. The bracket key split counts only
top-level commas, so a pre-index key and a register-offset key read as two
operands, like the line. An AArch64 alias fires only where the operand text
alone decides the target: register moves are ORR, moves with sp or wsp are
ADD immediate #0, an immediate move is MOVZ, MOVN or ORR by its exact
encoding, and one ZA tile under a governing predicate is MOVA. Any other
mov form shows nothing. Hover and inlay hints go through the same picker,
so they agree on every row.
"""

import os
import tempfile
import unittest
from pathlib import Path

from simdref.lsp import _hover_markdown, _hint_label, _mnemonic_from_asm_line
from simdref.storage import SQLITE_PATH, open_db

_USER_CATALOG = Path.home() / ".local" / "share" / "simdref" / "catalog.db"
_catalog_env = os.environ.get("SIMDREF_CATALOG", "")
_missing = Path(tempfile.gettempdir()) / "simdref-test-no-catalog.db"
CATALOG = (
    Path(_catalog_env)
    if _catalog_env and Path(_catalog_env).is_file()
    else (_missing if _catalog_env else (SQLITE_PATH if SQLITE_PATH.is_file() else _USER_CATALOG))
)

# (line, fragment in the hover form). None = no hover, no hint.
FORM_ROWS = [
    ("ldr s1, [x11], #4", "LDR (<ST>, [<XNSP_OPTION>], <HASH>#0)"),
    ("str x0, [sp, #-16]!", "STR (<XTORXZR__11>, [<XNSP_OPTION>, <HASH>#0]!)"),
    ("ldp x29, x30, [sp], #16", "LDP (<XT1ORXZR>, <XT2ORXZR>, [<XNSP_OPTION>], <HASH>#0)"),
    ("mov w0, #0x10", "MOVZ (<WDORWZR>"),
    # A register offset keeps the register-offset form (no '<HASH>' in the row).
    ("ldr w0, [x9, x10]", "LDR (<WTORWZR__4>, [<XNSP_OPTION>, <WORX_CHOICE>"),
]
# A shift/extend suffix belongs to the last register operand: the line counts
# as three operands. cmp aliases subs, and the register-offset ldr keeps its
# register-offset form even with a shift on the offset register.
SHIFT_ROWS = [
    ("add x0, x1, x2, lsl #2", "ADD (<XDORXZR__6>, <XNORXZR__12>, <XMORXZR__4>#0)"),
    ("add x0, x1, w2, uxtw #2", "ADD (<XDSP_OPTION>, <XNSP_OPTION__6>, <R_OPTION__2>"),
    ("cmp x0, x1, lsl #2", "SUBS (<XDORXZR__6>, <XNORXZR__12>, <XMORXZR__4>#0)"),
    ("ldr x0, [x1, x2, lsl #3]", "LDR (<XTORXZR__11>, [<XNSP_OPTION>, <WORX_CHOICE>"),
]
# Finding 5: the suffix kind selects shifted vs extended register rows.
EXTEND_ROWS = [
    ("sub x0, x1, w2, sxtx #2", "SUB (<XDSP_OPTION>, <XNSP_OPTION__6>, <R_OPTION__2>"),
    ("sub x0, x1, x2, sxtx #2", "SUB (<XDSP_OPTION>, <XNSP_OPTION__6>, <R_OPTION__2>"),
    ("add x0, x1, x2, lsr #1", "ADD (<XDORXZR__6>, <XNORXZR__12>, <XMORXZR__4>#0)"),
    ("adds x0, x1, w2, uxtw #1", "ADDS (<XDORXZR__6>, <XNSP_OPTION__6>, <R_OPTION__2>"),
]
# Finding 1: alias lines agree between hover and hint.
ALIAS_ROWS = [
    ("cmn x0, x1", "ADDS (<XDORXZR__6>"),
    ("tst x0, x1", "ANDS (<XDORXZR__6>"),
    ("cmp w0, w1", "SUBS (<WDORWZR>"),
    ("mov x0, x1", "ORR (<XDORXZR__6>"),
]
# Finding 3: sp/wsp moves are ADD immediate with #0, at the operand width.
SP_MOVE_ROWS = [
    ("mov x29, sp", "ADD (<XDSP_OPTION>, <XNSP_OPTION__3>, <HASH>#0#0)"),
    ("mov sp, x29", "ADD (<XDSP_OPTION>, <XNSP_OPTION__3>, <HASH>#0#0)"),
    ("mov w0, wsp", "ADD (<WDWSP_OPTION>, <WNWSP_OPTION>, <HASH>#0#0)"),
]
# Finding 4: the weak gate judges the aliased form on shared fp/sp spellings.
WEAK_ROWS = [
    ("mov fp, sp", "ADD (<XDSP_OPTION>"),
    ("cmp fp, sp", "SUBS"),
    ("tst fp, sp", "ANDS"),
]
# Finding 2: an ARM32 line (r0-r15) gets no AArch64 alias: no hover, no hint.
ARM32_ROWS = ["mov r9, r10", "mov r0, #1", "cmp r0, r1", "mov r3, sp"]
# Finding 6: an SVE line without ZA tiles picks its exact register-list row.
SVE_ROWS = [
    ("ld1b {z0.b}, p0/z, [x0]", "LD1B ({<OPT_SPACE>Z0.B<OPT_SPACE>}, P0/Z, [<XNSP__3>#0])"),
    ("ld1w {z0.s}, p0/z, [x0]", "LD1W ({<OPT_SPACE>Z0.S<OPT_SPACE>}, P0/Z, [<XNSP__3>#0])"),
    ("st1d {z0.d}, p0, [x0]", "ST1D ({<OPT_SPACE>Z0.D<OPT_SPACE>}, P0, [<XNSP__3>#0])"),
    ("ld1b {z0.b}, p0/z, [x1, x2]", "LD1B ({<OPT_SPACE>Z0.B<OPT_SPACE>}, P0/Z, [<XNSP__3>, X2])"),
]
# The class of a brace list is its shape: a multi-register list or range
# matches the counted multi-vector rows, and pn8 is a predicate-as-counter.
SME2_ROWS = [
    (
        "ld1b {z0.b, z1.b}, pn8/z, [x0]",
        "LD1B ({<OPT_SPACE><ZT1>.B-<ZT2>.B<OPT_SPACE>}, <PNG>/Z, [<XNSP__3>#0])",
    ),
    (
        "st1b {z0.b, z1.b}, pn8, [x0]",
        "ST1B ({<OPT_SPACE><ZT1>.B-<ZT2>.B<OPT_SPACE>}, <PNG>, [<XNSP__3>#0])",
    ),
    (
        "ld1b {z0.b-z3.b}, pn8/z, [x0]",
        "LD1B ({<OPT_SPACE><ZT1__2>.B-<ZT4>.B<OPT_SPACE>}, <PNG>/Z, [<XNSP__3>#0])",
    ),
    (
        "st1w {z0.s-z1.s}, pn8, [x0]",
        "ST1W ({<OPT_SPACE><ZT1>.S-<ZT2>.S<OPT_SPACE>}, <PNG>, [<XNSP__3>#0])",
    ),
]
# The alias target of a mov line is decided by the operand class: a single
# ZA tile under a governing predicate means MOVA, a general-register pair is
# ORR, sp or wsp is ADD immediate #0, and an immediate is MOVZ, MOVN or ORR
# by its exact encoding.
MOV_ALIAS_ROWS = [
    ("mov za0h.b[w12, 0], p0/m, z0.b", "MOVA (<ZAD__ZERO><HV__7>.B[W1, #0], P0/M, Z1.B)"),
    ("mov z0.b, p0/m, za0h.b[w12, 0]", "MOVA (Z0.B, P0/M, <ZAN__ZERO><HV__4>.B[W1, #0])"),
    ("mov x0, x1", "ORR (<XDORXZR__6>, <XNORXZR__12>, <XMORXZR__4>#0)"),
    ("mov x29, sp", "ADD (<XDSP_OPTION>, <XNSP_OPTION__3>, <HASH>#0#0)"),
    ("mov x0, #0x10, lsl #16", "MOVZ (<XDORXZR__6>, <HASH>#0<OPTIONAL_EXTEND__13>)"),
    ("mov x0, #0X1234", "MOVZ (<XDORXZR__6>, <HASH>#0<OPTIONAL_EXTEND__13>)"),
    ("mov w0, #-65536", "MOVZ (<WDORWZR>, <HASH>#0<OPTIONAL_EXTEND__12>)"),
    ("mov x0, #0xffffffffffff1234", "MOVN (<XDORXZR__6>, <HASH>#0<OPTIONAL_EXTEND__13>)"),
    ("mov w0, #0xffff1234", "MOVN (<WDORWZR>, <HASH>#0<OPTIONAL_EXTEND__12>)"),
    ("mov x0, #-0X8000000000000000", "MOVZ (<XDORXZR__6>, <HASH>#0<OPTIONAL_EXTEND__13>)"),
    ("mov w0, #0x00ff00ff", "ORR (<WDWSP_OPTION>, <WNORWZR>, <HASH>#0)"),
    ("mov x0, #0x00ff00ff00ff00ff", "ORR (<XDSP_OPTION>, <XNORXZR__11>, <HASH>#0)"),
]
# Every other mov form shows nothing: SIMD&FP forms, vector lanes, SVE and
# predicate forms, multi-tile ZA lists, and any immediate that does not
# parse. The server must survive every one of these inputs.
MOV_NULL_ROWS = [
    "mov z0.b, w2",
    "mov z0.b, p1/m, z2.b",
    "mov d0, v1.d[1]",
    "mov v0.b[0], v1.b[15]",
    "mov v0.8b, v1.8b",
    "mov b0, v1.b[15]",
    "mov p0.b, p1.b",
    "mov p0.b, p1/m, p2.b",
    "mov z0.b, #42",
    "mov za0h.b[w12, 0:1], {z0.b-z1.b}",
    "mov za.b[w8, 0, vgx2], {z0.b-z1.b}",
    "mov z0.d, sp",
    "mov x0, #0x",
    "mov x0, #",
    "mov x0, #" + "9" * 5000,
    "mov x0, #:lo12:sym",
]


class LeftoverArmFormPickTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CATALOG.is_file():
            raise unittest.SkipTest(f"simdref catalog missing at {CATALOG}")
        cls._conn = open_db(path=CATALOG)
        cls._conn.execute("PRAGMA query_only=1")
        cls._db_path = str(CATALOG)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_conn", None) is not None:
            cls._conn.close()

    def _hover_and_hint(self, line):
        mnemonic, operand_count, operands = _mnemonic_from_asm_line(line, False)
        hover = _hover_markdown(
            self._conn,
            mnemonic,
            True,
            operand_count=operand_count,
            operand_text=operands,
        )
        hint = _hint_label(self._db_path, mnemonic, operand_count, operands)
        return hover, hint

    def test_index_modes_and_immediates_pick_their_form(self):
        for line, fragment in FORM_ROWS:
            with self.subTest(line=line):
                hover, hint = self._hover_and_hint(line)
                self.assertIsNotNone(hover)
                self.assertIn(fragment, hover)
                self.assertIsNotNone(hint)

    def test_shift_and_extend_fold_into_the_register(self):
        for line, fragment in SHIFT_ROWS:
            with self.subTest(line=line):
                hover, hint = self._hover_and_hint(line)
                self.assertIsNotNone(hover)
                self.assertIn(fragment, hover)
                self.assertIsNotNone(hint)

    def test_register_offset_row_carries_no_immediate(self):
        hover, hint = self._hover_and_hint("ldr w0, [x9, x10]")
        self.assertIsNotNone(hover)
        self.assertNotIn("<HASH>", hover)
        self.assertIn("LDR (<WTORWZR__4>, [<XNSP_OPTION>, <WORX_CHOICE>", hover)

    def test_extend_suffix_picks_the_extended_row(self):
        for line, fragment in EXTEND_ROWS:
            with self.subTest(line=line):
                hover, hint = self._hover_and_hint(line)
                self.assertIsNotNone(hover)
                self.assertIn(fragment, hover)
                self.assertIsNotNone(hint)

    def test_alias_lines_agree_on_hover_and_hint(self):
        for line, fragment in ALIAS_ROWS:
            with self.subTest(line=line):
                hover, hint = self._hover_and_hint(line)
                self.assertIsNotNone(hover)
                self.assertIn(fragment, hover)
                self.assertIsNotNone(hint)

    def test_sp_moves_are_add_immediate(self):
        for line, fragment in SP_MOVE_ROWS:
            with self.subTest(line=line):
                hover, hint = self._hover_and_hint(line)
                self.assertIsNotNone(hover)
                self.assertIn(fragment, hover)
                self.assertIsNotNone(hint)

    def test_weak_gate_judges_the_aliased_form(self):
        for line, fragment in WEAK_ROWS:
            with self.subTest(line=line):
                hover, hint = self._hover_and_hint(line)
                self.assertIsNotNone(hover)
                self.assertIn(fragment, hover)
                self.assertIsNotNone(hint)

    def test_arm32_lines_get_no_aarch64_alias(self):
        for line in ARM32_ROWS:
            with self.subTest(line=line):
                hover, hint = self._hover_and_hint(line)
                self.assertIsNone(hover)
                self.assertIsNone(hint)

    def test_sve_lines_pick_their_register_list_row(self):
        for line, fragment in SVE_ROWS:
            with self.subTest(line=line):
                hover, hint = self._hover_and_hint(line)
                self.assertIsNotNone(hover)
                self.assertIn(fragment, hover)
                self.assertIsNotNone(hint)
                self.assertNotIn("<ZAT", hover)

    def test_sme2_lists_match_the_counted_rows(self):
        for line, fragment in SME2_ROWS:
            with self.subTest(line=line):
                hover, hint = self._hover_and_hint(line)
                self.assertIsNotNone(hover)
                self.assertIn(fragment, hover)
                self.assertIsNotNone(hint)
                self.assertNotIn("<ZAT", hover)

    def test_mov_keeps_only_decided_alias_rows(self):
        for line, fragment in MOV_ALIAS_ROWS:
            with self.subTest(line=line):
                hover, hint = self._hover_and_hint(line)
                self.assertIsNotNone(hover)
                self.assertIn(fragment, hover)
                if fragment.startswith("MOVA"):
                    self.assertNotIn("ORR (", hover)
                self.assertIsNotNone(hint)

    def test_undecided_mov_forms_show_nothing(self):
        for line in MOV_NULL_ROWS:
            with self.subTest(line=line[:40]):
                hover, hint = self._hover_and_hint(line)
                self.assertIsNone(hover)
                self.assertIsNone(hint)

    def test_x86_pred_operand_line_stays_x86(self):
        hover, hint = self._hover_and_hint("str p0.b")
        self.assertIsNotNone(hover)
        self.assertIn("STR (M16)", hover)
        self.assertNotIn("arm:", hover)


if __name__ == "__main__":
    unittest.main()
