"""AArch64 index-mode and '#' immediate lines must pick their own form.

A post-index line ("ldr s1, [x11], #4") hovers the post-index row, not the
register-offset row; a '#' immediate holds class i, so the immediate row
outranks a same-shape register row. The bracket key split counts only
top-level commas, so a pre-index key and a register-offset key read as two
operands, like the line. "mov <Wd>, #imm" is the GAS alias the catalog keys
as MOVZ (or MOVN for a negative value). Hover and inlay hints go through the
same picker, so they agree on every row.
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


if __name__ == "__main__":
    unittest.main()
