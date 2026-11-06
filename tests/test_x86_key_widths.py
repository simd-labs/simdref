"""Width-constrained x86 key operands pick the register rows, not the port rows.

The catalog keys name the implicit port registers ("IN (AL, DX)", "OUT (DX,
EAX)"). A key operand that is an exact x86 gpr spelling takes its width from
the regtable, so DX, ECX, RCX, EDX and RDX constrain the pick. The old hand
table mapped only AL, AX, EAX, RAX and CL and left the other port spellings
open, so these lines hovered the immediate-port rows ("IN (AL, 0)"). The
intel flag decides the syntax flavor: True is Intel ("out dx, al"), False is
AT&T ("out %al, %dx").
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

# (line, intel, row). Each row is the pick after the width rule; the old
# pick was the immediate-port row of the same width ("IN (AL, 0)" and kin).
B8_ROWS = [
    ("in %dx, %al", True, "IN (AX, DX)"),
    ("in %dx, %al", False, "IN (AL, DX)"),
    ("in %dx, %eax", True, "IN (AX, DX)"),
    ("in %dx, %eax", False, "IN (EAX, DX)"),
    ("in al, dx", True, "IN (AL, DX)"),
    ("in al, dx", False, "IN (AX, DX)"),
    ("in ax, dx", True, "IN (AX, DX)"),
    ("in ax, dx", False, "IN (AX, DX)"),
    ("in eax, dx", True, "IN (EAX, DX)"),
    ("in eax, dx", False, "IN (AX, DX)"),
    ("out %al, %dx", True, "OUT (DX, AX)"),
    ("out %al, %dx", False, "OUT (DX, AL)"),
    ("out %eax, %dx", True, "OUT (DX, AX)"),
    ("out %eax, %dx", False, "OUT (DX, EAX)"),
    ("out dx, al", True, "OUT (DX, AL)"),
    ("out dx, al", False, "OUT (DX, AX)"),
    ("out dx, ax", True, "OUT (DX, AX)"),
    ("out dx, ax", False, "OUT (DX, AX)"),
    ("out dx, eax", True, "OUT (DX, EAX)"),
    ("out dx, eax", False, "OUT (DX, AX)"),
]


class X86KeyWidthTests(unittest.TestCase):
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

    def test_width_constrained_port_rows_win(self):
        for line, intel, want in B8_ROWS:
            with self.subTest(line=line, intel=intel):
                mnemonic, operand_count, operands = _mnemonic_from_asm_line(line, intel)
                hover = _hover_markdown(
                    self._conn,
                    mnemonic,
                    True,
                    operand_count=operand_count,
                    operand_text=operands,
                    intel=intel,
                )
                self.assertIsNotNone(hover, f"{line} got no hover")
                self.assertEqual(hover.splitlines()[1], want)
                hint = _hint_label(self._db_path, mnemonic, operand_count, operands, intel=intel)
                self.assertIsNotNone(hint, f"{line} got no hint")
