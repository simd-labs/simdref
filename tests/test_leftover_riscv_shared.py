"""RISC-V lines on shared register spellings must not hover an AArch64 form.

s0-s11, sp and fp are register names RISC-V and AArch64 share. Such a name
keeps an architecture in the running only when one of the mnemonic's forms in
that architecture accepts the line's operands position by position; a line
made only of shared names shows nothing when no architecture accepts it. Hover
and inlay hints go through the same gate, so they agree on every row.
"""

import os
import tempfile
import unittest
from pathlib import Path

from simdref.lsp import _hover_markdown, _hint_label, _mnemonic_from_asm_line
from simdref.storage import SQLITE_PATH, load_instructions_by_mnemonic_from_db, open_db

_USER_CATALOG = Path.home() / ".local" / "share" / "simdref" / "catalog.db"
_catalog_env = os.environ.get("SIMDREF_CATALOG", "")
_missing = Path(tempfile.gettempdir()) / "simdref-test-no-catalog.db"
CATALOG = (
    Path(_catalog_env)
    if _catalog_env and Path(_catalog_env).is_file()
    else (_missing if _catalog_env else (SQLITE_PATH if SQLITE_PATH.is_file() else _USER_CATALOG))
)

# No line may name a foreign form.
RISCV_SHARED_RISCV = [
    "add s0, s1, s2",
    "sub s9, s10, s11",
    "mul s4, s5, s6",
    "add sp, sp, sp",
    "mul fp, fp, fp",
    "addw s0, s1, s2",
]
ARM_SCALAR_FP = [
    ("fneg d0, d1", "FNEG (D0, D1)"),
    ("fmov s0, s1", "FMOV (S0, S1)"),
    ("fmadd s0, s1, s2, s3", "FMADD (S0, S1, S2, S3)"),
    ("fdiv s0, s1, s2", "FDIV (S0, S1, S2)"),
    ("ldr s1, [x11], #4", "LDR"),
    ("sqadd s0, s1, s2", "SQADD"),
    ("uqsub s0, s1, s2", "UQSUB"),
    ("sqabs s0, s1", "SQABS"),
]


class RiscvSharedRegisterTests(unittest.TestCase):
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

    def test_riscv_shared_register_lines_keep_the_riscv_form(self):
        for line in RISCV_SHARED_RISCV:
            with self.subTest(line=line):
                hover, hint = self._hover_and_hint(line)
                has_riscv = any(
                    form.architecture == "riscv"
                    for form in load_instructions_by_mnemonic_from_db(self._conn, line.split()[0])
                )
                if has_riscv:
                    self.assertIsNotNone(hover, f"{line} got no hover")
                    self.assertEqual(hover.splitlines()[1], line.split()[0])
                    self.assertIsNotNone(hint, f"{line} got no hint")
                else:
                    self.assertIsNone(hover)
                    self.assertIsNone(hint)

    def test_x86_line_never_shows_a_riscv_or_arm_form(self):
        # "%ax" and "%bx" are x86 registers: the w-strip resolves to the x86
        # ADD form even though the exact key "addw" is RISC-V. The lookup gate
        # decides this line, so a reverted gate leaves it silent.
        hover, hint = self._hover_and_hint("addw %ax, %bx")
        self.assertIsNotNone(hover)
        self.assertRegex(hover.splitlines()[1], r"^ADD \((R\d+|[A-D]X|E[A-D]X)")
        self.assertIsNotNone(hint)

    def test_arm_scalar_fp_lines_keep_their_forms(self):
        for line, fragment in ARM_SCALAR_FP:
            with self.subTest(line=line):
                hover, hint = self._hover_and_hint(line)
                self.assertIsNotNone(hover)
                self.assertIn(fragment, hover)
                self.assertIsNotNone(hint)


if __name__ == "__main__":
    unittest.main()
