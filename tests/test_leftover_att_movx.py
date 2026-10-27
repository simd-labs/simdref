"""AT&T zero/sign-extend names resolve to the Intel MOVZX/MOVSX/MOVSXD rows.

A name carries both widths (movzbl: b source, l destination), so the lookup
maps it onto the catalog row that has those widths. The plain suffix strip
cannot do this: it lands on "movzb", a name no catalog row holds.
"""

from __future__ import annotations

import unittest

from test_lsp_stdio import CATALOG, _Server

ATT_URI = "file:///tmp/simdref_att_movx.s"

ATT_TEXT = (
    "movzbl (%rdx), %eax\n"
    "movzwq %ax, %rcx\n"
    "movsbw (%rdx), %cx\n"
    "movslq (%rdx), %rax\n"
    "movsb\n"
    "movsd %xmm1, %xmm0\n"
)


@unittest.skipUnless(CATALOG.is_file(), "needs isa update catalog")
class AttMovxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _Server(CATALOG)
        cls.server.start()
        cls.server.open(ATT_URI, "asm", ATT_TEXT)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def _hover(self, line, char=0):
        result = self.server.hover(ATT_URI, line, char)
        self.assertIsNotNone(result, f"hover on line {line} returned null")
        return result["contents"]["value"]

    def test_movzbl_mem_hovers_m8_form(self):
        value = self._hover(0)
        self.assertIn("MOVZX (R32, M8)", value)
        self.assertNotIn("MOVZX (R32, M16)", value)

    def test_movzwq_reg_hovers_r64_r16_form(self):
        value = self._hover(1)
        self.assertIn("MOVZX (R64, R16)", value)
        self.assertNotIn("MOVZX (R64, M16)", value)

    def test_movsbw_mem_hovers_m8_form(self):
        value = self._hover(2)
        self.assertIn("MOVSX (R16, M8)", value)
        self.assertNotIn("MOVSX (R16, R8", value)

    def test_movslq_mem_hovers_m32_form(self):
        value = self._hover(3)
        self.assertIn("MOVSXD (R64, M32)", value)

    def test_movsb_stays_the_string_op(self):
        value = self._hover(4)
        self.assertIn("MOVSB", value)
        self.assertNotIn("MOVSX", value)

    def test_movsd_xmm_stays_the_sse_move(self):
        value = self._hover(5)
        self.assertIn("MOVSD_XMM", value)
        self.assertNotIn("Move Data From String to String", value)


if __name__ == "__main__":
    unittest.main()
