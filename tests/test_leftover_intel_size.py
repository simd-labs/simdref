"""Intel-syntax size keywords (byte/word/...word ptr) must pick the sized M form.

``movzx eax, byte ptr [rdx]`` used to lose its memory width and hover the
wrong form, because ``_operand_class`` returned a bare ``m`` for any memory
operand and the scorer had no width tie-break for sized memory classes.
"""

from __future__ import annotations

import unittest

from test_lsp_stdio import CATALOG, _Server

# A .s extension reads as GAS regardless of languageId; the Intel rows need
# an Intel-mode buffer or the operand order flips and the store form wins.
INTEL_URI = "file:///tmp/simdref_intel_size.asm"
ATT_URI = "file:///tmp/simdref_att_control.s"

INTEL_TEXT = (
    "movzx eax, byte ptr [rdx]\n"
    "movzx eax, word ptr [rdx]\n"
    "vmovdqu ymm0, ymmword ptr [rdx]\n"
    "vmovdqu xmm0, xmmword ptr [rdx]\n"
)

ATT_TEXT = "movzbl (%rdx), %eax\n"


@unittest.skipUnless(CATALOG.is_file(), "needs isa update catalog")
class IntelSizeKeywordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _Server(CATALOG)
        cls.server.start()
        cls.server.open(INTEL_URI, "asm", INTEL_TEXT)
        cls.server.open(ATT_URI, "asm", ATT_TEXT)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def _hover(self, uri, line, char=0):
        result = self.server.hover(uri, line, char)
        self.assertIsNotNone(result, f"hover on line {line} returned null")
        return result["contents"]["value"]

    def test_movzx_byte_hovers_m8_form(self):
        value = self._hover(INTEL_URI, 0)
        self.assertIn("MOVZX (R32, M8)", value)
        self.assertNotIn("MOVZX (R32, M16)", value)

    def test_movzx_word_hovers_m16_form(self):
        value = self._hover(INTEL_URI, 1)
        self.assertIn("MOVZX (R32, M16)", value)
        self.assertNotIn("MOVZX (R32, M8)", value)

    def test_vmovdqu_ymmword_hovers_m256_form(self):
        value = self._hover(INTEL_URI, 2)
        self.assertIn("VMOVDQU (YMM, M256)", value)
        self.assertNotIn("VMOVDQU (XMM, M128)", value)

    def test_vmovdqu_xmmword_hovers_m128_form(self):
        value = self._hover(INTEL_URI, 3)
        self.assertIn("VMOVDQU (XMM, M128)", value)
        self.assertNotIn("VMOVDQU (YMM, M256)", value)

    def test_att_movzbl_control(self):
        # AT&T control: movzbl has no catalog row and the suffix strip lands on
        # movzb (a deliberate non-match), so it stays unresolved. The sized m
        # classes must not change that.
        result = self.server.hover(ATT_URI, 0, 0)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
