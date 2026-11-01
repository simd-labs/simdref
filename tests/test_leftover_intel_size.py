"""Intel-syntax size keywords (byte/word/...word ptr) must pick the sized M form.

``movzx eax, byte ptr [rdx]`` used to lose its memory width and hover the
wrong form, because ``_operand_class`` returned a bare ``m`` for any memory
operand and the scorer had no width tie-break for sized memory classes. A size
keyword only adds information: ``vmovdqu zmm0, zmmword ptr [rdx]`` has no M512
catalog row, so it hovers the same form as ``vmovdqu zmm0, [rdx]``, and the
operand position keeps deciding the load/store direction.

The movzx-word and vmovdqu-word lines pair a register of one width with a
size keyword of another, so they stay independently loud: ignore the size
keyword and the register width or the key order picks a different form.
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
    "movzx ax, word ptr [rdx]\n"
    "vmovdqu xmm0, ymmword ptr [rdx]\n"
    "vmovdqu ymm0, xmmword ptr [rdx]\n"
    "vmovdqu zmm0, zmmword ptr [rdx]\n"
    "vmovdqu zmmword ptr [rdx], zmm0\n"
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
        # The r16 destination names no M16 form; key order unsized picks
        # MOVZX (R16, M8), so this line fails on its own when the keyword
        # is ignored.
        value = self._hover(INTEL_URI, 1)
        self.assertIn("MOVZX (R32, M16)", value)
        self.assertNotIn("MOVZX (R16, M8)", value)

    def test_vmovdqu_ymmword_hovers_m256_form(self):
        # xmm0 reads as the XMM form; only the 256-bit keyword names M256.
        value = self._hover(INTEL_URI, 2)
        self.assertIn("VMOVDQU (YMM, M256)", value)
        self.assertNotIn("VMOVDQU (XMM, M128)", value)

    def test_vmovdqu_xmmword_hovers_m128_form(self):
        # ymm0 reads as the YMM form; only the 128-bit keyword names M128.
        value = self._hover(INTEL_URI, 3)
        self.assertIn("VMOVDQU (XMM, M128)", value)
        self.assertNotIn("VMOVDQU (YMM, M256)", value)

    def test_vmovdqu_zmmword_load_keeps_unsized_direction(self):
        # VMOVDQU has no ZMM row: the pick must equal the unsized line's
        # (``vmovdqu zmm0, [rdx]``), the load form, not the key-order store.
        value = self._hover(INTEL_URI, 4)
        self.assertIn("VMOVDQU (XMM, M128)", value)
        self.assertNotIn("VMOVDQU (M128, XMM)", value)

    def test_vmovdqu_zmmword_store_keeps_unsized_direction(self):
        value = self._hover(INTEL_URI, 5)
        self.assertIn("VMOVDQU (M128, XMM)", value)
        self.assertNotIn("VMOVDQU (XMM, M128)", value)

    def test_att_movzbl_hovers_movzx_m8_form(self):
        # AT&T control: movzbl resolves through the zero/sign-extend alias onto
        # the MOVZX row with the widths its name carries (b source, l dest).
        # The sized m classes must not change that.
        result = self.server.hover(ATT_URI, 0, 0)
        self.assertIsNotNone(result, "hover on movzbl returned null")
        self.assertIn("MOVZX (R32, M8)", result["contents"]["value"])


if __name__ == "__main__":
    unittest.main()
