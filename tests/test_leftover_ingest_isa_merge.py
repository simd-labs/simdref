"""Encoding-variant merges keep every variant's ISA labels (leftover fix).

uops.info lists one <instruction> node per encoding variant (EVEX, VEX,
APX). Variants of one form collapse into one catalog row. The merged row
must carry the union of the variants' ISA/extension labels, or the ISA
family filters lose instructions the variant rows still named.
"""

from __future__ import annotations

import unittest

from simdref.filters import FilterSpec
from simdref.ingest_catalog import parse_uops_xml

# uops.info instruction rows copied from vendor/uops/instructions.xml.
# The EVEX and VEX encodings of VFMADD132PD share one form: their cores
# both normalize to "VFMADD132PD (XMM, XMM, XMM)", so they merge into one
# row. The first variant carries only AVX512F_128; the FMA label comes
# from the second variant alone.
FMA_VARIANTS = """<?xml version="1.0"?>
<root>
<instruction asm="{evex} VFMADD132PD" category="VFMA" cpl="3" evex="1"
 extension="AVX512EVEX" iclass="VFMADD132PD"
 iform="VFMADD132PD_XMMf64_MASKmskw_XMMf64_XMMf64_AVX512" isa-set="AVX512F_128"
 string="VFMADD132PD_EVEX (XMM, XMM, XMM)"
 url="uops.info/html-instr/VFMADD132PD_EVEX_XMM_XMM_XMM.html"
 summary="Fused Multiply-Add of Packed DoublePrecision Floating-Point Values"
 url-ref="felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd"/>
<instruction asm="VFMADD132PD" category="VFMA" cpl="3" extension="FMA"
 iclass="VFMADD132PD" iform="VFMADD132PD_XMMdq_XMMdq_XMMdq" isa-set="FMA"
 string="VFMADD132PD (XMM, XMM, XMM)" vex="1"
 url="uops.info/html-instr/VFMADD132PD_XMM_XMM_XMM.html"
 summary="Fused Multiply-Add of Packed DoublePrecision Floating-Point Values"
 url-ref="felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd"/>
</root>
"""

# ADCX has an APX EVEX variant and an ADX variant whose labels, categories
# and summaries all differ.
ADCX_VARIANTS = """<?xml version="1.0"?>
<root>
<instruction asm="{load} ADCX" category="ADOX_ADCX" cpl="3" extension="ADOX_ADCX"
 iclass="ADCX" iform="ADCX_GPR32_32b" isa-set="ADOX_ADCX"
 string="ADCX (R32, R32)" summary="Unsigned Integer Addition of Two Operands With Carry Flag."/>
<instruction asm="{load} {evex} ADCX" category="APX" cpl="3" evex="1"
 extension="APXEVEX" iclass="ADCX" iform="ADCX_GPR32_32b_APX" isa-set="APX_F_ADX"
 string="ADCX_EVEX (R32, R32)" summary="Add with carry 32-bit integer operands."
 url="uops.info/html-instr/ADCX_EVEX_R32_R32.html"/>
</root>
"""


def _fma_row(records):
    rows = [r for r in records if r.key == "VFMADD132PD (XMM, XMM, XMM)"]
    assert len(rows) == 1, [r.key for r in records]
    return rows[0]


class UopsVariantMergeTest(unittest.TestCase):
    def test_variants_merge_keeps_fma_label(self):
        records = parse_uops_xml(FMA_VARIANTS)
        row = _fma_row(records)
        self.assertEqual(row.isa, ["AVX512F_128", "FMA"])

    def test_merged_fma_row_passes_avx_and_fma_filters(self):
        row = _fma_row(parse_uops_xml(FMA_VARIANTS))
        spec = FilterSpec()
        self.assertTrue(spec.matches(row, enabled_families=["AVX"]))
        self.assertFalse(spec.matches(row, enabled_families=["SSE"]))

    def test_variants_merge_unions_extension_category_summary(self):
        rows = [r for r in parse_uops_xml(ADCX_VARIANTS) if r.key == "ADCX (R32, R32)"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.isa, ["ADOX_ADCX", "APX_F_ADX"])
        self.assertIn("ADOX_ADCX", row.metadata["extension"])
        self.assertIn("APXEVEX", row.metadata["extension"])
        self.assertIn("ADOX_ADCX", row.metadata["category"])
        self.assertIn("APX", row.metadata["category"])
        self.assertIn("Unsigned Integer Addition of Two Operands With Carry Flag.", row.summary)
        self.assertIn("Add with carry 32-bit integer operands.", row.summary)
        # Single-valued fields keep the first variant; empty gaps fill.
        self.assertEqual(row.metadata["iform"], "ADCX_GPR32_32b")
        self.assertEqual(row.metadata["url"], "uops.info/html-instr/ADCX_EVEX_R32_R32.html")

    def test_catalog_gap_rows_still_produced(self):
        from test_leftover_catalog_gaps import UOPS_FIXTURE

        keys = {r.key for r in parse_uops_xml(UOPS_FIXTURE)}
        self.assertIn("MOVSD_XMM (XMM, XMM)", keys)
        self.assertIn("ADD (R16, R16)", keys)


if __name__ == "__main__":
    unittest.main()
