"""Encoding variants of one x86 form stay on rows of their own.

uops.info lists one <instruction> node per encoding variant (VEX, EVEX,
APX). The ordinary variant and the EVEX or APX variant differ in ISA labels,
category and visibility, so each variant keeps its own catalog row and its
own labels. Only opcode-map variants with identical labels share one row.
Each test feeds fixture rows copied from the upstream source data through
the real ingest and, where a reader needs the catalog, a real SQLite build.
"""

from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from simdref.cli import _emit_filtered_names, _print_search_results_runtime
from simdref.display import isa_visible
from simdref.filters import FilterSpec, load_categories_from_db
from simdref.ingest_catalog import parse_uops_xml
from simdref.models import Catalog
from simdref.storage import build_sqlite

# uops.info instruction rows copied from vendor/uops/instructions.xml.
# The EVEX and VEX encodings of VFMADD132PD share one assembly form but
# carry different ISA labels: the EVEX row is AVX512F_128-only, the VEX row
# carries the FMA label the AVX family filter needs.
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

# ADCX has a plain ADX variant and an APX EVEX variant. Their labels,
# categories and summaries differ, so each keeps its own row.
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

# The APX EVEX variant of AADD comes first in document order. The ordinary
# row must stay visible on its own.
AADD_VARIANTS = """<?xml version="1.0"?>
<root>
<instruction asm="{evex} AADD" category="APX" cpl="3" evex="1" extension="APXEVEX"
 iclass="AADD" iform="AADD_MEMi32_GPR32i32_APX" isa-set="APX_F_RAO_INT" opcode="FC"
 string="AADD_EVEX (M32, R32)"/>
<instruction asm="AADD" category="LEGACY" cpl="3" extension="RAO_INT" iclass="AADD"
 iform="AADD_MEM32_GPR32" isa-set="RAO_INT" opcode="0F38FC"
 string="AADD (M32, R32)" summary="Aadd 32-bit integer operands."/>
</root>
"""


_KEEP_DIRS: list[tempfile.TemporaryDirectory] = []


def _build_conn(records) -> sqlite3.Connection:
    tmp = tempfile.TemporaryDirectory()
    _KEEP_DIRS.append(tmp)
    path = Path(tmp.name) / "catalog.db"
    build_sqlite(
        Catalog(
            intrinsics=[], instructions=records, sources=[], generated_at="2026-10-28T00:00:00Z"
        ),
        path,
    )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


class UopsVariantRowsTest(unittest.TestCase):
    def test_variants_stay_separate_rows(self) -> None:
        records = parse_uops_xml(FMA_VARIANTS)
        self.assertEqual(len(records), 2)
        by_key = {record.key: record for record in records}
        self.assertEqual(by_key["VFMADD132PD (XMM, XMM, XMM)"].isa, ["FMA"])
        self.assertEqual(by_key["VFMADD132PD_EVEX (XMM, XMM, XMM)"].isa, ["AVX512F_128"])

    def test_list_command_finds_fma_row_under_avx_family(self) -> None:
        # Check 2: `llm list VFMADD132PD* --isa AVX` keeps one row, and the
        # ISA field splits into the single labels the reader expects.
        conn = _build_conn(parse_uops_xml(FMA_VARIANTS))
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            emitted = _emit_filtered_names(conn, "VFMADD132PD*", ["AVX"])
        self.assertEqual(emitted, 1)
        lines = [json.loads(line) for line in buf.getvalue().splitlines()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["name"], "VFMADD132PD (XMM, XMM, XMM)")
        self.assertEqual(lines[0]["isa"], ["FMA"])

    def test_plain_variant_passes_family_filter(self) -> None:
        spec = FilterSpec()
        by_key = {record.key: record for record in parse_uops_xml(FMA_VARIANTS)}
        self.assertTrue(
            spec.matches(by_key["VFMADD132PD (XMM, XMM, XMM)"], enabled_families=["AVX"])
        )
        self.assertFalse(
            spec.matches(by_key["VFMADD132PD (XMM, XMM, XMM)"], enabled_families=["SSE"])
        )

    def test_adcx_category_stays_exactly_matchable(self) -> None:
        # Check 1: the plain ADCX row keeps category ADOX_ADCX as a single
        # value the exact filters, the SQL IN predicate and the facets match.
        records = parse_uops_xml(ADCX_VARIANTS)
        plain = [record for record in records if record.key == "ADCX (R32, R32)"]
        self.assertEqual(len(plain), 1)
        row = plain[0]
        self.assertEqual(row.metadata["category"], "ADOX_ADCX")
        self.assertEqual(row.metadata["extension"], "ADOX_ADCX")
        self.assertTrue(FilterSpec().matches(row, enabled_categories=["ADOX_ADCX"]))
        conn = _build_conn(records)
        sql_rows = conn.execute(
            "SELECT key FROM instructions_data WHERE category = ?", ("ADOX_ADCX",)
        ).fetchall()
        self.assertEqual([r["key"] for r in sql_rows], ["ADCX (R32, R32)"])
        categories = {spec.category for spec in load_categories_from_db(conn)}
        self.assertIn("ADOX_ADCX", categories)
        self.assertFalse(any("/" in category for category in categories))

    def test_apx_first_variant_does_not_hide_plain_row(self) -> None:
        # Check 3: the merged-into-one AADD row was APX-labelled and hidden;
        # as separate rows the plain encoding stays visible to the CLI.
        records = parse_uops_xml(AADD_VARIANTS)
        by_key = {record.key: record for record in records}
        self.assertEqual(len(records), 2)
        self.assertTrue(isa_visible(by_key["AADD (M32, R32)"].isa))
        self.assertFalse(isa_visible(by_key["AADD_EVEX (M32, R32)"].isa))
        conn = _build_conn(records)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            count = _print_search_results_runtime(conn, "AADD", 100, as_json=True)
        self.assertEqual(count, 1)
        self.assertIn("AADD (M32, R32)", buf.getvalue())

    def test_catalog_gap_rows_still_produced(self) -> None:
        from test_leftover_catalog_gaps import UOPS_FIXTURE

        keys = {record.key for record in parse_uops_xml(UOPS_FIXTURE)}
        self.assertIn("MOVSD_XMM (XMM, XMM)", keys)
        self.assertIn("ADD (R16, R16)", keys)


if __name__ == "__main__":
    unittest.main()
