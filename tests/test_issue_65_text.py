"""Issue #65 catalog text: RISC-V sentinel synopsis and AMX tile summary.

Fixtures are copied byte-exact from the vendored uops.info XML, the Intel
Intrinsics Guide data.js and the RISC-V docs page."""

import contextlib
import dataclasses
import importlib.util
import io
import json
import unittest
from pathlib import Path

from simdref.ingest_catalog import link_records, parse_intel_payload, parse_uops_xml
from simdref.riscv import parse_riscv_instruction_payload

DOCS_URL = "https://docs.riscv.org/reference/isa/unpriv/v-st-ext.html"
VFSQRT_DOCS = '<div class="sect3">\n<h4 id="30-1-13-8-vector-floating-point-square-root-instruction"><a class="anchor" href="#30-1-13-8-vector-floating-point-square-root-instruction"></a>30.1.13.8. Vector Floating-Point Square-Root Instruction</h4>\n<div class="paragraph">\n<p>This is a unary vector-vector instruction.</p>\n</div>\n<div id="norm:vfsqrt_op" class="listingblock">\n<div class="content">\n<pre># Floating-point square root\nvfsqrt.v vd, vs2, vm   # Vector-vector square root</pre>\n</div>\n</div>\n</div>\n'
TDPFP16PS_UOPS = '<root><instruction asm="TDPFP16PS" category="AMX_TILE" cpl="3" extension="AMX_TILE" iclass="TDPFP16PS" iform="TDPFP16PS_TMMf32_TMM2f16_TMM2f16" isa-set="AMX_FP16" no_reg_match="1" opcode="5C" string="TDPFP16PS (TMM, TMM, TMM)" vex="1">\n      <operand idx="1" name="REG0" r="1" type="reg" w="1" width="0" xtype="f32">TMM0,TMM1,TMM2,TMM3,TMM4,TMM5,TMM6,TMM7</operand>\n      <operand idx="2" name="REG1" r="1" type="reg" width="0" xtype="2f16">TMM0,TMM1,TMM2,TMM3,TMM4,TMM5,TMM6,TMM7</operand>\n      <operand idx="3" name="REG2" r="1" type="reg" width="0" xtype="2f16">TMM0,TMM1,TMM2,TMM3,TMM4,TMM5,TMM6,TMM7</operand>\n    </instruction></root>'
TILE_DPFP16PS_INTRINSIC = '<intrinsics_list><intrinsic name="_tile_dpfp16ps" tech="AMX">\n\t<return type="void" />\n\t<parameter type="constexpr int" etype="FP32" varname="dst" />\n\t<parameter type="constexpr int" etype="FP16" varname="a" />\n\t<parameter type="constexpr int" etype="FP16" varname="b" />\n\t<description>Compute dot-product of FP16 (16-bit) floating-point pairs in tiles "a" and "b", accumulating the intermediate single-precision (32-bit) floating-point elements with elements in "dst", and store the 32-bit result back to tile "dst".</description>\n\t<operation>FOR m := 0 TO dst.rows - 1\n\ttmp := dst.row[m]\n\tFOR k := 0 TO (a.colsb / 4) - 1\n\t\tFOR n := 0 TO (dst.colsb / 4) - 1\n\t\t\ttmp.fp32[n] += FP32(a.row[m].fp16[2*k+0]) * FP32(b.row[k].fp16[2*n+0])\n\t\t\ttmp.fp32[n] += FP32(a.row[m].fp16[2*k+1]) * FP32(b.row[k].fp16[2*n+1])\n\t\tENDFOR\n\tENDFOR\n\twrite_row_and_zero(dst, m, tmp, dst.colsb)\nENDFOR\nzero_upper_rows(dst, dst.rows)\nzero_tileconfig_start()\n\t</operation>\n\t<instruction form="tmm, tmm, tmm" name="TDPFP16PS" xed="TDPFP16PS_TMMf32_TMM2f16_TMM2f16" />\n\t<CPUID>AMX-FP16</CPUID>\n\t<header>immintrin.h</header>\n\t<category>Application-Targeted</category>\n\t</intrinsic></intrinsics_list>'

VFSQRT_ITEM = '{\n  "mnemonic": "vfsqrt.v",\n  "form": "vfsqrt.v",\n  "summary": "No synopsis available.",\n  "isa": [\n    "V"\n  ],\n  "url": "https://docs.riscv.org/reference/isa/unpriv/v-st-ext.html",\n  "description": {\n    "Description": "No description available.",\n    "Operation": ""\n  },\n  "metadata": {\n    "category": "Floating-Point",\n    "extension": "V",\n    "policy": "agnostic",\n    "masking": "unmasked",\n    "tail_policy": "agnostic"\n  },\n  "policy": "agnostic",\n  "masking": "unmasked",\n  "encoding": "  match: 010011------00000001-----1010111\\n  variables:\\n    - name: vm\\n      location: 25-25\\n    - name: vs2\\n      location: 24-20\\n    - name: vd\\n      location: 11-7"\n}'
AES32ESI_ITEM = '{\n  "mnemonic": "aes32esi",\n  "form": "aes32esi",\n  "summary": "No synopsis available.",\n  "isa": [\n    "Zkne"\n  ],\n  "url": "https://github.com/riscv/riscv-unified-db/blob/798d6a54d709a1584e7fa81bd8cc4ec7b7803a42/spec/std/isa/inst/Zkne/aes32esi.yaml",\n  "description": {\n    "Description": "No description available.",\n    "Operation": ""\n  },\n  "metadata": {\n    "category": "Arithmetic",\n    "extension": "Zkne",\n    "policy": "agnostic",\n    "masking": "unmasked",\n    "tail_policy": "agnostic"\n  },\n  "policy": "agnostic",\n  "masking": "unmasked",\n  "encoding": "  match: --10001----------000-----0110011\\n  variables:\\n    - name: bs\\n      location: 31-30\\n    - name: xs2\\n      location: 24-20\\n    - name: xs1\\n      location: 19-15\\n    - name: xd\\n      location: 11-7"\n}'


def _riscv(item_json, with_docs):
    payload = {"format": "riscv-unified-db-v1", "instructions": [json.loads(item_json)]}
    if with_docs:
        payload["docs_pages"] = {DOCS_URL: VFSQRT_DOCS}
    return parse_riscv_instruction_payload(json.dumps(payload))[0]


class RiscvSentinelTests(unittest.TestCase):
    def test_sentinel_becomes_empty_summary(self):
        record = _riscv(AES32ESI_ITEM, with_docs=False)
        self.assertEqual(record.summary, "")

    def test_sentinel_stays_empty_when_docs_page_has_text(self):
        # VFSQRT_DOCS holds a Description paragraph; the summary must not borrow it.
        record = _riscv(VFSQRT_ITEM, with_docs=True)
        self.assertEqual(record.summary, "")

    def test_missing_summary_is_empty_not_a_dot(self):
        item = json.loads(AES32ESI_ITEM)
        del item["summary"]
        payload = {"format": "riscv-unified-db-v1", "instructions": [item]}
        record = parse_riscv_instruction_payload(json.dumps(payload))[0]
        self.assertEqual(record.summary, "")

    def test_real_summary_keeps_its_period(self):
        item = json.loads(AES32ESI_ITEM)
        item["summary"] = "AES encrypt"
        payload = {"format": "riscv-unified-db-v1", "instructions": [item]}
        record = parse_riscv_instruction_payload(json.dumps(payload))[0]
        self.assertEqual(record.summary, "AES encrypt.")


def _load_validator():
    path = Path(__file__).resolve().parents[1] / "tools" / "validate_sources.py"
    spec = importlib.util.spec_from_file_location("validate_sources_issue_65", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_riscv_validator(item, parse=None):
    module = _load_validator()
    text = json.dumps({"format": "riscv-unified-db-v1", "instructions": [item]})
    module.fetch_riscv_unified_db_data = lambda: (text, None)
    if parse is not None:
        module.parse_riscv_instruction_payload = parse
    with contextlib.redirect_stdout(buffer := io.StringIO()):
        _checked, failures = module.validate_riscv_instructions()
    return failures, buffer.getvalue()


class RiscvValidatorTests(unittest.TestCase):
    def test_sourced_empty_summary_passes(self):
        item = json.loads(AES32ESI_ITEM)
        for synopsis in ("No synopsis available.", "", None):
            with self.subTest(synopsis=synopsis):
                if synopsis is None:
                    del item["summary"]
                else:
                    item["summary"] = synopsis
                failures, out = _run_riscv_validator(item)
                self.assertEqual(failures, 0, out)

    def test_lost_source_text_still_fails(self):
        item = json.loads(AES32ESI_ITEM)
        item["summary"] = "AES encrypt"

        def parse_and_drop_text(text):
            return [
                dataclasses.replace(r, summary="") for r in parse_riscv_instruction_payload(text)
            ]

        failures, out = _run_riscv_validator(item, parse=parse_and_drop_text)
        self.assertEqual(failures, 1, out)
        self.assertIn("empty summary for aes32esi", out)


class EmptySummaryReaderTests(unittest.TestCase):
    def test_llm_markdown_prints_no_separator_for_empty_summary(self):
        from simdref.cli import _llm_format_markdown

        row = {"intrinsic": ["__riscv_vaadd_vv_i8m1"], "isa": ["V"], "summary": ""}
        md = _llm_format_markdown({"query": "vaadd.vv", "mode": "search", "results": [row]})
        line = md.splitlines()[-1]
        self.assertEqual(line, "- **__riscv_vaadd_vv_i8m1** `V`")
        row["summary"] = "Adds."
        md = _llm_format_markdown({"query": "vaadd.vv", "mode": "search", "results": [row]})
        self.assertEqual(md.splitlines()[-1], "- **__riscv_vaadd_vv_i8m1** `V` — Adds.")

    def test_search_subtitle_is_empty_for_empty_summary(self):
        from simdref.search import search_records

        record = _riscv(AES32ESI_ITEM, with_docs=False)
        self.assertEqual(record.summary, "")
        # A summary of only whitespace must also read as empty.
        padded = dataclasses.replace(record, summary="  ")
        for rec in (record, padded):
            with self.subTest(summary=rec.summary):
                hits = search_records([], [rec], "aes32esi")
                self.assertEqual([h.subtitle for h in hits], [""])


class X86TextTests(unittest.TestCase):
    def test_tdpfp16ps_takes_intel_description(self):
        from simdref.ingest_catalog import _backfill_amx_tile_summaries

        instructions = parse_uops_xml(TDPFP16PS_UOPS)
        intrinsics = parse_intel_payload(TILE_DPFP16PS_INTRINSIC)
        tdp = next(r for r in instructions if r.mnemonic == "TDPFP16PS")
        self.assertEqual(tdp.summary, "Tdpfp16Ps 0-bit operands.")
        link_records(intrinsics, instructions)
        _backfill_amx_tile_summaries(intrinsics, instructions)
        self.assertTrue(tdp.summary.startswith("Compute dot-product of FP16"), tdp.summary)


if __name__ == "__main__":
    unittest.main()
