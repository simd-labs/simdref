"""Leftover catalog gaps at ingest (leftovers.md items 2 and 5).

Each test feeds fixture rows copied from the upstream source data through the
ingest functions. It asserts that the forms the gap named now appear.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from simdref.arm_instructions import parse_arm_instruction_payload
from simdref.ingest_catalog import parse_uops_xml
from simdref.riscv import parse_riscv_instruction_payload

# uops.info instruction rows copied from vendor/uops/instructions.xml.
UOPS_FIXTURE = """<?xml version="1.0"?>
<root>
<instruction asm="{load} MOVSD" category="DATAXFER" cpl="3" extension="SSE2"
 iclass="MOVSD_XMM" iform="MOVSD_XMM_XMMsd_XMMsd_0F10" isa-set="SSE2" opcode="0F10"
 string="MOVSD_XMM_0F10 (XMM, XMM)"
 summary="Move or Merge Scalar Double Precision Floating-Point Value"/>
<instruction asm="{store} MOVSD" category="DATAXFER" cpl="3" extension="SSE2"
 iclass="MOVSD_XMM" iform="MOVSD_XMM_XMMsd_XMMsd_0F11" isa-set="SSE2" opcode="0F11"
 string="MOVSD_XMM_0F11 (XMM, XMM)"
 summary="Move or Merge Scalar Double Precision Floating-Point Value"/>
<instruction asm="MOVSD" category="DATAXFER" cpl="3" extension="SSE2"
 iclass="MOVSD_XMM" iform="MOVSD_XMM_XMMsd_MEMsd" isa-set="SSE2"
 string="MOVSD_XMM (M64, XMM)"
 summary="Move or Merge Scalar Double Precision Floating-Point Value"/>
<instruction asm="{load} ADD" category="BINARY" cpl="3" eosz="1" extension="BASE"
 iclass="ADD" iform="ADD_GPRv_GPRv_01" isa-set="I86" opcode="01"
 string="ADD_01 (R16, R16)" summary="Add"/>
<instruction asm="{store} ADD" category="BINARY" cpl="3" eosz="1" extension="BASE"
 iclass="ADD" iform="ADD_GPRv_GPRv_03" isa-set="I86" opcode="03"
 string="ADD_03 (R16, R16)" summary="Add"/>
<instruction asm="{load} {evex} ADD" category="BINARY" cpl="3" eosz="1" evex="1"
 extension="APXEVEX" iclass="ADD" iform="ADD_GPRv_GPRv_APX" isa-set="APX_F" nf="0"
 opcode="01" string="ADD_EVEX_01 (R16, R16)"/>
<instruction asm="ADD" category="BINARY" cpl="3" eosz="1" extension="BASE"
 iclass="ADD" iform="ADD_GPRv_MEMv" isa-set="I86" opcode="03"
 string="ADD (R16, M16)" summary="Add"/>
<instruction asm="ADD" category="BINARY" cpl="3" extension="BASE" iclass="ADD"
 iform="ADD_GPR8_GPR8_00" isa-set="I86" opcode="00"
 string="ADD_R0 (R8l, 0)" summary="Add"/>
</root>
"""


# AARCHMRS assembly symbol lists copied from the FAT A-profile archive
# (Instructions.json, LDBFADD_16 / LDFADD_32 / LDFADD_64).
def _sym(kind: str, value: str) -> dict:
    if kind == "lit":
        return {"_type": "Instruction.Symbols.Literal", "value": value}
    return {"_type": "Instruction.Symbols.RuleReference", "rule_id": value}


def _fpfar_symbols(first: str, second: tuple[str, str]) -> list[dict]:
    return [
        _sym("lit", first),
        _sym("ref", "SPACE"),
        _sym("ref", second[0]),
        _sym("ref", "COMMA"),
        _sym("ref", second[1]),
        _sym("ref", "COMMA"),
        _sym("lit", "["),
        _sym("ref", "XnSP_option"),
        _sym("lit", "]"),
    ]


AARCHMRS_FIXTURE = {
    "instructions": [
        {
            "_type": "InstructionSet",
            "name": "A64",
            "children": [
                {
                    "_type": "Instruction",
                    "name": "LDBFADD_16",
                    "assembly": {"symbols": _fpfar_symbols("LDBFADD", ("fpfar_hs", "fpfar_ht"))},
                },
                {
                    "_type": "Instruction",
                    "name": "LDFADD_32",
                    "assembly": {"symbols": _fpfar_symbols("LDFADD", ("fpfar_ss", "fpfar_st"))},
                },
                {
                    "_type": "Instruction",
                    "name": "LDFADD_64",
                    "assembly": {"symbols": _fpfar_symbols("LDFADD", ("fpfar_ds", "fpfar_dt"))},
                },
            ],
        }
    ]
}

# riscv-unified-db instruction YAML rows copied from
# spec/std/isa/inst/{I,M}/{add,sub,mul}.yaml.
RISCV_ADD_YAML = """\
$schema: "inst_schema.json#"
kind: instruction
name: add
long_name: Integer add
description: |
  Add the value in xs1 to xs2, and store the result in xd.
  Any overflow is thrown away.
definedBy:
  extension:
    name: I
assembly: xd, xs1, xs2
encoding:
  match: 0000000----------000-----0110011
operation(): X[xd] = X[xs1] + X[xs2];
"""

RISCV_SUB_YAML = """\
$schema: "inst_schema.json#"
kind: instruction
name: sub
long_name: Subtract
description: Subtract the value in xs2 from xs1, and store the result in xd
definedBy:
  extension:
    name: I
assembly: xd, xs1, xs2
encoding:
  match: 0100000----------000-----0110011
"""

RISCV_MUL_YAML = """\
$schema: "inst_schema.json#"
kind: instruction
name: mul
long_name: Signed multiply
description: |
  MUL performs an XLEN-bitxXLEN-bit multiplication of `xs1` by `xs2`.
definedBy:
  extension:
    name: Zmmul
assembly: xd, xs1, xs2
encoding:
  match: 0000001----------000-----0110011
"""


def _vendor_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "vendor_riscv_sources.py"
    spec = importlib.util.spec_from_file_location("vendor_riscv_sources", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UopsFormNamingTest(unittest.TestCase):
    """MOVSD (XMM, XMM) and ADD (R16, R16) rows named without encoding tags."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = parse_uops_xml(UOPS_FIXTURE)
        cls.by_key = {record.key: record for record in cls.records}

    def test_movsd_register_form_appears(self) -> None:
        rows = [
            record
            for record in self.records
            if record.mnemonic == "MOVSD" and record.form == "MOVSD_XMM (XMM, XMM)"
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].key, "MOVSD_XMM (XMM, XMM)")

    def test_add_r16_r16_form_appears(self) -> None:
        row = self.by_key.get("ADD (R16, R16)")
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.mnemonic, "ADD")
        variants = [record for record in self.records if record.db_key == "x86:add (r16, r16)"]
        self.assertEqual(len(variants), 1)

    def test_decorated_mnemonics_are_stripped(self) -> None:
        self.assertTrue(all(not record.mnemonic.startswith("{") for record in self.records))

    def test_rows_without_encoding_tags_keep_their_names(self) -> None:
        self.assertIn("ADD (R16, M16)", self.by_key)
        self.assertIn("ADD_R0 (R8l, 0)", self.by_key)
        self.assertIn("MOVSD_XMM (M64, XMM)", self.by_key)


class ArmFpfarMappingTest(unittest.TestCase):
    """FP accumulator operands map to scalar registers at ingest."""

    def test_fpfar_operands_render_as_registers(self) -> None:
        records = parse_arm_instruction_payload(json.dumps(AARCHMRS_FIXTURE))
        self.assertEqual(len(records), 3)
        half = next(r.form for r in records if r.mnemonic == "LDBFADD")
        single = next(r.form for r in records if r.form.startswith("LDFADD (S"))
        double = next(r.form for r in records if r.form.startswith("LDFADD (D"))
        self.assertTrue(half.startswith("LDBFADD (H1, H0"))
        self.assertTrue(single.startswith("LDFADD (S1, S0"))
        self.assertTrue(double.startswith("LDFADD (D1, D0"))
        for record in records:
            self.assertNotIn("<FPFAR", record.form)


class RiscvScalarInstructionsTest(unittest.TestCase):
    """Scalar add/sub/mul rows reach the bundle and the instruction parse."""

    def test_scalar_add_sub_mul_appear(self) -> None:
        tool = _vendor_tool()
        tool.git_rev = lambda _repo: "c0ffee"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inst = root / "spec" / "std" / "isa" / "inst"
            (inst / "I").mkdir(parents=True)
            (inst / "M").mkdir(parents=True)
            (inst / "I" / "add.yaml").write_text(RISCV_ADD_YAML)
            (inst / "I" / "sub.yaml").write_text(RISCV_SUB_YAML)
            (inst / "M" / "mul.yaml").write_text(RISCV_MUL_YAML)
            base = tool.collect_base_instructions(root)
        self.assertEqual(set(base), {"add", "sub", "mul"})
        # A scalar row keeps main's upstream-YAML link; no row lacks a URL.
        self.assertEqual(
            base["add"]["url"],
            f"{tool.UDB_PROJECT_URL}/blob/c0ffee/spec/std/isa/inst/I/add.yaml",
        )
        records = tool.synthesize_instruction_records(base, {})
        parsed = parse_riscv_instruction_payload(
            json.dumps({"format": "riscv-unified-db-v1", "instructions": records})
        )
        forms = {record.form for record in parsed}
        self.assertLessEqual({"add", "sub", "mul"}, forms)


if __name__ == "__main__":
    unittest.main()
