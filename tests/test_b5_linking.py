"""B5/3a and B5/3b: Arm instruction pick and template placeholders.

3a: a NEON intrinsic (vaddq_f32, vendor ref "FADD, operands
"Vd.4S,Vn.4S,Vm.4S") must link to the NEON vector form, not to an SVE
form that happens to share the FADD mnemonic.

3b: AARCHMRS rule tokens such as T_FMUL_HSD (float arrangement) and
T_OPTION__4 (AdvSIMD arrangement option) render as the concrete "4S"
instance like the existing T rule; tokens with no known instance drop
the operand text so no raw "<...>" template reaches a display.

Fixture records: inline vaddq_f32 plus two FADD InstructionRecords for
3a; tests/fixtures/b5/arm_fadd_aarchmrs.json holds three AARCHMRS FADD
nodes (SVE, AdvSIMD, SME) for 3b.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from simdref.ingest_catalog import link_records
from simdref.models import InstructionRecord, IntrinsicRecord

AARCHMRS = Path(__file__).parent / "fixtures" / "b5" / "arm_fadd_aarchmrs.json"


class ArmLinkPickTests(unittest.TestCase):
    def test_neon_intrinsic_links_to_the_neon_form_not_sve(self):
        intrinsic = IntrinsicRecord(
            name="vaddq_f32",
            signature="float32x4_t vaddq_f32(float32x4_t a, float32x4_t b)",
            description="Floating-point add.",
            header="arm_neon.h",
            architecture="arm",
            isa=["NEON"],
            instruction_refs=[
                {
                    "name": "FADD",
                    "form": "Vd.4S,Vn.4S,Vm.4S",
                    "architecture": "arm",
                }
            ],
        )
        sve = InstructionRecord(
            mnemonic="FADD",
            form="FADD (Z0.4S, Z1.4S, Z2.4S)",
            summary="Floating-point add.",
            architecture="arm",
            isa=["SVE", "NEON"],
        )
        neon = InstructionRecord(
            mnemonic="FADD",
            form="FADD (V0.4S, V1.4S, V2.4S)",
            summary="Floating-point add.",
            architecture="arm",
            isa=["NEON"],
        )

        link_records([intrinsic], [sve, neon])

        self.assertEqual(len(intrinsic.instruction_refs), 1)
        self.assertEqual(intrinsic.instruction_refs[0]["key"], neon.db_key)


class PlaceholderTests(unittest.TestCase):
    def test_arrangement_template_tokens_expand_to_4s(self):
        from simdref.arm_instructions import parse_arm_instruction_payload

        records = parse_arm_instruction_payload(AARCHMRS.read_text())
        forms = {record.form for record in records}
        self.assertIn("FADD (Z0.4S, Z1.4S, Z2.4S)", forms)
        self.assertIn("FADD (V0.4S, V1.4S, V2.4S)", forms)

    def test_no_template_placeholder_token_reaches_a_form(self):
        from simdref.arm_instructions import parse_arm_instruction_payload

        records = parse_arm_instruction_payload(AARCHMRS.read_text())
        for record in records:
            self.assertNotIn("<", record.form)
            self.assertNotIn(">", record.form)

    def test_unresolvable_operand_tokens_drop_the_operand_text(self):
        from simdref.arm_instructions import parse_arm_instruction_payload

        records = parse_arm_instruction_payload(AARCHMRS.read_text())
        sme = [record for record in records if record.isa == ["SME"]]
        self.assertEqual(len(sme), 1)
        self.assertEqual(sme[0].form, "FADD")


if __name__ == "__main__":
    unittest.main()
