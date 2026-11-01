"""AARCHMRS ISA labels: SVE and SME groups must not pick up NEON."""

from __future__ import annotations

import unittest

from simdref.arm_instructions import _infer_aarchmrs_isa


class InferIsaTests(unittest.TestCase):
    def test_sve_fp_group_is_sve_only(self):
        self.assertEqual(
            _infer_aarchmrs_isa(["A64", "A64", "sve", "sve_fp_pred", "sve_fp_2op_p_zds"]),
            ["SVE"],
        )

    def test_sme_sve_fp_group_keeps_sme_sve(self):
        self.assertEqual(
            _infer_aarchmrs_isa(
                ["A64", "A64", "sme", "mortlach_multi_sve_4", "mortlach_multi2_narrow_fp_cvrt"]
            ),
            ["SME", "SVE"],
        )

    def test_simd_dp_group_stays_neon(self):
        self.assertEqual(_infer_aarchmrs_isa(["A64", "A64", "simd_dp", "asimdsame"]), ["NEON"])

    def test_asimd_group_stays_neon(self):
        self.assertEqual(_infer_aarchmrs_isa(["A64", "A64", "asimd", "asimddiff"]), ["NEON"])


if __name__ == "__main__":
    unittest.main()
