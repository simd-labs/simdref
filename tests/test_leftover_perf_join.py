"""Fixture-row tests for the uops.info perf join, from the 391-row cross-check.

The cross-check (probes-perf/crosscheck.md) found zero join-key mismatch
patterns: no ref of the 391 points at a uops.info row that carries perf.
Each test below pins one examined near-miss pattern: the join must stay on
the exact form key and must not borrow a sibling row that holds values.
"""

import unittest

from simdref.ingest_catalog import link_records
from simdref.models import InstructionRecord, IntrinsicRecord
from simdref.perf import variant_perf_summary


def _intrinsic(name, refs):
    return IntrinsicRecord(
        name=name,
        signature=f"void {name}(void)",
        description="Synthetic perf-join fixture.",
        header="immintrin.h",
        isa=["AVX512F"],
        instruction_refs=[ref | {"architecture": "x86"} for ref in refs],
    )


def _row(mnemonic, form, iform="", measured=False):
    arch_details = {}
    if measured:
        arch_details = {
            "SKL": {
                "measurement": {"TP_unrolled": "1.25", "uops": "1"},
                "latencies": [{"cycles": "4"}],
                "doc": {},
                "iaca": [],
            }
        }
    return InstructionRecord(
        mnemonic=mnemonic,
        form=form,
        summary="Synthetic row.",
        isa=["AVX512F"],
        metadata={"iform": iform} if iform else {},
        arch_details=arch_details,
    )


class LeftoverPerfJoinTests(unittest.TestCase):
    def test_exact_form_join_carries_perf(self):
        # Control: the join attaches the exact form row and its perf.
        intrinsic = _intrinsic(
            "_rdrand32_step", [{"name": "RDRAND", "form": "r32", "xed": "RDRAND_GPRv"}]
        )
        rows = [
            _row("RDRAND", "RDRAND (R16)", "RDRAND_GPRv"),
            _row("RDRAND", "RDRAND (R32)", "RDRAND_GPRv", measured=True),
            _row("RDRAND", "RDRAND (R64)", "RDRAND_GPRv"),
        ]
        link_records([intrinsic], rows)
        self.assertEqual(intrinsic.instruction_refs[0]["key"], "x86:rdrand (r32)")
        self.assertEqual(variant_perf_summary(rows[1].arch_details), ("4", "1.25"))

    def test_width_sibling_rows_stay_separate(self):
        # Near-miss family 1: R16 rows hold values, the R32 form must not see them.
        r32 = _intrinsic(
            "_rdrand32_step", [{"name": "RDRAND", "form": "r32", "xed": "RDRAND_GPRv"}]
        )
        r16 = _intrinsic(
            "_rdrand16_step", [{"name": "RDRAND", "form": "r16", "xed": "RDRAND_GPRv"}]
        )
        rows = [
            _row("RDRAND", "RDRAND (R16)", "RDRAND_GPRv", measured=True),
            _row("RDRAND", "RDRAND (R32)", "RDRAND_GPRv"),
            _row("RDRAND", "RDRAND (R64)", "RDRAND_GPRv"),
        ]
        link_records([r32, r16], rows)
        self.assertEqual(r32.instruction_refs[0]["key"], "x86:rdrand (r32)")
        self.assertEqual(r16.instruction_refs[0]["key"], "x86:rdrand (r16)")

    def test_knc_mask_stem_does_not_borrow_avx512_rows(self):
        # Near-miss family 2: KORTEST is not KORTESTQ.
        intrinsic = _intrinsic("_mm512_kortestz", [{"name": "KORTEST", "form": "k, k", "xed": ""}])
        rows = [
            _row("KORTESTQ", "KORTESTQ (K, K)", "KORTESTQ_MASKmskw_MASKmskw_AVX512", measured=True)
        ]
        link_records([intrinsic], rows)
        self.assertNotIn("key", intrinsic.instruction_refs[0])
        self.assertEqual(intrinsic.instructions, ["KORTEST (K, K)"])
        self.assertEqual(intrinsic.instruction_refs[0]["resolution"], "unresolved")

    def test_knc_signed_name_does_not_borrow_unsigned_row(self):
        # Near-miss family 3: VPMULHD is not VPMULHW.
        intrinsic = _intrinsic(
            "_mm512_mulhd_epi32", [{"name": "VPMULHD", "form": "zmm, zmm, zmm", "xed": ""}]
        )
        rows = [_row("VPMULHW", "VPMULHW (ZMM, ZMM, ZMM)", "VPMULHW_ZMM_ZMM_ZMM", measured=True)]
        link_records([intrinsic], rows)
        self.assertNotIn("key", intrinsic.instruction_refs[0])
        self.assertEqual(intrinsic.instructions, ["VPMULHD (ZMM, ZMM, ZMM)"])

    def test_xed_path_beats_mask_spelling(self):
        # Near-miss family 4: `zmm {k}` vs `ZMM, K` must not lose the row when xed matches.
        intrinsic = _intrinsic(
            "_mm512_mask_addn_ps",
            [
                {
                    "name": "VADDNPS",
                    "form": "zmm {k}, zmm, zmm",
                    "xed": "VADDNPS_ZMMf32_MASKmskw_ZMMf32_AVX512",
                }
            ],
        )
        rows = [
            _row(
                "VADDNPS",
                "VADDNPS (ZMM, K, ZMM, ZMM)",
                "VADDNPS_ZMMf32_MASKmskw_ZMMf32_AVX512",
                measured=True,
            )
        ]
        link_records([intrinsic], rows)
        self.assertEqual(intrinsic.instruction_refs[0]["resolution"], "xed")
        self.assertEqual(intrinsic.instruction_refs[0]["key"], "x86:vaddnps (zmm, k, zmm, zmm)")

    def test_unmeasured_row_joins_but_yields_no_perf(self):
        # The 371 NOT-IN-UOPS rows: the join resolves, the row holds no perf.
        intrinsic = _intrinsic(
            "_aadd_i32", [{"name": "AADD", "form": "m32, r32", "xed": "AADD_MEM32_GPR32"}]
        )
        rows = [_row("AADD", "AADD (M32, R32)", "AADD_MEM32_GPR32")]
        link_records([intrinsic], rows)
        self.assertEqual(intrinsic.instruction_refs[0]["resolution"], "xed")
        self.assertEqual(variant_perf_summary(rows[0].arch_details), ("-", "-"))


if __name__ == "__main__":
    unittest.main()
