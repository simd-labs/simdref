"""Tests for the detail metadata rows the web shows.

Every test here fails on e47f97e, whose intrinsic detail omits the
Supported, Section, Arg Prep, Result and Instr Cat rows.
"""

from __future__ import annotations

import unittest

from simdref.models import IntrinsicRecord, InstructionRecord

try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - Textual is a runtime dep
    HAS_TEXTUAL = False


def _intrinsic() -> IntrinsicRecord:
    return IntrinsicRecord(
        name="vaddq_f32",
        signature="float32x4_t vaddq_f32(float32x4_t a, float32x4_t b)",
        description="Add.",
        header="arm_neon.h",
        url="",
        architecture="arm",
        isa=["NEON"],
        metadata={
            "classification": "unclassified",
            "supported_architectures": "v7/A32/A64",
            "classification_path": "Vector arithmetic / Add / Addition",
            "argument_preparation": "a -> register Vn.4S;b -> register Vm.4S",
            "result": "Vd.4S -> result",
        },
    )


def _linked() -> InstructionRecord:
    return InstructionRecord(
        mnemonic="VADD",
        form="VADD F0.4S, F1.4S, F2.4S",
        summary="Floating-point add.",
        architecture="arm",
        isa=["NEON"],
        metadata={"category": "Arithmetic"},
    )


def _detail_text(payload: list[tuple]) -> str:
    from rich.console import Console

    console = Console(record=True, width=200)
    for entry in payload:
        if entry[0] in {"static", "static_markup"}:
            console.print(entry[1])
    return console.export_text()


@unittest.skipUnless(HAS_TEXTUAL, "textual required")
class DetailMetadataTests(unittest.TestCase):
    def test_intrinsic_detail_shows_the_web_metadata_rows(self):
        from simdref.tui import SimdrefApp

        app = SimdrefApp()
        app._find_linked_instruction = lambda intrinsic, conn=None: _linked()
        text = _detail_text(app._build_intrinsic_payload(_intrinsic()))
        for label, value in (
            ("Supported", "v7/A32/A64"),
            ("Section", "Vector arithmetic / Add / Addition"),
            ("Arg Prep", "a -> register Vn.4S;b -> register Vm.4S"),
            ("Instr Cat", "Arithmetic"),
        ):
            self.assertIn(label, text)
            self.assertIn(value, text)
        # The Result metadata row is gone; doc blocks carry Returns.
        self.assertNotIn("Result", text)

    def test_rows_stay_out_when_the_record_has_no_value(self):
        from simdref.tui import SimdrefApp

        app = SimdrefApp()
        app._find_linked_instruction = lambda intrinsic, conn=None: None
        text = _detail_text(app._build_intrinsic_payload(_intrinsic()))
        # The linked instruction carries the only category here.
        self.assertNotIn("Instr Cat", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
