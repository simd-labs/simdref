"""Tests for the TeX to text converter on synthetic math strings."""

from __future__ import annotations

from simdref.cuda import apply_cuda_docs
from simdref.cuda_docs import tex_to_text
from simdref.models import IntrinsicRecord

RAW = "\\(2^{-21.41}\\) abs error for \\(x \\in [-\\pi, \\pi]\\)"
READABLE = "2⁻²¹·⁴¹ abs error for x ∈ [−π, π]"


def test_accuracy_formula_becomes_readable_text() -> None:
    assert tex_to_text(RAW) == READABLE


def test_scripts_map_digits_and_signs_only() -> None:
    assert tex_to_text("\\(x^2\\)") == "x²"
    assert tex_to_text("\\(x_{10}\\)") == "x₁₀"
    assert tex_to_text("\\(2^{n}\\)") == "2^(n)"
    assert tex_to_text("\\(a_{i+1}\\)") == "a_(i+1)"


def test_operators_and_greek_map_to_unicode() -> None:
    raw = "\\(2 + \\lfloor |2.97 \\cdot x| \\rfloor\\)"
    assert tex_to_text(raw) == "2 + ⌊ |2.97 · x| ⌋"
    assert tex_to_text("\\(x \\notin [-10, -2]\\)") == "x ∉ [−10, −2]"
    assert tex_to_text("\\(\\Gamma\\)") == "Γ"


def test_unknown_commands_stay_as_source_and_are_counted() -> None:
    unknown: list[str] = []
    assert tex_to_text("\\(a \\qquad b\\)", unknown) == "a \\qquad b"
    assert unknown == ["\\qquad"]


def test_plain_text_passes_through_unchanged() -> None:
    assert tex_to_text("2 ULP, otherwise") == "2 ULP, otherwise"
    assert tex_to_text("single-precision") == "single-precision"
    assert tex_to_text("Derived from __log2f(y * x)") == "Derived from __log2f(y * x)"


def test_join_converts_accuracy_and_keeps_the_original_tex() -> None:
    row = IntrinsicRecord(
        name="__zyx_sinf_v2",
        signature="float __zyx_sinf_v2(float)",
        description="",
        header="__clang_cuda_math.h",
        architecture="cuda",
        isa=["CUDA"],
        source="clang-cuda-headers",
    )
    docs = {
        "math": {
            "__zyx_sinf_v2": {
                "anchor": "zyx_sinf_v2",
                "page": "https://example.invalid/zyx_math.html",
                "signature": "__device__ float __zyx_sinf_v2(float x)",
                "description": "Compute the zyx sine.",
                "return_value": "Returns the zyx sine.",
                "parameters": "x: the zyx input.",
                "accuracy": "",
            },
        },
        "guide": {},
        "guide_archive": {},
        "libdevice": {},
        "ulp": {"__zyx_sinf_v2": {"accuracy": RAW, "caption": "Zyx accuracy"}},
    }
    apply_cuda_docs([row], {}, docs)
    assert row.doc_sections["Accuracy"] == READABLE
    assert row.metadata["doc_math"] == RAW
