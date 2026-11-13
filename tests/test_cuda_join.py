"""Tests for the CUDA doc join over synthetic rows and a synthetic call graph."""

from __future__ import annotations

from simdref.cuda import apply_cuda_docs
from simdref.models import IntrinsicRecord


def _row(name: str) -> IntrinsicRecord:
    return IntrinsicRecord(
        name=name,
        signature=f"int {name}(int)",
        description="CUDA device intrinsic from __clang_cuda_math.h.",
        header="__clang_cuda_math.h",
        url=f"https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#{name}",
        architecture="cuda",
        isa=["CUDA"],
        source="clang-cuda-headers",
    )


def _docs() -> dict:
    return {
        "math": {
            "__zyx_sin_v2": {
                "anchor": "zyx_target_sin_v2",
                "page": "https://example.invalid/zyx_math.html",
                "signature": "__device__ float __zyx_sin_v2(float x)",
                "description": "Compute the zyx sine. It works in zyx units.",
                "return_value": "Returns the zyx sine.",
                "parameters": "x: the zyx input.",
                "accuracy": "Accuracy data: below 2 ULP.",
            },
        },
        "guide": {
            "__zyx_vote_v1": {
                "anchor": "zyx-vote",
                "page": "https://example.invalid/zyx_guide.html",
                "signature": "unsigned int __zyx_vote_v1(unsigned mask);",
                "description": "Return the zyx vote of the warp.",
                "return_value": "",
                "parameters": "",
                "accuracy": "",
            },
        },
        "guide_archive": {},
        "libdevice": {},
        "ulp": {"__zyx_sin_v2": {"accuracy": "1 ULP", "caption": "Zyx accuracy"}},
    }


def test_exact_math_entry_fills_every_field() -> None:
    row = _row("__zyx_sin_v2")
    apply_cuda_docs([row], {}, _docs())
    assert row.url == "https://example.invalid/zyx_math.html#zyx_target_sin_v2"
    assert row.signature == "__device__ float __zyx_sin_v2(float x)"
    assert row.description == "Compute the zyx sine."
    assert row.doc_sections["Description"] == "Compute the zyx sine. It works in zyx units."
    assert row.doc_sections["Return"] == "Returns the zyx sine."
    assert row.doc_sections["Parameters"] == "x: the zyx input."
    assert row.doc_sections["Accuracy"] == "1 ULP"
    assert row.metadata["doc_source"] == "cuda-math-api"
    # C.5: the Result metadata row and the accuracy notes duplicate are
    # dropped. The C.1 entry carries the source text once.
    assert "result" not in row.metadata
    assert row.notes == []
    assert [entry["source"] for entry in row.doc_sources] == ["cuda-math-api"]
    assert row.doc_sources[0]["url"] == "https://example.invalid/zyx_math.html#zyx_target_sin_v2"


def test_math_entry_wins_over_guide_entry() -> None:
    docs = _docs()
    docs["guide"]["__zyx_sin_v2"] = docs["guide"]["__zyx_vote_v1"]
    row = _row("__zyx_sin_v2")
    apply_cuda_docs([row], {}, docs)
    assert row.metadata["doc_source"] == "cuda-math-api"
    assert row.url.endswith("#zyx_target_sin_v2")


def test_caller_fallback_stores_doc_via() -> None:
    row = _row("__zyx_helper_v1")
    calls = {"__zyx_vote_v1": ["__zyx_helper_v1"]}
    apply_cuda_docs([row], calls, _docs())
    assert row.metadata["doc_via"] == "__zyx_vote_v1"
    assert row.metadata["doc_source"] == "cuda-guide"
    assert row.url == "https://example.invalid/zyx_guide.html#zyx-vote"
    assert row.description == "Return the zyx vote of the warp."


def test_name_containment_maps_helper_onto_api_name() -> None:
    docs = _docs()
    docs["guide"]["zyxAdd"] = {
        "anchor": "zyx-add",
        "page": "https://example.invalid/zyx_guide.html",
        "signature": "int zyxAdd(int *p, int v);",
        "description": "Add a zyx value.",
        "return_value": "",
        "parameters": "",
        "accuracy": "",
    }
    row = _row("__iZyxAdd")
    apply_cuda_docs([row], {}, docs)
    assert row.metadata["doc_via"] == "zyxAdd"
    assert row.url == "https://example.invalid/zyx_guide.html#zyx-add"
    # The sibling's declaration is not a signature of this row: the row
    # keeps its own header signature (atomicAdd_block displayed
    # atomicAdd's declaration).
    assert row.signature == "int __iZyxAdd(int)"


def test_containment_uses_reachable_builtins() -> None:
    docs = _docs()
    docs["guide"]["__zyx_all_v1"] = dict(
        docs["guide"]["__zyx_vote_v1"],
        signature="unsigned int __zyx_all_v1(unsigned mask);",
    )
    row = _row("__zyx32_all_v1")
    calls = {"__zyx32_all_v1": ["__nvvm_zyx_all_v1_i32p"]}
    apply_cuda_docs([row], calls, docs)
    assert row.metadata["doc_via"] == "__zyx_all_v1"


def test_summary_feeds_the_row_description() -> None:
    docs = _docs()
    docs["math"]["__zyx_sin_v2"]["summary"] = "Compute the zyx sine of the input value."
    row = _row("__zyx_sin_v2")
    apply_cuda_docs([row], {}, docs)
    assert row.description == "Compute the zyx sine of the input value."


def test_doc_math_keeps_the_original_accuracy_tex() -> None:
    docs = _docs()
    raw = "\\(2^{-21.41}\\) abs error for \\(x \\in [-\\pi, \\pi]\\)"
    docs["ulp"]["__zyx_sin_v2"]["accuracy"] = raw
    row = _row("__zyx_sin_v2")
    apply_cuda_docs([row], {}, docs)
    assert row.doc_sections["Accuracy"] == "2⁻²¹·⁴¹ abs error for x ∈ [−π, π]"
    assert row.metadata["doc_math"] == raw


def test_no_doc_math_without_tex_in_accuracy() -> None:
    row = _row("__zyx_sin_v2")
    apply_cuda_docs([row], {}, _docs())
    assert "doc_math" not in row.metadata


def test_no_entry_keeps_no_url_and_filler() -> None:
    row = _row("__zyx_undocumented_v1")
    apply_cuda_docs([row], {}, _docs())
    assert row.url == ""
    assert row.description == "CUDA device intrinsic from __clang_cuda_math.h."
    assert row.doc_sections == {}
    assert "doc_source" not in row.metadata


def test_ulp_table_header_is_one_row_of_cells() -> None:
    # The ULP entry keeps the header as one flat list of cell strings.
    # The table block carries rows of cells, so the header becomes one
    # row of cells: iterating the flat strings gave one column per
    # letter in the web view of the __sinf accuracy table.
    docs = _docs()
    docs["ulp"]["__zyx_sin_v2"] = {
        "accuracy": "2 ULP",
        "caption": "Zyx accuracy",
        "math": [],
        "title": "Zyx appendix",
        "page": "https://example.invalid/zyx_ulp.html",
        "header": ["Function", "Meaning", "Maximum ULP Error"],
        "rows": [["__zyx_sin_v2(x)", "zyx sine", "2 ULP"]],
    }
    row = _row("__zyx_sin_v2")
    apply_cuda_docs([row], {}, docs)
    table = next(
        block
        for entry in row.doc_sources
        for section in entry["sections"]
        for block in section["blocks"]
        if block["kind"] == "table"
    )
    assert table["header"] == [["Function", "Meaning", "Maximum ULP Error"]]
    assert table["rows"] == [["__zyx_sin_v2(x)", "zyx sine", "2 ULP"]]


def test_summary_is_the_first_paragraph_block_never_list_text() -> None:
    # The legacy description string merges the lead-in paragraph with
    # the list items of the section (the atomicAdd summary read like a
    # truncated step list). The summary comes from the first paragraph
    # block only: list text never enters it.
    docs = _docs()
    docs["guide"]["__zyx_add_v1"] = {
        "anchor": "zyx-add",
        "page": "https://example.invalid/zyx_guide.html",
        "signature": "int __zyx_add_v1(int *p, int v);",
        "description": ("Does the add in one pass: Reads the old value. Adds the value."),
        "return_value": "",
        "parameters": "",
        "accuracy": "",
        "sections": [
            {
                "heading": "",
                "shared": False,
                "blocks": [
                    {"kind": "paragraph", "text": "Does the add in one pass:"},
                    {
                        "kind": "list",
                        "ordered": True,
                        "items": [{"text": "Reads the old value."}, {"text": "Adds the value."}],
                    },
                ],
            }
        ],
    }
    row = _row("__zyx_add_v1")
    apply_cuda_docs([row], {}, docs)
    assert row.description == "Does the add in one pass:"
    # The full description section keeps the legacy text unchanged.
    assert row.doc_sections["Description"].startswith("Does the add in one pass: Reads")
