"""Tests for CUDA doc parsing: math API pages, libdevice, ULP tables,
and the build-time fetch recipe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simdref.cuda_docs import (
    DOC_BUNDLE_FORMAT,
    build_docs,
    parse_docs_bundle,
    parse_libdevice_index,
    parse_libdevice_page,
    parse_math_api_page,
    parse_ulp_tables,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

MATH_INDEX_URL = "https://docs.nvidia.com/cuda/cuda-math-api/"
FAKE1_URL = "https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FAKE1.html"
ABS_URL = "https://example.invalid/cuda_math_api/group__CUDA__MATH__ABS.html"
GUIDE_URL = (
    "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html"
)
ARCHIVE_URL = "https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html"
ULP_URL = (
    "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html"
)
LIBDEVICE_INDEX_URL = (
    "https://docs.nvidia.com/cuda/libdevice-users-guide/libdevice_users_guide/functions.html"
)
LIBDEVICE_FREE_URL = (
    "https://docs.nvidia.com/cuda/libdevice-users-guide/libdevice_users_guide/"
    "function_zyx_decl_3.html"
)


def _load(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


class TestMathApiParse:
    """One entry per dl cpp function of the synthetic page."""

    def test_page_yields_both_functions(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_page.html"))
        assert sorted(out) == ["__zyx_cos_q7", "__zyx_sin_v2"]

    def test_sin_entry_fields(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_page.html"))
        entry = out["__zyx_sin_v2"]
        assert {k: v for k, v in entry.items() if k != "sections"} == {
            "anchor": "zyx_target_sin_v2",
            "signature": "__device__ float __zyx_sin_v2(float x)",
            "summary": "Compute the zyx sine of the input value.",
            "description": "The input is a value in zyx units.",
            "return_value": "Returns the zyx sine.",
            "parameters": "The zyx input value.",
            "accuracy": "Accuracy data: the error stays below 2 ULP.",
            "math": [],
            "heading": "__zyx_sin_v2",
            "title": "",
        }
        assert entry["sections"] == [
            {
                "heading": "",
                "shared": False,
                "blocks": [
                    {
                        "kind": "paragraph",
                        "owner": "__zyx_sin_v2",
                        "text": "Compute the zyx sine of the input value.",
                    },
                    {
                        "kind": "paragraph",
                        "owner": "__zyx_sin_v2",
                        "text": "The input is a value in zyx units.",
                    },
                    {
                        "kind": "note",
                        "note": "note",
                        "owner": "__zyx_sin_v2",
                        "text": "Accuracy data: the error stays below 2 ULP.",
                    },
                    {
                        "kind": "note",
                        "note": "note",
                        "owner": "__zyx_sin_v2",
                        "text": "Inputs in the zyx range are flushed to zero.",
                    },
                    {
                        "kind": "note",
                        "note": "seealso",
                        "owner": "__zyx_sin_v2",
                        "text": "The zyx phase table lists the domain.",
                    },
                    {
                        "kind": "deflist",
                        "owner": "__zyx_sin_v2",
                        "items": [
                            {"name": "Returns", "text": "Returns the zyx sine."},
                            {"name": "Parameters", "text": "The zyx input value."},
                        ],
                    },
                ],
            }
        ]

    def test_anchor_falls_back_to_dt_id(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_page.html"))
        assert out["__zyx_cos_q7"]["anchor"] == "_CPPv49__zyx_cos_q7f"

    def test_parameters_empty_without_field(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_page.html"))
        assert out["__zyx_cos_q7"]["parameters"] == ""

    def test_accuracy_ignores_plain_note(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_page.html"))
        accuracy = out["__zyx_sin_v2"]["accuracy"]
        assert "flushed" not in accuracy

    def test_accuracy_from_seealso(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_page.html"))
        assert out["__zyx_cos_q7"]["accuracy"] == ("The zyx cosine table lists 3 ULP error.")

    def test_signature_is_cleaned(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_page.html"))
        sig = out["__zyx_sin_v2"]["signature"]
        assert sig == " ".join(sig.split())


class TestBriefAndDetail:
    """The dd brief paragraph never enters the description."""

    def test_brief_moves_to_summary(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_tex_page.html"))
        entry = out["__zyx_tan_v5"]
        assert entry["summary"] == (
            "Calculate the fast approximation to the zyx tangent of the input argument."
        )
        assert entry["description"] == (
            "Calculate the fast approximation to the zyx tangent of the input argument "
            "x, measured in zyx radians."
        )

    def test_description_does_not_start_with_the_summary(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_tex_page.html"))
        entry = out["__zyx_tan_v5"]
        assert not entry["description"].startswith(entry["summary"])
        assert entry["summary"] not in entry["description"]

    def test_brief_without_detail_stays_in_description(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_tex_page.html"))
        entry = out["__zyx_abs_v6"]
        assert entry["summary"] == "Compute the zyx absolute value of the input."
        assert entry["description"] == "Compute the zyx absolute value of the input."


class TestTexMathMarking:
    """A math span keeps its TeX source and marks the field it sits in."""

    def test_math_fields_are_marked(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_tex_page.html"))
        assert out["__zyx_tan_v5"]["math"] == ["accuracy", "parameters", "return_value"]

    def test_tex_source_is_kept_verbatim(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_tex_page.html"))
        entry = out["__zyx_tan_v5"]
        assert entry["accuracy"] == (
            "Accuracy data: \\(2^{-21.41}\\) abs error for \\(x \\in [-\\pi, \\pi]\\)."
        )
        assert entry["return_value"] == "Returns \\(\\tan(x)\\)."
        assert entry["parameters"] == "The input \\(x\\)."

    def test_plain_entry_has_no_math(self) -> None:
        out = parse_math_api_page(_load("cuda_docs_math_tex_page.html"))
        assert out["__zyx_abs_v6"]["math"] == []


class TestUlpMathMarking:
    """The ULP accuracy cell marks math the same way."""

    def test_accuracy_math_is_marked_and_kept(self) -> None:
        out = parse_ulp_tables(_load("cuda_docs_ulp_math_table.html"))
        entry = out["__zyx_tan_v5"]
        assert entry["accuracy"] == (
            "▪ \\(2^{-21.41}\\) abs error for \\(x \\in [-\\pi, \\pi]\\) ▪ larger otherwise"
        )
        assert entry["math"] == ["accuracy"]

    def test_plain_accuracy_has_no_math(self) -> None:
        out = parse_ulp_tables(_load("cuda_docs_ulp_math_table.html"))
        assert out["__zyx_abs_v6"]["accuracy"] == "0 ULP"
        assert out["__zyx_abs_v6"]["math"] == []

    def test_table_cells_read_math_as_unicode(self) -> None:
        # The table block carries the readable text of a math cell, the
        # same text inline runs produce for a paragraph. The accuracy
        # field itself keeps the raw TeX for the math block.
        out = parse_ulp_tables(_load("cuda_docs_ulp_math_table.html"))
        entry = out["__zyx_tan_v5"]
        row = entry["rows"][0]
        assert "\\(" not in "".join(row)
        assert row[0] == "__zyx_tan_v5(x)"
        assert row[1] == "\\tan(x)"
        assert row[2] == "▪ 2⁻²¹·⁴¹ abs error for x ∈ [−π, π] ▪ larger otherwise"


class TestLibdeviceParse:
    """libdevice pages share the Sphinx shape and the output contract."""

    def test_same_contract(self) -> None:
        out = parse_libdevice_page(_load("cuda_docs_libdevice_page.html"))
        entry = out["__zyx_free_v7"]
        assert {k: v for k, v in entry.items() if k != "sections"} == {
            "anchor": "zyx_target_free_v7",
            "signature": "__device__ float @__zyx_free_v7(float x)",
            "summary": "Free the zyx value of the input.",
            "description": "Free the zyx value of the input.",
            "return_value": "Returns the freed zyx value.",
            "parameters": "",
            "accuracy": "",
            "math": [],
            "heading": "__zyx_free_v7",
            "title": "",
        }
        assert entry["sections"] == [
            {
                "heading": "",
                "shared": False,
                "blocks": [
                    {
                        "kind": "paragraph",
                        "owner": "__zyx_free_v7",
                        "text": "Free the zyx value of the input.",
                    },
                    {
                        "kind": "deflist",
                        "owner": "__zyx_free_v7",
                        "items": [{"name": "Returns", "text": "Returns the freed zyx value."}],
                    },
                ],
            }
        ]

    def test_at_prefix_is_not_part_of_the_name(self) -> None:
        out = parse_libdevice_page(_load("cuda_docs_libdevice_page.html"))
        assert "@__zyx_free_v7" not in out


class TestUlpTables:
    """Three data rows in one captioned table."""

    def test_three_rows(self) -> None:
        out = parse_ulp_tables(_load("cuda_docs_ulp_table.html"))
        assert out["__zyx_sin_v2"]["accuracy"] == "2 ULP"
        assert out["__zyx_cos_q7"]["accuracy"] == "3 ULP"
        assert out["__zyx_rsqrt_v1"]["accuracy"] == "0 ULP"

    def test_caption_is_kept(self) -> None:
        out = parse_ulp_tables(_load("cuda_docs_ulp_table.html"))
        caption = "Table 9 Zyx Functions Mapping and Accuracy (Maximum ULP)"
        assert out["__zyx_sin_v2"]["caption"] == caption

    def test_every_identifier_token_of_the_function_cell(self) -> None:
        out = parse_ulp_tables(_load("cuda_docs_ulp_table.html"))
        assert out["x"]["accuracy"] == "2 ULP"
        assert out["y"]["accuracy"] == "3 ULP"

    def test_table_without_caption_is_skipped(self) -> None:
        html = "<html><body><table><tr><td>__zyx_no_cap_v1(x)</td></tr></table></body></html>"
        assert parse_ulp_tables(html) == {}


class TestLibdeviceIndex:
    """Name to href map from the function list."""

    def test_function_list(self) -> None:
        out = parse_libdevice_index(_load("cuda_docs_libdevice_index.html"))
        assert out == {
            "__zyx_abs_v2": "function_zyx_decl_1.html",
            "__zyx_sin_v2": "function_zyx_decl_2.html",
            "__zyx_free_v7": "function_zyx_decl_3.html",
            "__zyx_ballot_v3": "function_zyx_decl_4.html",
        }

    def test_other_links_are_skipped(self) -> None:
        out = parse_libdevice_index(_load("cuda_docs_libdevice_index.html"))
        assert "Usage" not in out


class TestBuildDocs:
    """The fetch recipe with a fake fetch callable."""

    def _fetch(self):
        calls = []
        table = {
            MATH_INDEX_URL: _load("cuda_docs_math_index.html"),
            FAKE1_URL: _load("cuda_docs_math_page.html"),
            ABS_URL: _load("cuda_docs_math_page.html"),
            GUIDE_URL: _load("cuda_docs_guide_page.html"),
            ARCHIVE_URL: "<html><body><p>zyx archive page</p></body></html>",
            ULP_URL: _load("cuda_docs_ulp_table.html"),
            LIBDEVICE_INDEX_URL: _load("cuda_docs_libdevice_index.html"),
            LIBDEVICE_FREE_URL: _load("cuda_docs_libdevice_page.html"),
        }

        def fetch(url: str) -> str:
            calls.append(url)
            assert url in table, "unexpected fetch " + url
            return table[url]

        return fetch, calls

    def _bundle(self):
        fetch, calls = self._fetch()
        names = {
            "__zyx_sin_v2",
            "__zyx_free_v7",
            "__zyx_ballot_v3",
            "__zyx_missing_v9",
        }
        return build_docs(fetch, names), calls

    def test_bundle_shape(self) -> None:
        bundle, _ = self._bundle()
        assert bundle["format"] == DOC_BUNDLE_FORMAT == "cuda-docs-v1"
        assert [(p["kind"], p["url"]) for p in bundle["pages"]] == [
            ("math", FAKE1_URL),
            ("math", ABS_URL),
            ("guide", GUIDE_URL),
            ("guide-archive", ARCHIVE_URL),
            ("ulp", ULP_URL),
            ("libdevice-index", LIBDEVICE_INDEX_URL),
            ("libdevice", LIBDEVICE_FREE_URL),
        ]

    def test_metadata_counts(self) -> None:
        bundle, _ = self._bundle()
        assert bundle["metadata"] == {
            "names": 4,
            "pages": 7,
            "math_pages": 2,
            "libdevice_pages": 1,
            "matched_math": 1,
            "matched_guide": 1,
            "failed": [],
        }

    def test_dead_page_is_recorded_and_skipped(self) -> None:
        fetch, _ = self._fetch()

        def flaky(url: str) -> str:
            if url == ULP_URL:
                raise RuntimeError("zyx 404")
            return fetch(url)

        bundle = build_docs(flaky, {"__zyx_sin_v2"})
        assert bundle["metadata"]["failed"] == [{"url": ULP_URL, "error": "zyx 404"}]
        assert all(page["url"] != ULP_URL for page in bundle["pages"])

    def test_sha256_is_the_html_digest(self) -> None:
        bundle, _ = self._bundle()
        for page in bundle["pages"]:
            digest = hashlib.sha256(page["html"].encode("utf-8")).hexdigest()
            assert page["sha256"] == digest

    def test_libdevice_fetch_only_for_unmatched_name(self) -> None:
        _, calls = self._bundle()
        assert calls.count(LIBDEVICE_FREE_URL) == 1
        for url in calls:
            assert "function_zyx_decl_1" not in url
            assert "function_zyx_decl_2" not in url
            assert "function_zyx_decl_4" not in url

    def test_math_index_hrefs_are_resolved_and_deduped(self) -> None:
        _, calls = self._bundle()
        # The index links FAKE1 twice, once with a fragment.
        assert calls.count(FAKE1_URL) == 1
        assert calls.count(ABS_URL) == 1


class TestBundleRoundTrip:
    """parse_docs_bundle reads back what build_docs writes."""

    def _parsed(self):
        fetch, _ = TestBuildDocs()._fetch()
        names = {"__zyx_sin_v2", "__zyx_free_v7", "__zyx_ballot_v3"}
        bundle = build_docs(fetch, names)
        return parse_docs_bundle(json.dumps(bundle))

    def test_math_entries(self) -> None:
        out = self._parsed()
        entry = out["math"]["__zyx_sin_v2"]
        direct = parse_math_api_page(_load("cuda_docs_math_page.html"))["__zyx_sin_v2"]
        assert {k: v for k, v in entry.items() if k != "sections"} == {
            **{k: v for k, v in direct.items() if k != "sections"},
            "page": FAKE1_URL,
        }
        # The round trip keeps the C.1 blocks byte-for-byte.
        assert entry["sections"] == direct["sections"]

    def test_guide_entries(self) -> None:
        out = self._parsed()
        assert out["guide"]["__zyx_dup_v1"]["anchor"] == "zyx-warp-vote"
        assert out["guide_archive"] == {}

    def test_libdevice_entries(self) -> None:
        out = self._parsed()
        assert out["libdevice"]["__zyx_free_v7"]["anchor"] == "zyx_target_free_v7"

    def test_ulp_entries(self) -> None:
        out = self._parsed()
        assert out["ulp"]["__zyx_rsqrt_v1"] == {
            "accuracy": "0 ULP",
            "caption": "Table 9 Zyx Functions Mapping and Accuracy (Maximum ULP)",
            "math": [],
            "page": ULP_URL,
            "title": "",
            "header": ["Function", "Meaning", "Maximum ULP Error"],
            "rows": [["__zyx_rsqrt_v1(x)", "zyx inverse root", "0 ULP"]],
        }

    def test_libdevice_index_entries(self) -> None:
        out = self._parsed()
        assert out["libdevice_index"]["__zyx_abs_v2"] == "function_zyx_decl_1.html"

    def test_urls_and_sha256_per_kind(self) -> None:
        out = self._parsed()
        assert out["urls"] == {
            "math": FAKE1_URL,
            "guide": GUIDE_URL,
            "guide_archive": ARCHIVE_URL,
            "ulp": ULP_URL,
            "libdevice_index": LIBDEVICE_INDEX_URL,
            "libdevice": LIBDEVICE_FREE_URL,
        }
        ulp_html = _load("cuda_docs_ulp_table.html")
        assert out["sha256"]["ulp"] == hashlib.sha256(ulp_html.encode("utf-8")).hexdigest()

    def test_wrong_format_raises(self) -> None:
        try:
            parse_docs_bundle('{"format": "cuda-docs-v0", "pages": []}')
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    def test_unknown_kind_raises(self) -> None:
        text = json.dumps(
            {
                "format": DOC_BUNDLE_FORMAT,
                "pages": [{"kind": "zyx", "url": "u", "sha256": "s", "html": ""}],
            }
        )
        try:
            parse_docs_bundle(text)
        except ValueError:
            return
        raise AssertionError("expected ValueError")
