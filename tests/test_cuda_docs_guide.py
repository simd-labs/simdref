"""Tests for the Programming Guide parser over a synthetic page."""

from __future__ import annotations

from pathlib import Path

from simdref.cuda_docs import parse_guide_page

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _parse() -> dict[str, dict[str, str]]:
    html = (FIXTURES_DIR / "cuda_docs_guide_page.html").read_text()
    return parse_guide_page(html)


class TestGuideParse:
    """Names come from pre code blocks of sections."""

    def test_names_enter_from_blocks_and_prose(self) -> None:
        assert sorted(_parse()) == [
            "__zyx_ballot_v3",
            "__zyx_count_v4",
            "__zyx_dup_v1",
            "__zyx_late_v8",
            "__zyx_prose_only",
        ]

    def test_prose_name_enters_without_signature(self) -> None:
        out = _parse()
        entry = out["__zyx_prose_only"]
        assert entry["signature"] == ""
        assert entry["anchor"] == "zyx-shuffle"
        assert entry["description"] == (
            "A prose-only name like __zyx_prose_only never enters the map."
        )

    def test_plain_prose_word_is_absent(self) -> None:
        assert "zyx" not in _parse()
        assert "votes" not in _parse()

    def test_anchor_is_the_section_id(self) -> None:
        out = _parse()
        assert out["__zyx_ballot_v3"]["anchor"] == "zyx-warp-vote"

    def test_anchor_of_div_section(self) -> None:
        out = _parse()
        assert out["__zyx_late_v8"]["anchor"] == "zyx-shuffle"

    def test_signature_is_the_code_line(self) -> None:
        out = _parse()
        assert out["__zyx_ballot_v3"]["signature"] == (
            "unsigned int __zyx_ballot_v3(unsigned int pred);"
        )
        assert out["__zyx_count_v4"]["signature"] == "int __zyx_count_v4(int val);"

    def test_description_joins_paragraphs_after_the_block(self) -> None:
        out = _parse()
        assert out["__zyx_ballot_v3"]["description"] == (
            "These zyx vote calls return lane votes. They work on zyx warps only."
        )
        assert out["__zyx_count_v4"]["description"] == out["__zyx_ballot_v3"]["description"]

    def test_description_of_the_later_block(self) -> None:
        out = _parse()
        assert out["__zyx_late_v8"]["description"] == (
            "The late call runs after the first block. "
            "A prose-only name like __zyx_prose_only never enters the map."
        )

    def test_first_section_wins_for_duplicate_names(self) -> None:
        out = _parse()
        assert out["__zyx_dup_v1"]["anchor"] == "zyx-warp-vote"
        assert out["__zyx_dup_v1"]["signature"] == "int __zyx_dup_v1(int lane);"

    def test_entry_has_four_fields(self) -> None:
        for entry in _parse().values():
            assert sorted(entry) == ["anchor", "description", "math", "signature"]


class TestDeclarationPreference:
    """A declaring code line wins over an example call line."""

    def test_declaration_wins_over_example(self) -> None:
        html = (
            "<html><body>"
            '<section id="zyx-example"><pre>call_zyx_v1(p, 1);</pre>'
            "<p>An example block.</p></section>"
            '<section id="zyx-func"><pre>int call_zyx_v1(int *p, int v);</pre>'
            "<p>The real entry.</p></section>"
            "</body></html>"
        )
        out = parse_guide_page(html)
        assert out["call_zyx_v1"]["signature"] == "int call_zyx_v1(int *p, int v);"
        assert out["call_zyx_v1"]["anchor"] == "zyx-func"

    def test_example_line_is_kept_without_a_declaration(self) -> None:
        html = '<section id="zyx-example"><pre>call_zyx_v1(p, 1);</pre></section>'
        out = parse_guide_page(html)
        assert out["call_zyx_v1"]["signature"] == "call_zyx_v1(p, 1);"
        assert out["call_zyx_v1"]["anchor"] == "zyx-example"


class TestCalloutAndExampleCuts:
    """A description stops at the first callout or example sub-heading."""

    def _callouts(self) -> dict[str, dict[str, str]]:
        html = (FIXTURES_DIR / "cuda_docs_guide_callouts.html").read_text()
        return parse_guide_page(html)

    def test_description_stops_at_the_first_admonition(self) -> None:
        out = self._callouts()
        assert out["__zyx_mask_v9"]["description"] == (
            "These zyx shuffle calls move values inside a warp. They take a \\(2^{n}\\) mask."
        )

    def test_callout_and_example_text_is_not_in_the_description(self) -> None:
        desc = self._callouts()["__zyx_mask_v9"]["description"]
        assert "Hint" not in desc
        assert "memory barrier" not in desc
        assert "Example 3" not in desc
        assert "Compiler Explorer" not in desc

    def test_description_stops_at_an_example_rubric(self) -> None:
        out = self._callouts()
        assert out["__zyx_rub_v10"]["description"] == "The rubric cut works on this call."

    def test_description_stops_at_an_examples_section(self) -> None:
        out = self._callouts()
        assert out["__zyx_exa_v11"]["description"] == (
            "The example section cut works on this call."
        )

    def test_example_lead_in_and_code_stay_out_of_the_description(self) -> None:
        out = self._callouts()
        assert out["__zyx_lea_v12"]["description"] == "The lead-in cut works on this call."

    def test_guide_math_field_is_marked(self) -> None:
        out = self._callouts()
        assert out["__zyx_mask_v9"]["math"] == ["description"]
