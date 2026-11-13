"""Node 1 check (design E): the guide and math parsers keep the C.1 blocks.

A shuffle section yields one owned block list per function and one
shared block set, with the dt brief as the row summary. A math entry
keeps its seealso note and its note blocks as note blocks, and its
``Returns`` deflist item exactly once.
"""

from __future__ import annotations

from pathlib import Path

from simdref.cuda_docs import parse_guide_page, parse_math_api_page

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

SHUFFLE_NAMES = (
    "__zyx_shfl_v1",
    "__zyx_shfl_up_v2",
    "__zyx_shfl_down_v3",
    "__zyx_shfl_xor_v4",
)


def _load(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


class TestGuideSectionCut:
    """Four owned block lists and one shared block set per shuffle name."""

    def test_four_owned_lists_and_one_shared_set(self) -> None:
        entries = parse_guide_page(_load("cuda_docs_guide_cpp_ext.html"))
        shared_sets = []
        owned_lists = []
        for name in SHUFFLE_NAMES:
            entry = entries[name]
            shared = [s for s in entry["sections"] if s["shared"]]
            owned = [s for s in entry["sections"] if not s["shared"]]
            assert len(shared) == 1, name
            assert len(owned) == 1, name
            assert all(block.get("shared") for block in shared[0]["blocks"])
            assert all(block.get("owner") == name for block in owned[0]["blocks"])
            shared_sets.append(shared[0]["blocks"])
            owned_lists.append(owned[0]["blocks"])
        # One shared block set: every entry sees the same blocks.
        assert all(blocks == shared_sets[0] for blocks in shared_sets)
        # Four distinct owned lists: no two functions share a description.
        texts = [" ".join(block.get("text", "") for block in blocks) for blocks in owned_lists]
        assert len(set(texts)) == 4

    def test_dt_brief_is_the_row_summary(self) -> None:
        entries = parse_guide_page(_load("cuda_docs_guide_cpp_ext.html"))
        assert entries["__zyx_shfl_v1"]["summary"] == "Copy var from lane srcLane."
        assert entries["__zyx_shfl_up_v2"]["summary"] == ("Copy var from lane id minus delta.")
        assert entries["__zyx_shfl_down_v3"]["summary"] == ("Copy var from lane id plus delta.")
        assert entries["__zyx_shfl_xor_v4"]["summary"] == ("Copy var from lane id xor laneMask.")


class TestMathEntryBlocks:
    """The seealso note and the note blocks survive as note blocks."""

    def test_sin_keeps_seealso_and_notes(self) -> None:
        entry = parse_math_api_page(_load("cuda_docs_math_page.html"))["__zyx_sin_v2"]
        blocks = [block for s in entry["sections"] for block in s["blocks"]]
        notes = [b for b in blocks if b["kind"] == "note"]
        assert [n["note"] for n in notes] == ["note", "note", "seealso"]

    def test_sin_has_one_returns_item(self) -> None:
        entry = parse_math_api_page(_load("cuda_docs_math_page.html"))["__zyx_sin_v2"]
        blocks = [block for s in entry["sections"] for block in s["blocks"]]
        items = [item for b in blocks if b["kind"] == "deflist" for item in b["items"]]
        assert [i["name"] for i in items].count("Returns") == 1


class TestTableMathCells:
    """A table cell carries the readable text of its math, as runs do."""

    def test_math_cell_reads_as_unicode(self) -> None:
        from simdref.docblocks import blocks_of, build_tree

        html = (
            "<table><tr><td><p>rel err "
            '<span class="math notranslate nohighlight">\\(2^{-21.41}\\)</span>'
            "</p></td></tr></table>"
        )
        blocks = blocks_of(build_tree(html))
        table = next(b for b in blocks if b["kind"] == "table")
        assert table["rows"] == [["rel err 2⁻²¹·⁴¹"]]

    def test_plain_cell_text_is_untouched(self) -> None:
        from simdref.docblocks import blocks_of, build_tree

        html = "<table><tr><td><p>2 items</p></td></tr></table>"
        blocks = blocks_of(build_tree(html))
        table = next(b for b in blocks if b["kind"] == "table")
        assert table["rows"] == [["2 items"]]


class TestListOrdered:
    """The ordered flag comes from the element tag of every list."""

    def _items(self, html: str):
        from simdref.docblocks import blocks_of, build_tree

        blocks = blocks_of(build_tree(html))
        return next(b for b in blocks if b["kind"] == "list")

    def test_ordered_list_is_ordered(self) -> None:
        block = self._items("<ol><li>Reads the old value.</li><li>Adds.</li></ol>")
        assert block["ordered"] is True

    def test_unordered_list_is_not_ordered(self) -> None:
        block = self._items("<ul><li>int, unsigned, float.</li></ul>")
        assert block["ordered"] is False

    def test_nested_list_takes_its_own_flag(self) -> None:
        block = self._items("<ul><li>step<ul><li>sub</li></ul></li></ul>")
        nested = block["items"][0]["blocks"][0]
        assert block["ordered"] is False
        assert nested["ordered"] is False
