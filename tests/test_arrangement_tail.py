"""The shared arrangement-tail parser reads the element and the lane index."""

from simdref.lsp import (
    _arrangement_tail,
    _av_class,
    _is_a64_only_operand,
    _is_arrangement,
    _z_range_numbers,
)


def test_arrangement_tail_reads_element_and_lane():
    assert _arrangement_tail("4h") == ("h", "")
    assert _arrangement_tail("b") == ("b", "")
    assert _arrangement_tail("d[1]") == ("d", "1")
    assert _arrangement_tail("4x") == (None, "")
    assert _arrangement_tail("") == (None, "")


def test_arrangement_parsers_read_the_control_shapes():
    assert _z_range_numbers("z0.b-z3.b") == [0, 3]
    assert _av_class("z0.4h") == "avh"
    assert _is_arrangement("z0.4h")
    assert not _is_arrangement("z0.4h[1]")
    assert not _is_arrangement("v1.d[1]")
    assert _is_a64_only_operand("v1.d[1]")
    assert not _is_a64_only_operand("v1.d[x]")
