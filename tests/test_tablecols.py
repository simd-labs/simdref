"""Shape tests for simdref.tablecols: the vendors' value notations, no regex."""

from __future__ import annotations

from simdref import tablecols as tc


def test_digits():
    assert tc.value_shape("4")
    assert tc.row_tail("MOV 4 1 SKL") == ("4", "1", "SKL", "")


def test_digit_marker():
    assert tc.value_shape("4(1)")
    assert tc.row_tail("MOV 4(1) 1 SKL") == ("4(1)", "1", "SKL", "")


def test_digit_to_digit():
    # "d to d" spans two cells; row_tail reads it from the tail.
    assert tc.row_tail("MOV 4 to 8 1 SKL") == ("4 to 8", "1", "SKL", "")


def test_fraction():
    assert tc.value_shape("1/2")
    assert tc.row_tail("MOV 1/2 1 SKL") == ("1/2", "1", "SKL", "")
    assert tc.row_tail("MOV 4 1/2 SKL") == ("4", "1/2", "SKL", "")


def test_decimal():
    assert tc.value_shape("0.5")
    assert tc.row_tail("MOV 4 0.5 SKL") == ("4", "0.5", "SKL", "")


def test_fraction_to_fraction():
    # "d/d to d/d" spans two cells; row_tail reads it from the tail.
    assert tc.row_tail("MOV 4 1/4 to 1/2 SKL") == ("4", "1/4 to 1/2", "SKL", "")


def test_footnote_marker():
    assert tc.strip_footnote("4(1)") == "4"
    assert tc.strip_footnote("10") == "10"
    assert tc.strip_footnote("4(12)") == "4(12)"
