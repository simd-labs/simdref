"""Tests for the shared sm floor predicate."""

from __future__ import annotations

from simdref.filters import runs_on_sm


def test_empty_sides_pass() -> None:
    assert runs_on_sm("", "sm_80")
    assert runs_on_sm("sm_70", "")
    assert runs_on_sm("", "")
    assert runs_on_sm(None, None)


def test_floor_compare() -> None:
    assert runs_on_sm("sm_70", "sm_80")
    assert runs_on_sm("sm_70", "sm_70")
    assert not runs_on_sm("sm_90", "sm_80")


def test_plain_number_choice() -> None:
    assert runs_on_sm("sm_30", "80")
    assert not runs_on_sm("sm_90", "20")
