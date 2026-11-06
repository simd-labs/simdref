"""Pins for the non-regex alnum-run scanners in storage.py.

Each pin mirrors the behaviour of the removed [A-Za-z0-9]+ and
[a-zA-Z]+|[0-9]+ scans, including the ASCII-only gate.
"""

from __future__ import annotations

from simdref.storage import _alnum_runs, _fts_match_query, _tokenize_name


def test_alnum_runs_plain() -> None:
    assert _alnum_runs("a1_b2") == ["a1", "b2"]
    assert _alnum_runs("a1b2") == ["a1b2"]
    assert _alnum_runs("123abc") == ["123abc"]
    assert _alnum_runs("") == []
    assert _alnum_runs("___") == []
    # non-ASCII alphanumerics are not [A-Za-z0-9]
    assert _alnum_runs("\u00e91\u00e92") == ["1", "2"]


def test_alnum_runs_split_alpha_digit() -> None:
    assert _alnum_runs("mm256", split_alpha_digit=True) == ["mm", "256"]
    assert _alnum_runs("a1b", split_alpha_digit=True) == ["a", "1", "b"]
    assert _alnum_runs("1a", split_alpha_digit=True) == ["1", "a"]
    assert _alnum_runs("mm256_add", split_alpha_digit=True) == ["mm", "256", "add"]
    assert _alnum_runs("123abc", split_alpha_digit=True) == ["123", "abc"]


def test_tokenize_name() -> None:
    assert _tokenize_name("_mm256_add_epi32") == "mm 256 add epi 32"
    assert _tokenize_name("VADDPS (YMM, YMM, YMM)") == "vaddps ymm ymm ymm"


def test_fts_match_query() -> None:
    assert _fts_match_query("VADDPS (YMM, YMM, YMM)") == (
        '"vaddps"* AND "ymm"* AND "ymm"* AND "ymm"*'
    )
    assert _fts_match_query("!!!") == ""
