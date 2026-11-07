"""Edge cases of the annotate structural scans (no regular expressions)."""

from simdref.annotate import (
    _bcond_hits,
    _form_operand_tokens,
    _form_widths,
    _has_size_ptr,
    _is_number,
    _is_symbol,
    _summary_bits,
)


def test_bcond_hits_count_and_boundaries():
    assert _bcond_hits("b.eq 0x118 <dotn+0x118>") == 1
    assert _bcond_hits("b.eq b.ne") == 2
    assert _bcond_hits("b.b.eq") == 1
    assert _bcond_hits("b.eqx") == 0
    assert _bcond_hits("xb.eq") == 0
    assert _bcond_hits("b.cs b.cc") == 0
    assert _bcond_hits("B.EQ") == 0


def test_summary_bits_walks_back_over_dash_and_space():
    assert _summary_bits("Move 32-bit integer operands.") == "32"
    assert _summary_bits("3 - bit") == "3"
    assert _summary_bits("12-34-bit") == "34"
    assert _summary_bits("bit 32-bit") == "32"
    assert _summary_bits("032-bit") == "032"
    assert _summary_bits("3--bit") is None
    assert _summary_bits("3 x - bit") is None
    assert _summary_bits("plain words") is None


def test_form_operand_tokens_span_rule():
    assert _form_operand_tokens("XOR (R32, 0)") == ["R32", "0"]
    assert _form_operand_tokens("(x (y)") == ["X (Y"]
    assert _form_operand_tokens("(a) (b)") == ["B"]
    assert _form_operand_tokens("(a))") == []
    assert _form_operand_tokens("(() )") == []
    assert _form_operand_tokens("RET_NEAR") == []


def test_size_ptr_and_literals():
    assert _has_size_ptr("byte ptr [rdx]")
    assert _has_size_ptr("$byte ptr")
    assert not _has_size_ptr("fword ptr [rdx]")
    assert not _has_size_ptr("byteptr")
    assert not _has_size_ptr("xbyte ptr")
    assert _is_number("0x1f") and _is_number("-5") and not _is_number("0X1f")
    assert _is_symbol(".L2") and _is_symbol("$x") and not _is_symbol("a-b")
    assert _form_widths("MOV (M64, R64)") == ["64", "64"]
