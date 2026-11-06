"""operands.py: the operand-text helpers."""

from simdref.operands import (
    brace_members,
    first_member,
    intel_size_bits,
    is_int_literal,
    strip_braces,
)


def test_intel_size_bits_reads_the_leading_keyword():
    assert intel_size_bits("byte ptr [rdx]") == 8
    assert intel_size_bits("zmmword ptr [rdx]") == 512
    assert intel_size_bits("byte [rax]") == 8
    assert intel_size_bits("qword ptr ds:[rbx+8]") == 64
    # The keyword without its bracket names no width; so does a bare "ptr".
    assert intel_size_bits("byte[rdx]") is None
    assert intel_size_bits("byte ptr") is None
    assert intel_size_bits("ptr [rax]") is None


def test_brace_helpers_read_lists_and_shapes():
    assert brace_members("{z0.b, z1.b}") == "z0.b, z1.b"
    assert brace_members("z0.b") is None
    assert brace_members("{z0.b") is None
    assert first_member("z0.b-z1.b") == "z0.b"
    assert first_member("z0.b, z1.b") == "z0.b"
    assert strip_braces("zmm0{k1}{z}") == "zmm0"
    assert strip_braces("za0h.b[w12, 0]") == "za0h.b[w12, 0]"
    assert strip_braces("{a{b}") == ""


def test_is_int_literal_reads_the_literal_shape():
    assert is_int_literal("0x1F")
    assert is_int_literal("-5")
    assert is_int_literal("007")
    assert not is_int_literal("0x")
    assert not is_int_literal("12a")
    assert not is_int_literal("$5")
    assert not is_int_literal("-")
