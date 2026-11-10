"""Issue #65 M5: prefix words (data16, cs, ...) on a plain asm line."""

from simdref.annotate import parse_asm_line


def test_plain_line_prefix_tokens_not_mnemonic():
    ins = parse_asm_line("data16 cs nopw")
    assert ins.mnemonic == "nopw"
    assert ins.operands == ""


def test_plain_line_prefix_before_operands():
    ins = parse_asm_line("data16 nopw 0x0(%rax)")
    assert ins.mnemonic == "nopw"
    assert ins.operands == "0x0(%rax)"


def test_plain_line_lone_prefix_stays_prefix():
    # The lone-prefix rule from issue #52: a bare `data16` is the prefix
    # instruction by itself.
    ins = parse_asm_line("data16")
    assert ins.mnemonic == "data16"
