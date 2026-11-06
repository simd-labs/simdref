"""asmline.py: the shared assembly-text scanners."""

from simdref.asmline import (
    c_asm_literals,
    cut_asm_comment,
    cut_semicolon,
    is_mnemonic,
    split_line_operands,
    split_operands,
    statement_spans,
    word_at,
)


def _hash_keeps(prefix: str, rest: str) -> bool:
    return prefix.rstrip().endswith(",")


def test_split_operands_uses_top_level_commas():
    assert split_operands("[x0, x1], #4") == ["[x0, x1]", "#4"]
    # A pre-index key address must read as one operand, never two.
    assert split_operands("[<XNSP>, <HASH>#0]!") == ["[<XNSP>, <HASH>#0]!"]
    assert split_operands("  ") == []
    assert split_operands("a,") == ["a", ""]


def test_split_line_operands_keeps_the_legacy_shield():
    # The line split cuts at a comma that no closing bracket shields, so a
    # nested bracket list splits where the depth rule would not.
    assert split_line_operands("a, {b[1], c[2]}") == ["a", "{b[1]", "c[2]}"]
    assert split_line_operands("a, (b, c), d") == ["a", "(b, c)", "d"]
    assert split_operands("a, {b[1], c[2]}") == ["a", "{b[1], c[2]}"]


def test_word_at_finds_the_token_under_the_cursor():
    assert word_at("vaddps xmm0, xmm1", 7) == "xmm0"
    assert word_at("xmm0, xmm1", 4) == "xmm0"
    # A cursor right after the word still names it; past that it names nothing.
    assert word_at("xmm0,", 4) == "xmm0"
    assert word_at("xmm0,", 5) is None
    # A digit head starts no word: its letters name no token at column 0.
    assert word_at("0x10", 2) == "x10"
    assert word_at("0x10", 0) is None
    assert word_at("mov eax, 0x10", 9) is None
    # Word characters are ASCII: an accented letter ends the token before it.
    assert word_at("m\u00f6v", 1) == "m"
    assert word_at("\u00e9", 0) is None


def test_cut_asm_comment_keeps_a_hash_immediate():
    assert cut_asm_comment("add x0, #2 // tail", _hash_keeps) == "add x0, #2 "
    assert cut_asm_comment("add x0, #2", _hash_keeps) == "add x0, #2"
    assert cut_asm_comment("nop #c3", _hash_keeps) == "nop "
    assert cut_asm_comment("nop // done", _hash_keeps) == "nop "
    assert cut_asm_comment("nop /* x, y */ ret", _hash_keeps) == "nop  ret"
    assert cut_asm_comment("nop /* open", _hash_keeps) == "nop /* open"


def test_cut_semicolon_keeps_a_gas_semicolon_out():
    assert cut_semicolon("mov rax, rbx; add rcx, 1") == "mov rax, rbx"
    assert cut_semicolon("mov rax, rbx") == "mov rax, rbx"


def test_statement_spans_split_at_semicolon_and_escaped_newline():
    body = "xorl rax, rax\\naddq $8, rax; ret"
    spans = statement_spans(body)
    assert [body[low:high] for low, high in spans] == ["xorl rax, rax", "addq $8, rax", " ret"]


def test_c_asm_literals_finds_the_call_and_ignores_comments():
    text = 'int x; // asm("hidden")\nasm("xorl %eax, %eax\\n\\taddq $8, %rax");'
    literals = c_asm_literals(text)
    assert [body for _, body in literals] == ["xorl %eax, %eax\\n\\taddq $8, %rax"]
    assert text[literals[0][0] :][:3] == "xor"
    assert c_asm_literals("int a = 1;") == []
    assert c_asm_literals("asm(x)") == []
    # A comment inside the call hides nothing.
    assert [body for _, body in c_asm_literals('asm(/* c */ "nop")')] == ["nop"]
    # Adjacent literals keep their own bodies, like the old part scan.
    assert [body for _, body in c_asm_literals('asm("a" "b")')] == ["a", "b"]


def test_is_mnemonic_matches_the_mnemonic_spelling():
    assert is_mnemonic("movsd")
    assert is_mnemonic("vaddps")
    assert not is_mnemonic("1abc")
    assert not is_mnemonic("mo vd")
    assert not is_mnemonic("m\u00f6v")
    assert not is_mnemonic("")
