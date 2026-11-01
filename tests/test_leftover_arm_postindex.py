"""AArch64 '#' immediates must survive the asm comment strip (leftover item 4)."""

from simdref.lsp import _mnemonic_from_asm_line

# Asserted rows: (source line, mnemonic, operand count, operands).
ROWS = [
    ("ldr s1, [x11], #4", "ldr", 3, "s1, [x11], #4"),  # post-index
    ("ldr x0, [sp, #-16]!", "ldr", 2, "x0, [sp, #-16]!"),  # pre-index
    ("add x0, x1, x2, lsl #2", "add", 4, "x0, x1, x2, lsl #2"),  # shifted immediate
    ("add x0, x1, x2 # comment", "add", 3, "x0, x1, x2"),  # real comment strips
    ("addq $1, %rax # c", "addq", 2, "$1, %rax"),  # AT&T real comment strips
]


def test_hash_immediates_survive_the_comment_strip():
    for line, mnemonic, count, operands in ROWS:
        assert _mnemonic_from_asm_line(line) == (mnemonic, count, operands), line
