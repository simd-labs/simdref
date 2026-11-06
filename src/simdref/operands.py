"""Operand-text helpers for the form matcher. No regular expressions.

They parse the operand side of one asm line: Intel size keywords, brace
lists and integer literals. The register rules stay in lsp.py and
regtable.py.
"""

from __future__ import annotations

# Intel size keywords ("byte ptr [rdx]", NASM "byte [rdx]") name the memory width.
INTEL_SIZE_BITS = {
    "byte": 8,
    "word": 16,
    "dword": 32,
    "fword": 48,
    "qword": 64,
    "tbyte": 80,
    "oword": 128,
    "xmmword": 128,
    "ymmword": 256,
    "zmmword": 512,
}


def intel_size_bits(op: str) -> int | None:
    """The width a leading size keyword names ("byte ptr [rdx]" -> 8).

    The keyword takes whitespace, an optional "ptr" and an optional "seg:"
    before the bracket. A keyword without its bracket names no width.
    """
    for keyword, bits in INTEL_SIZE_BITS.items():
        if op[: len(keyword)].lower() != keyword:
            continue
        rest = op[len(keyword) :]
        index = _skip_space(rest, 0)
        if index == 0:
            continue
        if rest[index : index + 3].lower() == "ptr":
            probe = _skip_space(rest, index + 3)
            if probe > index + 3:
                index = probe
        letters = index
        while rest[letters : letters + 1].isascii() and rest[letters : letters + 1].isalpha():
            letters += 1
        if letters - index >= 2 and rest[letters : letters + 1] == ":":
            index = _skip_space(rest, letters + 1)
        if rest[index : index + 1] == "[":
            return bits
    return None


def brace_members(op: str) -> str | None:
    """The text inside one whole-brace operand ("{z0.b, z1.b}" -> its members)."""
    if op[:1] == "{" and op[-1:] == "}" and "}" not in op[1:-1]:
        return op[1:-1]
    return None


def first_member(inner: str) -> str:
    """The first register of a brace list ("z0.b-z1.b" -> "z0.b")."""
    return inner.strip().casefold().split(",", 1)[0].split("-", 1)[0].strip()


def strip_braces(op: str) -> str:
    """Remove every brace span ("zmm0{k1}{z}" -> "zmm0")."""
    out: list[str] = []
    cut, index = 0, 0
    while True:
        start = op.find("{", index)
        end = op.find("}", start + 1)
        if start == -1 or end == -1:
            break
        out.append(op[cut:start])
        cut = index = end + 1
    out.append(op[cut:])
    return "".join(out)


def is_int_literal(op: str) -> bool:
    """True for a signed decimal or 0x-hex literal ("-5", "0x1f")."""
    body = op[1:] if op.startswith("-") else op
    if body[:2].lower() == "0x":
        body = body[2:]
        return bool(body) and all(char in "0123456789abcdefABCDEF" for char in body)
    return bool(body) and body.isdecimal()


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index
