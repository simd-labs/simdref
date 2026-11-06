"""Catalog-key parsing for the form matcher. No regular expressions.

A key ("ADD (M32, I8)") names one form. This module splits its operands
and maps each key operand to the line classes it accepts. The accept sets
name the classes lsp.py mints in _operand_class.
"""

from __future__ import annotations

from simdref.asmline import split_operands

IMMEDIATE_ACCEPT = frozenset({"i", "lit0", "lit1"})
LITERAL_ACCEPT = {frozenset({"lit0"}), frozenset({"lit1"})}
ACCUMULATOR = {"al": "r8", "ax": "r16", "eax": "r32", "rax": "r64"}
X86_VECTOR_WIDTHS = {"xmm": "v128", "ymm": "v256", "zmm": "v512"}


def operands_of_key(key: str) -> list[str]:
    """The operands of a catalog key, split at top-level commas.

    AArch64 address operands hold their own commas ("[<XNSP>, <HASH>#0]!"),
    so a pre-index key must read as its two operands, never three.
    """
    start, end = key.find("("), key.rfind(")")
    inner = key[start + 1 : end].strip() if 0 <= start < end else ""
    return split_operands(inner)


def key_operand_bits(key: str) -> tuple:
    """The operand widths of a catalog key in order: R8l -> 8, M16 -> 16."""
    return tuple(_first_digits(op) for op in operands_of_key(key))


def _first_digits(text: str) -> int | None:
    start = 0
    while start < len(text) and not text[start].isdecimal():
        start += 1
    end = start
    while end < len(text) and text[end].isdecimal():
        end += 1
    return int(text[start:end]) if end > start else None


def form_accept(op: str) -> frozenset:
    """The line classes a catalog-key operand accepts, as one frozenset.

    An immediate-width key (i8..i64) takes every immediate class. A literal 0/1
    takes only its own value: a fixed-1 shift is not the line's "2". An AArch64
    placeholder maps to the register family it encodes (<XNSP> = x or sp).
    """
    o = op.casefold()
    if o in ("0", "1"):
        return frozenset({"lit" + o})
    if o[:1] == "i" and o[1:].isdecimal():
        return IMMEDIATE_ACCEPT
    if o in ACCUMULATOR or o == "cl":
        return frozenset({ACCUMULATOR.get(o, "r8")})
    base = o[:-1] if o[-1:] in ("h", "l") else o
    if base in ("r8", "r16", "r32", "r64"):
        return frozenset({base})
    if o in X86_VECTOR_WIDTHS or o in ("k", "mm"):
        return frozenset({X86_VECTOR_WIDTHS.get(o, o)})
    head, sep, tail = o.partition("_")
    if sep and head[:1] == "m" and head[1:].isdecimal():
        left, mark, right = tail.partition("to")
        if mark and left.isdecimal() and right.isdecimal():
            return frozenset({"mbc"})
    if o[:1] == "m" and o[1:].isdecimal():
        return frozenset({o, "m"})
    if o.startswith("moffs") and o[5:].isdecimal():
        return frozenset({"moffs"})
    if o == "rel" or (o.startswith("rel") and o[3:].isdecimal()):
        return frozenset({"rel"})
    if op.startswith("["):
        # An AArch64 bracket key encodes the index mode: pre-index carries
        # '!' ("[<XNSP>, <HASH>#0]!"), a register offset holds a bare comma
        # ("[<XNSP>, <WORX>]"), an immediate offset keeps its '#' ("[#0]"),
        # and a plain base has neither ("[<XNSP>]"). "m" stays acceptable to
        # the unsigned-offset key, whose '#imm' defaults to 0 on the line.
        if "!" in op:
            return frozenset({"mpre"})
        body = op[1 : op.index("]")] if "]" in op else op[1:]
        if "," in body:
            return frozenset({"mreg"})
        if "#" in body:
            return frozenset({"mo", "m"})
        return frozenset({"m"})
    name = _cut_key_repeats(op.upper())
    if "<HASH>" in name:
        return IMMEDIATE_ACCEPT
    # A ZA tile operand ("<ZAT...>[W1, #0]") is not a plain vector register.
    if "<ZAT" in name:
        return frozenset({"za"})
    if _is_counted_vector(name):
        # A counted register list accepts only the same count.
        zts = [
            char
            for index, char in enumerate(name)
            if char.isdecimal() and index >= 3 and name[index - 3 : index] == "<ZT"
        ]
        if zts:
            return frozenset({f"av{max(int(number) for number in zts)}"})
        count = name.count(",") + 1
        if count in (2, 4):
            return frozenset({f"av{count}"})
        # A Z-register arrangement element ("Z0.S") accepts its class ("avs"),
        # so the same-lane row outranks the other Z-register rows.
        element = _z_element(name)
        return frozenset({"av", f"av{element}"} if element else {"av"})
    # Numbered scalar FP placeholders: D0 or S1 style numbers, <BT> or <QT1> style aliases.
    if name[:1] in ("B", "H", "S", "D", "Q") and name[1:].isdecimal():
        return frozenset({name[0].lower()})
    if name[:1] == "<" and name[-1:] == ">" and len(name) >= 4:
        if (
            name[1] in ("B", "H", "S", "D", "Q")
            and name[2] == "T"
            and all(char.isdecimal() for char in name[3:-1])
        ):
            return frozenset({name[1].lower()})
    if name.startswith("D<") or name == "<FPFAR_DT>":
        return frozenset({"dfp"})
    if "PNG" in name:
        return frozenset({"png"})
    if name[:1] == "P" and name[1:2].isdecimal():
        return frozenset({"pred"})
    # An extend-flavored placeholder (a name with EXTEND) takes the extend
    # marker class; every register placeholder also takes the shift marker.
    extend = "EXTEND" in name
    if "WORX" in name or "R_OPTION" in name:
        return frozenset({"w", "x"} | ({"we", "xe"} if extend else {"ws", "xs"}))
    if "WZR" in name:
        return frozenset({"w", "wzr"} | ({"we"} if extend else {"ws"}))
    if "XZR" in name:
        return frozenset({"x", "xzr"} | ({"xe"} if extend else {"xs"}))
    if "WSP" in name:
        return frozenset({"w", "wsp"})
    if "SP" in name:
        return frozenset({"x", "sp"})
    return frozenset()


def _cut_key_repeats(name: str) -> str:
    """Drop trailing "#N" key repeats ("<HASH>#0#0" -> "<HASH>")."""
    while True:
        head, sep, tail = name.rpartition("#")
        if sep and tail.isdecimal():
            name = head
        else:
            return name


def _is_counted_vector(name: str) -> bool:
    """True for a counted list or arrangement key ("Z0.4S", "{...}", "V0<...")."""
    if name.startswith("{"):
        return True
    if name[:1] not in ("Z", "V"):
        return False
    end = 1
    while end < len(name) and name[end].isdecimal():
        end += 1
    if end == 1:
        return False
    tail = name[end : end + 1]
    return tail == "." or (name[:1] == "V" and tail == "<")


def _z_element(name: str) -> str | None:
    """The element letter of a Z arrangement ("Z0.4S" -> "S"), or None."""
    index = name.find("Z")
    while index != -1:
        end = index + 1
        while end < len(name) and name[end].isdecimal():
            end += 1
        if end > index + 1 and name[end : end + 1] == ".":
            letter = end + 1
            while letter < len(name) and name[letter].isdecimal():
                letter += 1
            if name[letter : letter + 1] in ("B", "H", "S", "D", "Q"):
                return name[letter].lower()
        index = name.find("Z", index + 1)
    return None
