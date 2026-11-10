"""Deterministic assembly-text scanners. No regular expressions.

These replace the lsp.py statement, comment, string and operand scans.
annotate.py and asm_identity.py reuse the same scans in later branches.
"""

from __future__ import annotations


def _is_word_head(char: str) -> bool:
    return char.isascii() and (char.isalpha() or char == "_")


def _is_word_tail(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in "_.")


def is_mnemonic(token: str) -> bool:
    """True for a mnemonic spelling: a letter, then letters, digits or dots."""
    return (
        bool(token)
        and token[0].isascii()
        and token[0].isalpha()
        and all(char.isascii() and (char.isalnum() or char == ".") for char in token[1:])
    )


def word_at(line: str, index: int) -> str | None:
    """The word token at a column: letter or underscore head, then word tails.

    A cursor right after the token still names it. A digit head ("0x10")
    starts no word, so its letters spell no token at that column.
    """
    if not 0 <= index <= len(line):
        return None
    start = index
    while start > 0 and _is_word_tail(line[start - 1]):
        start -= 1
    end = index
    while end < len(line) and _is_word_tail(line[end]):
        end += 1
    head = start
    while head < end and not _is_word_head(line[head]):
        head += 1
    return line[head:end] if head < end and head <= index else None


def split_operands(text: str) -> list[str]:
    """Split at top-level commas (bracket depth). An empty tail stays.

    Catalog keys use this split: a two-operand pre-index key must read as
    two operands, never three.
    """
    if not text.strip():
        return []
    parts: list[str] = []
    depth, cut = 0, 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[cut:index].strip())
            cut = index + 1
    parts.append(text[cut:].strip())
    return parts


def split_line_operands(text: str) -> list[str]:
    """Split a line's operand text at the commas the lsp matcher always split.

    A comma inside a bracket group stays only while the group closes before
    it opens again. Hover and hint picks pin this legacy shape: the depth
    rule above moves real picks on nested lists like "{z0.b[1], z1.b[2]}".
    """
    if not text.strip():
        return []
    parts: list[str] = []
    cut, index = 0, 0
    while index < len(text):
        if text[index] == ",":
            probe = index + 1
            while probe < len(text) and text[probe] not in "([{":
                if text[probe] in ")]}":
                    break
                probe += 1
            if probe >= len(text) or text[probe] in "([{":
                parts.append(text[cut:index].strip())
                cut = index + 1
        index += 1
    parts.append(text[cut:].strip())
    return parts


def cut_asm_comment(line: str, hash_opens) -> str:
    """Cut one asm comment. A '#' that hash_opens keeps stays in the line.

    hash_opens(prefix, rest) names the AArch64 case where a '#' opens an
    immediate operand. A block comment drops out and the scan continues.
    """
    out: list[str] = []
    pos, index = 0, 0
    while index < len(line):
        if line[index] == "#":
            if not hash_opens(line[:index], line[index + 1 :]):
                out.append(line[pos:index])
                return "".join(out)
            index += 1
            continue
        if line.startswith("/*", index):
            end = line.find("*/", index + 2)
            if end != -1:
                out.append(line[pos:index])
                pos = index = end + 2
                continue
        if line.startswith("//", index):
            out.append(line[pos:index])
            return "".join(out)
        index += 1
    out.append(line[pos:])
    return "".join(out)


def cut_semicolon(line: str) -> str:
    """Cut a NASM/MASM ';' comment. A GAS ';' keeps its statement role."""
    return line.partition(";")[0]


def statement_spans(body: str) -> list[tuple[int, int]]:
    """(start, end) of each statement in an asm-string body.

    A statement ends at a ';' or at the C escape "\\n" typed in the literal.
    """
    spans: list[tuple[int, int]] = []
    cut, index = 0, 0
    while index < len(body):
        if body[index] == ";":
            spans.append((cut, index))
            cut = index + 1
            index = cut
        elif body.startswith("\\n", index):
            spans.append((cut, index))
            cut = index + 2
            index = cut
        else:
            index += 1
    spans.append((cut, len(body)))
    return spans


def c_asm_literals(text: str) -> list[tuple[int, str]]:
    """(offset, body) of each string literal inside an asm(...) call.

    Comments blank out first (newlines stay), so a comment never hides a
    call. The bodies stay raw: the caller reads their escapes.
    """
    masked = _mask_c_comments(text)
    out: list[tuple[int, str]] = []
    index = 0
    while index < len(masked):
        hit = _asm_call_at(masked, index)
        if hit is not None:
            block_end, literals, _zones = hit
            out.extend(literals)
            index = block_end
        else:
            index += 1
    return out


def c_asm_zone_at(text: str, offset: int) -> str | None:
    """Completion zone holding the text offset, or None.

    'template' is an asm(...) template string, 'msvc' an MSVC __asm block
    or line, 'operands' the operand and clobber lists after the first ':'
    of an asm(...) call, 'comment' a C comment, 'string' a string literal
    that is no asm template. None is plain code. Shares _asm_call_at with
    c_asm_literals, so hover and completion parse the same calls.
    """
    span = c_span_at(text, offset)
    if span == "comment":
        return span
    zone = c_asm_zone_span_at(text, offset)
    return zone[0] if zone is not None else span


def c_asm_zone_span_at(text: str, offset: int) -> tuple[str, int, int] | None:
    """(kind, low, high) of the asm zone holding the offset, or None.

    The same walk as c_asm_zone_at. lsp.py reads the bounds: each MSVC
    __asm statement starts at its own zone's low bound.
    """
    masked = _mask_c_comments(text)
    index = 0
    while index < len(masked):
        hit = _asm_call_at(masked, index)
        if hit is not None:
            block_end, literals, zones = hit
            for zone in zones:
                if zone[1] <= offset <= zone[2]:
                    return zone
            for start, body in literals:
                if start <= offset <= start + len(body):
                    return "template", start, start + len(body)
            index = block_end
        else:
            index += 1
    return None


def c_span_at(text: str, offset: int) -> str | None:
    """'comment' or 'string' when a cursor offset sits inside one, else None.

    Same scan as _mask_c_comments. A cursor right after the opening
    delimiter is inside; a cursor after the closing one is outside. An
    unterminated block comment runs to the end, an unterminated string to
    the end of its line: both are still being typed. A char literal and a
    raw string are their own tokens: the double quote inside the char
    literal '"' never opens a string, and a raw string keeps its quotes.
    """
    index = 0
    while index < offset:
        line_end = text.find("\n", index)
        line_end = len(text) if line_end == -1 else line_end
        if text[index] == "'":
            end = _char_end(text, index) or line_end + 1
            if index < offset < end:
                return "string"
        elif text[index] == '"':
            end = _raw_string_end(text, index)
            if end is None:
                end = _string_end(text, index) or line_end + 1
            if index < offset < end:
                return "string"
        elif text.startswith("//", index):
            end = line_end
            if index + 2 <= offset <= end:
                return "comment"
        elif text.startswith("/*", index):
            close = text.find("*/", index + 2)
            end = len(text) if close == -1 else close + 2
            if index + 2 <= offset <= (end if close == -1 else close):
                return "comment"
        else:
            index += 1
            continue
        index = max(end, index + 1)
    return None


_QUALIFIERS = ("volatile", "__volatile__", "goto", "inline")
_ASM_KEYWORDS = ("__asm__", "__asm", "asm")


def _asm_keyword_at(text: str, index: int) -> str | None:
    """The asm keyword at index with word boundaries on both sides, or None."""
    if index > 0 and (text[index - 1].isalnum() or text[index - 1] == "_"):
        return None
    for keyword in _ASM_KEYWORDS:
        if text.startswith(keyword, index):
            after = index + len(keyword)
            tail = text[after : after + 1]
            if not (tail.isalnum() or tail == "_"):
                return keyword
    return None


def _asm_call_at(
    text: str, index: int
) -> tuple[int, list[tuple[int, str]], list[tuple[str, int, int]]] | None:
    """(block end, literals, zones) of the asm call at index.

    end is where the scan resumes: the last literal's closing quote, where
    the old regex match ended. zones are (kind, low, high) cursor regions,
    'operands' after the first ':' at bracket depth 1, 'msvc' for an MSVC
    __asm block or line.
    """
    if (
        index > 0
        and text[index - 1].isascii()
        and (text[index - 1].isalnum() or text[index - 1] == "_")
    ):
        return None
    for keyword in _ASM_KEYWORDS:
        if text.startswith(keyword, index):
            break
    else:
        return None
    cursor = index + len(keyword)
    while True:
        saved = cursor
        probe = _skip_space(text, cursor)
        if probe == cursor:
            break
        for qualifier in _QUALIFIERS:
            if text.startswith(qualifier, probe):
                cursor = probe + len(qualifier)
                break
        else:
            cursor = saved
            break
    cursor = _skip_space(text, cursor)
    if text[cursor : cursor + 1] == "{":
        # MSVC __asm { ... } block: bare instructions, no template string.
        close = _brace_end(text, cursor)
        if close is None:
            return None
        return close + 1, [], [("msvc", cursor + 1, close)]
    if text[cursor : cursor + 1] != "(":
        # MSVC __asm <insn>: the statement runs to end of line, or to the
        # next __asm keyword, which starts its own statement on the row.
        end = cursor
        while end < len(text) and text[end] != "\n":
            if _asm_keyword_at(text, end):
                break
            end += 1
        return end, [], [("msvc", cursor, end)]
    cursor = _skip_space(text, cursor + 1)
    literals: list[tuple[int, str]] = []
    block_end = cursor
    while True:
        end = _string_end(text, cursor, spans_lines=True)
        if end is None:
            break
        literals.append((cursor + 1, text[cursor + 1 : end - 1]))
        block_end = end
        probe = _skip_space(text, end)
        if text[probe : probe + 1] != '"':
            break
        cursor = probe
    if not literals:
        return None
    probe = _skip_space(text, block_end)
    if text[probe : probe + 1] not in (":", ")"):
        return None
    zones: list[tuple[str, int, int]] = []
    colon, close = _colon_and_close(text, block_end)
    if close is None:
        return None
    if colon is not None:
        zones.append(("operands", colon + 1, close))
    return block_end, literals, zones


def _colon_and_close(text: str, index: int) -> tuple[int | None, int | None]:
    """(first ':' at depth 1, closing ')') of the asm(...) paren group.

    index sits inside the group after the template strings. String
    literals are skipped: a ':' or ')' inside one is literal text.
    """
    depth, colon, close = 1, None, None
    cursor = index
    while cursor < len(text):
        char = text[cursor]
        if char == '"':
            end = _string_end(text, cursor, spans_lines=True)
            if end is None:
                return colon, None
            cursor = end
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                close = cursor
                break
        elif char == ":" and depth == 1 and colon is None:
            colon = cursor
        cursor += 1
    return colon, close


def _brace_end(text: str, index: int) -> int | None:
    """Index of the '}' matching the '{' at index, or None.

    String literals are skipped: a brace inside one is data.
    """
    depth, cursor = 0, index
    while cursor < len(text):
        char = text[cursor]
        if char == '"':
            end = _string_end(text, cursor, spans_lines=True)
            if end is None:
                return None
            cursor = end
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cursor
        cursor += 1
    return None


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _string_end(text: str, index: int, spans_lines: bool = False) -> int | None:
    """The index after the closing quote of a string at index, or None.

    A C escape pair ("\\t", "\\"") counts as one literal character. The
    asm-string case spans lines; the C-token case does not.
    """
    if text[index : index + 1] != '"':
        return None
    cursor = index + 1
    while cursor < len(text):
        char = text[cursor]
        if char == "\\":
            cursor += 2
        elif char == '"':
            return cursor + 1
        elif char == "\n" and not spans_lines:
            return None
        else:
            cursor += 1
    return None


def _char_end(text: str, index: int) -> int | None:
    """The index after the closing quote of the char literal at index, or None.

    A char literal is one token like a string: its quote never opens a
    string, and an escape pair ("\\'", "\\"") counts as one character.
    """
    if text[index : index + 1] != "'":
        return None
    cursor = index + 1
    while cursor < len(text):
        char = text[cursor]
        if char == "\\":
            cursor += 2
        elif char == "'":
            return cursor + 1
        elif char == "\n":
            return None
        else:
            cursor += 1
    return None


def _raw_string_end(text: str, index: int) -> int | None:
    """The index after the R"delim(...)delim" raw string whose quote is at index.

    None when the shape is not a raw string: the caller then reads the
    quote as a plain string. An unterminated raw string runs one past the
    end of the text, because its body spans lines.
    """
    if text[index - 1 : index] != "R":
        return None
    open_paren = text.find("(", index + 1, index + 18)
    if open_paren == -1:
        return None
    delim = text[index + 1 : open_paren]
    if len(delim) > 16 or any(char.isspace() or char in "()\\" for char in delim):
        return None
    close = text.find(")" + delim + '"', open_paren + 1)
    return len(text) + 1 if close == -1 else close + len(delim) + 2


def _mask_c_comments(text: str) -> str:
    """Blank C comments to spaces (newlines stay). Literal tokens stay."""
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "'":
            end = _char_end(text, index)
            if end is not None:
                out.append(text[index:end])
                index = end
                continue
        if char == '"':
            end = _raw_string_end(text, index)
            if end is None:
                end = _string_end(text, index)
            if end is not None:
                out.append(text[index:end])
                index = end
                continue
        if text.startswith("//", index):
            while index < len(text) and text[index] != "\n":
                out.append(" ")
                index += 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end != -1:
                while index < end + 2:
                    out.append("\n" if text[index] == "\n" else " ")
                    index += 1
                continue
        out.append(char)
        index += 1
    return "".join(out)
