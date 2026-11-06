"""HTML fragment to text for the vendored doc parsers.

One tag scanner replaces the per-module regex chains. Each call site passes
its tag policy as data and keeps its own whitespace tail. The scanners walk
literal ``<...>`` candidates the way the old regex chains did, so output
stays byte-identical on the vendored corpora.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Iterator


def tag_at(text: str, pos: int) -> tuple[int, int, str] | None:
    """Next ``<...>`` candidate at or after pos: (start, end, inner).

    A candidate has at least one inner character, like the old ``<[^>]+>``.
    """
    start = text.find("<", pos)
    while start >= 0:
        stop = text.find(">", start + 1)
        if stop < 0:
            return None
        if stop > start + 1:
            return start, stop + 1, text[start + 1 : stop]
        start = text.find("<", start + 1)
    return None


def _match_br(text: str, start: int) -> int | None:
    if text[start : start + 3].lower() != "<br":
        return None
    index = start + 3
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] == "/":
        index += 1
    return index + 1 if index < len(text) and text[index] == ">" else None


def _match_close(name: str) -> Callable[[str, int], int | None]:
    def match(text: str, start: int) -> int | None:
        probe = name.casefold()
        span = text[start : start + 2 + len(name)].casefold()
        if span != "</" + probe:
            return None
        index = start + 2 + len(name)
        while index < len(text) and text[index].isspace():
            index += 1
        return index + 1 if index < len(text) and text[index] == ">" else None

    return match


def _match_tags(names: tuple[str, ...], ignorecase: bool) -> Callable[[str, int], int | None]:
    def match(text: str, start: int) -> int | None:
        index = start + 1
        if index < len(text) and text[index] == "/":
            index += 1
        probe = text[index : index + max(len(name) for name in names)]
        if ignorecase:
            probe = probe.casefold()
        if not any(probe.startswith(name) for name in names):
            return None
        stop = text.find(">", index)
        return stop + 1 if stop >= 0 else None

    return match


def _match_any(text: str, start: int) -> int | None:
    stop = text.find(">", start + 1)
    return stop + 1 if stop > start + 1 else None


def _sub(text: str, match: Callable[[str, int], int | None], repl: str) -> str:
    out: list[str] = []
    pos = 0
    index = 0
    while index < len(text):
        if text[index] == "<":
            end = match(text, index)
            if end is not None:
                out.append(text[pos:index])
                out.append(repl)
                pos = end
                index = end
                continue
        index += 1
    out.append(text[pos:])
    return "".join(out)


def strip_html(
    text: str,
    rules: tuple[tuple[str, tuple[str, ...], str], ...],
    generic: str | None = None,
    ignorecase: bool = False,
) -> str:
    """Replace tag spans per rules, first match wins.

    ``("br", (), repl)`` matches ``<br\\s*/?>``; ``("close", (name,), repl)``
    matches ``</name\\s*>``; ``("tags", (names...), repl)`` matches
    ``</?name[^>]*>`` as a prefix. ``generic`` replaces every other span;
    ``None`` keeps those spans verbatim.
    """
    for kind, names, repl in rules:
        if kind == "br":
            matcher: Callable[[str, int], int | None] = _match_br
        elif kind == "close":
            matcher = _match_close(names[0])
        else:
            matcher = _match_tags(names, ignorecase)
        text = _sub(text, matcher, repl)
    if generic is not None:
        text = _sub(text, _match_any, generic)
    return text


def unwrap_anchors(text: str) -> str:
    """Drop ``<a ...>`` and its ``</a>`` around same-line text."""
    out: list[str] = []
    pos = 0
    while True:
        start = text.find("<a ", pos)
        if start < 0:
            out.append(text[pos:])
            return "".join(out)
        open_end = text.find(">", start + 3)
        if open_end < 0:
            out.append(text[pos:])
            return "".join(out)
        stop = text.find("\n", open_end + 1)
        if stop < 0:
            stop = len(text)
        close = text.find("</a>", open_end + 1, stop)
        if close < 0:
            out.append(text[pos : start + 1])
            pos = start + 1
            continue
        out.append(text[pos:start])
        out.append(text[open_end + 1 : close])
        pos = close + 4


def _runs(text: str) -> Iterator[tuple[bool, str]]:
    start = 0
    ws = bool(text) and text[0].isspace()
    for index, char in enumerate(text):
        if char.isspace() != ws:
            yield ws, text[start:index]
            start = index
            ws = not ws
    yield ws, text[start:]


def _single_spaced(chunk: str) -> str:
    out: list[str] = []
    pending = ""
    for char in chunk:
        if char in " \t":
            pending += char
            continue
        out.append(" " if len(pending) > 1 else pending)
        pending = ""
        out.append(char)
    out.append(" " if len(pending) > 1 else pending)
    return "".join(out)


def tighten(text: str) -> str:
    """Whitespace runs that hold a newline become one newline; long
    ``[ \\t]`` runs become one space."""
    out: list[str] = []
    for ws, chunk in _runs(text):
        if not ws:
            out.append(chunk)
        elif "\n" in chunk:
            out.append("\n")
        else:
            out.append(_single_spaced(chunk))
    return "".join(out).strip()


def tidy(text: str) -> str:
    """A whitespace run that holds a newline after its first character
    collapses to one newline; the tail after that newline stays."""
    out: list[str] = []
    for ws, chunk in _runs(text):
        cut = chunk.rfind("\n") if ws else -1
        if cut >= 1:
            out.append("\n")
            out.append(chunk[cut + 1 :])
        else:
            out.append(chunk)
    return "".join(out).strip()


def squeeze(text: str) -> str:
    """At most two consecutive newlines survive."""
    out: list[str] = []
    count = 0
    for char in text:
        if char == "\n":
            count += 1
            if count <= 2:
                out.append(char)
        else:
            count = 0
            out.append(char)
    return "".join(out).strip()


def squash(text: str) -> str:
    """Each whitespace run becomes one space."""
    out = [" " if ws else chunk for ws, chunk in _runs(text)]
    return "".join(out).strip()
