"""CUDA documentation page parsers and the build-time fetch recipe.

This module turns NVIDIA documentation pages into plain dictionaries.
Another module joins these dictionaries onto the intrinsic catalog.

Parsing rules:
* Use only ``html.parser`` and plain string methods.
* No regex and no extra dependency.
* Never keep NVIDIA prose in git. Fixtures for tests are synthetic.

Page shapes:
* CUDA Math API and libdevice pages hold Sphinx ``dl cpp function``
  entries.
* Programming Guide pages document functions in ``pre`` code blocks
  inside sections.
* The Mathematical Functions appendix holds ULP tables.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from html.parser import HTMLParser
from typing import Any

from simdref.urls import resolve_url

DOC_BUNDLE_FORMAT = "cuda-docs-v1"

_MATH_INDEX_URL = "https://docs.nvidia.com/cuda/cuda-math-api/"
_GUIDE_URL = (
    "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html"
)
_GUIDE_ARCHIVE_URL = (
    "https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html"
)
_ULP_URL = (
    "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html"
)
_LIBDEVICE_INDEX_URL = (
    "https://docs.nvidia.com/cuda/libdevice-users-guide/libdevice_users_guide/functions.html"
)


def _clean(s: str) -> str:
    """Collapse whitespace runs to single spaces and strip the ends."""
    return " ".join(s.split())


def _is_ident_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _is_identifier(s: str) -> bool:
    if not s or (not s[0].isalpha() and s[0] != "_"):
        return False
    for c in s:
        if not _is_ident_char(c):
            return False
    return True


def _identifier_tokens(text: str) -> list[str]:
    """All whole identifier tokens of the text, in order of appearance."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isalpha() or c == "_":
            j = i
            while j < n and _is_ident_char(text[j]):
                j += 1
            out.append(text[i:j])
            i = j
        else:
            i += 1
    return out


def _has_token(text: str, name: str) -> bool:
    """True when name appears in text as a whole identifier token."""
    pos = 0
    while True:
        i = text.find(name, pos)
        if i == -1:
            return False
        left_ok = i == 0 or not _is_ident_char(text[i - 1])
        j = i + len(name)
        right_ok = j == len(text) or not _is_ident_char(text[j])
        if left_ok and right_ok:
            return True
        pos = i + 1


# TeX operator commands with one Unicode character each. Only the
# commands the CUDA accuracy fields use. The mapping is TeX syntax.
_TEX_OPERATORS = {
    "cdot": "\u00b7",
    "in": "\u2208",
    "lfloor": "\u230a",
    "notin": "\u2209",
    "rfloor": "\u230b",
}
_SUPERSCRIPT = {
    "0": "\u2070",
    "1": "\u00b9",
    "2": "\u00b2",
    "3": "\u00b3",
    "4": "\u2074",
    "5": "\u2075",
    "6": "\u2076",
    "7": "\u2077",
    "8": "\u2078",
    "9": "\u2079",
    "+": "\u207a",
    "-": "\u207b",
    ".": "\u00b7",
}
_SUBSCRIPT = {
    "0": "\u2080",
    "1": "\u2081",
    "2": "\u2082",
    "3": "\u2083",
    "4": "\u2084",
    "5": "\u2085",
    "6": "\u2086",
    "7": "\u2087",
    "8": "\u2088",
    "9": "\u2089",
    "+": "\u208a",
    "-": "\u208b",
    ".": "\u00b7",
}


def _tex_command(name: str, unknown: list[str] | None) -> str:
    """Unicode form of one TeX command, or its source when it has none."""
    mapped = _TEX_OPERATORS.get(name)
    if mapped is None:
        kind = "CAPITAL" if name[:1].isupper() else "SMALL"
        try:
            mapped = unicodedata.lookup(f"GREEK {kind} LETTER {name.upper()}")
        except KeyError:
            mapped = None
    if mapped is None:
        if unknown is not None:
            unknown.append("\\" + name)
        return "\\" + name
    return mapped


def _tex_script_body(text: str, i: int) -> tuple[str, int]:
    """The script body at i: the content of one brace group or one character.

    shortcut: a brace escaped as ``\\{`` counts as a group brace, the doc
    fields hold none; switch to an escape-aware count when one appears.
    """
    if i < len(text) and text[i] == "{":
        depth = 0
        j = i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1 : j], j + 1
            j += 1
        return text[i + 1 :], len(text)
    return text[i : i + 1], i + 1


def tex_to_text(text: str, unknown: list[str] | None = None) -> str:
    """Readable text for the TeX math source of one doc field.

    Drops the ``\\(`` and ``\\)`` delimiters. Maps a Greek letter or an
    operator command to its Unicode character. Maps a script to the
    superscript or subscript characters where every character has one,
    else writes ``^(...)`` or ``_(...)``. A command with no Unicode form
    stays as source and goes into ``unknown`` when given.
    """
    out = []
    i = 0
    n = len(text)
    in_math = False
    while i < n:
        c = text[i]
        if c == "\\":
            j = i + 1
            if j < n and text[j].isalpha():
                k = j
                while k < n and text[k].isalpha():
                    k += 1
                out.append(_tex_command(text[i + 1 : k], unknown))
                i = k
                continue
            if j < n and text[j] in "()":
                in_math = text[j] == "("
                i = j + 1
                continue
            out.append(text[i : j + 1])
            if unknown is not None:
                unknown.append(text[i : j + 1])
            i = j + 1
            continue
        if c in "^_" and in_math:
            inner, i = _tex_script_body(text, i + 1)
            forms = _SUPERSCRIPT if c == "^" else _SUBSCRIPT
            if inner and all(ch in forms for ch in inner):
                out.append("".join(forms[ch] for ch in inner))
            else:
                out.append(c + "(" + tex_to_text(inner, unknown) + ")")
            continue
        if c in "{}":
            i += 1
            continue
        out.append("\u2212" if in_math and c == "-" else c)
        i += 1
    return "".join(out)


class _FunctionEntryParser(HTMLParser):
    """Collects Sphinx ``dl cpp function`` entries from one page."""

    def __init__(self):
        super().__init__()
        self.entries = []
        self._dl_depth = 0
        self._div_depth = 0
        self._p_depth = 0
        self._in_func = False
        self._func_dl_depth = 0
        self._in_func_dt = False
        self._in_func_dd = False
        self._in_fieldlist = False
        self._fieldlist_dl_depth = 0
        self._in_field_dt = False
        self._in_field_dd = False
        self._field_name = None
        self._in_admonition = False
        self._admonition_div_depth = 0
        self._adm_kind = None
        self._in_adm_title = False
        self._in_headerlink = False
        self._in_sig_name = False
        self._sig_buf = []
        self._desc_paras = []
        self._para_buf = None
        self._ret_buf = []
        self._param_buf = []
        self._adm_buf = []
        self._admonitions = []
        self._name_buf = []
        self._math_fields: set[str] = set()
        self._math_open = False
        self._anchor = None
        self._dt_id = None
        self._func_name = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "dl":
            self._dl_depth += 1
            cls = a.get("class") or ""
            if not self._in_func and "cpp" in cls and "function" in cls:
                self._start_function()
            elif self._in_func and "field-list" in cls:
                self._in_fieldlist = True
                self._fieldlist_dl_depth = self._dl_depth
                self._field_name = None
        elif tag == "div":
            self._div_depth += 1
            if self._in_func and not self._in_admonition:
                cls = a.get("class") or ""
                if "admonition" in cls:
                    self._in_admonition = True
                    self._admonition_div_depth = self._div_depth
                    if "note" in cls:
                        self._adm_kind = "note"
                    elif "seealso" in cls:
                        self._adm_kind = "seealso"
                    else:
                        self._adm_kind = None
                    self._adm_buf = []
        elif tag == "dt":
            if self._in_func and self._dl_depth == self._func_dl_depth:
                self._in_func_dt = True
                if a.get("id"):
                    self._dt_id = a["id"]
            elif (
                self._in_func and self._in_fieldlist and self._dl_depth == self._fieldlist_dl_depth
            ):
                self._in_field_dt = True
                self._field_name = None
        elif tag == "dd":
            if self._in_func and self._dl_depth == self._func_dl_depth and not self._in_fieldlist:
                self._in_func_dd = True
            elif (
                self._in_func and self._in_fieldlist and self._dl_depth == self._fieldlist_dl_depth
            ):
                self._in_field_dd = True
        elif tag == "a":
            if "headerlink" in (a.get("class") or ""):
                self._in_headerlink = True
        elif tag == "span":
            cls = a.get("class") or ""
            if "math" in cls:
                self._math_open = True
            if self._in_func_dt:
                if "target" in cls:
                    self._anchor = a.get("id")
                if "sig-name" in cls and "descname" in cls:
                    self._in_sig_name = True
                    self._name_buf = []
        elif tag == "p":
            self._p_depth += 1
            if self._p_depth == 1 and self._target_name() == "description":
                self._para_buf = []
            if self._in_admonition and "admonition-title" in (a.get("class") or ""):
                self._in_adm_title = True

    def handle_endtag(self, tag):
        if tag == "dl":
            if self._in_fieldlist and self._dl_depth == self._fieldlist_dl_depth:
                self._in_fieldlist = False
                self._in_field_dt = False
                self._in_field_dd = False
                self._field_name = None
            if self._in_func and self._dl_depth == self._func_dl_depth:
                self._finish_function()
            self._dl_depth -= 1
        elif tag == "div":
            if self._in_admonition and self._div_depth == self._admonition_div_depth:
                text = _clean("".join(self._adm_buf))
                if self._adm_kind in ("note", "seealso") and text:
                    self._admonitions.append(text)
                self._in_admonition = False
                self._adm_kind = None
                self._in_adm_title = False
            self._div_depth -= 1
        elif tag == "dt":
            self._in_func_dt = False
            self._in_field_dt = False
        elif tag == "dd":
            if self._in_func_dd and self._dl_depth == self._func_dl_depth:
                self._in_func_dd = False
            self._in_field_dd = False
        elif tag == "a":
            self._in_headerlink = False
        elif tag == "span":
            if self._math_open:
                self._math_open = False
                key = self._target_name()
                if key:
                    self._math_fields.add(key)
            if self._in_sig_name:
                self._in_sig_name = False
                if self._name_buf:
                    self._func_name = "".join(self._name_buf).strip()
        elif tag == "p":
            if self._p_depth == 1 and self._para_buf is not None:
                self._desc_paras.append("".join(self._para_buf))
                self._para_buf = None
            else:
                buf = self._text_target()
                if buf is not None:
                    buf.append(" ")
            self._in_adm_title = False
            self._p_depth -= 1

    def handle_data(self, data):
        if not self._in_func or self._in_headerlink:
            return
        if self._in_func_dt:
            self._sig_buf.append(data)
            if self._in_sig_name:
                self._name_buf.append(data)
            return
        if self._in_field_dt:
            if data.strip():
                self._field_name = data.strip()
            return
        buf = self._text_target()
        if buf is not None:
            buf.append(data)

    def _target_name(self) -> str | None:
        """Logical field of the text captured in the current context."""
        if self._in_fieldlist and self._in_field_dd:
            if self._field_name == "Returns":
                return "return_value"
            if self._field_name == "Parameters":
                return "parameters"
            return None
        if self._in_admonition:
            return None if self._in_adm_title else "accuracy"
        if self._in_func_dd and self._p_depth > 0:
            return "description"
        return None

    def _text_target(self):
        """Text buffer for data in the current capture context.

        Data joins without a space. The end of a paragraph adds the
        gap, so nested spans keep their own spacing.
        """
        key = self._target_name()
        if key is None:
            return None
        if key == "description":
            return self._para_buf
        return {
            "return_value": self._ret_buf,
            "parameters": self._param_buf,
            "accuracy": self._adm_buf,
        }[key]

    def _start_function(self):
        self._in_func = True
        self._func_dl_depth = self._dl_depth
        self._sig_buf = []
        self._desc_paras = []
        self._para_buf = None
        self._ret_buf = []
        self._param_buf = []
        self._adm_buf = []
        self._admonitions = []
        self._name_buf = []
        self._math_fields = set()
        self._math_open = False
        self._anchor = None
        self._dt_id = None
        self._func_name = None
        self._in_func_dt = False
        self._in_func_dd = False
        self._in_fieldlist = False
        self._in_admonition = False
        self._adm_kind = None
        self._field_name = None

    def _finish_function(self):
        self._in_func = False
        self._in_func_dt = False
        self._in_func_dd = False
        self._in_fieldlist = False
        self._in_admonition = False
        name = self._func_name or ""
        # libdevice pages mark names with the LLVM "@" prefix.
        name = name.removeprefix("@")
        accuracy = []
        for text in self._admonitions:
            low = text.lower()
            if "accuracy" in low or "ulp" in low:
                accuracy.append(text)
        # The first paragraph of the dd is the brief; the paragraphs
        # after it are the detailed description. The description keeps
        # the detail only when one follows the brief.
        paras = [text for text in (_clean(p) for p in self._desc_paras) if text]
        if len(paras) > 1:
            summary, description = paras[0], " ".join(paras[1:])
        elif paras:
            summary = description = paras[0]
        else:
            summary = description = ""
        self.entries.append(
            {
                "name": name,
                "anchor": self._anchor or self._dt_id or "",
                "signature": _clean("".join(self._sig_buf)),
                "summary": summary,
                "description": description,
                "return_value": _clean("".join(self._ret_buf)),
                "parameters": _clean("".join(self._param_buf)),
                "accuracy": _clean(" ".join(accuracy)),
                "math": sorted(self._math_fields),
            }
        )
        self._func_name = None


def parse_math_api_page(html: str) -> dict[str, dict[str, str]]:
    """Parse one Sphinx function page into a name to entry mapping.

    The page holds ``dl cpp function`` entries. The returned mapping
    holds one entry per function name. A name with several entries
    keeps the first one in document order. Each entry carries the C.1
    blocks of its dd, in page order, plus the page heading.
    """
    parser = _FunctionEntryParser()
    parser.feed(html)
    parser.close()
    from simdref.docblocks import build_tree, math_entry_views, page_title

    root = build_tree(html)
    views = math_entry_views(root)["entries"]
    title = page_title(root)
    out = {}
    for index, e in enumerate(parser.entries):
        name = e["name"]
        if not name or name in out:
            continue
        view = views[index] if index < len(views) else {}
        blocks = view.get("blocks") or []
        out[name] = {
            "anchor": e["anchor"],
            "signature": e["signature"],
            "summary": e["summary"],
            "description": e["description"],
            "return_value": e["return_value"],
            "parameters": e["parameters"],
            "accuracy": e["accuracy"],
            "math": list(e["math"]),
            "title": title,
            "heading": name,
            "sections": [
                {
                    "heading": "",
                    "shared": False,
                    "blocks": [{**block, "owner": name} for block in blocks],
                }
            ],
        }
        if "signature_runs" in view:
            out[name]["signature_runs"] = view["signature_runs"]
    return out


def parse_libdevice_page(html: str) -> dict[str, dict[str, str]]:
    """libdevice function pages use the same Sphinx shape."""
    return parse_math_api_page(html)


class _GuideParser(HTMLParser):
    """Collects code blocks and paragraphs per section of a guide page."""

    def __init__(self):
        super().__init__()
        # Each event is ("pre" | "p" | "stop", open section ids, text).
        # A "stop" event marks an admonition, an example sub-heading or
        # an examples child section. Math spans record the event index
        # of the capture that held them in math_events.
        self.events = []
        self.math_events: set[int] = set()
        self._frames = []
        self._div_depth = 0
        self._capture = None
        self._capture_math = False
        self._math_open = False
        self._buf = []

    def _ids(self) -> tuple:
        return tuple(f[1] for f in self._frames if f[1])

    def _emit(self, kind: str, text: str) -> None:
        index = len(self.events)
        self.events.append((kind, self._ids(), text))
        if self._capture_math:
            self.math_events.add(index)
        self._capture_math = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag == "section":
            self._frames.append(("section", a.get("id"), None))
            if _is_example_section(a.get("id") or ""):
                self.events.append(("stop", self._ids(), "examples"))
        elif tag == "div":
            self._div_depth += 1
            if "section" in cls:
                self._frames.append(("div", a.get("id"), self._div_depth))
                if _is_example_section(a.get("id") or ""):
                    self.events.append(("stop", self._ids(), "examples"))
            elif "admonition" in cls:
                self.events.append(("stop", self._ids(), "admonition"))
        elif tag == "span":
            if "math" in cls:
                self._math_open = True
        elif tag == "pre":
            self._capture = "pre"
            self._buf = []
        elif tag == "p":
            if "rubric" in cls:
                self.events.append(("stop", self._ids(), "rubric"))
            if self._capture is None:
                self._capture = "p"
                self._buf = []
        if a.get("id"):
            self.events.append(("id", self._ids(), a["id"]))

    def handle_endtag(self, tag):
        if tag == "section":
            if self._frames and self._frames[-1][0] == "section":
                self._frames.pop()
        elif tag == "div":
            if (
                self._frames
                and self._frames[-1][0] == "div"
                and self._frames[-1][2] == self._div_depth
            ):
                self._frames.pop()
            self._div_depth -= 1
        elif tag == "span":
            if self._math_open:
                self._math_open = False
                if self._capture is not None:
                    self._capture_math = True
        elif tag in ("pre", "p") and self._capture == tag:
            self._emit(tag, "".join(self._buf))
            self._capture = None

    def handle_data(self, data):
        if self._capture is not None:
            self._buf.append(data)


def _code_function_names(text: str) -> list[str]:
    """Function names used in a code block, in order of appearance.

    A function name is an identifier token directly followed by "(". No
    name list is used. Other identifiers in the block are not names.
    """
    names = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isalpha() or c == "_":
            j = i
            while j < n and _is_ident_char(text[j]):
                j += 1
            k = j
            while k < n and text[k] in " \t":
                k += 1
            if k < n and text[k] == "(":
                name = text[i:j]
                if name not in names:
                    names.append(name)
            i = j
        else:
            i += 1
    return names


def _prose_names(text: str) -> list[str]:
    """Function names mentioned in one prose paragraph.

    A name is an identifier token that starts with "__", or a token of
    two or more characters directly followed by "(". No name list is
    used. A one character token before "(" is a math variable, not a
    function name.
    """
    names = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isalpha() or c == "_":
            j = i
            while j < n and _is_ident_char(text[j]):
                j += 1
            k = j
            while k < n and text[k] in " \t":
                k += 1
            token = text[i:j]
            if token.startswith("__") or (len(token) > 1 and k < n and text[k] == "("):
                if token not in names:
                    names.append(token)
            i = j
        else:
            i += 1
    return names


def _is_generated_id(value: str) -> bool:
    """A docutils generated target id, like id17."""
    return value.startswith("id") and value[2:].isdigit()


def _is_example_section(sid: str) -> bool:
    """A child section that holds the examples of the section above it."""
    return "example" in sid


def _is_detail_section(sid: str) -> bool:
    """A child section that holds detail of the section above it."""
    return (
        sid in {"synopsis", "description", "example", "restrictions"}
        or sid.endswith("-synopsis")
        or sid.endswith("-description")
        or sid.endswith("-example")
    )


def _climbed_chain(ids: tuple) -> tuple:
    """The section chain of a hit, with detail children removed."""
    chain = list(ids)
    while len(chain) > 1 and _is_detail_section(chain[-1]):
        chain.pop()
    return tuple(chain)


def _anchor_for(events: list, index: int, chain: tuple) -> str:
    """The anchor id for one hit.

    The page keeps a stable id as a target element inside the section,
    for example "trap-function" in section "trap". Take the last
    non-generated id seen inside the section before the hit. A section
    without one anchors on its own id.
    """
    best = chain[-1] if chain else ""
    for k in range(index):
        kind, ids, value = events[k]
        if kind == "id" and ids == chain and not _is_generated_id(value):
            best = value
    return best


def _skip_paren_group(text: str, start: int) -> int:
    """Return the index just past the balanced ``( ... )`` group at *start*."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def drop_attribute_groups(text: str) -> str:
    """Remove every ``__attribute__(( ... ))`` group from the text."""
    out = []
    i = 0
    while True:
        a = text.find("__attribute__(", i)
        if a < 0:
            return "".join(out) + text[i:]
        out.append(text[i:a])
        i = _skip_paren_group(text, a + len("__attribute__"))


_CXX_DECL_QUALIFIERS = frozenset(("static", "inline", "extern", "constexpr"))
_STATEMENT_KEYWORDS = frozenset(
    (
        "if",
        "else",
        "for",
        "while",
        "switch",
        "return",
        "do",
        "goto",
        "break",
        "continue",
        "case",
        "default",
        "sizeof",
        "try",
        "catch",
        "throw",
    )
)


def declaration_qualifiers(text: str) -> set[str]:
    """The qualifier tokens of one C source text.

    The C++ declaration qualifiers plus every token a qualifier macro
    of the text expands to. A macro qualifies when its expansion holds
    only C++ qualifiers and ``__``-prefixed tokens: the ``__DEVICE__``
    pattern. No hand list of CUDA macros. The text must have its
    comments stripped.
    """
    out: set[str] = set(_CXX_DECL_QUALIFIERS)
    buf = ""
    pending = text.split("\n") + [""]
    for raw in pending:
        line = raw.rstrip()
        buf = buf + " " + line.strip() if buf else line
        if line.endswith("\\") and not line.endswith("\\\\"):
            buf = buf[:-1]
            continue
        define = buf.strip()
        buf = ""
        if not define.startswith("#define "):
            continue
        tokens = drop_attribute_groups(define[len("#define ") :]).split()
        if tokens and all(
            _is_identifier(token) and (token in _CXX_DECL_QUALIFIERS or token.startswith("__"))
            for token in tokens
        ):
            out.update(tokens)
    return out


def _call_params(text: str, start: int) -> str | None:
    """Parameter text of the call group that opens at *start*, else None."""
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None


def normalize_declaration(chunk: str, qualifiers: set[str]) -> tuple[list[str], str, str] | None:
    """One declaration normalized: (return type tokens, name, params).

    Joins the line breaks, splits on whitespace, drops the attribute
    groups and the leading qualifier tokens, then reads the return
    type, the name before the call parens and the parameter text. A
    leading ``__``-prefixed token is a qualifier when a type follows it,
    so a macro qualifier needs no name list. None when the text declares
    nothing: no call parens, a statement keyword shape, or a return
    type that is not a type.
    """
    text = drop_attribute_groups(" ".join(chunk.split()))
    open_at = text.find("(")
    if open_at < 0:
        return None
    params = _call_params(text, open_at)
    if params is None:
        return None
    lead = text[:open_at].rstrip()
    stop = len(lead)
    while stop > 0 and (lead[stop - 1].isalnum() or lead[stop - 1] == "_"):
        stop -= 1
    name = lead[stop:]
    if not name or name[0].isdigit() or name in _STATEMENT_KEYWORDS:
        return None
    tokens = lead[:stop].split()
    cut = 0
    while cut < len(tokens) - 1 and (tokens[cut] in qualifiers or tokens[cut].startswith("__")):
        cut += 1
    type_tokens = tokens[cut:]
    for token in type_tokens:
        if token in _STATEMENT_KEYWORDS:
            return None
        if any(not (c.isalnum() or c == "_" or c in "*&<>:") for c in token):
            return None
    return type_tokens, name, " ".join(params.split())


def declares_name(text: str, name: str) -> bool:
    """The text declares the name: a return type precedes its call.

    Routed through the normalizer. A declaration has a return type
    before the name. An example call has nothing before it. The doc
    join uses this to tell a function entry from an example line.
    """
    normalized = normalize_declaration(text, declaration_qualifiers(text))
    return normalized is not None and normalized[1] == name and bool(normalized[0])


def parse_guide_page(html: str) -> dict[str, dict[str, str]]:
    """Parse a Programming Guide page into a name to entry mapping.

    A name is a function name in a ``pre`` code block of a section, or a
    name mentioned in the prose of a section. The section is a
    ``section`` or ``div.section`` element with an id. The entry holds
    the anchor id, the code line with the name as signature (empty for a
    prose name), and the description text of the section after the code
    block. A name in several sections keeps the first entry in document
    order. A declaring code line wins over an example call line. Code
    block hits win over prose hits.
    """
    parser = _GuideParser()
    parser.feed(html)
    parser.close()
    events = parser.events

    def _pick_signature(lines, name):
        for line in lines:
            if not _has_token(line, name):
                continue
            clean_line = _clean(line)
            if declares_name(clean_line, name):
                return clean_line
        return ""

    out = {}
    for require_decl in (True, False):
        for i, (kind, ids, text) in enumerate(events):
            if kind != "pre" or not ids:
                continue
            chain = _climbed_chain(ids)
            anchor = _anchor_for(events, i, chain)
            lines = text.splitlines()
            for name in _code_function_names(text):
                if name in out:
                    continue
                signature = _pick_signature(lines, name)
                if require_decl and not declares_name(signature, name):
                    continue
                desc = []
                desc_idx = []
                for k in range(i + 1, len(events)):
                    kind2, ids2, text2 = events[k]
                    if kind2 == "stop" and ids2[: len(chain)] == chain:
                        break
                    if kind2 == "pre" and ids2[: len(chain)] == chain:
                        # A later code block heads an example. The run ends
                        # before it and its lead-in paragraphs go with it.
                        while desc:
                            between = events[desc_idx[-1] + 1 : k]
                            adjacent = all(item[0] == "id" for item in between)
                            if desc[-1].endswith(":") or adjacent:
                                desc.pop()
                                desc_idx.pop()
                            else:
                                break
                        break
                    if kind2 != "p" or not text2.strip():
                        continue
                    chain2 = _climbed_chain(ids2)
                    if chain2[: len(chain)] == chain:
                        desc.append(text2)
                        desc_idx.append(k)
                out[name] = {
                    "anchor": anchor,
                    "signature": signature,
                    "description": _clean(" ".join(desc)),
                    "math": ["description"]
                    if any(k in parser.math_events for k in desc_idx)
                    else [],
                    "section": chain[-1] if chain else "",
                }
    for i, (kind, ids, text) in enumerate(events):
        if kind != "p" or not ids or not text.strip():
            continue
        chain = _climbed_chain(ids)
        anchor = _anchor_for(events, i, chain)
        for name in _prose_names(text):
            if name in out:
                continue
            out[name] = {
                "anchor": anchor,
                "signature": "",
                "description": _clean(text),
                "math": ["description"] if i in parser.math_events else [],
                "section": chain[-1] if chain else "",
            }
    from simdref.docblocks import build_tree, cut_runs, guide_views, page_title

    root = build_tree(html)
    views = guide_views(root)
    title = page_title(root)
    for name, entry in list(out.items()):
        section = entry.pop("section", "")
        view = views.get(section) or {}
        entry["title"] = title
        entry["heading"] = str(view.get("heading") or "")
        entry["summary"] = str((view.get("briefs") or {}).get(name) or "")
        entry["sections"] = cut_runs(view.get("runs") or [], name)
    for section, view in views.items():
        for name, brief in (view.get("briefs") or {}).items():
            if name in out:
                continue
            out[name] = {
                "anchor": section,
                "signature": "",
                "description": "",
                "math": [],
                "title": title,
                "heading": str(view.get("heading") or ""),
                "summary": str(brief or ""),
                "sections": cut_runs(view.get("runs") or [], name),
            }
    return out


class _UlpTableParser(HTMLParser):
    """Collects captioned tables as lists of rows of cell texts."""

    def __init__(self):
        super().__init__()
        self.tables = []
        self._table = None
        self._row = None
        self._cell = None
        self._cell_is_th = False
        self._cell_math = False
        self._math_open = False
        self._in_caption = False
        self._in_headerlink = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._table = {"caption": None, "rows": []}
        elif tag == "caption" and self._table is not None:
            self._in_caption = True
            self._table["caption"] = ""
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._cell_is_th = tag == "th"
            self._cell_math = False
        elif tag == "span":
            if "math" in (a.get("class") or ""):
                self._math_open = True
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")
        elif tag == "a" and self._in_caption:
            if "headerlink" in (a.get("class") or ""):
                self._in_headerlink = True

    def handle_endtag(self, tag):
        if tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None
        elif tag == "caption":
            self._in_caption = False
        elif tag == "tr" and self._row is not None:
            if self._table is not None:
                self._table["rows"].append(self._row)
            self._row = None
        elif tag == "span":
            if self._math_open:
                self._math_open = False
                if self._cell is not None:
                    self._cell_math = True
        elif tag in ("td", "th") and self._cell is not None:
            if self._row is not None:
                self._row.append((self._cell_is_th, _clean("".join(self._cell)), self._cell_math))
            self._cell = None
        elif tag == "a":
            self._in_headerlink = False

    def handle_data(self, data):
        if self._in_caption and self._table is not None:
            if not self._in_headerlink:
                self._table["caption"] += data
        elif self._cell is not None:
            self._cell.append(data)


def parse_ulp_tables(html: str) -> dict[str, dict[str, Any]]:
    """Parse the ULP tables of a page into a name token to entry map.

    Each ``table`` with a ``caption`` takes part. The header row names
    the function column and the accuracy column. Each data row maps
    every whole identifier token of its function cell to the text of
    the accuracy cell and the table caption. A row without an accuracy
    cell is skipped. A name token in several rows keeps the first row.
    The entry keeps the page title, the header row and every matching
    data row, for the C.1 table block.
    """
    parser = _UlpTableParser()
    parser.feed(html)
    parser.close()
    from simdref.docblocks import build_tree, page_title

    title = page_title(build_tree(html))
    out: dict[str, dict[str, Any]] = {}
    for table in parser.tables:
        if table["caption"] is None:
            continue
        caption = _clean(table["caption"])
        func_col = 0
        acc_col = None
        saw_header = False
        header_cells: list[str] = []
        for row in table["rows"]:
            raw_cells = [text for _, text, _ in row]
            # The block cells read math as unicode, the way inline runs
            # render it. The accuracy field keeps the raw TeX, so the
            # math block can render it again.
            cells = [_clean(tex_to_text(text)) if math else text for _, text, math in row]
            is_header = any(is_th for is_th, _, _ in row)
            if is_header and not saw_header:
                saw_header = True
                header_cells = cells
                for idx, (is_th, text, _) in enumerate(row):
                    low = text.lower()
                    if is_th and "function" in low:
                        func_col = idx
                        break
                for idx, (is_th, text, _) in enumerate(row):
                    low = text.lower()
                    if is_th and ("ulp" in low or "accuracy" in low):
                        acc_col = idx
                        break
                continue
            if is_header or not cells:
                continue
            fidx = func_col if func_col < len(raw_cells) else 0
            acc = None
            acc_math = False
            if acc_col is not None and acc_col < len(raw_cells) and acc_col != fidx:
                acc = raw_cells[acc_col]
                acc_math = row[acc_col][2]
            else:
                # No accuracy column in the header. Take the first row
                # cell that names an accuracy or a ULP bound.
                for idx, text in enumerate(raw_cells):
                    if idx == fidx:
                        continue
                    low = text.lower()
                    if "ulp" in low or "accuracy" in low:
                        acc = text
                        acc_math = row[idx][2]
                        break
            if acc is None:
                continue
            for token in _identifier_tokens(raw_cells[fidx]):
                if token not in out:
                    out[token] = {
                        "accuracy": acc,
                        "caption": caption,
                        "math": ["accuracy"] if acc_math else [],
                        "title": title,
                        "header": header_cells,
                        "rows": [cells],
                    }
                elif out[token]["caption"] == caption:
                    out[token]["rows"].append(cells)
    return out


class _LinkParser(HTMLParser):
    """Collects (text, href) pairs of anchor elements."""

    def __init__(self):
        super().__init__()
        self.links = []
        self._href = None
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            a = dict(attrs)
            self._href = a.get("href")
            self._buf = []

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((_clean("".join(self._buf)), self._href))
            self._href = None

    def handle_data(self, data):
        if self._href is not None:
            self._buf.append(data)


def parse_libdevice_index(html: str) -> dict[str, str]:
    """Parse the libdevice function list into a name to href mapping.

    The function pages carry names like "function_...html". The link
    text holds the name as its last word.
    """
    parser = _LinkParser()
    parser.feed(html)
    parser.close()
    out = {}
    for text, href in parser.links:
        fragless = href.partition("#")[0]
        base = fragless.rsplit("/", 1)[-1]
        if not base.startswith("function_") or not base.endswith(".html"):
            continue
        name = text.rsplit(" ", 1)[-1] if text else ""
        if not _is_identifier(name):
            continue
        if name not in out:
            out[name] = href
    return out


def _page_hrefs(html: str) -> list[str]:
    """All href values of the page, in document order."""
    out = []
    pos = 0
    while True:
        i = html.find('href="', pos)
        if i == -1:
            break
        start = i + len('href="')
        end = html.find('"', start)
        if end == -1:
            break
        out.append(html[start:end])
        pos = end + 1
    return out


def build_docs(fetch, names: set[str]) -> dict:
    """Fetch and store the doc pages at build time.

    ``fetch(url)`` returns the page HTML. The result is a bundle dict
    with the raw HTML of each page. The caller writes the bundle to a
    gitignored path. Nothing here keeps a list of function names.
    """
    pages = []
    failed = []

    def _try(url):
        # One dead page must not block the bundle or the catalog build.
        try:
            return fetch(url)
        except Exception as exc:
            failed.append({"url": url, "error": str(exc)})
            return ""

    def _record(kind, url, html):
        if not html:
            return
        pages.append(
            {
                "kind": kind,
                "url": url,
                "sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
                "html": html,
            }
        )

    index_html = _try(_MATH_INDEX_URL)
    math_urls = []
    for href in _page_hrefs(index_html):
        fragless = href.partition("#")[0]
        if not fragless.rsplit("/", 1)[-1].startswith("group__CUDA__MATH"):
            continue
        url = resolve_url(_MATH_INDEX_URL, fragless)
        if url not in math_urls:
            math_urls.append(url)
    math_htmls = []
    for url in math_urls:
        html = _try(url)
        math_htmls.append(html)
        _record("math", url, html)

    guide_html = _try(_GUIDE_URL)
    _record("guide", _GUIDE_URL, guide_html)
    archive_html = _try(_GUIDE_ARCHIVE_URL)
    _record("guide-archive", _GUIDE_ARCHIVE_URL, archive_html)
    ulp_html = _try(_ULP_URL)
    _record("ulp", _ULP_URL, ulp_html)
    lib_index_html = _try(_LIBDEVICE_INDEX_URL)
    _record("libdevice-index", _LIBDEVICE_INDEX_URL, lib_index_html)

    math_matched = set()
    for html in math_htmls:
        math_matched |= set(parse_math_api_page(html))
    guide_matched = set(parse_guide_page(guide_html))
    guide_matched |= set(parse_guide_page(archive_html))

    lib_index = parse_libdevice_index(lib_index_html)
    lib_urls = []
    for name in sorted(names):
        if name in math_matched or name in guide_matched:
            continue
        if name not in lib_index:
            continue
        url = resolve_url(_LIBDEVICE_INDEX_URL, lib_index[name].partition("#")[0])
        if url in lib_urls:
            continue
        lib_urls.append(url)
        _record("libdevice", url, _try(url))

    return {
        "format": DOC_BUNDLE_FORMAT,
        "pages": pages,
        "metadata": {
            "names": len(names),
            "pages": len(pages),
            "math_pages": len(math_urls),
            "libdevice_pages": len(lib_urls),
            "matched_math": len(names & math_matched),
            "matched_guide": len(names & guide_matched),
            "failed": failed,
        },
    }


_KIND_KEYS = {
    "math": "math",
    "libdevice": "libdevice",
    "guide": "guide",
    "guide-archive": "guide_archive",
    "ulp": "ulp",
    "libdevice-index": "libdevice_index",
}

_PARSERS = {
    "math": parse_math_api_page,
    "libdevice": parse_libdevice_page,
    "guide": parse_guide_page,
    "guide-archive": parse_guide_page,
    "ulp": parse_ulp_tables,
    "libdevice-index": parse_libdevice_index,
}


def parse_docs_bundle(text: str) -> dict:
    """Parse a bundle JSON string into per kind parse results.

    The result holds one parse mapping per kind, plus "urls" and
    "sha256" per kind. Kinds with several pages keep the first page in
    those two maps. All pages stay in the bundle "pages" list.
    """
    data = json.loads(text)
    if data.get("format") != DOC_BUNDLE_FORMAT:
        raise ValueError("unsupported bundle format: " + repr(data.get("format")))
    out = {
        "math": {},
        "libdevice": {},
        "guide": {},
        "guide_archive": {},
        "ulp": {},
        "libdevice_index": {},
        "urls": {},
        "sha256": {},
    }
    for page in data.get("pages", []):
        kind = page.get("kind")
        if kind not in _PARSERS:
            raise ValueError("unknown page kind: " + repr(kind))
        key = _KIND_KEYS[kind]
        for name, entry in _PARSERS[kind](page["html"]).items():
            if name in out[key]:
                continue
            if isinstance(entry, dict):
                entry = dict(entry)
                entry["page"] = page["url"]
            out[key][name] = entry
        out["urls"].setdefault(key, page["url"])
        out["sha256"].setdefault(key, page["sha256"])
    return out
