"""Doc block extraction: the C.1 block shape from the source page DOM.

One tiny element tree per page, then structure queries per source. No
regex, no per-family rules. Block order is document order in lists.

Block kinds (design C.1): paragraph, list, code, table, note, deflist,
math. Owners come from document containment (design C.2): a ``dl``
function term owns its item, a one-name declaration ``pre`` owns its
run, a child section owns what it declares, the rest is shared.
"""

from __future__ import annotations

from simdref.cuda_docs import _clean, _code_function_names, declares_name, tex_to_text

# Admonition class to the C.1 note kind.
_NOTE_KINDS = {
    "note": "note",
    "seealso": "seealso",
    "hint": "hint",
    "warning": "warning",
    "tip": "note",
    "important": "note",
    "attention": "warning",
    "caution": "warning",
    "danger": "warning",
}


class _Node:
    __slots__ = ("attrs", "kids", "tag")

    def __init__(self, tag: str, attrs: dict):
        self.tag = tag
        self.attrs = attrs
        self.kids: list = []

    def classes(self) -> list[str]:
        return (self.attrs.get("class") or "").split()


_VOID_TAGS = frozenset(
    [
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    ]
)


class _TreeBuilder:
    """A minimal element tree. Text nodes are str kids, in document order."""

    def __init__(self) -> None:
        self.root = _Node("doc", {})
        self._stack = [self.root]

    def start(self, tag: str, attrs: dict) -> None:
        node = _Node(tag, attrs)
        self._stack[-1].kids.append(node)
        if tag not in _VOID_TAGS:
            self._stack.append(node)

    def end(self, tag: str) -> None:
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def data(self, text: str) -> None:
        self._stack[-1].kids.append(text)


def build_tree(html: str) -> _Node:
    """The element tree of one HTML page."""
    from html.parser import HTMLParser

    class _Feed(HTMLParser):
        def __init__(self, builder: _TreeBuilder) -> None:
            super().__init__(convert_charrefs=True)
            self._b = builder

        def handle_starttag(self, tag, attrs):
            self._b.start(tag, dict(attrs))

        def handle_startendtag(self, tag, attrs):
            self._b.start(tag, dict(attrs))
            self._b.end(tag)

        def handle_endtag(self, tag):
            self._b.end(tag)

        def handle_data(self, data):
            self._b.data(data)

    builder = _TreeBuilder()
    feed = _Feed(builder)
    feed.feed(html)
    feed.close()
    return builder.root


def node_text(node) -> str:
    """Text of a node as the page renders it: text kids joined in order.

    A ``a.headerlink`` glyph is not text. Elsewhere tags never separate
    words, so the text kids concatenate as-is and the caller cleans.
    """
    parts: list[str] = []
    for kid in node.kids:
        if isinstance(kid, str):
            parts.append(kid)
        elif kid.tag == "a" and "headerlink" in kid.classes():
            continue
        else:
            parts.append(node_text(kid))
    return "".join(parts)


def heading_text(node) -> str:
    """Heading text minus the section number and the headerlink glyph."""
    parts: list[str] = []
    for kid in node.kids:
        if isinstance(kid, str):
            parts.append(kid)
        elif (
            kid.tag == "span"
            and "section-number" in kid.classes()
            or kid.tag == "a"
            and "headerlink" in kid.classes()
        ):
            continue
        else:
            parts.append(node_text(kid))
    return _clean("".join(parts))


_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8")


def _is_admonition(node) -> bool:
    return node.tag == "div" and "admonition" in node.classes()


def _is_highlight(node) -> bool:
    return node.tag == "div" and any(cls.startswith("highlight") for cls in node.classes())


def _is_section(node) -> bool:
    return node.tag == "section" or (node.tag == "div" and "section" in node.classes())


def _first_pre(node):
    """The first ``pre`` in the subtree, else None."""
    if node.tag == "pre":
        return node
    for kid in node.kids:
        if isinstance(kid, str):
            continue
        hit = _first_pre(kid)
        if hit is not None:
            return hit
    return None


def _code_text(node) -> str:
    """Verbatim text of a code block: line breaks kept, edges stripped."""
    text = node_text(_first_pre(node) or node)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def _note_block(node) -> dict:
    kind = "note"
    for cls in node.classes():
        if cls in _NOTE_KINDS:
            kind = _NOTE_KINDS[cls]
            break
    body = _Node("div", {})
    for kid in node.kids:
        if isinstance(kid, str):
            body.kids.append(kid)
        elif kid.tag == "p" and "admonition-title" in kid.classes():
            continue
        else:
            body.kids.append(kid)
    return _with_runs({"kind": "note", "note": kind, "text": _clean(node_text(body))}, body)


def _with_runs(block: dict, node) -> dict:
    """The block with its optional runs list, when the source marks up."""
    runs = inline_runs(node)
    if has_markup(runs):
        block["runs"] = runs
    return block


def _own_kids(node) -> list:
    """Kids of a list item minus its nested lists (manager decision 1)."""
    return [k for k in node.kids if isinstance(k, str) or k.tag not in ("ul", "ol")]


def _list_block(node) -> dict:
    """One list block.

    An item is a plain string when it has no inline markup and no
    nested list, else the object {text, runs?, blocks?}. A nested list
    goes into its item's ``blocks``. Order lives in ``items``. The
    ordered flag comes from the element tag of the list.
    """
    items = []
    for kid in node.kids:
        if isinstance(kid, str):
            continue
        nested = [sub for sub in kid.kids if not isinstance(sub, str) and sub.tag in ("ul", "ol")]
        own = _Node(kid.tag, kid.attrs)
        own.kids = _own_kids(kid)
        text = _clean(node_text(own))
        if not text and not nested:
            continue
        runs = inline_runs(own)
        if not nested and not has_markup(runs):
            items.append(text)
            continue
        item = {"text": text}
        if has_markup(runs):
            item["runs"] = runs
        if nested:
            blocks: list[dict] = []
            for sub in nested:
                blocks.extend(block_of(sub))
            item["blocks"] = blocks
        items.append(item)
    return {"kind": "list", "ordered": node.tag == "ol", "items": items}


def _has_math(node) -> bool:
    """True when a subtree holds a math span or math element."""
    for kid in node.kids:
        if isinstance(kid, str):
            continue
        if kid.tag == "math" or (kid.tag == "span" and "math" in kid.classes()):
            return True
        if _has_math(kid):
            return True
    return False


def _cell_text(cell) -> str:
    """Text of one table cell: math reads as unicode, as inline runs do."""
    text = _clean(node_text(cell))
    return _clean(tex_to_text(text)) if _has_math(cell) else text


def _table_block(node) -> dict:
    caption = ""
    header: list[list[str]] = []
    rows: list[list[str]] = []
    for kid in node.kids:
        if isinstance(kid, str):
            continue
        if kid.tag == "caption":
            caption = _clean(node_text(kid))
        elif kid.tag in ("thead", "tbody", "tfoot"):
            for row in kid.kids:
                if isinstance(row, str) or row.tag != "tr":
                    continue
                cells = [_cell_text(cell) for cell in row.kids if not isinstance(cell, str)]
                if any(cell.tag == "th" for cell in row.kids if not isinstance(cell, str)):
                    header.append(cells)
                else:
                    rows.append(cells)
        elif kid.tag == "tr":
            cells = [_cell_text(cell) for cell in kid.kids if not isinstance(cell, str)]
            rows.append(cells)
    return {"kind": "table", "caption": caption, "header": header, "rows": rows}


def _deflist_items(node) -> list[dict]:
    """{name, text} pairs of one ``dl``, nested ``dl`` items flattened in."""
    items: list[dict] = []
    term = None
    for kid in node.kids:
        if isinstance(kid, str):
            continue
        if kid.tag == "dt":
            term = _clean(node_text(kid))
        elif kid.tag == "dd" and term is not None:
            nested = [sub for sub in kid.kids if not isinstance(sub, str) and sub.tag == "dl"]
            if nested:
                for sub in nested:
                    items.extend(_deflist_items(sub))
            else:
                items.append(_with_runs({"name": term, "text": _clean(node_text(kid))}, kid))
            term = None
    return [item for item in items if item["name"] or item["text"]]


def _deflist_block(node) -> dict:
    return {"kind": "deflist", "items": _deflist_items(node)}


def _display_math(node) -> dict | None:
    """The math block of a standalone display math element, else None.

    Only a display math element that stands alone in the source becomes
    a math block. An inline ``span.math`` becomes a math run (manager
    decision 2).
    """
    if node.tag == "math" or (node.tag != "span" and "math" in node.classes()):
        raw = node_text(node).strip()
        return {"kind": "math", "text": tex_to_text(raw), "tex": raw}
    return None


_RUN_TAGS = {"em": "em", "i": "em", "strong": "strong", "b": "strong"}


def inline_runs(node) -> list[list]:
    """Inline runs of one text element: [kind, text], [kind, text, href]
    or ["math", readable, tex].

    Splits at ``code.docutils.literal``, ``span.math``, ``em``,
    ``strong`` and ``a``. A math run sits at its position and never
    merges: its text is the readable form while ``tex`` keeps the raw
    source. The plain ``text`` field stays the readable fallback.
    """
    out: list[list] = []

    def add(kind: str, text: str, href: str | None = None) -> None:
        if not text:
            return
        run = [kind, text] if href is None else [kind, text, href]
        if out and out[-1][0] == kind and len(out[-1]) == len(run):
            out[-1][1] += text
        else:
            out.append(run)

    def walk(n) -> None:
        for kid in n.kids:
            if isinstance(kid, str):
                add("text", kid)
                continue
            cls = kid.classes()
            if kid.tag == "a" and "headerlink" in cls:
                continue
            if kid.tag == "code" and "literal" in cls:
                add("code", node_text(kid))
            elif kid.tag == "span" and "math" in cls:
                raw = node_text(kid)
                out.append(["math", tex_to_text(raw), raw])
            elif kid.tag == "a":
                add("link", node_text(kid), kid.attrs.get("href") or "")
            elif kid.tag in _RUN_TAGS:
                add(_RUN_TAGS[kid.tag], node_text(kid))
            else:
                walk(kid)

    walk(node)
    return out


def has_markup(runs: list[list]) -> bool:
    """True when the runs carry more than plain text."""
    return any(run[0] != "text" for run in runs)


def highlight_lang(node, default: str = "text") -> str:
    """Code language of a block: its own ``highlight-<lang>`` class.

    The source names the language; no hand label table. A class of
    ``text`` names no language, so the source container decides (the
    caller's default: 'ptx' inside the PTX ISA source, 'text'
    elsewhere).
    """
    for cls in node.classes():
        if cls.startswith("highlight-"):
            name = cls[len("highlight-") :]
            return default if name in ("", "text") else name
    return default


def _tab_blocks(node, default_lang: str) -> list[dict]:
    """Blocks of an ``sd-tab-set``: one group per tab, in page order.

    Each code block of a tab carries ``caption`` = the tab label
    (manager decision 3).
    """
    out: list[dict] = []
    caption = ""
    for kid in node.kids:
        if isinstance(kid, str):
            continue
        if kid.tag == "label":
            caption = _clean(node_text(kid))
            continue
        if kid.tag == "div" and "sd-tab-content" in kid.classes():
            for block in blocks_of(kid, default_lang):
                if caption and block.get("kind") == "code":
                    block["caption"] = caption
                out.append(block)
        else:
            out.extend(block_of(kid, default_lang))
    return out


def block_of(node, default_lang: str = "text") -> list[dict]:
    """Blocks of one block-level element.

    A wrapper element splices its children's blocks. An unknown leaf
    with text becomes a paragraph, so no source text is dropped. A
    heading is structure: its text belongs to the section entry, so it
    is not a block.
    """
    if node.tag in _HEADING_TAGS:
        return []
    math = _display_math(node)
    if math is not None:
        return [math]
    if _is_admonition(node):
        return [_note_block(node)]
    if node.tag == "p":
        return [_with_runs({"kind": "paragraph", "text": _clean(node_text(node))}, node)]
    if node.tag in ("ul", "ol"):
        return [_list_block(node)]
    if node.tag == "dl":
        return [_deflist_block(node)]
    if node.tag == "table":
        return [_table_block(node)]
    if node.tag == "pre" or _is_highlight(node):
        return [
            {
                "kind": "code",
                "text": _code_text(node),
                "lang": highlight_lang(node, default_lang),
            }
        ]
    if node.tag == "div" and "sd-tab-set" in node.classes():
        return _tab_blocks(node, default_lang)
    if _is_section(node):
        return blocks_of(node, default_lang)
    out = []
    for kid in node.kids:
        if isinstance(kid, str):
            text = _clean(kid)
            if text:
                out.append({"kind": "paragraph", "text": text})
        else:
            out.extend(block_of(kid, default_lang))
    return out


def blocks_of(node, default_lang: str = "text") -> list[dict]:
    """Blocks of an element's children, in document order."""
    out: list[dict] = []
    for kid in node.kids:
        if isinstance(kid, str):
            text = _clean(kid)
            if text:
                out.append({"kind": "paragraph", "text": text})
        else:
            out.extend(block_of(kid, default_lang))
    return out


def elements(node) -> list:
    """Element children of a node, in document order."""
    return [kid for kid in node.kids if not isinstance(kid, str)]


def heading_of(node) -> str:
    """Text of the first h1..h8 heading inside the node, else \"\"."""
    for kid in elements(node):
        if kid.tag in ("h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"):
            return heading_text(kid)
        if _is_section(kid):
            continue
        hit = heading_of(kid)
        if hit:
            return hit
    return ""


def page_heading(root) -> str:
    """Text of the page title: the first heading in the document.

    The page h1 sits inside the page section, so this search descends
    into sections too (``heading_of`` does not).
    """
    for kid in elements(root):
        if kid.tag in _HEADING_TAGS:
            return heading_text(kid)
        hit = page_heading(kid)
        if hit:
            return hit
    return ""


def page_title(root) -> str:
    """The page's own name: its h1 text, else its ``title`` element.

    Trimmed. The section number span and the headerlink glyph are not
    part of the name the source gives the page. No hand label table.
    """
    hit = page_heading(root)
    if hit:
        return hit
    for kid in elements(root):
        found = _find_title(kid)
        if found:
            return found
    return ""


def _find_title(node) -> str:
    if node.tag == "title":
        return _clean(node_text(node))
    for kid in elements(node):
        found = _find_title(kid)
        if found:
            return found
    return ""


def declared_names(text: str) -> list[str]:
    """Names one code block declares: a type word precedes the call."""
    out: list[str] = []
    for name in _code_function_names(text):
        for line in text.splitlines():
            if declares_name(line, name):
                if name not in out:
                    out.append(name)
                break
    return out


def declared_names_of(node) -> list[str]:
    """Names the code blocks of a subtree declare, in document order."""
    out: list[str] = []
    if node.tag == "pre" or _is_highlight(node):
        for name in declared_names(_code_text(node)):
            if name not in out:
                out.append(name)
        return out
    for kid in elements(node):
        for name in declared_names_of(kid):
            if name not in out:
                out.append(name)
    return out


def dl_items(node) -> list[tuple[str, object]]:
    """(term text, dd node) pairs of one ``dl``, in document order."""
    out: list[tuple[str, object]] = []
    term = None
    for kid in node.kids:
        if isinstance(kid, str):
            continue
        if kid.tag == "dt":
            term = node_text(kid)
        elif kid.tag == "dd" and term is not None:
            out.append((_clean(term), kid))
            term = None
    return out


def term_names(term: str) -> list[str]:
    """Function names a ``dl`` term names: identifier tokens before \"(\"."""
    return _code_function_names(term)


def term_brief(term: str) -> str:
    """Brief of a function term: the text after the name call and \":\".

    A term such as "__foo(): Does x." gives "Does x.". A term with no
    text after the call gives "".
    """
    names = term_names(term)
    if not names:
        return ""
    index = term.find(names[0]) + len(names[0])
    while index < len(term) and term[index] in " \t":
        index += 1
    if index >= len(term) or term[index] != "(":
        return ""
    depth = 0
    while index < len(term):
        if term[index] == "(":
            depth += 1
        elif term[index] == ")":
            depth -= 1
            if depth == 0:
                break
        index += 1
    rest = term[index + 1 :].strip()
    if rest.startswith(":"):
        rest = rest[1:].strip()
    return rest


def guide_section_runs(section) -> list[dict]:
    """Owner-tagged block runs of one guide section (design C.2).

    Each run is {"owners": [names] or [], "blocks": [...]}. The rules,
    by containment only: a ``dl`` function term owns its item; a code
    block that declares exactly one name owns its run to the next run
    start; a child section owns what it declares; the rest is shared.
    """
    runs: list[dict] = []

    def push(owners, blocks) -> None:
        for block in blocks:
            if runs and runs[-1]["owners"] == owners:
                runs[-1]["blocks"].append(block)
            else:
                runs.append({"owners": owners, "blocks": [block]})

    def walk(nodes, owner) -> None:
        current = owner
        for node in nodes:
            if isinstance(node, str):
                continue
            if node.tag == "dl":
                items = dl_items(node)
                named = [(term_names(term), dd) for term, dd in items]
                if any(names for names, _dd in named):
                    for names, dd in named:
                        push(names or [], blocks_of(dd))
                    current = []
                else:
                    push(current, [_deflist_block(node)])
            elif _is_section(node):
                walk(node.kids, declared_names_of(node))
                current = []
            elif node.tag == "pre" or _is_highlight(node):
                text = _code_text(node)
                names = declared_names(text)
                owners = names[:1] if len(names) == 1 else []
                push(
                    owners,
                    [
                        {
                            "kind": "code",
                            "text": text,
                            "lang": highlight_lang(node, "text"),
                        }
                    ],
                )
                current = owners
            else:
                push(current, block_of(node))

    walk(section.kids, [])
    return runs


def cut_runs(runs: list[dict], name: str) -> list[dict]:
    """The C.1 ``sections`` of one name: its owned runs plus the shared.

    Each section is {"heading", "shared", "blocks"}; each block carries
    "owner" (the cut name) or "shared": True.
    """
    sections: list[dict] = []
    for run in runs:
        owners = run["owners"]
        if owners and name not in owners:
            continue
        shared = not owners
        blocks = [
            ({**block, "shared": True} if shared else {**block, "owner": name})
            for block in run["blocks"]
        ]
        sections.append({"heading": "" if shared else name, "shared": shared, "blocks": blocks})
    return sections


def _dl_terms_in(node) -> list[str]:
    """Term texts of every ``dl`` in the subtree, in document order."""
    out: list[str] = []
    for kid in elements(node):
        if kid.tag == "dl":
            out.extend(term for term, _dd in dl_items(kid))
        else:
            out.extend(_dl_terms_in(kid))
    return out


def guide_views(root) -> dict[str, dict]:
    """Per doc section id: {"heading", "runs", "briefs"}.

    A section is a ``section`` or ``div.section`` element with an id.
    Nested sections keep their own entry; their blocks also take part in
    the parent walk under the names they declare (C.2 rule 3).
    """
    out: dict[str, dict] = {}

    def collect(node) -> None:
        for kid in elements(node):
            if _is_section(kid):
                sid = kid.attrs.get("id") or ""
                if sid and sid not in out:
                    briefs: dict[str, str] = {}
                    for term in _dl_terms_in(kid):
                        brief = term_brief(term)
                        for name in term_names(term):
                            briefs.setdefault(name, brief)
                    out[sid] = {
                        "heading": heading_of(kid),
                        "runs": guide_section_runs(kid),
                        "briefs": briefs,
                    }
            collect(kid)

    collect(root)
    return out


def math_entry_views(root) -> dict:
    """Block views of the ``dl.cpp function`` entries of one page.

    Returns {"heading": page title, "entries": [{"blocks", "heading"}]}
    in document order. An entry is a ``dl.cpp function`` not nested
    inside another one; its blocks are the dd children in page order.
    """
    entries: list[dict] = []

    def walk(node, inside_entry: bool) -> None:
        for kid in elements(node):
            cls = kid.classes()
            is_entry = kid.tag == "dl" and "cpp" in cls and "function" in cls
            if is_entry and not inside_entry:
                subs = elements(kid)
                dt = next((sub for sub in subs if sub.tag == "dt"), None)
                dd = next((sub for sub in subs if sub.tag == "dd"), None)
                view: dict = {"blocks": blocks_of(dd) if dd is not None else []}
                if dt is not None:
                    runs = inline_runs(dt)
                    if has_markup(runs):
                        view["signature_runs"] = runs
                entries.append(view)
            else:
                walk(kid, inside_entry or is_entry)

    walk(root, False)
    return {"heading": page_heading(root), "entries": entries}


def merge_doc_entries(entries: list[dict], new: dict) -> None:
    """Fold one doc entry into a row's list: one entry per source page.

    The fold copies the incoming entry first: the row may mutate the
    fold (a second link of the same page extends the sections), and
    the same entry object can live in another row's list. An entry for
    the same page extends. A second line from the same source section
    only adds its own Syntax line; other sections repeat what the
    section already holds. Later sections of the page append in
    document order. A different page appends a new entry.
    """
    import copy

    new = copy.deepcopy(new)
    page = str(new.get("page") or new.get("url") or "")
    for old in entries:
        if str(old.get("source") or "") != str(new.get("source") or ""):
            continue
        if str(old.get("page") or old.get("url") or "") != page:
            continue
        if old.get("url") == new.get("url"):
            _merge_same_section(old, new)
        else:
            old.setdefault("sections", []).extend(new.get("sections") or [])
        return
    entries.append(new)


def _merge_same_section(old: dict, new: dict) -> None:
    for section in new.get("sections") or []:
        if section.get("heading") != "Syntax":
            continue
        for old_section in old.get("sections") or []:
            if old_section.get("heading") != "Syntax":
                continue
            blocks = [b for b in section.get("blocks") or [] if b.get("kind") == "code"]
            old_blocks = [b for b in old_section.get("blocks") or [] if b.get("kind") == "code"]
            for block in blocks:
                text = str(block.get("text") or "")
                known = [str(b.get("text") or "") for b in old_blocks]
                if text and text not in known:
                    if old_blocks:
                        prior = str(old_blocks[0].get("text") or "")
                        old_blocks[0]["text"] = f"{prior}\n{text}" if prior else text
                    else:
                        old_section.setdefault("blocks", []).append(block)
