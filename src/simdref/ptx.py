"""NVIDIA PTX ISA Sphinx HTML parsing and derived instruction rows.

The PTX ISA reference is one Sphinx page. An instruction section holds a
"Title: op" heading, a Syntax block with template lines and qualifier
sets, and notes blocks with PTX ISA version and sm floors. Rows carry
grammar and version facts only; the doc prose stays out of the catalog.
"""

from __future__ import annotations

import html
import json
from dataclasses import replace
from typing import Any

from simdref.asmline import split_operands
from simdref.catkey import build_key as _build_key
from simdref.catkey import normalize_isa as _normalize_isa
from simdref.htmltext import strip_html, tag_at
from simdref.models import InstructionRecord

BUNDLE_FORMAT = "ptx-isa-v1"
PTX_ISA_DOC_URL = "https://docs.nvidia.com/cuda/parallel-thread-execution/"

_RUBRIC_SYNTAX = '<p class="rubric">Syntax</p>'
_RUBRIC_PTX_NOTES = '<p class="rubric">PTX ISA Notes</p>'
_RUBRIC_TARGET_NOTES = '<p class="rubric">Target ISA Notes</p>'
_RUBRIC_OPEN = '<p class="rubric">'


def _plain(fragment: str) -> str:
    """Tag-stripped, unescaped text of one HTML fragment, one space per gap."""
    clean = strip_html(fragment, (("tags", ("span", "code"), ""),), generic="")
    return " ".join(html.unescape(clean).split())


def _token_char(char: str) -> bool:
    return char.isalnum() or char in "._:"


def _looks_like_op(token: str) -> bool:
    """Op-shaped: dotted segments of letters, digits, '_' or ':'."""
    if not token:
        return False
    for segment in token.split("."):
        if not segment or not all(_token_char(char) for char in segment):
            return False
    return True


def _next_section(doc: str, pos: int) -> tuple[int, int, str] | None:
    """Next <section id="..."> at or after pos: (start, end, id).

    end sits past the matching </section>. An unterminated section runs
    to the end of the document.
    """
    cursor = pos
    while True:
        start = doc.find('<section id="', cursor)
        if start < 0:
            return None
        open_end = doc.find('">', start)
        if open_end < 0:
            cursor = start + 1
            continue
        section_id = doc[start + len('<section id="') : open_end]
        depth = 1
        probe = open_end + 2
        while depth > 0:
            nxt_open = doc.find("<section", probe)
            nxt_close = doc.find("</section>", probe)
            if nxt_close < 0:
                return start, len(doc), section_id
            if 0 <= nxt_open < nxt_close:
                depth += 1
                probe = nxt_open + 8
            else:
                depth -= 1
                probe = nxt_close + 10
        return start, probe, section_id


def _heading(text: str) -> str:
    """Plain text of the first <h1>..<h8> in the section, else ""."""
    probe = 0
    while True:
        hit = tag_at(text, probe)
        if hit is None:
            return ""
        start, end, inner = hit
        if len(inner) >= 2 and inner[0] == "h" and inner[1] in "12345678":
            close = text.find(f"</h{inner[1]}>", end)
            if close < 0:
                return ""
            return _plain(text[end:close])
        probe = start + 1


def _op_prefix(token: str) -> str:
    index = 0
    while index < len(token) and (
        token[index].islower() or token[index].isdigit() or token[index] in "._:"
    ):
        index += 1
    return token[:index]


def _heading_parts(heading: str) -> tuple[str, list[str]] | None:
    """(category, op names) from "Title: op, op" text, else None.

    Each op part takes its leading op-shaped prefix, so a trailing
    headerlink glyph drops off.
    """
    index = 0
    while index < len(heading) and (heading[index].isdigit() or heading[index] == "."):
        index += 1
    text = heading[index:].strip()
    head, sep, tail = text.rpartition(":")
    if not sep or not tail.strip():
        return None
    ops = [_op_prefix(part.strip()) for part in tail.split(",")]
    if not ops or not all(_looks_like_op(op) for op in ops):
        return None
    return head.strip(), ops


def _syntax_block(body: str) -> str | None:
    """Text of the first <pre> after the Syntax rubric, else None."""
    mark = body.find(_RUBRIC_SYNTAX)
    if mark < 0:
        return None
    probe = mark + len(_RUBRIC_SYNTAX)
    while True:
        hit = tag_at(body, probe)
        if hit is None:
            return None
        start, end, inner = hit
        if inner.casefold().startswith("pre"):
            close = body.find("</pre>", end)
            if close < 0:
                return None
            return html.unescape(strip_html(body[end:close], (), generic=""))
        probe = start + 1


def _split_top_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _parse_syntax(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """(template lines, qualifier sets) from one Syntax pre block."""
    templates: list[str] = []
    qualifiers: dict[str, list[str]] = {}
    lines = [line.strip() for line in text.split("\n")]
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line or line.startswith("//"):
            index += 1
            continue
        equals = line.find("=")
        # The value follows the LAST "=", so `.a = .b = { ... }` matches too.
        after_equals = line.split("=")[-1].lstrip() if equals >= 0 else ""
        if equals >= 0 and after_equals.startswith("{"):
            chunk = line[equals + 1 :]
            depth = chunk.count("{") - chunk.count("}")
            follow = index + 1
            while depth > 0 and follow < len(lines):
                chunk += " " + lines[follow]
                depth += lines[follow].count("{") - lines[follow].count("}")
                follow += 1
            open_b = chunk.find("{")
            close_b = chunk.rfind("}")
            inner = chunk[open_b + 1 : close_b] if open_b >= 0 and close_b > open_b else ""
            # `.dtype = .atype = .btype = { ... }` names three sets.
            for part in line.split("=")[:-1]:
                name = part.strip().lstrip(".")
                if name and not qualifiers.get(name):
                    qualifiers[name] = _split_top_commas(inner)
            index = follow
            continue
        # Template lines wrap in the source: join until the statement's ';'.
        # A following line with an "=" starts a set definition or its
        # annotation, so a pending fragment before it is dropped.
        pending = line
        while _top_level_char(pending, ";") < 0:
            if index + 1 >= len(lines):
                pending = ""
                break
            follow_line = lines[index + 1]
            if not follow_line or follow_line.startswith("//") or "=" in follow_line:
                pending = ""
                break
            index += 1
            pending += " " + follow_line
        if pending:
            templates.append(pending)
        index += 1
    return templates, qualifiers


def _top_level_char(tail: str, char: str) -> int:
    depth = 0
    for index, probe in enumerate(tail):
        if probe in "[{":
            depth += 1
        elif probe in "]}":
            depth -= 1
        elif probe == char and depth == 0:
            return index
    return -1


def _leading_token(template: str) -> str | None:
    """Op-shaped leading token of a template line, else None.

    Optional brace groups directly in the token run ("frob{.fast}.vec")
    are skipped, not part of the token.
    """
    parts: list[str] = []
    index = 0
    while index < len(template):
        char = template[index]
        if char.isalnum() or char in "._:":
            start = index
            while index < len(template) and (template[index].isalnum() or template[index] in "._:"):
                index += 1
            parts.append(template[start:index])
        elif char == "{":
            close = template.find("}", index)
            if close < 0:
                return None
            index = close + 1
            if index >= len(template) or template[index] != ".":
                break
        else:
            break
    token = "".join(parts)
    return token.casefold() if _looks_like_op(token) else None


def _template_parts(template: str, set_names: frozenset[str]) -> tuple[str, str] | None:
    """(mnemonic, form) from one template line, else None.

    The mnemonic is the leading literal token minus the segments that
    name a qualifier set of the block. The form is the operand tail up
    to the first top-level semicolon.
    """
    token = _leading_token(template)
    if token is None:
        return None
    # Tail starts at the first non-token char outside a brace group.
    probe = 0
    depth = 0
    while probe < len(template):
        char = template[probe]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif depth == 0 and not (char.isalnum() or char in "._:"):
            break
        probe += 1
    tail = template[probe:]
    semicolon = _top_level_char(tail, ";")
    form = " ".join(tail[:semicolon].split()) if semicolon >= 0 else " ".join(tail.split())
    chain = [seg for seg in token.split(".") if seg not in set_names]
    if not chain:
        chain = token.split(".")
    return ".".join(chain), form


def _notes_prose(body: str, marker: str) -> str:
    """Plain text from a notes rubric to the next rubric or section end."""
    mark = body.find(marker)
    if mark < 0:
        return ""
    start = mark + len(marker)
    stop = len(body)
    for stop_marker in (_RUBRIC_OPEN, "</section>"):
        candidate = body.find(stop_marker, start)
        if candidate >= 0 and candidate < stop:
            stop = candidate
    return _plain(body[start:stop])


def _ptx_versions(prose: str) -> list[str]:
    """Version numbers stated after "PTX ISA version", in order."""
    versions: list[str] = []
    marker = "PTX ISA version"
    index = 0
    while True:
        index = prose.find(marker, index)
        if index < 0:
            return versions
        cursor = index + len(marker)
        while cursor < len(prose) and prose[cursor].isspace():
            cursor += 1
        end = cursor
        while end < len(prose) and (prose[end].isdigit() or prose[end] == "."):
            end += 1
        number = prose[cursor:end].rstrip(".")
        if "." in number and number not in versions:
            versions.append(number)
        index = end if end > cursor else cursor + 1


def _sm_floors(prose: str) -> list[str]:
    """sm_XX tokens stated in the prose, in order."""
    floors: list[str] = []
    index = 0
    while True:
        index = prose.find("sm_", index)
        if index < 0:
            return floors
        if index > 0 and (prose[index - 1].isalnum() or prose[index - 1] == "_"):
            index += 3
            continue
        if index + 3 >= len(prose) or not prose[index + 3].isdigit():
            index += 3
            continue
        end = index + 3
        while end < len(prose) and (prose[end].isalnum() or prose[end] == "_"):
            end += 1
        token = prose[index:end]
        if token not in floors:
            floors.append(token)
        index = end


def _section_rows(body: str, section_id: str) -> list[dict[str, Any]]:
    parts = _heading_parts(_heading(body))
    if parts is None:
        return []
    category, ops = parts
    syntax_text = _syntax_block(body)
    if syntax_text is None:
        return []
    templates, qualifiers = _parse_syntax(syntax_text)
    if not templates:
        return []
    set_names = frozenset(qualifiers)
    versions = _ptx_versions(_notes_prose(body, _RUBRIC_PTX_NOTES))
    floors = _sm_floors(_notes_prose(body, _RUBRIC_TARGET_NOTES))
    url = f"#{section_id}"
    rows: list[dict[str, Any]] = []
    for template in templates:
        parsed = _template_parts(template, set_names)
        if parsed is None:
            continue
        mnemonic, form = parsed
        rows.append(
            {
                "mnemonic": mnemonic,
                "form": form,
                "category": category,
                "syntax": template,
                "qualifiers": dict(qualifiers),
                "ptx_isa": list(versions),
                "sm": list(floors),
                "url": url,
            }
        )
    return rows


def _merge_key_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per catalog key: catkey.build_key(mnemonic, form) is the
    row identity, as for every ISA. Template lines that share a key
    (`add.type1` and `add{.sat}.type2`) merge: Syntax lists every line,
    qualifier sets and version facts are unioned in document order. The
    `urls` list names the section of each Syntax line, so a concrete
    spelling can still reach the section that documents it.
    """
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        key = _build_key(str(row.get("mnemonic", "")), str(row.get("form", "")))
        if not key:
            continue
        entry = merged.get(key)
        if entry is None:
            entry = dict(row)
            entry["syntax"] = [row["syntax"]] if row.get("syntax") else []
            entry["urls"] = [str(row.get("url") or "")] if row.get("syntax") else []
            categories = str(row.get("category", ""))
            entry["category"] = [categories] if categories else []
            merged[key] = entry
            order.append(key)
            continue
        syntax = row.get("syntax")
        if syntax and syntax not in entry["syntax"]:
            entry["syntax"].append(syntax)
            entry["urls"].append(str(row.get("url") or ""))
        category = str(row.get("category", ""))
        if category and category not in entry["category"]:
            entry["category"].append(category)
        for name, tokens in (row.get("qualifiers") or {}).items():
            known = entry["qualifiers"].setdefault(name, [])
            for token in tokens:
                if token not in known:
                    known.append(token)
        for field in ("ptx_isa", "sm"):
            for value in row.get(field) or []:
                if value not in entry[field]:
                    entry[field].append(value)
    out: list[dict[str, Any]] = []
    for key in order:
        entry = merged[key]
        entry["syntax"] = "\n".join(entry["syntax"])
        entry["category"] = ", ".join(entry["category"])
        out.append(entry)
    return out


def derive_rows(html: str) -> list[dict[str, Any]]:
    """Derived field dicts for every PTX instruction section in the page.

    One row per catalog key. A section without a Syntax block yields no
    rows. A section that fails to parse is skipped; the walk continues.
    """
    rows: list[dict[str, Any]] = []
    pos = 0
    while True:
        section = _next_section(html, pos)
        if section is None:
            return _merge_key_rows(rows)
        start, end, section_id = section
        pos = start + 1
        try:
            rows.extend(_section_rows(html[start:end], section_id))
        except Exception:
            continue


def _generate_summary(row: dict[str, Any]) -> str:
    """One generated line from the row's own facts. No doc prose."""
    mnemonic = str(row.get("mnemonic", ""))
    category = str(row.get("category", ""))
    parts = [f"{mnemonic} instruction in {category}."] if category else [f"{mnemonic} instruction."]
    names = [name for name, tokens in row.get("qualifiers", {}).items() if tokens]
    if names:
        parts.append("Qualifier sets: " + ", ".join(names) + ".")
    versions = row.get("ptx_isa") or []
    if versions:
        parts.append(f"Introduced in PTX ISA {versions[0]}.")
    floors = row.get("sm") or []
    if floors:
        parts.append(f"Requires {floors[0]} or higher.")
    return " ".join(parts)


def _row_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("instructions") or []
    else:
        return []
    return [item for item in items if isinstance(item, dict)]


def parse_ptx_instructions_payload(text: str) -> list[InstructionRecord]:
    """Records from a ptx-isa-v1 bundle dict or raw Sphinx HTML.

    The bundle holds derived row dicts under the "instructions" key.
    Raw HTML is parsed through derive_rows first.
    """
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        rows = _row_list(json.loads(text))
    else:
        rows = derive_rows(text)
    records: list[InstructionRecord] = []
    for row in rows:
        mnemonic = str(row.get("mnemonic") or "").strip()
        if not mnemonic:
            continue
        raw_qualifiers = row.get("qualifiers")
        qualifiers: dict[str, list[str]] = {}
        if isinstance(raw_qualifiers, dict):
            for name, tokens in raw_qualifiers.items():
                if isinstance(tokens, list):
                    cleaned = [str(item).strip() for item in tokens if str(item).strip()]
                    if cleaned:
                        qualifiers[str(name)] = cleaned
        versions = [str(item).strip() for item in (row.get("ptx_isa") or []) if str(item).strip()]
        floors = [str(item).strip() for item in (row.get("sm") or []) if str(item).strip()]
        syntax = str(row.get("syntax") or "").strip()
        metadata: dict[str, str] = {
            "url": str(row.get("url") or "").strip(),
            "ptx_isa": ", ".join(versions),
            "sm": ", ".join(floors),
        }
        line_urls = row.get("urls")
        if isinstance(line_urls, list) and line_urls:
            metadata["urls"] = json.dumps([str(item).strip() for item in line_urls])
        if qualifiers:
            metadata["qualifiers"] = json.dumps(qualifiers, sort_keys=True)
        full = dict(row)
        full["qualifiers"] = qualifiers
        full["ptx_isa"] = versions
        full["sm"] = floors
        records.append(
            InstructionRecord(
                mnemonic=mnemonic,
                form=str(row.get("form") or "").strip(),
                summary=_generate_summary(full),
                architecture="ptx",
                isa=_normalize_isa(["PTX"]),
                metadata=metadata,
                description={"Syntax": syntax} if syntax else {},
                source="ptx-isa",
            )
        )
    return records


def op_chain(token: str) -> tuple[str, tuple[str, ...]]:
    """(base op, modifier segments) of a dotted opcode, lowercase."""
    segments = [seg for seg in token.strip().casefold().split(".") if seg]
    if not segments:
        return "", ()
    return segments[0], tuple(segments[1:])


def _qualifier_tokens(record: InstructionRecord) -> set[str]:
    raw = record.metadata.get("qualifiers", "")
    if not raw:
        return set()
    try:
        qualifiers = json.loads(raw)
    except ValueError:
        return set()
    if not isinstance(qualifiers, dict):
        return set()
    tokens: set[str] = set()
    for value in qualifiers.values():
        if not isinstance(value, list):
            continue
        for item in value:
            token = str(item).strip().lstrip(".")
            if token:
                tokens.add(token)
    return tokens


def _coverage(record: InstructionRecord, modifiers: tuple[str, ...]) -> int:
    """Line modifiers the row's qualifier sets allow."""
    tokens = _qualifier_tokens(record)
    if not tokens or not modifiers:
        return 0
    count = 0
    for modifier in modifiers:
        if any(
            modifier == token
            or token.startswith(modifier + "{")
            or token.startswith(modifier + "::")
            for token in tokens
        ):
            count += 1
    return count


def _operand_count(text: str) -> int:
    return sum(1 for part in split_operands(text) if part)


def pick_form(rows: list[InstructionRecord], token: str, operands: str) -> InstructionRecord | None:
    """The row best matching a line opcode and its operand string.

    Candidates share the line's base op. Rows whose mnemonic is a chain
    prefix of the token beat the rest. Within a pool, higher
    qualifier-vocabulary coverage wins, then the closer operand count.
    """
    base, modifiers = op_chain(token)
    if not base:
        return None
    lowered = token.casefold()
    candidates = [
        row
        for row in rows
        # A NOCASE mnemonic query can return other architectures' rows
        # (bare "nop"): a PTX line keeps ptx rows only.
        if row.architecture == "ptx" and op_chain(row.mnemonic)[0] == base
    ]
    if not candidates:
        return None
    prefixes = [
        row
        for row in candidates
        if row.mnemonic.casefold() == lowered or lowered.startswith(row.mnemonic.casefold() + ".")
    ]
    pool = prefixes if prefixes else candidates
    line_count = _operand_count(operands)
    return min(
        pool,
        key=lambda row: (
            -_coverage(row, modifiers),
            abs(_operand_count(row.form) - line_count),
            row.form,
        ),
    )


def _qualifier_map(record: InstructionRecord) -> dict[str, set[str]]:
    """Casefolded set name -> accepted tokens, from the record's metadata."""
    raw = record.metadata.get("qualifiers", "")
    if not raw:
        return {}
    try:
        qualifiers = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(qualifiers, dict):
        return {}
    out: dict[str, set[str]] = {}
    for name, tokens in qualifiers.items():
        if not isinstance(tokens, list):
            continue
        values = {str(item).strip().lstrip(".").casefold() for item in tokens if str(item).strip()}
        if values:
            out[str(name).casefold()] = values
    return out


def _line_vocab(
    template: str, qualifiers: dict[str, set[str]]
) -> tuple[list[str], set[str], set[str]] | None:
    """(required literals, literal names, accepted tokens) of one template's
    opcode run.

    Run segments name a literal or a qualifier set. Brace groups in the run
    add an optional literal or set tokens. The run ends where the operands
    start.
    """
    required: list[str] = []
    literals: set[str] = set()
    accepted: set[str] = set()
    index = 0
    while index < len(template):
        char = template[index]
        if char.isalnum() or char in "._:":
            start = index
            while index < len(template) and (template[index].isalnum() or template[index] in "._:"):
                index += 1
            for segment in template[start:index].split("."):
                name = segment.casefold()
                if not name:
                    continue
                if name in qualifiers:
                    accepted |= qualifiers[name]
                else:
                    required.append(name)
                    accepted.add(name)
        elif char == "{":
            close = template.find("}", index)
            if close < 0:
                return None
            for item in _split_top_commas(template[index + 1 : close]):
                name = item.strip().lstrip(".").casefold()
                if name in qualifiers:
                    accepted |= qualifiers[name]
                elif name:
                    literals.add(name)
                    accepted.add(name)
            index = close + 1
        else:
            break
    if not required and not accepted:
        return None
    literals.update(required)
    return required, literals, accepted


def _part_accepted(part: str, accepted: set[str]) -> bool:
    return part in accepted or any(
        token.startswith(part + "{") or token.startswith(part + "::") for token in accepted
    )


def _line_urls(record: InstructionRecord, syntax: str) -> list[str]:
    """One url per Syntax line; a bundle without per-line urls repeats the
    row's own url."""
    count = len(syntax.split("\n")) if syntax else 0
    urls: list[str] = []
    raw = record.metadata.get("urls", "")
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            urls = [str(item) for item in parsed]
    if len(urls) != count:
        urls = [record.metadata.get("url", "")] * count
    return urls


def pick_spelling(
    rows: list[InstructionRecord], spelling: str, operands: str = ""
) -> InstructionRecord | None:
    """The record a concrete PTX spelling names, else None.

    The spelling splits on '.'. Each part must fall in the matched template
    line's vocabulary: a literal op segment, an optional brace literal, or a
    token of one of the line's qualifier sets; the line's required literals
    must appear among the parts. The most specific line wins: the most parts
    matched as literals, then the most required literals, then the closest
    operand count. The returned record carries the matched line's section
    url. The LSP and the `isa` CLI both resolve through this one function.
    """
    parts = [part for part in spelling.strip().casefold().split(".") if part]
    if not parts:
        return None
    line_count = _operand_count(operands)
    best: tuple | None = None
    chosen: tuple[InstructionRecord, str] | None = None
    for row in rows:
        if row.architecture != "ptx":
            continue
        qualifiers = _qualifier_map(row)
        syntax = (row.description or {}).get("Syntax") or ""
        for line, url in zip(syntax.split("\n"), _line_urls(row, syntax)):
            vocab = _line_vocab(line, qualifiers)
            if vocab is None:
                continue
            required, literals, accepted = vocab
            if any(literal not in parts for literal in required):
                continue
            if not all(_part_accepted(part, accepted) for part in parts):
                continue
            rank = (
                -sum(1 for part in parts if part in literals),
                -len(required),
                abs(_operand_count(row.form) - line_count),
                row.key,
            )
            if best is None or rank < best:
                best = rank
                chosen = (row, url)
    if chosen is None:
        return None
    row, url = chosen
    if url and url != row.metadata.get("url"):
        return replace(row, metadata={**row.metadata, "url": url})
    return row
