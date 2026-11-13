"""CUDA C++ device intrinsics parsing and derived record rows.

Parses the JSON bundle produced by tools/vendor_cuda_sources.py, which
extracts CUDA API names from four clang headers and resolves PTX
instruction links by parsing LLVM TableGen .td files. Joins the NVIDIA
documentation pages, parsed by simdref.cuda_docs, into the rows. The
join runs at build time. No NVIDIA text is stored in git.
"""

from __future__ import annotations

import json
from typing import Any

from simdref.cuda_docs import declares_name, tex_to_text
from simdref.docblocks import merge_doc_entries
from simdref.models import IntrinsicRecord
from simdref.urls import resolve_url

BUNDLE_FORMAT = "clang-cuda-headers-v2"
CUDA_INTRINSICS_DOC_URL = "https://docs.nvidia.com/cuda/cuda-c-programming-guide/"


def parse_cuda_intrinsics_payload(text: str) -> list[IntrinsicRecord]:
    """Parse the CUDA intrinsics JSON bundle into IntrinsicRecord rows."""
    data = json.loads(text)
    if data.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"Unexpected bundle format: {data.get('format')!r}")

    rows: list[IntrinsicRecord] = []
    for item in data.get("intrinsics", []):
        metadata = item.get("metadata", {})
        classification = metadata.get("classification", "")
        record_metadata = {"classification": classification}
        for key in ("min_sm", "min_ptx", "min_sm_from", "min_ptx_from"):
            value = str(item.get(key) or metadata.get(key) or "").strip()
            if value:
                record_metadata[key] = value

        row = IntrinsicRecord(
            name=item["name"],
            signature=item.get("signature", ""),
            description=item.get("description", ""),
            header=item.get("header", ""),
            url=item.get("url", ""),
            architecture="cuda",
            isa=["CUDA"],
            category=item.get("category", ""),
            subcategory=item.get("subcategory", ""),
            instructions=item.get("instructions", []),
            instruction_refs=item.get("instruction_refs", []),
            metadata=record_metadata,
            doc_sections=item.get("doc_sections", {}),
            notes=item.get("notes", []),
            aliases=item.get("aliases", []),
            source="clang-cuda-headers",
        )
        rows.append(row)

    return rows


def parse_cuda_calls(text: str) -> dict[str, list[str]]:
    """name to call-name list from the bundle call graph.

    The vendor step records, per header function, the names its body
    calls. This is the graph the doc join and the SM floors walk.
    """
    data = json.loads(text)
    if data.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"Unexpected bundle format: {data.get('format')!r}")
    out: dict[str, list[str]] = {}
    for item in data.get("intrinsics", []):
        out[item["name"]] = [str(name) for name in item.get("calls") or []]
    return out


def _signature_params(signature: str) -> str:
    """The argument list of a signature, its parameter names and types.

    The math pages carry no Parameters field. The signature does. This
    takes the text between the first "(" and the last ")" of it.
    """
    start = signature.find("(")
    end = signature.rfind(")")
    if start == -1 or end <= start:
        return ""
    args = " ".join(signature[start + 1 : end].split())
    if not args or args == "void":
        return ""
    return args


def _first_sentence(text: str) -> str:
    """The first sentence of a description, with its full stop."""
    joined = " ".join(text.split())
    if not joined:
        return ""
    stop = joined.find(". ")
    if stop == -1:
        return joined
    return joined[: stop + 1]


def _stripped(name: str) -> str:
    """A name as one lowercase run of characters, for containment."""
    return name.replace("_", "").lower()


def _reachable_builtins(name: str, calls: dict[str, list[str]]) -> set[str]:
    """The __nvvm_ and __nv_ tokens the name reaches over the call graph."""
    seen = {name}
    stack = [name]
    out: set[str] = set()
    while stack:
        current = stack.pop()
        for token in calls.get(current, []):
            if token.startswith("__nvvm_") or token.startswith("__nv_"):
                out.add(token)
            elif token in calls and token not in seen:
                seen.add(token)
                stack.append(token)
    return out


def _nearest_documented_caller(name: str, calls: dict[str, list[str]], documented: set[str]) -> str:
    """The nearest documented name that calls this one, else "".

    Breadth first over the caller side of the call graph. The nearest
    level with a documented name decides. Ties take the shortest name.
    """
    callers: dict[str, list[str]] = {}
    for caller, callees in calls.items():
        for callee in callees:
            callers.setdefault(callee, []).append(caller)
    seen = {name}
    frontier = [name]
    while frontier:
        next_frontier: list[str] = []
        candidates: list[str] = []
        for current in frontier:
            for caller in callers.get(current, []):
                if caller in documented and caller not in candidates:
                    candidates.append(caller)
                if caller not in seen:
                    seen.add(caller)
                    next_frontier.append(caller)
        if candidates:
            return min(candidates, key=lambda item: (len(item), item))
        frontier = next_frontier
    return ""


def _contained_doc_name(
    name: str, calls: dict[str, list[str]], documented: dict[str, tuple[dict, str]]
) -> str:
    """A documented name that this name reaches by name containment.

    Last join tier. The documented name, without underscores, must sit
    inside the row name or inside one of the builtins the row reaches,
    from a word start of that name. The row name is tried first, so a
    helper such as __iAtomicExch maps to atomicExch and not to the type
    cast its body calls. The longest such name wins. This maps the
    header helpers onto the API names the docs use.
    """
    for needles in (
        [name],
        sorted(_reachable_builtins(name, calls)),
    ):
        best = ""
        best_len = 0
        for doc_name, (doc_entry, _source) in sorted(documented.items()):
            # A tier-3 name must be a documented function, not a word of
            # an example or a feature table cell.
            signature = str(doc_entry.get("signature") or "")
            if not declares_name(signature, doc_name):
                continue
            token = _stripped(doc_name)
            if len(token) < 4:
                continue
            if any(_starts_at_word(token, needle) for needle in needles) and len(token) > best_len:
                best = doc_name
                best_len = len(token)
        if best:
            return best
    return ""


def _starts_at_word(token: str, name: str) -> bool:
    """The token matches the name from a word start, underscores aside.

    Word starts are the name start, the letter after an underscore or
    a digit, and a case hump. Matching skips underscores of the name,
    so a multiword builtin name still matches its word run.
    """
    n = len(name)
    for pos in range(n):
        if pos:
            prev = name[pos - 1]
            hump = name[pos].isupper() and not prev.isupper()
            if prev != "_" and not prev.isdigit() and not hump:
                continue
        i = pos
        ok = True
        for ch in token:
            while i < n and name[i] == "_":
                i += 1
            # A width run in the name, such as the 32 of match32, does
            # not split the word for the doc name.
            if i < n and name[i].isdigit() and name[i].lower() != ch:
                while i < n and (name[i].isdigit() or name[i] == "_"):
                    i += 1
            if i >= n or name[i].lower() != ch:
                ok = False
                break
            i += 1
        if ok:
            return True
    return False


def _first_paragraph_text(entry: dict[str, Any]) -> str:
    """Text of the first paragraph block of the entry, else "".

    The summary is a paragraph, never list text: the flat description
    string of a guide entry merges a lead-in paragraph with the list
    items that follow it (the atomicAdd steps).
    """
    for section in entry.get("sections") or []:
        for block in section.get("blocks") or []:
            if block.get("kind") == "paragraph" and str(block.get("text") or "").strip():
                return str(block["text"])
    return ""


def apply_cuda_docs(
    rows: list[IntrinsicRecord], calls: dict[str, list[str]], docs: dict[str, Any]
) -> None:
    """Fill documentation fields of the CUDA rows from the parsed pages.

    Join tiers per name: an exact doc entry, then the nearest documented
    caller over the header call graph, then a documented name contained
    in the row name or in a builtin it reaches. The chosen entry stores
    its caller in metadata["doc_via"] on the fallback tiers. A name with
    no entry keeps no URL and keeps the header signature.
    """
    ulp = docs.get("ulp") or {}
    documented: dict[str, tuple[dict, str]] = {}
    for mapping, source in (
        (docs.get("math") or {}, "cuda-math-api"),
        (docs.get("guide") or {}, "cuda-guide"),
        (docs.get("guide_archive") or {}, "cuda-guide-archive"),
        (docs.get("libdevice") or {}, "libdevice"),
    ):
        for doc_name, entry in mapping.items():
            if doc_name not in documented:
                documented[doc_name] = (entry, source)
    for row in rows:
        name = row.name
        via = ""
        hit = documented.get(name)
        if hit is None:
            via = _nearest_documented_caller(name, calls, set(documented))
            if via:
                hit = documented[via]
        if hit is None:
            via = _contained_doc_name(name, calls, documented)
            if via:
                hit = documented[via]
        if hit is None:
            row.url = ""
            continue
        entry, source = hit
        anchor = str(entry.get("anchor") or "")
        page = str(entry.get("page") or "")
        row.url = resolve_url(page, "#" + anchor) if anchor else resolve_url(page)
        signature = str(entry.get("signature") or "")
        # A row shows a declaration of its own name, never an example
        # call or a sibling's declaration: the header signature stays
        # when the doc entry declares another name.
        if signature and declares_name(signature, name):
            row.signature = signature
        description = str(entry.get("description") or "")
        summary = str(entry.get("summary") or "")
        row.description = _first_sentence(summary or _first_paragraph_text(entry) or description)
        sections: dict[str, str] = {}
        if description:
            sections["Description"] = description
        parameters = str(entry.get("parameters") or "")
        if not parameters:
            parameters = _signature_params(str(entry.get("signature") or ""))
        if parameters:
            sections["Parameters"] = parameters
        if entry.get("return_value"):
            sections["Return"] = str(entry["return_value"])
        ulp_entry = ulp.get(name) or {}
        accuracy = str(ulp_entry.get("accuracy") or "")
        if not accuracy:
            accuracy = str(entry.get("accuracy") or "")
        if accuracy:
            readable = tex_to_text(accuracy)
            sections["Accuracy"] = readable
            # The web renders the original TeX later; keep it verbatim.
            if readable != accuracy:
                row.metadata["doc_math"] = accuracy
        row.doc_sections = sections
        row.metadata["doc_source"] = source
        if via:
            row.metadata["doc_via"] = via
        row.doc_sources = _doc_sources(entry, source, row.url, via, page)
        ulp_blocks = _ulp_doc_entry(ulp_entry)
        if ulp_blocks is not None:
            row.doc_sources.append(ulp_blocks)


def _doc_sources(
    entry: dict[str, Any], source: str, url: str, via: str, page: str
) -> list[dict[str, Any]]:
    """The row's own C.1 entry for the page that documents it."""
    sections = entry.get("sections") or []
    block_entry: dict[str, Any] = {
        "source": source,
        "url": url,
        "page": resolve_url(page),
        "title": str(entry.get("title") or ""),
        "heading": str(entry.get("heading") or ""),
        "shared": bool(via) or any(bool(item.get("shared")) for item in sections),
        "sections": sections,
    }
    if entry.get("signature"):
        block_entry["signature"] = str(entry["signature"])
    if entry.get("signature_runs"):
        block_entry["signature_runs"] = entry["signature_runs"]
    if via:
        block_entry["via"] = via
    return [block_entry]


def _ulp_doc_entry(ulp_entry: dict[str, Any]) -> dict[str, Any] | None:
    """The ULP table of the name as one C.1 table entry, else None."""
    rows = [list(cells) for cells in (ulp_entry.get("rows") or [])]
    if not rows:
        return None
    caption = str(ulp_entry.get("caption") or "")
    blocks: list[dict[str, Any]] = [
        {
            "kind": "table",
            "caption": caption,
            # The entry keeps the header as one flat list of cell
            # strings; the table block carries rows of cells. Wrapping
            # once keeps the cells as strings: iterating the flat list
            # would split each cell into its letters.
            "header": [list(ulp_entry.get("header") or [])],
            "rows": rows,
        }
    ]
    accuracy = str(ulp_entry.get("accuracy") or "")
    if ulp_entry.get("math") and accuracy:
        blocks.append({"kind": "math", "text": tex_to_text(accuracy), "tex": accuracy})
    return {
        "source": "ulp",
        "url": str(ulp_entry.get("page") or ""),
        "page": str(ulp_entry.get("page") or ""),
        "title": str(ulp_entry.get("title") or ""),
        "heading": caption,
        "shared": False,
        "sections": [{"heading": "", "shared": False, "blocks": blocks}],
    }


def append_linked_ptx_docs(rows: list[IntrinsicRecord], instructions: list[Any]) -> None:
    """Append the linked PTX entries to the CUDA rows (design C.3).

    One entry per PTX page. Instruction order first, then document
    order. A row with no linked instruction keeps no PTX entry.
    """
    by_name = {item.mnemonic: item for item in instructions if item.architecture == "ptx"}
    for row in rows:
        for ref in row.instruction_refs:
            target = str(ref.get("name") or "") if isinstance(ref, dict) else str(ref or "")
            linked = by_name.get(target)
            if linked is None:
                continue
            for doc_entry in linked.doc_sources:
                merge_doc_entries(row.doc_sources, doc_entry)
