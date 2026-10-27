"""Arm instruction-source parsing and normalization."""

from __future__ import annotations

import json
import re
from typing import Any

from simdref.models import InstructionRecord


def _normalize_isa(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,/|]\s*|\s{2,}", value) if part.strip()]
    return []


def _canonical_instruction_key(name: str, form: str) -> str:
    instruction_name = name.strip().upper()
    instruction_form = re.sub(r"\s*,\s*", ", ", form.strip())
    instruction_form = re.sub(r"\s+", " ", instruction_form)
    if not instruction_name:
        return ""
    if not instruction_form:
        return instruction_name
    return f"{instruction_name} ({instruction_form.upper()})"


def _generated_summary(mnemonic: str) -> str:
    core = mnemonic.strip().upper()
    if not core:
        return "Arm instruction."
    if core.startswith("ADD"):
        return "Add operands."
    if core.startswith("SUB"):
        return "Subtract operands."
    if core.startswith("MUL"):
        return "Multiply operands."
    if core.startswith("LD"):
        return "Load data."
    if core.startswith("ST"):
        return "Store data."
    return f"{core.title()} instruction."


def _strip_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value", "body", "summary", "description", "brief"):
            text = _strip_text(value.get(key))
            if text:
                return text
        return ""
    if isinstance(value, list):
        parts = [_strip_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    return ""


def _collapse_section_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_collapse_section_value(item) for item in value]
        return "\n\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        if "title" in value and any(key in value for key in ("body", "content", "text", "value")):
            title = _strip_text(value.get("title"))
            body = _strip_text(
                value.get("body") or value.get("content") or value.get("text") or value.get("value")
            )
            return "\n".join(part for part in (title, body) if part).strip()
        flattened = []
        for item in value.values():
            text = _collapse_section_value(item)
            if text:
                flattened.append(text)
        return "\n\n".join(flattened).strip()
    return ""


def _description_sections(item: dict[str, Any]) -> dict[str, str]:
    sections: dict[str, str] = {}

    for key in ("description", "descriptions", "doc_sections", "sections"):
        raw = item.get(key)
        if isinstance(raw, dict):
            for title, value in raw.items():
                text = _collapse_section_value(value)
                if text:
                    sections[str(title).strip()] = text
        elif isinstance(raw, list):
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                title = _strip_text(entry.get("title") or entry.get("name") or entry.get("heading"))
                body = _collapse_section_value(
                    entry.get("body")
                    or entry.get("content")
                    or entry.get("text")
                    or entry.get("value")
                )
                if title and body:
                    sections[title] = body

    operation = _collapse_section_value(item.get("operation") or item.get("pseudocode"))
    if operation:
        sections.setdefault("Operation", operation)

    detail = _strip_text(item.get("detail") or item.get("details"))
    if detail:
        sections.setdefault("Details", detail)

    return sections


def _infer_arm_isa(item: dict[str, Any]) -> list[str]:
    explicit = _normalize_isa(
        item.get("isa") or item.get("isas") or item.get("extensions") or item.get("feature_tags")
    )
    if explicit:
        return explicit

    haystack_parts = [
        _strip_text(item.get("category")),
        _strip_text(item.get("section")),
        _strip_text(item.get("group")),
        _strip_text(item.get("classification")),
        _strip_text(item.get("url")),
    ]
    haystack = " ".join(part.upper() for part in haystack_parts if part)

    isa: list[str] = []
    for token in (
        "SME2",
        "SME",
        "SVE2",
        "SVE",
        "MVE",
        "HELIUM",
        "NEON",
        "ADVSIMD",
        "SIMD-FP",
        "SIMD&FP",
    ):
        if token in haystack:
            if token in {"HELIUM", "MVE"}:
                candidate = "MVE"
            elif token in {"ADVSIMD", "SIMD-FP", "SIMD&FP"}:
                candidate = "NEON"
            else:
                candidate = token
            if candidate not in isa:
                isa.append(candidate)

    if isa:
        return isa
    return ["A64"]


def _collect_aliases(item: dict[str, Any]) -> list[str]:
    raw = item.get("aliases") or item.get("alias_mnemonics") or []
    if isinstance(raw, str):
        raw = [raw]
    aliases = [str(value).strip() for value in raw if str(value).strip()]
    alias = _strip_text(item.get("alias"))
    if alias:
        aliases.append(alias)
    return list(dict.fromkeys(aliases))


def _collect_metadata(item: dict[str, Any], *, url: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for source_key, target_key in (
        ("category", "category"),
        ("section", "section"),
        ("group", "group"),
        ("classification", "classification"),
        ("encoding", "encoding"),
        ("operand_form", "operand_form"),
        ("instruction_class", "instruction_class"),
        ("architecture_state", "architecture_state"),
    ):
        value = _strip_text(item.get(source_key))
        if value:
            metadata[target_key] = value
    if url:
        metadata["url"] = url
    return metadata


def _normalize_instruction_item(item: dict[str, Any]) -> InstructionRecord | None:
    mnemonic = _strip_text(
        item.get("mnemonic")
        or item.get("base_instruction")
        or item.get("instruction")
        or item.get("name")
        or item.get("opcode")
    ).upper()
    if not mnemonic:
        return None

    operands = _strip_text(item.get("operands") or item.get("operand_form") or item.get("form"))
    form = (
        _canonical_instruction_key(mnemonic, operands)
        if operands and not operands.upper().startswith(f"{mnemonic} (")
        else operands
    )
    summary = _strip_text(
        item.get("summary") or item.get("brief") or item.get("title")
    ) or _generated_summary(mnemonic)
    if not summary.endswith("."):
        summary += "."
    url = _strip_text(item.get("url") or item.get("source_url") or item.get("reference_url"))
    return InstructionRecord(
        mnemonic=mnemonic,
        form=form,
        summary=summary,
        architecture="arm",
        isa=_infer_arm_isa(item),
        metadata=_collect_metadata(item, url=url),
        aliases=_collect_aliases(item),
        description=_description_sections(item),
        source="arm-a64",
    )


def _candidate_instruction_lists(payload: Any) -> list[list[dict[str, Any]]]:
    lists: list[list[dict[str, Any]]] = []
    if isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            lists.append(payload)
        return lists
    if not isinstance(payload, dict):
        return lists

    for key in (
        "instructions",
        "base_instructions",
        "instruction_set",
        "InstructionSet",
        "items",
        "records",
    ):
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            lists.append(value)

    for value in payload.values():
        if isinstance(value, dict):
            lists.extend(_candidate_instruction_lists(value))

    return lists


def _records_from_payload(payload: Any) -> list[InstructionRecord]:
    if isinstance(payload, dict) and payload.get("format") == "arm-aarchmrs-instructions-v1":
        instructions_json = str(payload.get("instructions_json") or "[]")
        return _records_from_payload(json.loads(instructions_json))

    if isinstance(payload, dict) and payload.get("format") == "arm-instructions-fixture-v1":
        return _records_from_payload(payload.get("instructions") or [])

    aarchmrs = _records_from_aarchmrs_tree(payload)
    if aarchmrs:
        return aarchmrs

    for candidates in _candidate_instruction_lists(payload):
        records = [_normalize_instruction_item(item) for item in candidates]
        normalized = [record for record in records if record is not None]
        if normalized:
            return normalized
    return []


# ---------------------------------------------------------------------------
# AARCHMRS (official Arm machine-readable spec) tree walker
# ---------------------------------------------------------------------------

_RULE_PLACEHOLDER: dict[str, str] = {
    # Scalar
    "Xd": "x0",
    "Xn": "x1",
    "Xm": "x2",
    "Xa": "x3",
    "Xt": "x0",
    "Xt2": "x1",
    "Xs": "x1",
    "Xt1": "x0",
    "Wd": "w0",
    "Wn": "w1",
    "Wm": "w2",
    "Wa": "w3",
    "Wt": "w0",
    "Wt2": "w1",
    "Ws": "w1",
    "Wt1": "w0",
    # Advanced SIMD vector registers
    "Vd": "v0",
    "Vn": "v1",
    "Vm": "v2",
    "Va": "v3",
    "Vt": "v0",
    "Vt2": "v1",
    "Vt3": "v2",
    "Vt4": "v3",
    "Dd": "d0",
    "Dn": "d1",
    "Dm": "d2",
    "Da": "d3",
    # FP accumulator register operands; t/s letters follow the Xt/Xs numbering.
    "fpfar_dt": "d0",
    "fpfar_ds": "d1",
    "fpfar_st": "s0",
    "fpfar_ss": "s1",
    "fpfar_ht": "h0",
    "fpfar_hs": "h1",
    "Sd": "s0",
    "Sn": "s1",
    "Sm": "s2",
    "Sa": "s3",
    "Hd": "h0",
    "Hn": "h1",
    "Hm": "h2",
    "Ha": "h3",
    "Bd": "b0",
    "Bn": "b1",
    "Bm": "b2",
    "Qd": "q0",
    "Qn": "q1",
    "Qm": "q2",
    "Qt": "q0",
    "Qt2": "q1",
    # SVE predicate / vector
    "Zd": "z0",
    "Zn": "z1",
    "Zm": "z2",
    "Za": "z3",
    "Zdn": "z0",
    "Zt": "z0",
    "Pd": "p0",
    "Pn": "p1",
    "Pm": "p2",
    "Pg": "p0",
    "Pt": "p0",
    # SME
    "ZAd": "za0",
    "ZAn": "za0",
    "ZAt": "za0",
    # Stack pointer / zero register
    "SP": "sp",
    "XZR": "xzr",
    "WZR": "wzr",
    "XSP": "sp",
    "WSP": "wsp",
    # Condition codes / misc commonly referenced
    "cond": "eq",
    "nzcv": "0",
    # Arrangement specifiers and element size hints
    "T": "4S",
    "Ta": "4S",
    "Tb": "4S",
    "Ts": "4S",
    "T__1": "4S",
    "T__2": "4S",
    "T__3": "4S",
    "T__4": "4S",
    "size": "4S",
    "size__1": "4S",
}

_LITERAL_KEEP = {
    "COMMA": ",",
    "SPACE": " ",
    "LBRACKET": "[",
    "RBRACKET": "]",
    "LBRACE": "{",
    "RBRACE": "}",
    "HASH": "#",
    "EXCLAM": "!",
}


def _render_assembly(assembly: dict[str, Any]) -> str:
    """Render an AARCHMRS ``assembly.symbols`` list to a concrete asm string.

    Literals are kept verbatim; RuleReferences substitute from
    ``_RULE_PLACEHOLDER`` when known, fall back to ``#0`` for numeric rules
    (imm / off / lsb / width / shift) and to a neutral token otherwise.
    """
    symbols = assembly.get("symbols") if isinstance(assembly, dict) else None
    if not isinstance(symbols, list):
        return ""
    parts: list[str] = []
    for sym in symbols:
        if not isinstance(sym, dict):
            continue
        sym_type = str(sym.get("_type", ""))
        if sym_type.endswith("Literal"):
            parts.append(str(sym.get("value", "")))
            continue
        if sym_type.endswith("RuleReference"):
            rule = str(sym.get("rule_id", ""))
            if rule in _LITERAL_KEEP:
                parts.append(_LITERAL_KEEP[rule])
                continue
            base = re.sub(r"__\d+$", "", rule)
            if rule in _RULE_PLACEHOLDER:
                parts.append(_RULE_PLACEHOLDER[rule])
            elif base in _RULE_PLACEHOLDER:
                parts.append(_RULE_PLACEHOLDER[base])
            elif re.search(r"imm|offset|off|lsb|width|shift|amount|rot", rule, re.IGNORECASE):
                parts.append("#0")
            elif re.search(r"label|addr", rule, re.IGNORECASE):
                parts.append(".")
            else:
                parts.append(f"<{rule}>")
            continue
    return "".join(parts).strip()


def _infer_aarchmrs_isa(group_path: list[str]) -> list[str]:
    joined = " ".join(group_path).lower()
    isa: list[str] = []
    if "sme" in joined:
        isa.append("SME2" if "sme2" in joined else "SME")
    if "sve2" in joined:
        isa.append("SVE2")
    elif "sve" in joined:
        isa.append("SVE")
    if (
        "advsimd" in joined
        or "asimd" in joined
        or "neon" in joined
        or "fp_" in joined
        or "simd" in joined
    ):
        isa.append("NEON")
    if "mve" in joined:
        isa.append("MVE")
    return isa or ["A64"]


def _records_from_aarchmrs_tree(payload: Any) -> list[InstructionRecord]:
    if not isinstance(payload, dict):
        return []
    top = payload.get("instructions")
    if not isinstance(top, list) or not top:
        return []
    # Detect AARCHMRS: top-level entries are InstructionSet nodes with children.
    if not all(
        isinstance(node, dict) and str(node.get("_type", "")).endswith("InstructionSet")
        for node in top
    ):
        return []

    records: list[InstructionRecord] = []
    seen_keys: set[tuple[str, str]] = set()

    def walk(node: dict[str, Any], group_path: list[str]) -> None:
        node_type = str(node.get("_type", ""))
        if node_type.endswith("Instruction"):
            assembly = node.get("assembly")
            if not isinstance(assembly, dict):
                return
            rendered = _render_assembly(assembly)
            if not rendered:
                return
            head, _, tail = rendered.partition(" ")
            mnemonic = head.strip().upper()
            if not mnemonic or not re.match(r"^[A-Z][A-Z0-9]*$", mnemonic):
                return
            operand_form = tail.strip()
            form = _canonical_instruction_key(mnemonic, operand_form) if operand_form else mnemonic
            key = (mnemonic, form.casefold())
            if key in seen_keys:
                return
            seen_keys.add(key)
            summary = _generated_summary(mnemonic)
            if not summary.endswith("."):
                summary += "."
            metadata = {
                "operation_id": str(node.get("operation_id") or "").strip(),
                "instruction_set": group_path[0] if group_path else "A64",
                "group": group_path[-1] if group_path else "",
            }
            metadata = {k: v for k, v in metadata.items() if v}
            records.append(
                InstructionRecord(
                    mnemonic=mnemonic,
                    form=form,
                    summary=summary,
                    architecture="arm",
                    isa=_infer_aarchmrs_isa(group_path),
                    metadata=metadata,
                    source="arm-a64",
                )
            )
            return
        if node_type.endswith("InstructionAlias"):
            # Skip aliases — they resolve to another instruction we already parse.
            return
        name = str(node.get("name") or "").strip()
        next_path = group_path + [name] if name else group_path
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                walk(child, next_path)

    for top_node in top:
        if isinstance(top_node, dict):
            walk(top_node, [str(top_node.get("name") or "")])
    return records


def parse_arm_instruction_payload(text: str) -> list[InstructionRecord]:
    payload = json.loads(text)
    return _records_from_payload(payload)
