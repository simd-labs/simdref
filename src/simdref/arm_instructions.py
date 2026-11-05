"""Arm instruction-source parsing and normalization."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from simdref.ingest_sources import read_local_arm_a64_xml_briefs
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


# Arm ARM titles for mnemonics the AARCHMRS payload and the ACLE intrinsics
# data do not name (the payload ships no titles and no operation briefs; the
# ACLE data is intrinsic-driven and skips scalar instructions).
_AARCHMRS_TITLES: dict[str, str] = {
    "MOVI": "Move Immediate",
    "FMLA": "Floating-point fused Multiply-Add",
    "LD4": "Load Multiple Structures",
    "LDP": "Load Pair of Registers",
}

# The ACLE intrinsics file pairs every intrinsic with the A64 instruction it
# lowers to and describes the intrinsic. Single-instruction entries give a
# real summary sentence keyed by mnemonic.
_ARM_INTRINSICS_JSON_PATHS: tuple[Path, ...] = (
    Path(__file__).resolve().parents[2] / "vendor" / "arm" / "intrinsics.json",
)

_ACLE_SUMMARY_CAP = 160

# ISA words _infer_aarchmrs_isa returns for SIMD/SVE/SME forms. ACLE
# intrinsics text describes those vector forms, so only rows of one of
# these families take the ACLE sentence.
_SIMD_ISAS = frozenset({"NEON", "SVE", "SVE2", "SME", "SME2", "MVE"})


def _acle_first_sentence(description: str) -> str:
    clean = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(description))).strip()
    sentence = clean.split(". ")[0].rstrip(".")
    if len(sentence) > _ACLE_SUMMARY_CAP:
        sentence = sentence[:_ACLE_SUMMARY_CAP].rsplit(" ", 1)[0]
    return sentence


def _acle_summaries_from_text(text: str) -> dict[str, str]:
    """First-sentence summaries keyed by A64 mnemonic from ACLE intrinsics.

    An entry counts only when it lowers to exactly one mnemonic; entries that
    lower to several instructions describe the intrinsic, so their text would
    mislabel the target (vtbx1_s8 lowers to TBL plus MOVI).
    """
    try:
        entries = json.loads(text)
    except ValueError:
        return {}
    if not isinstance(entries, list):
        return {}
    buckets: dict[str, Counter[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if "A64" not in (entry.get("Architectures") or []):
            continue
        mnemonics: set[str] = set()
        for group in entry.get("instructions") or []:
            if not isinstance(group, dict):
                continue
            for instruction in group.get("list") or []:
                if not isinstance(instruction, dict):
                    continue
                base = str(instruction.get("base_instruction") or "").strip().upper()
                if base:
                    mnemonics.add(base)
        if len(mnemonics) != 1:
            continue
        sentence = _acle_first_sentence(str(entry.get("description") or ""))
        if sentence:
            buckets.setdefault(mnemonics.pop(), Counter())[sentence] += 1
    return {mnemonic: counts.most_common(1)[0][0] for mnemonic, counts in buckets.items()}


@lru_cache(maxsize=1)
def _acle_instruction_summaries() -> dict[str, str]:
    for path in _ARM_INTRINSICS_JSON_PATHS:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        summaries = _acle_summaries_from_text(text)
        if summaries:
            return summaries
    return {}


@lru_cache(maxsize=1)
def _a64xml_summaries() -> dict[str, str]:
    """A64 ISA XML ``<desc><brief>`` one-liners, unescaped, keyed by name."""
    return {key: html.unescape(text) for key, text in read_local_arm_a64_xml_briefs().items()}


# A64 condition-code spellings for B.cond with their encoded cond field value
# (Arm ARM cond field, B.cond bits 3:0; llvm-mc -show-inst prints the same
# value as the Bcc immediate). The hs/lo spellings share the cs/cc values, as
# llvm-objdump prints for the same field values.
_CONDITION_CODES: tuple[tuple[str, int], ...] = (
    ("eq", 0),
    ("ne", 1),
    ("hs", 2),
    ("cs", 2),
    ("lo", 3),
    ("cc", 3),
    ("mi", 4),
    ("pl", 5),
    ("vs", 6),
    ("vc", 7),
    ("hi", 8),
    ("ls", 9),
    ("ge", 10),
    ("lt", 11),
    ("gt", 12),
    ("le", 13),
    ("al", 14),
    ("nv", 15),
)


def _generated_summary(mnemonic: str, isa: list[str] | None = None) -> str:
    core = mnemonic.strip().upper()
    if not core:
        return "Arm instruction."
    if isa is None or _SIMD_ISAS.intersection(isa):
        # Scalar/base rows share the mnemonic with the vector form but must
        # not take its ACLE text; they fall through to table and generated.
        acle = _acle_instruction_summaries().get(core)
        if acle:
            return f"{acle}."
    title = _AARCHMRS_TITLES.get(core)
    if title:
        return f"{title}."
    brief = _a64xml_summaries().get(core)
    if brief:
        return f"{brief}."
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
    return f"{core.title()} is an Arm instruction."


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
    isa = _infer_arm_isa(item)
    summary = _strip_text(
        item.get("summary") or item.get("brief") or item.get("title")
    ) or _generated_summary(mnemonic, isa)
    if not summary.endswith("."):
        summary += "."
    url = _strip_text(item.get("url") or item.get("source_url") or item.get("reference_url"))
    return InstructionRecord(
        mnemonic=mnemonic,
        form=form,
        summary=summary,
        architecture="arm",
        isa=isa,
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


def _is_aarchmrs_tree(payload: Any) -> bool:
    top = payload.get("instructions") if isinstance(payload, dict) else None
    return (
        isinstance(top, list)
        and bool(top)
        and all(
            isinstance(node, dict) and str(node.get("_type", "")).endswith("InstructionSet")
            for node in top
        )
    )


def _records_from_payload(payload: Any) -> list[InstructionRecord]:
    if isinstance(payload, dict) and payload.get("format") == "arm-aarchmrs-instructions-v1":
        instructions_json = str(payload.get("instructions_json") or "[]")
        return _records_from_payload(json.loads(instructions_json))

    if isinstance(payload, dict) and payload.get("format") == "arm-instructions-fixture-v1":
        return _records_from_payload(payload.get("instructions") or [])

    if _is_aarchmrs_tree(payload):
        return _records_from_aarchmrs_tree(payload)

    for candidates in _candidate_instruction_lists(payload):
        records = [_normalize_instruction_item(item) for item in candidates]
        normalized = [record for record in records if record is not None]
        if normalized:
            return normalized
    return []


# ---------------------------------------------------------------------------
# AARCHMRS (official Arm machine-readable spec) tree walker
# ---------------------------------------------------------------------------


def _encodeset_mask_match(encoding: Any) -> tuple[int, int, int] | None:
    """Return (bits, mask, match) for an AARCHMRS encodeset, or None.

    Every encodeset entry value with a fixed '0'/'1' pattern adds mask and
    match bits; 'x' entries (operand fields) add nothing. An entry with a
    non-binary value or a length that disagrees with its range makes the
    whole encodeset unusable as a key.
    """
    if not isinstance(encoding, dict):
        return None
    width = encoding.get("width")
    values = encoding.get("values")
    if width != 32 or not isinstance(values, list):
        return None
    mask = match = 0
    for entry in values:
        if not isinstance(entry, dict):
            return None
        value = str(((entry.get("value") or {}).get("value")) or "").strip("'")
        rng = entry.get("range") or {}
        start, field_width = rng.get("start"), rng.get("width")
        if not isinstance(start, int) or not isinstance(field_width, int):
            return None
        if len(value) != field_width:
            return None
        for i, char in enumerate(value):
            bit = start + field_width - 1 - i
            if char == "x":
                continue
            if char not in "01":
                return None
            mask |= 1 << bit
            if char == "1":
                match |= 1 << bit
    return 32, mask, match


def _encodeset_fields(encoding: Any) -> dict[str, tuple[int, int]]:
    """Named encodeset fields as name -> (lsb, width) ranges."""
    fields: dict[str, tuple[int, int]] = {}
    if not isinstance(encoding, dict):
        return fields
    for entry in encoding.get("values") or []:
        rng = entry.get("range") or {}
        name = entry.get("name")
        start, width = rng.get("start"), rng.get("width")
        if name and isinstance(start, int) and isinstance(width, int):
            fields[str(name)] = (start, width)
    return fields


def _alias_condition_bits(condition: Any, fields: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """(mask, match) bits an alias condition fixes on named encodeset fields.

    Only == / IN conjuncts outside any negation give constant bits. OR, NOT
    and identifier compares constrain no bit (a MOV alias fixes Zn equal to
    Zd, which no constant mask can hold).
    """
    constraints: list[tuple[str, list[str]]] = []

    def collect(node: Any) -> None:
        if not isinstance(node, dict) or node.get("_type") != "AST.BinaryOp":
            return
        op = node.get("op")
        if op == "&&":
            collect(node.get("left"))
            collect(node.get("right"))
            return
        left, right = node.get("left"), node.get("right")
        if not (isinstance(left, dict) and left.get("_type") == "AST.Identifier"):
            return
        if op == "==":
            members = [right]
        elif op == "IN" and isinstance(right, dict) and right.get("_type") == "AST.Set":
            members = right.get("values") or []
        else:
            return
        literals: list[str] = []
        for member in members:
            literal = str(member.get("value") or "").strip("'") if isinstance(member, dict) else ""
            if not literal:
                return
            literals.append(literal)
        constraints.append((str(left.get("value")), literals))

    collect(condition)
    mask = match = 0
    for name, literals in constraints:
        rng = fields.get(name)
        if rng is None or any(len(value) != rng[1] for value in literals):
            continue
        start, width = rng
        for index in range(width):
            chars = {value[width - 1 - index] for value in literals}
            if len(chars) != 1 or not chars <= {"0", "1"}:
                continue
            mask |= 1 << (start + index)
            if chars == {"1"}:
                match |= 1 << (start + index)
    return mask, match


def _alias_feature_tags(condition: Any) -> list[str]:
    """Uppercased feature names from IsFeatureImplemented guards, e.g. SME."""
    tags: set[str] = set()

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("_type") == "AST.Function" and node.get("name") == "IsFeatureImplemented":
                for arg in node.get("arguments") or []:
                    name = str(arg.get("value") or "") if isinstance(arg, dict) else ""
                    if name.startswith("FEAT_"):
                        tags.add(name[5:].upper())
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(condition)
    return sorted(tags)


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
    # Parameterised arrangement rules carry the element-size options in the
    # name (T_FMUL_HSD = H/S/D floats, T_OPTION = AdvSIMD options). The 4S
    # instance matches the width policy of the plain T rule above.
    "T_FMUL_HSD": "4S",
    "T_OPTION": "4S",
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

# The vendored AARCHMRS revisions mix case in rule ids (T_OPTION__4 in one,
# T_option__4 in FAT-2025-09, T_FMUL_HSD vs T_fmul_HSD), so the placeholder
# lookup folds case.
_RULE_PLACEHOLDER_BY_FOLD = {
    name.casefold(): instance for name, instance in _RULE_PLACEHOLDER.items()
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
            if rule.casefold() in _RULE_PLACEHOLDER_BY_FOLD:
                parts.append(_RULE_PLACEHOLDER_BY_FOLD[rule.casefold()])
            elif base.casefold() in _RULE_PLACEHOLDER_BY_FOLD:
                parts.append(_RULE_PLACEHOLDER_BY_FOLD[base.casefold()])
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
    # NEON covers Advanced SIMD and the scalar FP groups under simd_dp.
    # SVE and SME group paths carry "fp_" fragments of their own
    # (sve_fp_*, mortlach_*_fp_*), so those never take the NEON label.
    if not isa and (
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


def _aarchmrs_title(node: dict[str, Any], operations: dict[str, Any]) -> str:
    """AARCHMRS title text for one instruction node, when the data carries it.

    The node title wins; the operation entry named by ``operation_id`` is the
    only other place a title can live.
    """
    title = _strip_text(node.get("title")).rstrip(".")
    if title:
        return title
    operation = operations.get(str(node.get("operation_id") or "").strip())
    if isinstance(operation, dict):
        return _strip_text(operation.get("title") or operation.get("brief")).rstrip(".")
    return ""


def _alias_children(
    node: dict[str, Any],
    group_path: list[str],
    operations: dict[str, Any],
    seen_keys: dict[tuple[str, str], tuple[int, int]],
    records: list[InstructionRecord],
) -> None:
    """One named row per alias child of an instruction node (issue #65).

    Each row carries the alias's own assembly template, the target's title
    and isa tags, and the feature tags of its own condition. Its encoding
    is the target's pattern plus the field values the alias condition
    fixes, so it strictly covers the target's pattern and byte identity
    picks the alias row
    (asm_identity._refine_subset drops the covered target). Aliases no
    constant mask separates from their target or a sibling keep no encoding
    and stay text rows.
    """
    aliases = [
        child
        for child in node.get("children", []) or []
        if isinstance(child, dict) and str(child.get("_type", "")).endswith("InstructionAlias")
    ]
    if not aliases:
        return
    target_enc = _encodeset_mask_match(node.get("encoding"))
    fields = _encodeset_fields(node.get("encoding"))
    encodings: list[tuple[int, int]] = []
    for alias in aliases:
        dmask, dmatch = _alias_condition_bits(alias.get("condition"), fields)
        enc = (0, 0)
        if target_enc is not None:
            bits, tmask, tmatch = target_enc
            mask, match = tmask | dmask, (tmatch & tmask) | (dmatch & dmask)
            # A condition contradicting the target pattern is bad data; such
            # an alias keeps no encoding instead of a wrong one.
            if (match & tmask) == (tmatch & tmask) and (mask, match) != (tmask, tmatch):
                enc = (mask, match)
        encodings.append(enc)
    shared = {enc for enc, count in Counter(encodings).items() if enc != (0, 0) and count > 1}
    isa = _infer_aarchmrs_isa(group_path)
    target_title = _aarchmrs_title(node, operations)
    for alias, enc in zip(aliases, encodings):
        mask, match = (0, 0) if enc in shared else enc
        assembly = alias.get("assembly")
        rendered = _render_assembly(assembly) if isinstance(assembly, dict) else ""
        if not rendered:
            continue
        head, _, tail = rendered.partition(" ")
        up = head.strip().upper()
        end = 0
        while end < len(up) and up[end].isalnum():
            end += 1
        if not end or not up[0].isalpha():
            continue
        mnemonic = up[:end]
        operand_form = tail.strip()
        form = _canonical_instruction_key(mnemonic, operand_form) if operand_form else mnemonic
        row_isa = list(isa)
        for tag in _alias_feature_tags(alias.get("condition")):
            if tag not in row_isa:
                row_isa.append(tag)
        # The alias's own A64 XML brief when the release names it; else the
        # target's title when the data has one; else the alias's own name,
        # because a generated "Msr is an Arm instruction." would name the target.
        summary = (
            _a64xml_summaries().get(mnemonic)
            or target_title
            or _generated_summary(mnemonic, row_isa)
        )
        if not summary.endswith("."):
            summary += "."
        key = (mnemonic, form.casefold())
        seen_enc = seen_keys.get(key)
        if seen_enc is not None:
            if seen_enc == (mask, match):
                continue
            # Sibling encodings of one rendered form stay separate rows;
            # their keys are never merged (spec K4).
            form = f"{form} [{str(alias.get('name') or '').strip()}]"
            key = (mnemonic, form.casefold())
        seen_keys[key] = (mask, match)
        metadata = {
            "operation_id": str(alias.get("operation_id") or "").strip(),
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
                isa=row_isa,
                metadata=metadata,
                source="arm-a64",
                enc_bits=32 if mask else 0,
                enc_mask=mask,
                enc_match=match,
            )
        )


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
    seen_keys: dict[tuple[str, str], tuple[int, int]] = {}
    raw_operations = payload.get("operations")
    operations: dict[str, Any] = raw_operations if isinstance(raw_operations, dict) else {}

    def walk(node: dict[str, Any], group_path: list[str]) -> None:
        node_type = str(node.get("_type", ""))
        if node_type.endswith("Instruction"):
            # Alias children get their own named rows first (issue #65).
            start = len(records)
            _alias_children(node, group_path, operations, seen_keys, records)
            alias_rows = records[start:]
            enc = _encodeset_mask_match(node.get("encoding"))
            if enc is None:
                enc_bits = enc_mask = enc_match = 0
            else:
                enc_bits, enc_mask, enc_match = enc
            assembly = node.get("assembly")
            if not isinstance(assembly, dict):
                return
            rendered = _render_assembly(assembly)
            if not rendered:
                return
            head, _, tail = rendered.partition(" ")
            # Head tokens can glue a literal or rule placeholder onto the
            # mnemonic (RET<Rn_option>, DCPS1#0, B.<cond_option>): take the
            # leading alphanumeric run instead of dropping the whole node.
            mnemonic_match = re.match(r"[A-Z][A-Z0-9]*", head.strip().upper())
            if not mnemonic_match:
                return
            mnemonic = mnemonic_match.group(0)
            isa = _infer_aarchmrs_isa(group_path)
            operand_form = tail.strip()
            form = _canonical_instruction_key(mnemonic, operand_form) if operand_form else mnemonic
            metadata = {
                "operation_id": str(node.get("operation_id") or "").strip(),
                "instruction_set": group_path[0] if group_path else "A64",
                "group": group_path[-1] if group_path else "",
            }
            metadata = {k: v for k, v in metadata.items() if v}
            if str(node.get("operation_id") or "") == "B_cond":
                # B.cond assembles under one encoding with the condition in a
                # free field; assemblers spell it B.EQ, B.NE, ... and the
                # mnemonic matcher keys on that dotted form (issue #57a).
                # Each row fixes its own cond value in bits 3:0 (the family
                # encodeset leaves them open), so byte identity tells the
                # rows apart (issue #60).
                for code, cond in _CONDITION_CODES:
                    cond_mnemonic = f"B.{code.upper()}"
                    records.append(
                        InstructionRecord(
                            mnemonic=cond_mnemonic,
                            form=(
                                _canonical_instruction_key(cond_mnemonic, operand_form)
                                if operand_form
                                else cond_mnemonic
                            ),
                            summary="Branch conditionally.",
                            architecture="arm",
                            isa=isa,
                            metadata=dict(metadata),
                            source="arm-a64",
                            enc_bits=enc_bits,
                            enc_mask=enc_mask | 0xF,
                            enc_match=enc_match | cond,
                        )
                    )
                # shortcut: a B.cond alias keeps no aliasof link (the family
                # names no single target row), upgrade if the data gains one
                return
            key = (mnemonic, form.casefold())
            seen_enc = seen_keys.get(key)
            if seen_enc is not None:
                if seen_enc == (enc_mask, enc_match):
                    # The earlier same-key row is the aliases' target too.
                    for alias_row in alias_rows:
                        alias_row.metadata["aliasof"] = form
                    return
                # Sibling encodings of one rendered form stay separate rows;
                # their keys are never merged (spec K4).
                form = f"{form} [{str(node.get('name') or '').strip()}]"
                key = (mnemonic, form.casefold())
            seen_keys[key] = (enc_mask, enc_match)
            summary = _aarchmrs_title(node, operations) or _generated_summary(mnemonic, isa)
            if not summary.endswith("."):
                summary += "."
            records.append(
                InstructionRecord(
                    mnemonic=mnemonic,
                    form=form,
                    summary=summary,
                    architecture="arm",
                    isa=isa,
                    metadata=metadata,
                    source="arm-a64",
                    enc_bits=enc_bits,
                    enc_mask=enc_mask,
                    enc_match=enc_match,
                )
            )
            # The alias row is the same encoding as this target row, so the
            # perf merge gives it the target's timing through this link.
            for alias_row in alias_rows:
                alias_row.metadata["aliasof"] = form
            return
        if node_type.endswith("InstructionAlias"):
            # Alias nodes outside their target instruction carry no target
            # context; the tree nests them under the target (issue #65).
            return
        name = str(node.get("name") or "").strip()
        next_path = group_path + [name] if name else group_path
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                walk(child, next_path)

    for top_node in top:
        if not isinstance(top_node, dict):
            continue
        set_name = str(top_node.get("name") or "")
        # A32/T32 encodings cannot be scoped from a bare word (their patterns
        # overlap A32 condition space), and the byte key is defined on the A64
        # instruction set only (spec K2/K4).
        if set_name != "A64":
            continue
        walk(top_node, [set_name])
    return records


def parse_arm_instruction_payload(text: str) -> list[InstructionRecord]:
    payload = json.loads(text)
    return _records_from_payload(payload)
