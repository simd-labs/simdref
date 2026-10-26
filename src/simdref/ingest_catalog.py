"""Catalog parsing, linking, and assembly."""

from __future__ import annotations

import csv
import html
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urljoin

from simdref.arm_instructions import parse_arm_instruction_payload
from simdref.ingest_pdf import find_pdf_source_path, load_or_parse_pdf_source, merge_pdf_enrichment
from simdref.ingest_sources import (
    fetch_arm_a64_data,
    fetch_arm_acle_data,
    fetch_intel_data,
    fetch_riscv_rvv_intrinsics_data,
    fetch_riscv_unified_db_data,
    fetch_uops_xml,
    now_iso,
)
from simdref.models import Catalog, InstructionRecord, IntrinsicRecord, SourceVersion
from simdref.riscv import parse_riscv_instruction_payload, parse_riscv_intrinsics_payload

_UOPS_METADATA_KEYS = frozenset(
    {"category", "cpl", "extension", "iclass", "iform", "url", "url-ref"}
)
_UOPS_OPERAND_KEYS = ("idx", "r", "w", "type", "width", "xtype", "name")
_ARM_ACLE_INTRINSIC_BASE_URL = (
    "https://developer.arm.com/architectures/instruction-sets/intrinsics/"
)
_ARM_NEON_REFERENCE_URL = "https://arm-software.github.io/acle/neon_intrinsics/advsimd.html"


def _iter_xml_elements(source: str | Path, tag: str):
    context = ET.iterparse(
        io.StringIO(source) if isinstance(source, str) else source, events=("start", "end")
    )
    root = None
    for event, elem in context:
        if event == "start" and root is None:
            root = elem
            continue
        if event != "end" or elem.tag != tag:
            continue
        yield elem
        elem.clear()
        if root is not None:
            root.clear()


def _intern_small(value: str) -> str:
    return sys.intern(value) if value and len(value) <= 32 else value


def _normalize_isa(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_intern_small(str(item)) for item in value if item]
    if isinstance(value, str):
        return list(_normalize_isa_string(value))
    return []


@lru_cache(maxsize=512)
def _normalize_isa_string(value: str) -> tuple[str, ...]:
    parts = re.split(r"[,/|]\s*|\s{2,}", value)
    return tuple(_intern_small(part.strip()) for part in parts if part.strip())


def _canonical_instruction_key(name: str, form: str) -> str:
    instruction_name = name.strip().upper()
    instruction_form = re.sub(r"\s*,\s*", ", ", form.strip())
    instruction_form = re.sub(r"\s+", " ", instruction_form)
    if not instruction_name:
        return ""
    if not instruction_form:
        return instruction_name
    return f"{instruction_name} ({instruction_form.upper()})"


@lru_cache(maxsize=128)
def _normalize_operand_xtype(value: str, width: str = "") -> str:
    xtype = value.strip()
    if not xtype:
        return xtype
    if xtype == "int":
        # uops.info marks every scalar integer operand xtype=int; the width holds the size
        return f"i{width}" if width in {"8", "16", "32", "64"} else "i32"
    match = re.fullmatch(r"\d+([iu]\d+)", xtype)
    if match:
        return match.group(1)
    return xtype


def _summary_too_terse(summary: str, mnemonic: str) -> bool:
    text = summary.strip().strip(".")
    if not text:
        return True
    words = re.findall(r"[A-Za-z0-9+\-]+", text)
    if len(words) <= 1:
        return True
    if text.casefold() == mnemonic.casefold():
        return True
    if len(words) == 2 and words[0].casefold() == words[1].casefold() == mnemonic.casefold():
        return True
    return False


def _summary_prefix(mnemonic: str, operand_details: list[dict[str, str]], summary: str) -> str:
    upper = mnemonic.upper()
    lowered = summary.casefold()
    has_mask_operand = any(op.get("xtype") == "i1" for op in operand_details)
    if (
        upper.endswith("_Z")
        and "zero-mask" not in lowered
        and "zeromask" not in lowered
        and "zeroing" not in lowered
    ):
        return "Zero-masked "
    if has_mask_operand and "mask" not in lowered:
        return "Masked "
    return ""


def _operand_kind(op: dict[str, str]) -> str:
    kind = op.get("type", "").strip().lower()
    if kind == "reg":
        return "register"
    if kind == "mem":
        return "memory"
    if kind == "imm":
        return "immediate"
    if kind == "agen":
        return "address"
    if kind == "flags":
        return "flags"
    return kind or "operand"


def _element_type_phrase(xtype: str, width: str) -> str:
    xtype = xtype.strip().lower()
    mapping = {
        "f16": "FP16",
        "f32": "single-precision floating-point",
        "f64": "double-precision floating-point",
        "i1": "mask",
        "i8": "8-bit integer",
        "u8": "8-bit integer",
        "i16": "16-bit integer",
        "u16": "16-bit integer",
        "i32": "32-bit integer",
        "u32": "32-bit integer",
        "i64": "64-bit integer",
        "u64": "64-bit integer",
        "i128": "128-bit integer",
        "u128": "128-bit integer",
    }
    if xtype in mapping:
        return mapping[xtype]
    if width.isdigit() and xtype.startswith(("i", "u")):
        return f"{width}-bit integer"
    return ""


def _shared_operand_phrase(operand_details: list[dict[str, str]]) -> str:
    semantic_ops = [
        op for op in operand_details if _operand_kind(op) in {"register", "memory", "immediate"}
    ]
    if not semantic_ops:
        return "operands"
    kinds = {_operand_kind(op) for op in semantic_ops}
    xtypes = {
        op.get("xtype", "").strip().lower()
        for op in semantic_ops
        if op.get("xtype", "").strip().lower() and op.get("xtype", "").strip().lower() != "i1"
    }
    widths = {op.get("width", "").strip() for op in semantic_ops if op.get("width", "").strip()}
    if len(xtypes) == 1:
        phrase = _element_type_phrase(next(iter(xtypes)), next(iter(widths), ""))
        if phrase:
            return "mask operands" if phrase == "mask" else f"{phrase} operands"
    if len(widths) == 1:
        width = next(iter(widths))
        if width.isdigit():
            return f"{width}-bit operands"
    if kinds == {"register"}:
        return "register operands"
    if kinds == {"memory"}:
        return "memory operands"
    if kinds == {"immediate"}:
        return "immediate operands"
    if kinds == {"register", "memory"}:
        return "register and memory operands"
    if kinds == {"register", "immediate"}:
        return "register and immediate operands"
    if kinds == {"memory", "immediate"}:
        return "memory and immediate operands"
    if kinds == {"register", "memory", "immediate"}:
        return "register, memory, and immediate operands"
    return "operands"


def _verb_for_mnemonic(mnemonic: str, architecture: str = "x86") -> str:
    core = mnemonic.upper()
    for suffix in ("_ER_Z", "_ER", "_Z"):
        if core.endswith(suffix):
            core = core[: -len(suffix)]
            break
    if architecture == "x86" and core.startswith("V") and len(core) > 3:
        core = core[1:]
    for key, verb in [
        ("ADC", "Add with carry"),
        ("ADD", "Add"),
        ("SUB", "Subtract"),
        ("SBB", "Subtract with borrow"),
        ("MUL", "Multiply"),
        ("IMUL", "Multiply"),
        ("DIV", "Divide"),
        ("IDIV", "Divide"),
        ("MOV", "Move"),
        ("CMP", "Compare"),
        ("AND", "Bitwise AND"),
        ("OR", "Bitwise OR"),
        ("XOR", "Bitwise XOR"),
        ("TEST", "Test"),
        ("MIN", "Compute minimum of"),
        ("MAX", "Compute maximum of"),
        ("BLEND", "Blend"),
        ("EXPAND", "Expand"),
        ("LOAD", "Load"),
        ("STORE", "Store"),
        ("SHUFFLE", "Shuffle"),
        ("PERM", "Permute"),
    ]:
        if core.startswith(key):
            return verb
    return core.replace("_", " ").title()


def _generated_instruction_summary(
    mnemonic: str, operand_details: list[dict[str, str]], architecture: str = "x86"
) -> str:
    return f"{_verb_for_mnemonic(mnemonic, architecture=architecture)} {_shared_operand_phrase(operand_details)}".strip()


def _instruction_summary(
    mnemonic: str,
    raw_summary: str,
    operand_details: list[dict[str, str]],
    architecture: str = "x86",
) -> str:
    base = raw_summary.strip().strip(".")
    prefix = _summary_prefix(mnemonic, operand_details, base)
    if not _summary_too_terse(base, mnemonic):
        return f"{prefix}{base}".strip() + "."
    return f"{prefix}{_generated_instruction_summary(mnemonic, operand_details, architecture=architecture)}.".strip()


def parse_intel_payload(text: str) -> list[IntrinsicRecord]:
    stripped = text.strip()
    if stripped.startswith("var data_js"):
        match = re.search(r'var\s+data_js\s*=\s*"(?P<body>.*)";\s*$', stripped, re.DOTALL)
        if not match:
            raise ValueError("could not locate Intel XML payload in data.js")
        xml_blob = match.group("body").replace("\\\n", "")
        stripped = bytes(xml_blob, "utf-8").decode("unicode_escape").strip()
    if stripped.startswith("<?xml") or stripped.startswith("<intrinsics_list"):
        records: list[IntrinsicRecord] = []
        for node in _iter_xml_elements(stripped, "intrinsic"):
            name = node.attrib.get("name", "").strip()
            if not name:
                continue
            operation_text = ""
            op_node = node.find("./operation")
            if op_node is not None:
                # ``<operation>`` carries SDM-style pseudocode; preserve internal
                # whitespace (it encodes the algorithm) but trim outer padding.
                raw = "".join(op_node.itertext())
                operation_text = raw.strip("\n").rstrip() if raw else ""
            return_node = node.find("./return")
            ret = (
                node.attrib.get("rettype", "").strip()
                or (return_node.attrib.get("type", "").strip() if return_node is not None else "")
                or "void"
            )
            params = []
            instruction_refs: list[dict[str, str]] = []
            for param in node.findall("./parameter"):
                ptype = param.attrib.get("type", "").strip()
                pname = param.attrib.get("varname", "").strip()
                params.append(" ".join(part for part in [ptype, pname] if part))
            description = " ".join(
                (part.text or "").strip()
                for part in node.findall("./description")
                if (part.text or "").strip()
            )
            cpuid = [
                cpuid.text.strip()
                for cpuid in node.findall("./CPUID")
                if (cpuid.text or "").strip()
            ]
            notes: list[str] = []
            if node.attrib.get("sequence"):
                notes.append(node.attrib["sequence"].strip())
            for key in ("sequence", "sequence_note"):
                child = node.find(f"./{key}")
                if child is not None and (child.text or "").strip():
                    notes.append(child.text.strip())
            instructions: list[str] = []
            for inst in node.findall("./instruction"):
                inst_name = inst.attrib.get("name", "").strip()
                inst_form = inst.attrib.get("form", "").strip()
                inst_xed = inst.attrib.get("xed", "").strip()
                if not inst_name:
                    continue
                instruction_refs.append({"name": inst_name, "form": inst_form, "xed": inst_xed})
                instructions.append(_canonical_instruction_key(inst_name, inst_form) or inst_name)
            records.append(
                IntrinsicRecord(
                    name=name,
                    signature=f"{ret} {name}({', '.join(params)})",
                    description=description,
                    header=(
                        (node.findtext("./header") or "").strip() or node.attrib.get("header", "")
                    ),
                    url=_intel_intrinsic_url(name),
                    architecture="x86",
                    isa=_normalize_isa(
                        cpuid or node.attrib.get("isa", "") or node.attrib.get("tech", "")
                    ),
                    category=(
                        (node.findtext("./category") or "").strip()
                        or node.attrib.get("category", "")
                    ),
                    subcategory=node.attrib.get("tech", "").strip(),
                    instructions=instructions,
                    instruction_refs=[ref | {"architecture": "x86"} for ref in instruction_refs],
                    doc_sections=({"Operation": operation_text} if operation_text else {}),
                    notes=notes,
                    aliases=[],
                )
            )
        return records

    json_blob = stripped
    if stripped.startswith("var ") or stripped.startswith("window."):
        match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
        if not match:
            raise ValueError("could not locate JSON payload in Intel data")
        json_blob = match.group(1)

    payload = json.loads(json_blob)
    candidates = (
        payload.get("intrinsics") or payload.get("data") or payload.get("records") or []
        if isinstance(payload, dict)
        else payload
    )
    records: list[IntrinsicRecord] = []
    for item in candidates:
        name = str(item.get("name") or item.get("intrinsic") or "").strip()
        if not name:
            continue
        signature = str(item.get("signature") or item.get("prototype") or "").strip()
        if not signature:
            return_type = str(item.get("returnType") or item.get("rettype") or "void").strip()
            params = item.get("parameters") or item.get("params") or []
            rendered_params = []
            if isinstance(params, list):
                for param in params:
                    if isinstance(param, dict):
                        ptype = str(param.get("type", "")).strip()
                        pname = str(param.get("name") or param.get("varname") or "").strip()
                        rendered_params.append(" ".join(part for part in [ptype, pname] if part))
                    else:
                        rendered_params.append(str(param).strip())
            signature = f"{return_type} {name}({', '.join(p for p in rendered_params if p)})"
        instructions = (
            item.get("instructions") or item.get("instruction") or item.get("Instruction") or []
        )
        if isinstance(instructions, str):
            instructions = [instructions]
        notes = item.get("notes") or item.get("operationNotes") or []
        if isinstance(notes, str):
            notes = [notes]
        aliases = item.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        description = str(
            item.get("description") or item.get("summary") or item.get("technology", "")
        ).strip()
        operation_text = (
            str(item.get("operation") or item.get("Operation") or "").strip("\n").rstrip()
        )
        records.append(
            IntrinsicRecord(
                name=name,
                signature=signature,
                description=description,
                header=str(item.get("header") or item.get("include") or "").strip(),
                url=str(item.get("url") or "").strip() or _intel_intrinsic_url(name),
                architecture="x86",
                isa=_normalize_isa(
                    item.get("isa") or item.get("tech") or item.get("instructionSet") or []
                ),
                category=str(item.get("category") or "").strip(),
                subcategory=str(item.get("tech") or item.get("subcategory") or "").strip(),
                instructions=[str(value).strip() for value in instructions if str(value).strip()],
                instruction_refs=[
                    {"name": str(value).strip(), "form": "", "xed": "", "architecture": "x86"}
                    for value in instructions
                    if str(value).strip()
                ],
                doc_sections=({"Operation": operation_text} if operation_text else {}),
                notes=[str(value).strip() for value in notes if str(value).strip()],
                aliases=[str(value).strip() for value in aliases if str(value).strip()],
            )
        )
    return records


def parse_arm_intrinsics_payload(text: str) -> list[IntrinsicRecord]:
    stripped = text.strip()
    if stripped.startswith("{"):
        payload = json.loads(text)
        if payload.get("format") == "arm-intrinsics-json-v1":
            return parse_arm_intrinsics_json_bundle(payload)
        if payload.get("format") == "acle-neon-csv-v1":
            return parse_arm_neon_intrinsics_bundle(payload)
    payload = json.loads(text)
    candidates = payload.get("intrinsics") if isinstance(payload, dict) else payload
    records: list[IntrinsicRecord] = []
    for item in candidates or []:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        instruction_refs = []
        instructions = []
        for ref in item.get("instruction_refs") or item.get("instructions") or []:
            if isinstance(ref, str):
                ref = {"name": ref}
            ref_name = str(ref.get("name") or "").strip()
            ref_form = str(ref.get("form") or "").strip()
            if not ref_name:
                continue
            rendered = _canonical_instruction_key(ref_name, ref_form) or ref_name
            instructions.append(rendered)
            instruction_refs.append(
                {
                    "name": ref_name,
                    "form": ref_form,
                    "architecture": "arm",
                }
            )
        records.append(
            IntrinsicRecord(
                name=name,
                signature=str(item.get("signature") or "").strip(),
                description=str(item.get("description") or "").strip(),
                header=str(item.get("header") or "").strip(),
                url=str(item.get("url") or "").strip(),
                architecture="arm",
                isa=_normalize_isa(item.get("isa") or []),
                category=str(item.get("category") or "").strip(),
                subcategory=str(item.get("subcategory") or item.get("group") or "").strip(),
                instructions=instructions,
                instruction_refs=instruction_refs,
                metadata={
                    str(key): str(value).strip()
                    for key, value in (item.get("metadata") or {}).items()
                    if str(value).strip()
                },
                notes=[
                    str(value).strip() for value in item.get("notes") or [] if str(value).strip()
                ],
                aliases=[
                    str(value).strip() for value in item.get("aliases") or [] if str(value).strip()
                ],
                source="arm-acle",
            )
        )
    return records


def _arm_intrinsic_url(name: str) -> str:
    return urljoin(_ARM_ACLE_INTRINSIC_BASE_URL, quote(name))


_INTEL_INTRINSIC_BASE_URL = (
    "https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html"
)


def _intel_intrinsic_url(name: str) -> str:
    """Stable Intel Intrinsics Guide deep link for *name* (uses ``#text=`` fragment)."""
    return f"{_INTEL_INTRINSIC_BASE_URL}#text={quote(name)}"


def _arm_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    return slug or "intrinsics"


def _parse_c_signature(signature: str) -> tuple[str, str, list[str]]:
    match = re.match(
        r"^(?P<ret>.+?)\s+(?P<name>[A-Za-z0-9_]+)\((?P<params>.*)\)$", signature.strip()
    )
    if not match:
        raise ValueError(f"could not parse C signature: {signature}")
    params = [
        part.strip()
        for part in match.group("params").split(",")
        if part.strip() and part.strip() != "void"
    ]
    return match.group("ret").strip(), match.group("name").strip(), params


def _arm_instruction_refs_from_field(value: str) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for part in [chunk.strip() for chunk in value.split(";") if chunk.strip()]:
        pieces = part.split(None, 1)
        name = pieces[0].strip()
        form = pieces[1].strip() if len(pieces) > 1 else ""
        refs.append({"name": name, "form": form, "architecture": "arm"})
    return refs


def _normalize_arm_intrinsic_name(name: str) -> str:
    return name.strip().replace("[_", "_").replace("]", "")


def _strip_markdown_html(value: str) -> str:
    text = value.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"</?(?:code|strong|em|span|p|div)[^>]*>", "", text)
    text = re.sub(r"<a [^>]*>(.*?)</a>", r"\1", text)
    text = html.unescape(re.sub(r"&nbsp;", " ", text))
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def _arm_html_to_text(value: str) -> str:
    text = value.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = text.replace("</p>", "\n\n").replace("</pre>", "\n").replace("</h4>", "\n")
    text = re.sub(r"<a [^>]*>(.*?)</a>", r"\1", text)
    text = re.sub(r"</?(?:pre|p|code|strong|em|span|div|h4|ul|ol|li|b|i)[^>]*>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _arm_live_instruction_refs(groups: list[dict[str, Any]]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for group in groups:
        for entry in group.get("list") or []:
            name = str(entry.get("base_instruction") or "").strip()
            operands = str(entry.get("operands") or "").strip()
            if name:
                refs.append({"name": name, "form": operands, "architecture": "arm"})
    return refs


def _arm_live_operation_sections(
    operation_id: str, operations: dict[str, dict[str, Any]]
) -> dict[str, str]:
    content = str((operations.get(operation_id) or {}).get("content") or "").strip()
    if not content:
        return {}
    text = _arm_html_to_text(content)
    if text.startswith("Operation\n"):
        text = text[len("Operation\n") :].strip()
    return {"ACLE Operation": text} if text else {}


def _arm_live_notes(item: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    notes: list[str] = []
    sections: dict[str, str] = {}
    required = item.get("required_streaming_features") or {}
    if required:
        intro = _arm_html_to_text(str(required.get("intro") or ""))
        features = str(required.get("features") or "").strip()
        body = "\n".join(
            part for part in [intro, f"Features: {features}" if features else ""] if part
        )
        if body:
            sections[str(required.get("title") or "Required streaming features")] = body
            notes.append("Requires streaming features")
    sme_modes = [str(value).strip() for value in item.get("sme_modes") or [] if str(value).strip()]
    if sme_modes:
        sections["Required keyword attributes"] = "\n".join(sme_modes)
    return notes, sections


def parse_arm_intrinsics_json_bundle(payload: dict[str, Any]) -> list[IntrinsicRecord]:
    intrinsics = json.loads(str(payload.get("intrinsics_json") or "[]"))
    operations_payload = json.loads(str(payload.get("operations_json") or "[]"))
    operations = {
        str(item.get("item", {}).get("id") or ""): item.get("item", {})
        for item in operations_payload
        if isinstance(item, dict) and item.get("item")
    }
    records: list[IntrinsicRecord] = []
    for item in intrinsics:
        raw_name = str(item.get("name") or "").strip()
        if not raw_name:
            continue
        name = _normalize_arm_intrinsic_name(raw_name)
        args = [str(value).strip() for value in item.get("arguments") or [] if str(value).strip()]
        group_path = [
            part.strip()
            for part in str(item.get("instruction_group") or "").split("|")
            if part.strip()
        ]
        category = group_path[-1] if group_path else ""
        subcategory = " / ".join(group_path[:-1])
        simd_isa = [
            str(value).strip() for value in item.get("SIMD_ISA") or [] if str(value).strip()
        ]
        isa: list[str] = []
        if "Neon" in simd_isa:
            isa.append("NEON")
        if "SVE" in simd_isa:
            isa.append("SVE")
        if "SVE2" in simd_isa:
            isa.append("SVE2")
        if not isa:
            continue
        instruction_groups = [
            group for group in (item.get("instructions") or []) if isinstance(group, dict)
        ]
        instruction_refs = _arm_live_instruction_refs(instruction_groups)
        docs: dict[str, str] = {}
        for group in instruction_groups:
            preamble = str(group.get("preamble") or "").strip()
            entries = []
            for entry in group.get("list") or []:
                base_instruction = str(entry.get("base_instruction") or "").strip()
                operands = str(entry.get("operands") or "").strip()
                url = str(entry.get("url") or "").strip()
                rendered = " ".join(part for part in [base_instruction, operands] if part).strip()
                if url:
                    rendered = f"{rendered}\nURL: {url}" if rendered else f"URL: {url}"
                if rendered:
                    entries.append(rendered)
            if preamble and entries:
                docs[preamble] = "\n\n".join(entries)
        docs.update(
            _arm_live_operation_sections(str(item.get("Operation") or "").strip(), operations)
        )
        notes, extra_sections = _arm_live_notes(item)
        docs.update(extra_sections)
        arg_prep = item.get("Arguments_Preparation") or {}
        arg_prep_text = ";".join(
            f"{arg} -> {', '.join(f'{key} {value}' for key, value in mapping.items())}"
            for arg, mapping in arg_prep.items()
            if isinstance(mapping, dict)
        )
        result_text = ";".join(
            f"{key} -> {value}" for row in item.get("results") or [] for key, value in row.items()
        )
        records.append(
            IntrinsicRecord(
                name=name,
                signature=f"{str((item.get('return_type') or {}).get('value') or '').strip()} {name}({', '.join(args)})".strip(),
                description=str(item.get("description") or "").strip(),
                header="arm_neon.h" if "NEON" in isa else "arm_sve.h",
                url=_arm_intrinsic_url(name),
                architecture="arm",
                isa=isa,
                category=category,
                subcategory=subcategory,
                instructions=[
                    _canonical_instruction_key(ref["name"], ref["form"]) or ref["name"]
                    for ref in instruction_refs
                ],
                instruction_refs=instruction_refs,
                metadata={
                    "argument_preparation": arg_prep_text,
                    "result": result_text,
                    "supported_architectures": "/".join(
                        str(value).strip()
                        for value in item.get("Architectures") or []
                        if str(value).strip()
                    ),
                    "reference_url": str(
                        (instruction_groups[0].get("list") or [{}])[0].get("url") or ""
                    ).strip()
                    if instruction_groups
                    else "",
                    "classification_path": " / ".join(group_path),
                    "operation_id": str(item.get("Operation") or "").strip(),
                    "simd_isa": ", ".join(simd_isa),
                },
                doc_sections=docs,
                notes=notes,
                source="arm-intrinsics-site",
            )
        )
    return records


def _parse_neon_markdown_docs(markdown: str) -> dict[str, dict[str, str]]:
    docs: dict[str, dict[str, str]] = {}
    headings: list[tuple[int, str]] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line[level:].strip()
            headings = [(lvl, text) for lvl, text in headings if lvl < level]
            headings.append((level, title))
            continue
        if not line.startswith("| <code>"):
            continue
        cells = [cell.strip() for cell in raw_line.strip().split("|")[1:-1]]
        if len(cells) < 5:
            continue
        name_match = re.search(r">([A-Za-z0-9_]+)</a>", cells[0])
        if not name_match:
            continue
        name = name_match.group(1)
        section_path = " / ".join(text for level, text in headings if level >= 4)
        docs[name] = {
            "ACLE Documentation": "\n".join(
                part
                for part in [
                    f"Section: {section_path}" if section_path else "",
                    f"Intrinsic: {_strip_markdown_html(cells[0])}",
                    f"Argument preparation:\n{_strip_markdown_html(cells[1])}",
                    f"AArch64 instruction:\n{_strip_markdown_html(cells[2])}",
                    f"Result:\n{_strip_markdown_html(cells[3])}"
                    if _strip_markdown_html(cells[3])
                    else "",
                    f"Supported architectures:\n{_strip_markdown_html(cells[4])}",
                ]
                if part
            )
        }
    return docs


def _family_stem(name: str) -> str:
    return name.split("_", 1)[0] if "_" in name else name


def _parse_sve_markdown_docs(markdown: str) -> dict[str, dict[str, str]]:
    docs: dict[str, dict[str, str]] = {}
    headings: list[tuple[int, str]] = []
    paragraph: list[str] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[level:].strip()
            headings = [(lvl, text) for lvl, text in headings if lvl < level]
            headings.append((level, title))
            paragraph = []
            index += 1
            continue
        if stripped.startswith("```"):
            block_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block_lines.append(lines[index].rstrip())
                index += 1
            block = "\n".join(block_lines).strip()
            section_path = " / ".join(text for level, text in headings if level >= 3)
            notes = "\n".join(line for line in paragraph if line.strip()).strip()
            families = sorted(set(re.findall(r"\b(sv[a-z0-9]+)(?=\[|_|\()", block)))
            for family in families:
                docs.setdefault(family, {})
                if notes:
                    docs[family]["ACLE Notes"] = notes
                docs[family]["ACLE Prototypes"] = "\n".join(
                    part
                    for part in [f"Section: {section_path}" if section_path else "", block]
                    if part
                )
            paragraph = []
            index += 1
            continue
        if stripped:
            paragraph.append(stripped)
        elif paragraph and paragraph[-1]:
            paragraph.append("")
        index += 1
    return docs


def parse_arm_neon_intrinsics_bundle(payload: dict[str, Any]) -> list[IntrinsicRecord]:
    intrinsics_csv = str(payload.get("intrinsics_csv") or "")
    classification_csv = str(payload.get("classification_csv") or "")
    neon_markdown = str(payload.get("neon_markdown") or "")
    acle_markdown = str(payload.get("acle_markdown") or "")
    neon_docs = _parse_neon_markdown_docs(neon_markdown) if neon_markdown else {}
    sve_docs = _parse_sve_markdown_docs(acle_markdown) if acle_markdown else {}
    classifications: dict[str, str] = {}
    for row in csv.reader(io.StringIO(classification_csv), delimiter="\t"):
        if not row or row[0].startswith("<"):
            continue
        if len(row) >= 2:
            classifications[row[0].strip()] = row[1].strip()

    records: list[IntrinsicRecord] = []
    current_section = ""
    current_section_text = ""
    for row in csv.reader(io.StringIO(intrinsics_csv), delimiter="\t"):
        if not row:
            continue
        tag = row[0].strip()
        if tag == "<SECTION>":
            current_section = row[1].strip() if len(row) > 1 else ""
            current_section_text = row[2].strip() if len(row) > 2 else ""
            continue
        if tag.startswith("<"):
            continue
        if len(row) < 5:
            continue
        signature, arg_prep, instruction_field, result_field, supported_arches = (
            value.strip() for value in row[:5]
        )
        if "A64" not in supported_arches:
            continue
        return_type, name, params = _parse_c_signature(signature)
        path = [current_section] if current_section else []
        if classification := classifications.get(name):
            path.extend(part.strip() for part in classification.split("|") if part.strip())
        category = path[-1] if path else "NEON intrinsics"
        subcategory = " / ".join(path[:-1])
        reference_url = f"{_ARM_NEON_REFERENCE_URL}#{_arm_slug(category)}"
        instruction_refs = _arm_instruction_refs_from_field(instruction_field)
        records.append(
            IntrinsicRecord(
                name=name,
                signature=f"{return_type} {name}({', '.join(params)})",
                description=f"{category}.",
                header="arm_neon.h",
                url=_arm_intrinsic_url(name),
                architecture="arm",
                isa=["NEON"],
                category=category,
                subcategory=subcategory,
                instructions=[
                    _canonical_instruction_key(ref["name"], ref["form"]) or ref["name"]
                    for ref in instruction_refs
                ],
                instruction_refs=instruction_refs,
                metadata={
                    "argument_preparation": arg_prep,
                    "result": result_field,
                    "supported_architectures": supported_arches,
                    "reference_url": reference_url,
                    "section": current_section,
                    "section_description": current_section_text,
                    "classification_path": " / ".join(path),
                },
                doc_sections=neon_docs.get(name, {}),
                notes=[current_section_text] if current_section_text else [],
                source="arm-acle",
            )
        )
    for item in payload.get("extra_intrinsics") or []:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        instruction_refs = []
        instructions = []
        for ref in item.get("instruction_refs") or item.get("instructions") or []:
            if isinstance(ref, str):
                ref = {"name": ref}
            ref_name = str(ref.get("name") or "").strip()
            ref_form = str(ref.get("form") or "").strip()
            if not ref_name:
                continue
            instruction_refs.append({"name": ref_name, "form": ref_form, "architecture": "arm"})
            instructions.append(_canonical_instruction_key(ref_name, ref_form) or ref_name)
        records.append(
            IntrinsicRecord(
                name=name,
                signature=str(item.get("signature") or "").strip(),
                description=str(item.get("description") or "").strip(),
                header=str(item.get("header") or "").strip(),
                url=str(item.get("url") or "").strip(),
                architecture="arm",
                isa=_normalize_isa(item.get("isa") or []),
                category=str(item.get("category") or "").strip(),
                subcategory=str(item.get("subcategory") or "").strip(),
                instructions=instructions,
                instruction_refs=instruction_refs,
                metadata={
                    str(key): str(value).strip()
                    for key, value in (item.get("metadata") or {}).items()
                    if str(value).strip()
                },
                doc_sections={
                    str(key): str(value).strip()
                    for key, value in (
                        item.get("doc_sections") or sve_docs.get(_family_stem(name)) or {}
                    ).items()
                    if str(value).strip()
                },
                notes=[
                    str(value).strip() for value in item.get("notes") or [] if str(value).strip()
                ],
                source="arm-acle",
            )
        )
    records.extend(parse_arm_sve_instruction_map(acle_markdown, sve_docs=sve_docs))
    return records


def parse_arm_sve_instruction_map(
    markdown: str, *, sve_docs: dict[str, dict[str, str]] | None = None
) -> list[IntrinsicRecord]:
    if "### Mapping of SVE instructions to intrinsics" not in markdown:
        return []
    start = markdown.find("### Mapping of SVE instructions to intrinsics")
    if start < 0:
        return []
    table_start = markdown.find("| **Instruction**", start)
    if table_start < 0:
        return []
    lines = markdown[table_start:].splitlines()
    records: list[IntrinsicRecord] = []
    seen: set[str] = set()
    row_re = re.compile(
        r"^\|\s*(?P<instruction>[^|]+?)\s*\|\s*\[`(?P<name>[^`]+)`\]\((?P<url>[^)]+)\)\s*\|$"
    )
    for line in lines[2:]:
        if not line.startswith("|"):
            break
        match = row_re.match(line.strip())
        if not match:
            continue
        instruction = match.group("instruction").strip()
        name = match.group("name").strip()
        url = match.group("url").strip()
        if name in seen or not name.startswith("sv"):
            continue
        seen.add(name)
        instruction_head, _, instruction_tail = instruction.partition("(")
        instruction_name = instruction_head.strip().split()[0]
        instruction_form = instruction_tail.rsplit(")", 1)[0].strip() if instruction_tail else ""
        isa = (
            "SVE2"
            if any(
                token in instruction_name
                for token in ("ADDB", "ADDT", "HNB", "HNT", "LB", "LT", "WB", "WT")
            )
            or instruction_name.startswith("SADD")
            or instruction_name.startswith("UADD")
            else "SVE"
        )
        records.append(
            IntrinsicRecord(
                name=name,
                signature=name,
                description=f"{instruction}.",
                header="arm_sve.h",
                url=url,
                architecture="arm",
                isa=[isa],
                category="Instruction mapping",
                subcategory="SVE / Instruction family",
                instructions=[instruction],
                instruction_refs=[
                    {"name": instruction_name, "form": instruction_form, "architecture": "arm"}
                ],
                metadata={
                    "reference_url": "https://arm-software.github.io/acle/main/acle.html#mapping-of-sve-instructions-to-intrinsics",
                    "mapping_instruction": instruction,
                },
                doc_sections=dict((sve_docs or {}).get(name, {})),
                source="arm-acle",
            )
        )
    return records


def parse_uops_xml(source: str | Path) -> list[InstructionRecord]:
    records: list[InstructionRecord] = []
    for node in _iter_xml_elements(source, "instruction"):
        mnemonic = _intern_small((node.attrib.get("asm") or node.attrib.get("name") or "").strip())
        if not mnemonic:
            continue
        form = (
            node.attrib.get("string")
            or node.attrib.get("form")
            or node.attrib.get("cpl")
            or node.attrib.get("category")
            or ""
        ).strip()
        raw_summary = node.attrib.get("summary", "").strip()
        isa = _normalize_isa(
            node.attrib.get("isa-set", "")
            or node.attrib.get("extension", "")
            or node.attrib.get("isa", "")
        )
        operand_details: list[dict[str, str]] = []
        metadata = {
            key: (
                _intern_small(value.strip())
                if key in {"category", "cpl", "extension", "iclass"}
                else value.strip()
            )
            for key, value in node.attrib.items()
            if key in _UOPS_METADATA_KEYS
        }
        if raw_summary:
            metadata["uops_summary"] = raw_summary
        arch_details: dict[str, dict[str, Any]] = {}
        for child in node:
            if child.tag == "operand":
                xtype = _normalize_operand_xtype(
                    child.attrib.get("xtype", "").strip(), child.attrib.get("width", "").strip()
                )
                operand_payload = {
                    key: (
                        _intern_small(value.strip())
                        if key in {"type", "width", "name"}
                        else value.strip()
                    )
                    for key, value in child.attrib.items()
                    if key in _UOPS_OPERAND_KEYS
                }
                if xtype:
                    operand_payload["xtype"] = _intern_small(xtype)
                operand_details.append(operand_payload)
            elif child.tag == "architecture":
                arch = (
                    child.attrib.get("name")
                    or child.attrib.get("uarch")
                    or child.attrib.get("arch")
                )
                if not arch:
                    continue
                arch = _intern_small(arch)
                arch_entry: dict[str, Any] = {
                    "measurement": {},
                    "latencies": [],
                    "doc": {},
                    "iaca": [],
                }
                for grandchild in child:
                    if grandchild.tag == "measurement":
                        arch_entry["measurement"] = dict(grandchild.attrib)
                        for latency in grandchild.findall("./latency"):
                            arch_entry["latencies"].append(dict(latency.attrib))
                    elif grandchild.tag == "doc":
                        arch_entry["doc"] = dict(grandchild.attrib)
                    elif grandchild.tag == "IACA":
                        arch_entry["iaca"].append(dict(grandchild.attrib))
                arch_details[arch] = arch_entry
        records.append(
            InstructionRecord(
                mnemonic=mnemonic,
                form=form,
                summary=_instruction_summary(
                    mnemonic, raw_summary, operand_details, architecture="x86"
                ),
                architecture="x86",
                isa=isa,
                operand_details=operand_details,
                metadata=metadata,
                arch_details=arch_details,
            )
        )
    return records


def _instruction_indexes(
    instructions: list[InstructionRecord],
) -> tuple[
    dict[tuple[str, str], list[InstructionRecord]],
    dict[tuple[str, str], list[InstructionRecord]],
    dict[tuple[str, str], list[InstructionRecord]],
]:
    by_mnemonic: dict[tuple[str, str], list[InstructionRecord]] = {}
    by_key: dict[tuple[str, str], list[InstructionRecord]] = {}
    by_iform: dict[tuple[str, str], list[InstructionRecord]] = {}
    for record in instructions:
        arch = record.architecture
        by_mnemonic.setdefault((arch, record.mnemonic.casefold()), []).append(record)
        by_key.setdefault((arch, record.key.casefold()), []).append(record)
        if record.form:
            by_mnemonic.setdefault((arch, record.key.casefold()), []).append(record)
        if record.metadata.get("iform"):
            by_iform.setdefault((arch, record.metadata["iform"].casefold()), []).append(record)
    return by_mnemonic, by_key, by_iform


def _resolve_instruction_ref(
    intrinsic: IntrinsicRecord,
    ref: dict[str, str],
    *,
    by_mnemonic: dict[tuple[str, str], list[InstructionRecord]],
    by_key: dict[tuple[str, str], list[InstructionRecord]],
    by_iform: dict[tuple[str, str], list[InstructionRecord]],
) -> tuple[list[InstructionRecord], dict[str, str]]:
    matched: list[InstructionRecord] = []
    ref_arch = ref.get("architecture", intrinsic.architecture).strip() or intrinsic.architecture
    xed = ref.get("xed", "").strip()
    name = ref.get("name", "").strip()
    form = ref.get("form", "").strip()
    resolution = "unresolved"
    if xed and ref_arch == "x86":
        matched = by_iform.get((ref_arch, xed.casefold()), [])
        if matched:
            resolution = "xed"
    if not matched and name and form:
        key_candidates = [(_canonical_instruction_key(name, form) or name).casefold()]
        if ref_arch == "riscv":
            key_candidates.insert(0, form.casefold())
        for candidate_key in key_candidates:
            matched = by_key.get((ref_arch, candidate_key), [])
            if matched:
                resolution = "key"
                break
    if not matched and name:
        matched = by_mnemonic.get((ref_arch, name.casefold()), [])
        if matched:
            resolution = "mnemonic"
        if ref_arch == "arm" and len(matched) > 1 and intrinsic.isa:
            intrinsic_isas = {value.casefold() for value in intrinsic.isa}
            narrowed = [
                instruction
                for instruction in matched
                if intrinsic_isas & {value.casefold() for value in instruction.isa}
            ]
            if narrowed:
                matched = narrowed
                resolution = "arm-isa"
    if ref_arch == "riscv" and len(matched) > 1:
        ref_isa = {value.casefold() for value in _normalize_isa(ref.get("isa", ""))}
        if ref_isa:
            narrowed = [
                instruction
                for instruction in matched
                if ref_isa & {value.casefold() for value in instruction.isa}
            ]
            if narrowed:
                matched = narrowed
                resolution = f"{resolution}-riscv-isa"
        ref_policy = ref.get("policy", "").strip()
        if len(matched) > 1 and ref_policy:
            narrowed = [
                instruction
                for instruction in matched
                if instruction.metadata.get("policy", "").strip() == ref_policy
            ]
            if narrowed:
                matched = narrowed
                resolution = f"{resolution}-riscv-policy"
        ref_tail_policy = ref.get("tail_policy", "").strip()
        if len(matched) > 1 and ref_tail_policy:
            narrowed = [
                instruction
                for instruction in matched
                if instruction.metadata.get("tail_policy", "").strip() == ref_tail_policy
            ]
            if narrowed:
                matched = narrowed
                resolution = f"{resolution}-riscv-tail"
        ref_mask_policy = ref.get("mask_policy", "").strip()
        if len(matched) > 1 and ref_mask_policy:
            narrowed = [
                instruction
                for instruction in matched
                if instruction.metadata.get("mask_policy", "").strip() == ref_mask_policy
            ]
            if narrowed:
                matched = narrowed
                resolution = f"{resolution}-riscv-mask"
        ref_masking = ref.get("masking", "").strip()
        if len(matched) > 1 and ref_masking:
            narrowed = [
                instruction
                for instruction in matched
                if instruction.metadata.get("masking", "").strip() == ref_masking
            ]
            if narrowed:
                matched = narrowed
                resolution = f"{resolution}-riscv-masking"
    if ref_arch == "x86" and len(matched) > 1:
        lowered_name = intrinsic.name.casefold()
        if "_maskz_" in lowered_name:
            narrowed = [
                instruction
                for instruction in matched
                if instruction.key.startswith(f"{instruction.mnemonic}_Z ")
            ]
            if narrowed:
                matched = narrowed
                resolution = f"{resolution}-maskz"
        elif "_mask_" in lowered_name or "_mask2_" in lowered_name:
            narrowed = [
                instruction
                for instruction in matched
                if ", K," in instruction.key
                and not instruction.key.startswith(f"{instruction.mnemonic}_Z ")
            ]
            if narrowed:
                matched = narrowed
                resolution = f"{resolution}-mask"
        else:
            narrowed = [
                instruction
                for instruction in matched
                if ", K," not in instruction.key
                and not instruction.key.startswith(f"{instruction.mnemonic}_Z ")
            ]
            if narrowed:
                matched = narrowed
                resolution = f"{resolution}-plain"
    if ref_arch == "x86" and len(matched) > 1:
        lowered_name = intrinsic.name.casefold()
        width_markers = (
            ("64", ("R64", "REX64")),
            ("32", ("R32", "Rel32")),
            ("16", ("R16", "Rel16")),
            ("8", ("R8",)),
        )
        for width, markers in width_markers:
            if width not in lowered_name:
                continue
            narrowed = [
                instruction
                for instruction in matched
                if any(marker in instruction.key for marker in markers)
            ]
            if narrowed:
                matched = narrowed
                resolution = f"{resolution}-width{width}"
                break
    resolved = {
        "architecture": ref_arch,
        "name": name,
        "form": form,
        "xed": xed,
        "match_count": str(len(matched)),
        "resolution": resolution if matched else "unresolved",
    }
    return matched, resolved


def link_records(intrinsics: list[IntrinsicRecord], instructions: list[InstructionRecord]) -> None:
    by_mnemonic, by_key, by_iform = _instruction_indexes(instructions)
    for intrinsic in intrinsics:
        linked: list[str] = []
        resolved_refs: list[dict[str, str]] = []
        refs = intrinsic.instruction_refs or [
            {"name": name, "form": "", "xed": ""} for name in intrinsic.instructions
        ]
        for ref in refs:
            matched, resolved = _resolve_instruction_ref(
                intrinsic,
                ref,
                by_mnemonic=by_mnemonic,
                by_key=by_key,
                by_iform=by_iform,
            )
            ref_arch = resolved["architecture"]
            if not matched:
                fallback = (
                    _canonical_instruction_key(resolved["name"], resolved["form"])
                    or resolved["name"]
                )
                if fallback:
                    linked.append(fallback)
                    resolved_refs.append(dict(ref) | resolved)
                continue
            for instruction in matched:
                if intrinsic.name not in instruction.linked_intrinsics:
                    instruction.linked_intrinsics.append(intrinsic.name)
                linked.append(instruction.key)
                resolved_refs.append(
                    dict(ref)
                    | resolved
                    | {
                        "architecture": instruction.architecture,
                        "key": instruction.db_key,
                        "display_key": instruction.key,
                    }
                )
        intrinsic.instructions = list(dict.fromkeys(linked))
        intrinsic.instruction_refs = resolved_refs


def _ingest_perf_sources(
    instructions: list[InstructionRecord],
    *,
    status: Callable[[str], None],
) -> list[SourceVersion]:
    """Run the llvm scheduling pipeline and merge rows into *instructions*.

    Any ingester failure propagates so silent breakage (missing LLVM
    binary, empty exegesis output, scheduler-model mismatch) surfaces at
    build time instead of producing a silently empty catalog.
    """
    from simdref.perf_sources import (
        ingest_llvm_mca,
        merge_perf_rows,
    )

    versions: list[SourceVersion] = []

    status("Driving llvm-exegesis + llvm-mca across modeled cores")
    llvm_rows, llvm_version = ingest_llvm_mca(instructions)
    status(f"Collected {len(llvm_rows)} llvm-mca perf rows")
    if llvm_rows:
        merge_perf_rows(instructions, llvm_rows)
        versions.append(
            SourceVersion(
                source="llvm-mca",
                version=llvm_version or "unknown",
                fetched_at=now_iso(),
                url="https://llvm.org/docs/CommandGuide/llvm-mca.html",
            )
        )

    return versions


def build_catalog(
    include_sdm: bool = False,
    *,
    status: Callable[[str], None] | None = None,
) -> Catalog:
    emit = status or (lambda _msg: None)
    emit("Fetching Intel intrinsics data")
    intel_text, intel_source = fetch_intel_data()
    emit(f"Fetched Intel intrinsics data from {intel_source.url}")
    emit("Fetching uops.info instruction data")
    uops_text, uops_source = fetch_uops_xml()
    emit(f"Fetched uops.info instruction data from {uops_source.url}")
    emit("Fetching Arm ACLE intrinsic data")
    arm_acle_text, arm_acle_source = fetch_arm_acle_data()
    emit(f"Fetched Arm ACLE intrinsic data from {arm_acle_source.url}")
    emit("Fetching Arm A64 instruction data")
    arm_a64_text, arm_a64_source = fetch_arm_a64_data()
    emit(f"Fetched Arm A64 instruction data from {arm_a64_source.url}")
    emit("Fetching RISC-V RVV intrinsic data")
    riscv_intrinsics_text, riscv_intrinsics_source = fetch_riscv_rvv_intrinsics_data()
    emit(f"Fetched RISC-V RVV intrinsic data from {riscv_intrinsics_source.url}")
    emit("Fetching RISC-V unified-db instruction data")
    riscv_instructions_text, riscv_instructions_source = fetch_riscv_unified_db_data()
    emit(f"Fetched RISC-V unified-db instruction data from {riscv_instructions_source.url}")
    emit("Parsing intrinsic catalog")
    intrinsics = parse_intel_payload(intel_text)
    arm_intrinsics = parse_arm_intrinsics_payload(arm_acle_text)
    intrinsics.extend(arm_intrinsics)
    intrinsics.extend(parse_riscv_intrinsics_payload(riscv_intrinsics_text))
    emit(f"Parsed {len(intrinsics)} intrinsics")
    emit("Parsing instruction catalog")
    instructions = parse_uops_xml(uops_text)
    instructions.extend(parse_arm_instruction_payload(arm_a64_text))
    instructions.extend(parse_riscv_instruction_payload(riscv_instructions_text))
    emit(f"Parsed {len(instructions)} instructions")
    emit("Linking intrinsics to instructions")
    link_records(intrinsics, instructions)
    emit("Linked intrinsics and instructions")

    perf_sources_version = list(_ingest_perf_sources(instructions, status=emit))

    if include_sdm:
        sdm_path = find_pdf_source_path("intel-sdm")
        if sdm_path is not None:
            try:
                emit(f"Preparing Intel SDM descriptions from {sdm_path}")
                result = load_or_parse_pdf_source("intel-sdm", sdm_path, status=status)
                merge_pdf_enrichment(instructions, "intel-sdm", result)
                emit("Merged Intel SDM descriptions into instruction records")
            except Exception:
                pass

    emit("Assembling final catalog")
    return Catalog(
        intrinsics=sorted(intrinsics, key=lambda item: item.name),
        instructions=sorted(
            instructions, key=lambda item: (item.architecture, item.mnemonic, item.form)
        ),
        sources=[
            intel_source,
            uops_source,
            arm_acle_source,
            arm_a64_source,
            riscv_intrinsics_source,
            riscv_instructions_source,
            *perf_sources_version,
        ],
        generated_at=now_iso(),
    )
