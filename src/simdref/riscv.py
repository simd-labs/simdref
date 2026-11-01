"""RISC-V RVV instruction/intrinsic parsing and normalization."""

from __future__ import annotations

import html
import json
import re
from typing import Any

from simdref.models import InstructionRecord, IntrinsicRecord

RISCV_RVV_INTRINSICS_PROJECT_URL = "https://github.com/riscv-non-isa/riscv-rvv-intrinsic-doc"


def _normalize_isa(value: Any) -> list[str]:
    if isinstance(value, dict):
        normalized: list[str] = []
        for key in ("isa", "extensions", "extension", "name", "value"):
            normalized.extend(_normalize_isa(value.get(key)))
        if normalized:
            return list(dict.fromkeys(normalized))
        return [str(key).strip() for key, enabled in value.items() if enabled and str(key).strip()]
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            normalized.extend(_normalize_isa(item))
        return list(dict.fromkeys(normalized))
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,/|]\s*|\s{2,}", value) if part.strip()]
    return []


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key).strip(): str(item).strip()
        for key, item in value.items()
        if str(key).strip() and str(item).strip()
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _strip_tags(text: str) -> str:
    clean = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    clean = re.sub(r"</p\s*>", "\n\n", clean, flags=re.IGNORECASE)
    clean = re.sub(r"</?(?:code|span|strong|em)[^>]*>", "", clean, flags=re.IGNORECASE)
    clean = re.sub(
        r"</?(?:pre|div|section|article|ul|ol|li)[^>]*>", "\n", clean, flags=re.IGNORECASE
    )
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = html.unescape(clean)
    clean = clean.replace("\r", "")
    clean = re.sub(r"[ \t]+\n", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"\s*\n\s*", "\n", clean)
    clean = re.sub(r"\n+", "\n", clean)
    return clean.strip()


def _extract_html_section(doc: str, title: str) -> str:
    pattern = re.compile(
        rf"<h[1-6][^>]*>\s*{re.escape(title)}\s*</h[1-6]>(?P<body>.*?)(?=<h[1-6][^>]*>|$)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(doc)
    if not match:
        return ""
    return _strip_tags(match.group("body"))


def _section_slice(doc: str, mnemonic: str) -> str:
    lowered = doc.casefold()
    needle = mnemonic.casefold()
    index = lowered.find(needle)
    if index < 0:
        return ""
    start = max(
        doc.rfind('<div class="sect4">', 0, index),
        doc.rfind('<div class="sect3">', 0, index),
        doc.rfind('<div class="sect2">', 0, index),
    )
    if start < 0:
        start = max(0, index - 4000)
    end_candidates = [
        candidate
        for marker in ('<div class="sect4">', '<div class="sect3">', '<div class="sect2">')
        if (candidate := doc.find(marker, index + len(needle))) >= 0
    ]
    end = min(end_candidates) if end_candidates else min(len(doc), index + 12000)
    return doc[start:end]


def _extract_instruction_section_semantics(doc: str, mnemonic: str) -> dict[str, str]:
    section = _section_slice(doc, mnemonic)
    if not section:
        return {}

    description = ""
    paragraphs = re.findall(
        r"<div class=\"paragraph\">(.*?)</div>", section, re.IGNORECASE | re.DOTALL
    )
    for raw in paragraphs:
        text = _strip_tags(raw)
        if not text:
            continue
        description = text
        if mnemonic.casefold() in text.casefold():
            break

    operation = ""
    op_heading = re.search(r"<dt[^>]*>\s*Operation\s*</dt>", section, re.IGNORECASE | re.DOTALL)
    if op_heading:
        pre_match = re.search(
            r"<pre[^>]*>(.*?)</pre>", section[op_heading.end() :], re.IGNORECASE | re.DOTALL
        )
        if pre_match:
            operation = _strip_tags(pre_match.group(1))
    if not operation:
        for pre in re.findall(r"<pre[^>]*>(.*?)</pre>", section, re.IGNORECASE | re.DOTALL):
            text = _strip_tags(pre)
            if mnemonic.casefold() in text.casefold():
                operation = text
                break
    if not operation:
        listings = re.findall(
            r"<div class=\"listingblock\">(.*?)</div>\s*</div>", section, re.IGNORECASE | re.DOTALL
        )
        for listing in listings:
            text = _strip_tags(listing)
            if mnemonic.casefold() in text.casefold():
                operation = text
                break
    result: dict[str, str] = {}
    if description:
        result["Description"] = description
    if operation:
        result["Operation"] = operation
    return result


def _normalize_sections(item: dict[str, Any]) -> dict[str, str]:
    sections: dict[str, str] = {}
    for key in ("description", "doc_sections", "sections", "descriptions"):
        value = item.get(key)
        if isinstance(value, dict):
            sections.update(_string_map(value))
    if not sections:
        description = _string(item.get("description_text") or item.get("description"))
        if description:
            sections["Description"] = description
        operation = _string(
            item.get("operation") or item.get("operation_text") or item.get("pseudocode")
        )
        if operation:
            sections["Operation"] = operation
    return sections


def _docs_page_map(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    docs_pages = payload.get("docs_pages") or payload.get("doc_pages") or payload.get("docs")
    if isinstance(docs_pages, dict):
        return {
            _string(key): _string(value)
            for key, value in docs_pages.items()
            if _string(key) and _string(value)
        }
    if isinstance(docs_pages, list):
        mapped: dict[str, str] = {}
        for item in docs_pages:
            if not isinstance(item, dict):
                continue
            url = _string(item.get("url") or item.get("reference_url"))
            page = _string(item.get("html") or item.get("content"))
            if url and page:
                mapped[url] = page
        return mapped
    return {}


def _doc_candidates(url: str, docs_pages: dict[str, str]) -> list[str]:
    if not url:
        return []
    base = url.split("#", 1)[0]
    candidates = [url]
    if base != url:
        candidates.append(base)
    return [candidate for candidate in candidates if candidate in docs_pages]


def _instruction_semantics(
    item: dict[str, Any], url: str, docs_pages: dict[str, str]
) -> dict[str, str]:
    sections = _normalize_sections(item)
    if sections.get("Description") and sections.get("Operation"):
        return sections
    mnemonic = _string(
        item.get("mnemonic") or item.get("name") or item.get("instruction") or item.get("syntax")
    ).casefold()
    for candidate in _doc_candidates(url, docs_pages):
        page = docs_pages[candidate]
        if not sections.get("Description"):
            description = _extract_html_section(page, "Description")
            if description:
                sections["Description"] = description
        if not sections.get("Operation"):
            operation = _extract_html_section(page, "Operation")
            if operation:
                sections["Operation"] = operation
        if not sections.get("Description") or not sections.get("Operation"):
            extracted = _extract_instruction_section_semantics(page, mnemonic)
            for key, value in extracted.items():
                if not sections.get(key):
                    sections[key] = value
        if sections.get("Description") and sections.get("Operation"):
            break
    return sections


def _normalize_policy(value: Any) -> str:
    normalized = _string(value).casefold().replace("-", "").replace("_", "")
    aliases = {
        "": "agnostic",
        "ta": "agnostic",
        "ma": "agnostic",
        "tama": "agnostic",
        "tum": "tum",
        "tu": "tu",
        "mu": "mu",
        "tumu": "tumu",
    }
    return aliases.get(normalized, normalized)


def _normalize_masking(value: Any) -> str:
    normalized = _string(value).casefold().replace("-", "").replace("_", "")
    if normalized in {"", "nomask", "unmasked", "false"}:
        return "unmasked"
    if normalized in {"mask", "masked", "true"}:
        return "masked"
    return normalized


def _tail_policy(policy: str) -> str:
    if "tu" in policy:
        return "undisturbed"
    return "agnostic"


def _mask_policy(policy: str, masking: str) -> str:
    if masking != "masked":
        return ""
    if "mu" in policy:
        return "undisturbed"
    return "agnostic"


def _instruction_form(item: dict[str, Any], mnemonic: str) -> str:
    explicit = _string(item.get("form"))
    if explicit:
        return explicit
    policy = _normalize_policy(item.get("policy") or item.get("tail_policy"))
    masking = _normalize_masking(item.get("masking"))
    suffix: list[str] = []
    if masking and masking != "unmasked":
        suffix.append(masking)
    if policy and policy not in {"agnostic", "default"}:
        suffix.append(policy)
    if suffix:
        return f"{mnemonic} [{' '.join(suffix)}]"
    return mnemonic


_RV_ENC_KEYED_RE = re.compile(
    r"^[ ]{0,2}(RV32|RV64|RV32E|RV64E):[ \t]*\n((?:^[ ]{3,}.*\n?)*)", re.MULTILINE
)
_RV_ENC_MATCH_RE = re.compile(r"^[ \t]*match:[ \t]*\"?([01xX-]+)\"?", re.MULTILINE)


def _rv_bits_to_ints(bits: str) -> tuple[int, int, int]:
    mask = match = 0
    for char in bits:
        mask <<= 1
        match <<= 1
        if char == "1":
            mask |= 1
            match |= 1
        elif char == "0":
            mask |= 1
    return len(bits), mask, match


def _rv_encoding_patterns(block: str) -> list[list]:
    """Parse a kept YAML encoding block into [base, bits, mask, match] rows.

    A flat ``match:`` applies to every base; a per-base block (RV32/RV64/...)
    yields one pattern per base named in the block.
    """
    patterns: list[list] = []
    for keyed in _RV_ENC_KEYED_RE.finditer(block):
        matched = _RV_ENC_MATCH_RE.search(keyed.group(2))
        if matched:
            bits, mask, match = _rv_bits_to_ints(matched.group(1).strip())
            patterns.append([keyed.group(1), bits, mask, match])
    if not patterns:
        matched = _RV_ENC_MATCH_RE.search(block)
        if matched:
            bits, mask, match = _rv_bits_to_ints(matched.group(1).strip())
            patterns.append(["", bits, mask, match])
    return patterns


def parse_riscv_instruction_payload(text: str) -> list[InstructionRecord]:
    payload = json.loads(text)
    docs_pages = _docs_page_map(payload)
    if isinstance(payload, dict) and payload.get("format") == "riscv-unified-db-v1":
        candidates = payload.get("instructions") or []
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = payload.get("instructions") or payload.get("records") or []

    records: list[InstructionRecord] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        mnemonic = _string(
            item.get("mnemonic")
            or item.get("name")
            or item.get("instruction")
            or item.get("syntax")
            or item.get("asm")
        ).casefold()
        if not mnemonic:
            continue
        metadata = _string_map(item.get("metadata"))
        url = _string(item.get("url") or item.get("reference_url") or item.get("reference"))
        if url:
            metadata["url"] = url
        policy = _normalize_policy(
            item.get("policy")
            or item.get("tail_policy")
            or metadata.get("policy")
            or metadata.get("tail_policy")
            or "agnostic"
        )
        masking = _normalize_masking(item.get("masking") or metadata.get("masking") or "unmasked")
        metadata.setdefault("policy", policy)
        metadata.setdefault("masking", masking)
        metadata.setdefault("tail_policy", _tail_policy(policy))
        mask_policy = _string(
            item.get("mask_policy") or metadata.get("mask_policy") or _mask_policy(policy, masking)
        )
        if mask_policy:
            metadata.setdefault("mask_policy", mask_policy)
        if item.get("extension"):
            metadata.setdefault("extension", _string(item.get("extension")))
        description = _instruction_semantics(item, url, docs_pages)
        record = InstructionRecord(
            mnemonic=mnemonic,
            form=_instruction_form(item, mnemonic),
            summary=_string(
                item.get("summary")
                or item.get("brief")
                or item.get("description_text")
                or f"{mnemonic} instruction."
            ).rstrip(".")
            + ".",
            architecture="riscv",
            isa=_normalize_isa(
                item.get("isa")
                or item.get("extensions")
                or item.get("extension")
                or metadata.get("extensions")
                or ["V"]
            ),
            operand_details=[
                {key: str(value).strip() for key, value in operand.items() if str(value).strip()}
                for operand in (item.get("operand_details") or [])
                if isinstance(operand, dict)
            ],
            metadata=metadata,
            aliases=_string_list(item.get("aliases")),
            description=description,
            source="riscv-unified-db",
        )
        # RVV masked/unmasked variants share one encoding (spec K3): the key
        # lives on the base-instruction row only.
        encoding_block = _string(item.get("encoding"))
        if encoding_block and record.form == record.mnemonic:
            patterns = _rv_encoding_patterns(encoding_block)
            if patterns:
                record.enc_bits = patterns[0][1]
                record.enc_mask = patterns[0][2]
                record.enc_match = patterns[0][3]
                record.enc_extra = patterns
        records.append(record)
    return records


def _infer_intrinsic_policy(name: str) -> tuple[str, str]:
    lowered = name.casefold()
    policy = "agnostic"
    masking = (
        "masked"
        if lowered.endswith(("_m", "_mu", "_tum", "_tumu")) or "_m_" in lowered
        else "unmasked"
    )
    for suffix in ("_tumu", "_tum", "_mu", "_tu"):
        if lowered.endswith(suffix):
            policy = suffix.removeprefix("_")
            break
    return policy, masking


def _normalize_instruction_ref(raw_ref: dict[str, Any], intrinsic_name: str) -> dict[str, str]:
    ref_name = _string(raw_ref.get("name") or raw_ref.get("mnemonic") or raw_ref.get("instruction"))
    form = _string(raw_ref.get("form") or raw_ref.get("syntax") or ref_name)
    inferred_policy, inferred_masking = _infer_intrinsic_policy(intrinsic_name)
    policy = _normalize_policy(
        raw_ref.get("policy") or raw_ref.get("tail_policy") or inferred_policy
    )
    masking = _normalize_masking(raw_ref.get("masking") or inferred_masking)
    ref = {
        "architecture": "riscv",
        "name": ref_name,
        "form": form or ref_name,
        "isa": _string(raw_ref.get("isa") or raw_ref.get("extension")),
        "policy": policy,
        "masking": masking,
        "tail_policy": _string(raw_ref.get("tail_policy") or _tail_policy(policy)),
        "mask_policy": _string(raw_ref.get("mask_policy") or _mask_policy(policy, masking)),
    }
    return {key: value for key, value in ref.items() if value}


def parse_riscv_intrinsics_payload(text: str) -> list[IntrinsicRecord]:
    payload = json.loads(text)
    if isinstance(payload, dict) and payload.get("format") == "riscv-rvv-intrinsics-v1":
        candidates = payload.get("intrinsics") or []
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = payload.get("intrinsics") or payload.get("records") or []

    records: list[IntrinsicRecord] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = _string(item.get("name"))
        if not name:
            continue
        refs: list[dict[str, str]] = []
        rendered_instructions: list[str] = []
        for raw_ref in item.get("instruction_refs") or []:
            if not isinstance(raw_ref, dict):
                continue
            ref = _normalize_instruction_ref(raw_ref, name)
            if not ref["name"]:
                continue
            if not ref["form"]:
                ref["form"] = ref["name"]
            rendered_instructions.append(ref["form"])
            refs.append(ref)
        if not refs:
            for raw_instruction in item.get("instructions") or []:
                ref = _normalize_instruction_ref(
                    {"name": raw_instruction, "form": raw_instruction}, name
                )
                if ref["name"]:
                    rendered_instructions.append(ref["form"])
                    refs.append(ref)
        inferred_policy, inferred_masking = _infer_intrinsic_policy(name)
        metadata = _string_map(item.get("metadata"))
        metadata.setdefault("policy", inferred_policy)
        metadata.setdefault("masking", inferred_masking)
        metadata.setdefault("tail_policy", _tail_policy(inferred_policy))
        mask_policy = _mask_policy(inferred_policy, inferred_masking)
        if mask_policy:
            metadata.setdefault("mask_policy", mask_policy)
        records.append(
            IntrinsicRecord(
                name=name,
                signature=_string(item.get("signature") or name),
                description=_string(item.get("description")),
                header=_string(item.get("header") or "riscv_vector.h"),
                url=_string(
                    item.get("url") or item.get("reference_url") or RISCV_RVV_INTRINSICS_PROJECT_URL
                ),
                architecture="riscv",
                isa=_normalize_isa(item.get("isa") or ["V"]),
                category=_string(item.get("category") or "RVV"),
                subcategory=_string(item.get("subcategory")),
                instructions=rendered_instructions,
                instruction_refs=refs,
                metadata=metadata,
                doc_sections=_string_map(item.get("doc_sections") or item.get("sections")),
                notes=_string_list(item.get("notes")),
                aliases=_string_list(item.get("aliases")),
                source="rvv-intrinsic-doc",
            )
        )
    return records
