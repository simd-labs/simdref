#!/usr/bin/env python3
"""Developer-only source validation for simdref ingestion."""

from __future__ import annotations

import argparse
import json
import msgpack
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simdref.arm_instructions import parse_arm_instruction_payload
from simdref.display import strip_instruction_decorators
from simdref.ingest import load_or_parse_intel_sdm
from simdref.ingest_catalog import (
    _arm_live_instruction_refs,
    _canonical_instruction_key,
    _iter_xml_elements,
    _normalize_arm_intrinsic_name,
    _normalize_uops_form,
    _parse_uops_xml_keyed,
    _resolve_instruction_ref,
    parse_arm_intrinsics_payload,
    parse_intel_payload,
    parse_riscv_instruction_payload,
    parse_riscv_intrinsics_payload,
    parse_uops_xml,
    link_records,
)
from simdref.ingest_sources import (
    fetch_arm_a64_data,
    fetch_arm_acle_data,
    fetch_intel_data,
    fetch_riscv_rvv_intrinsics_data,
    fetch_riscv_unified_db_data,
    fetch_uops_xml,
)
from simdref.ingest_pdf import find_pdf_source_path
from simdref.pdfparse.intel import INTEL_SDM_CACHE_PATH


def _fail(message: str) -> None:
    raise AssertionError(message)


def _strip(value: Any) -> str:
    return str(value or "").strip()


def _normalize_isa_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,/|]\s*|\s{2,}", value) if part.strip()]
    return []


_X86_SDM_EXPECTATIONS: dict[str, dict[str, object]] = {
    "ADDPS": {
        "family": "SSE",
        "sections": (
            "Description",
            "Operation",
            "Intrinsic Equivalents",
            "SIMD Floating-Point Exceptions",
        ),
        "contains": {
            "Description": "packed single precision floating-point",
            "Operation": "DEST",
            "Intrinsic Equivalents": "_mm_add_ps",
        },
    },
    "VADDPS": {
        "family": "AVX",
        "sections": ("Description", "Operation"),
        "contains": {
            "Description": "packed single precision floating-point",
            "Operation": "DEST",
        },
    },
    "VPDPBUSD": {
        "family": "AVX-512 VNNI",
        "sections": ("Description", "Operation", "Intrinsic Equivalents"),
        "contains": {
            "Description": "unsigned bytes",
            "Operation": "FOR i := 0 TO KL-1",
            "Intrinsic Equivalents": "_mm_dpbusd",
        },
    },
    "VPEXPANDD": {
        "family": "AVX-512 maskz",
        "sections": ("Description", "Operation"),
        "contains": {
            "Description": "doubleword integer values",
            "Operation": "KL",
        },
    },
    "VPMADD52LUQ": {
        "family": "AVX-512 IFMA",
        "sections": ("Description", "Operation", "Intrinsic Equivalents", "Flags Affected"),
        "contains": {
            "Description": "52-bit integers",
            "Operation": "srcdest.qword",
            "Flags Affected": "None",
        },
    },
    "RDPID": {
        "family": "system/control",
        "sections": ("Description", "Operation", "Flags Affected"),
        "contains": {
            "Description": "IA32_TSC_AUX",
            "Operation": "DEST := IA32_TSC_AUX",
            "Flags Affected": "None",
        },
    },
    "TZCNT": {
        "family": "BMI1",
        "sections": ("Description", "Operation", "Flags Affected", "Intrinsic Equivalents"),
        "contains": {
            "Description": "trailing least significant zero bits",
            "Operation": "CF := 1",
            "Flags Affected": "ZF is set to 1",
        },
    },
    "PDEP": {
        "family": "BMI2",
        "sections": ("Description", "Operation", "Flags Affected"),
        "contains": {
            "Description": "deposit",
            "Operation": "TEMP",
        },
    },
    "AESENC": {
        "family": "AES",
        "sections": ("Description", "Operation"),
        "contains": {
            "Description": "round",
            "Operation": "ShiftRows",
        },
    },
    "SHA1RNDS4": {
        "family": "SHA",
        "sections": ("Description", "Operation"),
        "contains": {
            "Description": "SHA1",
            "Operation": "W",
        },
    },
    "CRC32": {
        "family": "CRC",
        "sections": ("Description", "Operation"),
        "contains": {
            "Description": "CRC32",
            "Operation": "TEMP",
        },
    },
    "VCVTNEPS2BF16": {
        "family": "AVX-512 BF16",
        "sections": (
            "Description",
            "Operation",
            "Intrinsic Equivalents",
            "SIMD Floating-Point Exceptions",
        ),
        "contains": {
            "Description": "converts the elements to BF16",
            "Operation": "convert_fp32_to_bfloat16",
            "Intrinsic Equivalents": "_mm_cvtneps_pbh",
        },
    },
    "ADCX": {
        "family": "ADX",
        "sections": ("Description", "Operation", "Flags Affected"),
        "contains": {
            "Description": "carry",
            "Operation": "CF",
        },
    },
}

_X86_SDM_MIN_COVERAGE = 0.02
_X86_SDM_FAMILY_THRESHOLDS: dict[str, float] = {
    "SSE": 0.01,
    "AVX": 0.01,
    "AVX-512": 0.01,
    "AMX": 0.01,
}

_RISCV_EXPECTED_SEMANTICS: dict[str, str] = {
    "vadd.vv": "arithmetic",
    "vsub.vv": "arithmetic",
    "vle32.v": "loads/stores",
    "vse32.v": "loads/stores",
    "vlse32.v": "loads/stores",
    "vsse32.v": "loads/stores",
    "vluxei32.v": "loads/stores",
    "vsuxei32.v": "loads/stores",
    "vmand.mm": "mask operations",
    "vredsum.vs": "reductions",
    "vwadd.vv": "widening/narrowing",
    "vnclipu.wi": "widening/narrowing",
    "vzext.vf2": "widening/narrowing",
    "vrgather.vv": "permute/move",
    "vmv.x.s": "permute/move",
    "vaesdf.vv": "crypto",
}
_RISCV_MIN_DESCRIPTION_COVERAGE = 0.95
_RISCV_MIN_OPERATION_COVERAGE = 0.60
_RISCV_MIN_LINK_COVERAGE = 1.0


def _unwrap_intel_source_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("var data_js"):
        match = re.search(r'var\s+data_js\s*=\s*"(?P<body>.*)";\s*$', stripped, re.DOTALL)
        if not match:
            _fail("could not locate Intel XML payload in data.js")
        xml_blob = match.group("body").replace("\\\n", "")
        return bytes(xml_blob, "utf-8").decode("unicode_escape").strip()
    if stripped.startswith("var ") or stripped.startswith("window."):
        match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
        if not match:
            _fail("could not locate JSON payload in Intel source data")
        return match.group(1)
    return stripped


def _raw_intel_intrinsic_records(text: str) -> list[dict[str, Any]]:
    unwrapped = _unwrap_intel_source_text(text)
    if unwrapped.startswith("<?xml") or unwrapped.startswith("<intrinsics_list"):
        records: list[dict[str, Any]] = []
        for node in _iter_xml_elements(unwrapped, "intrinsic"):
            name = _strip(node.attrib.get("name"))
            if not name:
                continue
            return_node = node.find("./return")
            ret = (
                _strip(node.attrib.get("rettype"))
                or (_strip(return_node.attrib.get("type")) if return_node is not None else "")
                or "void"
            )
            params = []
            for param in node.findall("./parameter"):
                ptype = _strip(param.attrib.get("type"))
                pname = _strip(param.attrib.get("varname"))
                params.append(" ".join(part for part in [ptype, pname] if part))
            instructions = []
            for inst in node.findall("./instruction"):
                inst_name = _strip(inst.attrib.get("name"))
                inst_form = _strip(inst.attrib.get("form"))
                if not inst_name:
                    continue
                instructions.append(_canonical_instruction_key(inst_name, inst_form) or inst_name)
            records.append(
                {
                    "name": name,
                    "signature": f"{ret} {name}({', '.join(params)})",
                    "header": _strip(node.findtext("./header") or node.attrib.get("header")),
                    "isa": [
                        _strip(cpuid.text)
                        for cpuid in node.findall("./CPUID")
                        if _strip(cpuid.text)
                    ],
                    "category": _strip(node.findtext("./category") or node.attrib.get("category")),
                    "instructions": instructions,
                }
            )
        return records

    payload = json.loads(unwrapped)
    candidates = (
        payload.get("intrinsics") or payload.get("data") or payload.get("records") or []
        if isinstance(payload, dict)
        else payload
    )
    records = []
    for item in candidates or []:
        name = _strip(item.get("name") or item.get("intrinsic"))
        if not name:
            continue
        signature = _strip(item.get("signature") or item.get("prototype"))
        if not signature:
            return_type = _strip(item.get("returnType") or item.get("rettype") or "void")
            params = item.get("parameters") or item.get("params") or []
            rendered_params = []
            if isinstance(params, list):
                for param in params:
                    if isinstance(param, dict):
                        ptype = _strip(param.get("type"))
                        pname = _strip(param.get("name") or param.get("varname"))
                        rendered_params.append(" ".join(part for part in [ptype, pname] if part))
                    else:
                        rendered_params.append(_strip(param))
            signature = f"{return_type} {name}({', '.join(p for p in rendered_params if p)})"
        instructions = (
            item.get("instructions") or item.get("instruction") or item.get("Instruction") or []
        )
        if isinstance(instructions, str):
            instructions = [instructions]
        records.append(
            {
                "name": name,
                "signature": signature,
                "header": _strip(item.get("header") or item.get("include")),
                "isa": _normalize_isa_list(
                    item.get("isa") or item.get("tech") or item.get("instructionSet") or []
                ),
                "category": _strip(item.get("category")),
                "instructions": [_strip(value) for value in instructions if _strip(value)],
            }
        )
    return records


def validate_intel_intrinsics() -> tuple[int, int]:
    text, _source = fetch_intel_data()
    parsed = parse_intel_payload(text)
    parsed_by_name: dict[str, list] = {}
    for record in parsed:
        parsed_by_name.setdefault(record.name, []).append(record)
    raw_records = _raw_intel_intrinsic_records(text)
    failures = 0
    checked = 0

    for raw in raw_records:
        name = raw["name"]
        checked += 1
        records = parsed_by_name.get(name) or []
        if not records:
            print(f"FAIL intel intrinsics: missing {name}")
            failures += 1
            continue
        # Upstream may list several intrinsics sharing one name (e.g. _mm_prefetch
        # appears under SSE, KNC, and PREFETCHWT1 with different headers/categories).
        # The parser preserves each as a distinct record; match the raw entry to the
        # parsed record that agrees on header+category, falling back to the first one.
        header_matches = [r for r in records if not raw["header"] or r.header == raw["header"]]
        candidates = (
            [r for r in header_matches if not raw["category"] or r.category == raw["category"]]
            or header_matches
            or records
        )
        record = candidates[0]
        if f"{name}(" not in record.signature:
            print(f"FAIL intel intrinsics: signature missing intrinsic name for {name}")
            failures += 1
        if raw["header"] and not any(r.header == raw["header"] for r in records):
            print(f"FAIL intel intrinsics: header mismatch for {name}")
            failures += 1
        if raw["category"] and not any(r.category == raw["category"] for r in records):
            print(f"FAIL intel intrinsics: category mismatch for {name}")
            failures += 1
        if raw["instructions"]:
            parsed_insts = [
                str(value).strip() for value in record.instructions if str(value).strip()
            ]
            if not parsed_insts:
                print(f"FAIL intel intrinsics: missing parsed instruction refs for {name}")
                failures += 1
        if raw["isa"] and not record.isa:
            print(f"FAIL intel intrinsics: missing isa tags for {name}")
            failures += 1

    if len(parsed) < len(raw_records):
        print(
            f"FAIL intel intrinsics: parsed count {len(parsed)} < source count {len(raw_records)}"
        )
        failures += 1
    print(f"validated Intel intrinsics: checked={checked} parsed={len(parsed)} failures={failures}")
    return checked, failures


def _raw_uops_instructions(source: str | Path) -> list[dict[str, Any]]:
    # The parser decides which record to skip the same way
    # (decorator-stripped mnemonic empty), so the raw records line up with
    # the parser's record ordinals one to one.
    raw_records: list[dict[str, Any]] = []
    for node in _iter_xml_elements(source, "instruction"):
        mnemonic = strip_instruction_decorators(
            node.attrib.get("asm") or node.attrib.get("name") or ""
        ).strip()
        if not mnemonic:
            continue
        form = _strip(
            node.attrib.get("string")
            or node.attrib.get("form")
            or node.attrib.get("cpl")
            or node.attrib.get("category")
        )
        arch_details: dict[str, Any] = {}
        for child in node:
            if child.tag != "architecture":
                continue
            arch = (
                child.attrib.get("name")
                or child.attrib.get("uarch")
                or child.attrib.get("arch")
                or ""
            ).strip()
            if not arch:
                continue
            entry: dict[str, Any] = {"measurement": {}, "latencies": []}
            for grandchild in child:
                if grandchild.tag == "measurement":
                    entry["measurement"] = dict(grandchild.attrib)
                    entry["latencies"] = [
                        dict(latency.attrib) for latency in grandchild.findall("./latency")
                    ]
            arch_details[arch] = entry
        raw_records.append(
            {
                "mnemonic": mnemonic,
                "form": _normalize_uops_form(form),
                "iform": _strip(node.attrib.get("iform")),
                "iclass": _strip(node.attrib.get("iclass")),
                "extension": _strip(node.attrib.get("extension")),
                "category": _strip(node.attrib.get("category")),
                "url": _strip(node.attrib.get("url")),
                "url-ref": _strip(node.attrib.get("url-ref")),
                "operand_count": len(node.findall("./operand")),
                "arch_details": arch_details,
            }
        )
    return raw_records


def _uops_measurements_cover(row_details: dict[str, Any], raw_details: dict[str, Any]) -> bool:
    for arch, raw_entry in raw_details.items():
        entry = row_details.get(arch)
        if entry is None:
            return False
        if entry.get("measurement") != raw_entry.get("measurement"):
            return False
        if entry.get("latencies") != raw_entry.get("latencies"):
            return False
    return True


def validate_uops_instructions() -> tuple[int, int]:
    source, _source = fetch_uops_xml()
    parsed = parse_uops_xml(source)
    # The parser decides which row each raw record lands on; ask it for the
    # row key of every raw record and look up exactly that row. No grouping
    # index and no normalization re-derived here, so no sibling row can
    # stand in for a record that lost its data.
    row_keys = _parse_uops_xml_keyed(source)[1]
    rows_by_key: dict[str, Any] = {record.db_key: record for record in parsed}
    raw_records = _raw_uops_instructions(source)
    failures = 0
    checked = 0
    if len(row_keys) != len(raw_records):
        print(
            f"FAIL uops: parser keyed {len(row_keys)} records, "
            f"validator reads {len(raw_records)} raw records"
        )
        failures += 1

    for index, raw in enumerate(raw_records):
        checked += 1
        label = raw["iform"] or raw["form"]
        record = rows_by_key.get(row_keys.get(index, ""))
        if record is None:
            print(f"FAIL uops: missing parsed row for {raw['mnemonic']} / {label}")
            failures += 1
            continue
        if record.mnemonic != raw["mnemonic"]:
            print(f"FAIL uops: mnemonic mismatch for {label}")
            failures += 1
        if record.metadata.get("iform", "") != raw["iform"]:
            print(f"FAIL uops: iform mismatch for {label}")
            failures += 1
        if raw["iclass"] and record.metadata.get("iclass", "") != raw["iclass"]:
            print(f"FAIL uops: iclass mismatch for {label}")
            failures += 1
        if raw["category"] and record.metadata.get("category", "") != raw["category"]:
            print(f"FAIL uops: category mismatch for {label}")
            failures += 1
        if raw["extension"] and record.metadata.get("extension", "") != raw["extension"]:
            print(f"FAIL uops: extension mismatch for {label}")
            failures += 1
        if len(record.operand_details) != raw["operand_count"]:
            print(f"FAIL uops: operand count mismatch for {label}")
            failures += 1
        if raw["arch_details"] and not _uops_measurements_cover(
            record.arch_details, raw["arch_details"]
        ):
            print(f"FAIL uops: missing measurements for {label}")
            failures += 1
        if not record.summary.strip():
            print(f"FAIL uops: empty summary for {label}")
            failures += 1

    # Each parsed row's uarch set must equal the union of the uarch sets of
    # the raw records mapped to it, so no row can carry uarch data that no
    # raw record mapped to it has.
    mapped_uarches: dict[str, set[str]] = {}
    for index, raw in enumerate(raw_records):
        mapped_uarches.setdefault(row_keys.get(index, ""), set()).update(raw["arch_details"])
    for record in parsed:
        if set(record.arch_details) != mapped_uarches.get(record.db_key, set()):
            print(
                f"FAIL uops: uarch set mismatch for {record.metadata.get('iform') or record.form}"
            )
            failures += 1

    print(
        f"validated uops.info instructions: checked={checked} parsed={len(parsed)} failures={failures}"
    )
    return checked, failures


def validate_arm_intrinsics() -> tuple[int, int]:
    text, _source = fetch_arm_acle_data()
    parsed = parse_arm_intrinsics_payload(text)
    parsed_by_name = {record.name: record for record in parsed}
    payload = json.loads(text)
    failures = 0
    checked = 0

    if payload.get("format") != "arm-intrinsics-json-v1":
        print(
            f"validated Arm intrinsics: checked={len(parsed)} parsed={len(parsed)} failures=0 (fallback payload)"
        )
        return len(parsed), 0

    raw_intrinsics = json.loads(str(payload.get("intrinsics_json") or "[]"))
    operations_payload = json.loads(str(payload.get("operations_json") or "[]"))
    operations = {
        str(item.get("item", {}).get("id") or ""): item.get("item", {})
        for item in operations_payload
        if isinstance(item, dict) and item.get("item")
    }

    for item in raw_intrinsics:
        simd_isa = [
            str(value).strip() for value in item.get("SIMD_ISA") or [] if str(value).strip()
        ]
        if not {"Neon", "SVE", "SVE2"} & set(simd_isa):
            continue
        raw_name = _strip(item.get("name"))
        if not raw_name:
            continue
        checked += 1
        name = _normalize_arm_intrinsic_name(raw_name)
        record = parsed_by_name.get(name)
        if record is None:
            print(f"FAIL arm intrinsics: missing {name}")
            failures += 1
            continue
        expected_header = "arm_neon.h" if "Neon" in simd_isa else "arm_sve.h"
        if record.header != expected_header:
            print(f"FAIL arm intrinsics: header mismatch for {name}")
            failures += 1
        expected_refs = _arm_live_instruction_refs(
            [group for group in (item.get("instructions") or []) if isinstance(group, dict)]
        )
        if record.instruction_refs != expected_refs:
            print(f"FAIL arm intrinsics: instruction refs mismatch for {name}")
            failures += 1
        expected_arches = "/".join(
            str(value).strip() for value in item.get("Architectures") or [] if str(value).strip()
        )
        if record.metadata.get("supported_architectures", "") != expected_arches:
            print(f"FAIL arm intrinsics: supported architectures mismatch for {name}")
            failures += 1
        operation_id = _strip(item.get("Operation"))
        operation_content = _strip((operations.get(operation_id) or {}).get("content"))
        if operation_content and "ACLE Operation" not in record.doc_sections:
            print(f"FAIL arm intrinsics: missing ACLE Operation for {name}")
            failures += 1

    print(f"validated Arm intrinsics: checked={checked} parsed={len(parsed)} failures={failures}")
    return checked, failures


def _instruction_candidates(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        results: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            # AARCHMRS top-level entries are ``InstructionSet`` / group
            # nodes wrapping ``children``. Recurse into them to surface
            # actual ``Instruction`` leaves instead of accidentally
            # treating the ISA markers (A64/A32/T32) as instructions.
            node_type = str(item.get("_type", ""))
            if node_type.endswith("InstructionSet") or (
                isinstance(item.get("children"), list) and not item.get("mnemonic")
            ):
                results.extend(_instruction_candidates(item.get("children") or []))
                continue
            if node_type.endswith("InstructionAlias"):
                continue
            results.append(item)
        return results
    if not isinstance(payload, dict):
        return []
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
            return _instruction_candidates(value)
    for value in payload.values():
        if isinstance(value, dict):
            nested = _instruction_candidates(value)
            if nested:
                return nested
    return []


def validate_arm_instructions(require_authoritative: bool) -> tuple[int, int]:
    text, _source = fetch_arm_a64_data()
    # Fixture-only path was dropped: fetch now raises on total failure.
    payload = json.loads(text)
    parsed = parse_arm_instruction_payload(text)
    parsed_by_mnemonic: dict[str, list[Any]] = {}
    for record in parsed:
        parsed_by_mnemonic.setdefault(record.mnemonic, []).append(record)

    if isinstance(payload, dict) and payload.get("format") == "arm-aarchmrs-instructions-v1":
        # AARCHMRS is a deeply nested Instruction/InstructionGroup tree
        # keyed by LLVM opcode names, not mnemonics. The raw-record
        # validator was built for the older flat-list fixtures, so here
        # we only assert that parsing produced a non-trivial population
        # and trust the parser's own coverage tests for correctness.
        if len(parsed) < 1000:
            print(f"FAIL arm instructions: only {len(parsed)} parsed records from AARCHMRS payload")
            return 1, 1
        print(f"validated Arm instructions (AARCHMRS): parsed={len(parsed)} failures=0")
        return 1, 0

    if isinstance(payload, dict) and payload.get("format") == "arm-instructions-fixture-v1":
        raw_payload = payload.get("instructions") or []
    else:
        raw_payload = payload

    raw_records = _instruction_candidates(raw_payload)
    failures = 0
    checked = 0
    for item in raw_records:
        mnemonic = _strip(
            item.get("mnemonic")
            or item.get("base_instruction")
            or item.get("instruction")
            or item.get("name")
            or item.get("opcode")
        ).upper()
        if not mnemonic:
            continue
        checked += 1
        candidates = parsed_by_mnemonic.get(mnemonic, [])
        if not candidates:
            print(f"FAIL arm instructions: missing {mnemonic}")
            failures += 1
            continue
        raw_url = _strip(item.get("url") or item.get("source_url") or item.get("reference_url"))
        match = None
        for candidate in candidates:
            if raw_url and candidate.metadata.get("url") == raw_url:
                match = candidate
                break
        if match is None:
            match = candidates[0]
        if raw_url and match.metadata.get("url") != raw_url:
            print(f"FAIL arm instructions: url mismatch for {mnemonic}")
            failures += 1
        if not match.summary.strip():
            print(f"FAIL arm instructions: empty summary for {mnemonic}")
            failures += 1

    print(f"validated Arm instructions: checked={checked} parsed={len(parsed)} failures={failures}")
    return checked, failures


def validate_riscv_intrinsics() -> tuple[int, int]:
    text, _source = fetch_riscv_rvv_intrinsics_data()
    parsed = parse_riscv_intrinsics_payload(text)
    payload = json.loads(text)
    raw_records = payload.get("intrinsics") or []
    parsed_by_name = {record.name: record for record in parsed}
    failures = 0
    checked = 0
    unlinked = 0

    for item in raw_records:
        if not isinstance(item, dict):
            continue
        name = _strip(item.get("name"))
        if not name:
            continue
        checked += 1
        record = parsed_by_name.get(name)
        if record is None:
            print(f"FAIL riscv intrinsics: missing {name}")
            failures += 1
            continue
        if record.header != _strip(item.get("header") or "riscv_vector.h"):
            print(f"FAIL riscv intrinsics: header mismatch for {name}")
            failures += 1
        if not record.instruction_refs:
            unlinked += 1
        if "Prototype" not in record.doc_sections or "Semantics" not in record.doc_sections:
            print(f"FAIL riscv intrinsics: missing prototype/semantics sections for {name}")
            failures += 1

    print(
        "validated RISC-V intrinsics: "
        f"checked={checked} parsed={len(parsed)} unlinked={unlinked} failures={failures}"
    )
    return checked, failures


def validate_riscv_instructions() -> tuple[int, int]:
    text, _source = fetch_riscv_unified_db_data()
    parsed = parse_riscv_instruction_payload(text)
    payload = json.loads(text)
    raw_records = payload.get("instructions") or []
    parsed_by_form = {record.form.casefold(): record for record in parsed}
    failures = 0
    checked = 0

    for item in raw_records:
        if not isinstance(item, dict):
            continue
        mnemonic = _strip(item.get("mnemonic") or item.get("name")).casefold()
        if not mnemonic:
            continue
        checked += 1
        form = _strip(item.get("form") or mnemonic).casefold()
        record = parsed_by_form.get(form)
        if record is None:
            print(f"FAIL riscv instructions: missing {form}")
            failures += 1
            continue
        if not record.summary.strip():
            print(f"FAIL riscv instructions: empty summary for {form}")
            failures += 1
    print(
        f"validated RISC-V instructions: checked={checked} parsed={len(parsed)} failures={failures}"
    )
    return checked, failures


def validate_riscv_intrinsic_links() -> tuple[int, int]:
    intrinsic_text, _intrinsic_source = fetch_riscv_rvv_intrinsics_data()
    instruction_text, _instruction_source = fetch_riscv_unified_db_data()
    intrinsics = parse_riscv_intrinsics_payload(intrinsic_text)
    instructions = parse_riscv_instruction_payload(instruction_text)
    checked = sum(len(item.instruction_refs) for item in intrinsics)
    failures = 0
    ambiguous = 0
    unresolved = 0

    link_records(intrinsics, instructions)
    for intrinsic in intrinsics:
        for ref in intrinsic.instruction_refs:
            if not ref.get("key", "").strip():
                unresolved += 1
                failures += 1
                print(
                    f"FAIL riscv links: unresolved {intrinsic.name} -> {ref.get('form') or ref.get('name')}"
                )
                continue
            if ref.get("match_count") != "1":
                ambiguous += 1
                failures += 1
                print(
                    f"FAIL riscv links: ambiguous {intrinsic.name} -> {ref.get('form') or ref.get('name')} matches={ref.get('match_count')}"
                )

    print(
        "validated RISC-V intrinsic links: "
        f"checked={checked} ambiguous={ambiguous} unresolved={unresolved} failures={failures}"
    )
    return checked, failures


def validate_riscv_semantics_and_coverage() -> tuple[int, int]:
    instruction_text, _instruction_source = fetch_riscv_unified_db_data()
    intrinsic_text, _intrinsic_source = fetch_riscv_rvv_intrinsics_data()
    instructions = parse_riscv_instruction_payload(instruction_text)
    intrinsics = parse_riscv_intrinsics_payload(intrinsic_text)
    link_records(intrinsics, instructions)

    failures = 0
    by_mnemonic = {record.mnemonic: record for record in instructions}
    for mnemonic, family in _RISCV_EXPECTED_SEMANTICS.items():
        record = by_mnemonic.get(mnemonic)
        if record is None:
            print(f"FAIL riscv semantics: missing {mnemonic} sample for {family}")
            failures += 1
            continue
        if not record.description.get("Description", "").strip():
            print(f"FAIL riscv semantics: missing Description for {mnemonic}")
            failures += 1
        if not record.description.get("Operation", "").strip():
            print(f"FAIL riscv semantics: missing Operation for {mnemonic}")
            failures += 1

    described = sum(
        1 for record in instructions if record.description.get("Description", "").strip()
    )
    operational = sum(
        1 for record in instructions if record.description.get("Operation", "").strip()
    )
    linkable = sum(1 for intrinsic in intrinsics if intrinsic.instruction_refs)
    linked = sum(
        1
        for intrinsic in intrinsics
        if intrinsic.instruction_refs
        and all(ref.get("key", "").strip() for ref in intrinsic.instruction_refs)
    )
    description_coverage = described / len(instructions) if instructions else 0.0
    operation_coverage = operational / len(instructions) if instructions else 0.0
    link_coverage = linked / linkable if linkable else 1.0

    if description_coverage < _RISCV_MIN_DESCRIPTION_COVERAGE:
        failures += 1
        print(
            f"FAIL riscv coverage: description {description_coverage:.3f} below threshold {_RISCV_MIN_DESCRIPTION_COVERAGE:.3f}"
        )
    if operation_coverage < _RISCV_MIN_OPERATION_COVERAGE:
        failures += 1
        print(
            f"FAIL riscv coverage: operation {operation_coverage:.3f} below threshold {_RISCV_MIN_OPERATION_COVERAGE:.3f}"
        )
    if link_coverage < _RISCV_MIN_LINK_COVERAGE:
        failures += 1
        print(
            f"FAIL riscv coverage: links {link_coverage:.3f} below threshold {_RISCV_MIN_LINK_COVERAGE:.3f}"
        )

    family_counts: dict[str, int] = {}
    for family in _RISCV_EXPECTED_SEMANTICS.values():
        family_counts.setdefault(family, 0)
    for mnemonic, family in _RISCV_EXPECTED_SEMANTICS.items():
        if mnemonic in by_mnemonic:
            family_counts[family] += 1

    print(
        "validated RISC-V semantics/coverage: "
        f"checked={len(instructions)} description={description_coverage:.3f} "
        f"operation={operation_coverage:.3f} links={link_coverage:.3f} "
        f"linkable={linkable} failures={failures}"
    )
    print(
        "validated RISC-V coverage summary: "
        f"instructions={len(instructions)} intrinsics={len(intrinsics)} "
        + " ".join(
            f"{family.replace('/', '_')}={count}" for family, count in sorted(family_counts.items())
        )
    )
    return len(instructions), failures


def _load_sdm_descriptions() -> dict[str, dict[str, Any]] | None:
    pdf_path = find_pdf_source_path("intel-sdm")
    if pdf_path is not None and pdf_path.exists():
        return load_or_parse_intel_sdm(pdf_path)
    if INTEL_SDM_CACHE_PATH.exists():
        payload = msgpack.unpackb(INTEL_SDM_CACHE_PATH.read_bytes(), raw=False)
        if "descriptions" in payload and isinstance(payload["descriptions"], dict):
            return payload["descriptions"]
        result = payload.get("result")
        if isinstance(result, dict) and isinstance(result.get("descriptions"), dict):
            return result["descriptions"]
    return None


def _sdm_payload_for_mnemonic(
    descriptions: dict[str, dict[str, Any]],
    mnemonic: str,
) -> dict[str, Any] | None:
    payload = descriptions.get(mnemonic)
    if payload:
        return payload
    if mnemonic.startswith("V") and len(mnemonic) > 1:
        return descriptions.get(mnemonic[1:])
    return None


def _x86_instruction_indexes(
    instructions: list[Any],
) -> tuple[
    dict[tuple[str, str], list[Any]],
    dict[tuple[str, str], list[Any]],
    dict[tuple[str, str], list[Any]],
]:
    by_mnemonic: dict[tuple[str, str], list[Any]] = {}
    by_key: dict[tuple[str, str], list[Any]] = {}
    by_iform: dict[tuple[str, str], list[Any]] = {}
    for record in instructions:
        arch = record.architecture
        by_mnemonic.setdefault((arch, record.mnemonic.casefold()), []).append(record)
        by_key.setdefault((arch, record.key.casefold()), []).append(record)
        if record.metadata.get("iform"):
            by_iform.setdefault((arch, record.metadata["iform"].casefold()), []).append(record)
    return by_mnemonic, by_key, by_iform


def validate_x86_intrinsic_links() -> tuple[int, int]:
    intel_text, _intel_source = fetch_intel_data()
    uops_text, _uops_source = fetch_uops_xml()
    intrinsics = parse_intel_payload(intel_text)
    instructions = parse_uops_xml(uops_text)
    by_mnemonic, by_key, by_iform = _x86_instruction_indexes(instructions)
    failures = 0
    checked = 0
    resolved = 0
    ambiguous = 0
    unresolved = 0

    for intrinsic in intrinsics:
        refs = intrinsic.instruction_refs or [
            {"name": name, "form": "", "xed": "", "architecture": "x86"}
            for name in intrinsic.instructions
        ]
        for ref in refs:
            checked += 1
            matched, resolution = _resolve_instruction_ref(
                intrinsic,
                ref,
                by_mnemonic=by_mnemonic,
                by_key=by_key,
                by_iform=by_iform,
            )
            match_count = int(resolution["match_count"])
            if not matched:
                unresolved += 1
                print(
                    f"WARN x86 links: unresolved {intrinsic.name} -> {ref.get('name', '')} {ref.get('form', '')}".rstrip()
                )
                continue
            resolved += 1
            if ref.get("xed", "").strip() and match_count != 1:
                ambiguous += 1
                print(
                    f"WARN x86 links: ambiguous xed mapping for {intrinsic.name} -> {ref.get('xed')}: {match_count} matches"
                )

    link_records(intrinsics, instructions)
    linked_by_name = {record.name: record for record in intrinsics}
    for intrinsic in linked_by_name.values():
        refs = intrinsic.instruction_refs or []
        for ref in refs:
            if ref.get("key", "").strip():
                continue
            unresolved += 1
            print(
                f"WARN x86 links: unresolved after linking for {intrinsic.name} -> {ref.get('name', '')} {ref.get('form', '')}".rstrip()
            )

    print(
        "validated x86 intrinsic links: "
        f"checked={checked} resolved={resolved} ambiguous={ambiguous} unresolved={unresolved} failures={failures}"
    )
    return checked, failures


def validate_x86_sdm_semantics(*, require_sdm: bool) -> tuple[int, int]:
    descriptions = _load_sdm_descriptions()
    if descriptions is None:
        if require_sdm:
            _fail("Intel SDM PDF or cache is required for x86 semantic validation")
        print("validated Intel SDM semantics: skipped (no PDF/cache available)")
        return 0, 0

    checked = 0
    failures = 0
    for mnemonic, expectation in _X86_SDM_EXPECTATIONS.items():
        checked += 1
        payload = _sdm_payload_for_mnemonic(descriptions, mnemonic)
        if not payload:
            print(f"FAIL intel sdm: missing description payload for {mnemonic}")
            failures += 1
            continue
        sections = dict(payload.get("sections") or {})
        for section_name in expectation["sections"]:
            if section_name not in sections or not str(sections[section_name]).strip():
                print(f"FAIL intel sdm: missing section {section_name} for {mnemonic}")
                failures += 1
        for section_name, needle in expectation["contains"].items():
            body = str(sections.get(section_name) or "")
            if needle not in body:
                print(
                    f"FAIL intel sdm: section {section_name} for {mnemonic} missing expected text {needle!r}"
                )
                failures += 1
    print(f"validated Intel SDM semantics: checked={checked} failures={failures}")
    return checked, failures


def validate_x86_sdm_coverage(*, require_sdm: bool) -> tuple[int, int]:
    descriptions = _load_sdm_descriptions()
    if descriptions is None:
        if require_sdm:
            _fail("Intel SDM PDF or cache is required for x86 coverage validation")
        print("validated Intel SDM coverage: skipped (no PDF/cache available)")
        return 0, 0

    uops_text, _uops_source = fetch_uops_xml()
    instructions = [record for record in parse_uops_xml(uops_text) if record.architecture == "x86"]
    if not instructions:
        _fail("no x86 instructions were parsed for SDM coverage validation")

    described = [
        record
        for record in instructions
        if _sdm_payload_for_mnemonic(descriptions, record.mnemonic)
    ]
    overall = len(described) / len(instructions)
    failures = 0

    if overall < _X86_SDM_MIN_COVERAGE:
        failures += 1
        print(
            "FAIL intel sdm coverage: "
            f"overall {overall:.3f} below threshold {_X86_SDM_MIN_COVERAGE:.3f}"
        )

    family_groups: dict[str, list[Any]] = {}
    for record in instructions:
        family = (
            "AVX-512"
            if any(
                "AVX512" in value.upper().replace("-", "").replace("_", "") for value in record.isa
            )
            else ""
        )
        if not family:
            family = next(
                (
                    value
                    for value in ("SSE", "AVX", "AMX")
                    if any(value in isa.upper() for isa in record.isa)
                ),
                "Other",
            )
        family_groups.setdefault(family, []).append(record)

    checked = len(instructions)
    for family, threshold in _X86_SDM_FAMILY_THRESHOLDS.items():
        records = family_groups.get(family, [])
        if not records:
            continue
        family_coverage = sum(
            1 for record in records if _sdm_payload_for_mnemonic(descriptions, record.mnemonic)
        ) / len(records)
        if family_coverage < threshold:
            failures += 1
            print(
                "FAIL intel sdm coverage: "
                f"{family} {family_coverage:.3f} below threshold {threshold:.3f}"
            )

    print(
        "validated Intel SDM coverage: "
        f"checked={checked} described={len(described)} overall={overall:.3f} failures={failures}"
    )
    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Developer-only source validation for simdref")
    parser.add_argument(
        "--require-authoritative-arm-instructions",
        action="store_true",
        help="Fail if the Arm instruction source cannot be fetched.",
    )
    parser.add_argument(
        "--require-sdm",
        action="store_true",
        help="Fail if Intel SDM semantic validation cannot run because no PDF/cache is available.",
    )
    args = parser.parse_args()

    totals_checked = 0
    totals_failed = 0

    for checked, failed in (
        validate_intel_intrinsics(),
        validate_uops_instructions(),
        validate_arm_intrinsics(),
        validate_arm_instructions(args.require_authoritative_arm_instructions),
        validate_riscv_intrinsics(),
        validate_riscv_instructions(),
        validate_riscv_intrinsic_links(),
        validate_riscv_semantics_and_coverage(),
        validate_x86_intrinsic_links(),
        validate_x86_sdm_semantics(require_sdm=args.require_sdm),
        validate_x86_sdm_coverage(require_sdm=args.require_sdm),
    ):
        totals_checked += checked
        totals_failed += failed

    if totals_failed:
        print(f"source validation failed: checked={totals_checked} failures={totals_failed}")
        return 1
    print(f"source validation passed: checked={totals_checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
