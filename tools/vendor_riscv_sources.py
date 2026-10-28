#!/usr/bin/env python3
"""Generate vendored RISC-V source bundles from official upstream clones."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UDB_ROOT = Path("/tmp/riscv-unified-db")
DEFAULT_RVV_ROOT = Path("/tmp/riscv-rvv-intrinsic-doc")
DEFAULT_OUT_DIR = REPO_ROOT / "vendor" / "riscv"
RVV_PROJECT_URL = "https://github.com/riscv-non-isa/riscv-rvv-intrinsic-doc"
UDB_PROJECT_URL = "https://github.com/riscv/riscv-unified-db"
V_DOC_URL = "https://docs.riscv.org/reference/isa/unpriv/v-st-ext.html"
VECTOR_CRYPTO_DOC_URL = "https://docs.riscv.org/reference/isa/unpriv/vector-crypto.html"
BFLOAT16_DOC_URL = "https://docs.riscv.org/reference/isa/unpriv/bfloat16.html"
INTRINSIC_SIGNATURE_RE = re.compile(
    r"(?P<sig>[A-Za-z_][A-Za-z0-9_ *]*\s+(?P<name>__riscv_[A-Za-z0-9_]+)\s*\(.*?\);)",
    re.DOTALL,
)
POLICY_SUFFIXES = ("_tumu", "_tum", "_mu", "_tu")


def git_rev(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def extract_yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip().strip('"') if match else ""


def extract_yaml_block(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*\|\n(?P<body>(?:^[ ]+.*\n?)*)", text, re.MULTILINE)
    if not match:
        return ""
    lines = [
        line[2:] if line.startswith("  ") else line for line in match.group("body").splitlines()
    ]
    return "\n".join(lines).strip()


def normalize_extension(extension: str) -> str:
    cleaned = extension.strip()
    if cleaned.casefold().startswith("zvl"):
        return "V"
    return cleaned or "V"


def sentence(text: str) -> str:
    cleaned = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if not cleaned:
        return ""
    if "." in cleaned:
        return cleaned.split(".", 1)[0].strip() + "."
    return cleaned.rstrip(".") + "."


def instruction_category(mnemonic: str, isa: list[str]) -> str:
    lowered = mnemonic.casefold()
    isa_tokens = {token.casefold() for token in isa}
    if any(token.startswith("zvk") or token in {"zvbb", "zvbc"} for token in isa_tokens):
        return "Vector Crypto"
    if any(token.startswith("zvfbf") for token in isa_tokens):
        return "BFloat16"
    if lowered.startswith(("vl", "vs")):
        return "Loads and Stores"
    if lowered.startswith("vm") and ".m" in lowered:
        return "Mask Operations"
    if "red" in lowered:
        return "Reductions"
    if lowered.startswith(("vrgather", "vslide", "vcompress", "vmv.")):
        return "Permute/Move"
    if lowered.startswith(("vw", "vn", "vzext", "vsext")):
        return "Widening/Narrowing"
    if lowered.startswith(("vf", "vmf")):
        return "Floating-Point"
    return "Arithmetic"


def instruction_doc_url(isa: list[str], source_url: str) -> str:
    lowered = {token.casefold() for token in isa}
    if any(token.startswith("zvfbf") for token in lowered):
        return BFLOAT16_DOC_URL
    if any(token.startswith("zvk") or token in {"zvbb", "zvbc"} for token in lowered):
        return VECTOR_CRYPTO_DOC_URL
    if "v" in lowered:
        return V_DOC_URL
    # Newer vector extensions (Zvabd, Zvdot4a8i, ...) have no docs.riscv.org page.
    return source_url


def parse_instruction_file(path: Path, source_url: str) -> dict[str, object]:
    text = path.read_text()
    mnemonic = extract_yaml_scalar(text, "name").casefold()
    if not mnemonic:
        return {}
    description = extract_yaml_block(text, "description")
    operation = extract_yaml_block(text, "operation()")
    long_name = extract_yaml_scalar(text, "long_name")
    extension_match = re.search(r"definedBy:\n(?:^[ ]+.*\n)*?^\s+name:\s*(.+)$", text, re.MULTILINE)
    extension = normalize_extension(
        extension_match.group(1).strip() if extension_match else path.parent.name
    )
    isa = [extension]
    return {
        "mnemonic": mnemonic,
        "summary": sentence(long_name or description or f"{mnemonic} instruction."),
        "description": description.strip(),
        "operation": operation.strip(),
        "isa": isa,
        "category": instruction_category(mnemonic, isa),
        "url": instruction_doc_url(isa, source_url),
        "extension": extension,
    }


def iter_vector_instruction_files(udb_root: Path) -> list[Path]:
    inst_root = udb_root / "spec" / "std" / "isa" / "inst"
    files: list[Path] = []
    for path in inst_root.rglob("*.yaml"):
        extension_dir = path.parent.name
        if extension_dir.startswith(("V", "Zv", "Zve")):
            files.append(path)
    return sorted(files)


def infer_intrinsic_policy(name: str) -> tuple[str, str]:
    lowered = name.casefold()
    policy = "agnostic"
    masking = (
        "masked"
        if lowered.endswith(("_m", "_mu", "_tum", "_tumu")) or "_m_" in lowered
        else "unmasked"
    )
    for suffix in POLICY_SUFFIXES:
        if lowered.endswith(suffix):
            policy = suffix.removeprefix("_")
            break
    return policy, masking


def instruction_variant_form(mnemonic: str, policy: str, masking: str) -> str:
    if policy in {"", "agnostic", "default"}:
        return f"{mnemonic} [masked]" if masking == "masked" else mnemonic
    return f"{mnemonic} [{policy}]"


def classify_intrinsic_path(rel_path: Path) -> tuple[int, str, str]:
    text = rel_path.as_posix().casefold()
    rules = [
        ("vector-crypto", "Vector Crypto", ""),
        ("bfloat16", "BFloat16", ""),
        ("00_vector_loads_and_stores", "Loads and Stores", ""),
        ("01_vector_loads_and_stores_segment", "Loads and Stores", "Segment"),
        ("02_vector_integer_arithmetic", "Arithmetic", ""),
        ("03_vector_fixed-point_arithmetic", "Fixed-Point", ""),
        ("04_vector_floating-point", "Floating-Point", ""),
        ("05_vector_reduction_operations", "Reductions", ""),
        ("06_vector_mask", "Mask Operations", ""),
        ("07_vector_permutation", "Permute/Move", ""),
        ("08_miscellaneous_vector_utility", "Utility", ""),
        ("09_zvdot4a8i", "Dot Product", ""),
        ("10_zvfofp8min", "OFP8 Conversion", ""),
        ("11_zvfofp8min", "OFP8 Conversion", ""),
        ("12_zvfofp8min", "OFP8 Conversion", ""),
        ("13_zvfofp8min", "OFP8 Conversion", ""),
        ("14_zvabd", "Absolute Difference", ""),
    ]
    for rank, (needle, category, subcategory) in enumerate(rules):
        if needle in text:
            return rank, category, subcategory
    return 999, "RVV", ""


def infer_intrinsic_isa(rel_path: Path, matched_instruction: dict[str, object] | None) -> list[str]:
    if matched_instruction is not None:
        return list(matched_instruction["isa"])
    text = rel_path.as_posix()
    if "vector-crypto" in text:
        return ["Zvkned"]
    if "bfloat16" in text:
        return ["Zvfbfmin"]
    match = re.search(r"/(zv[a-z0-9]+)", text.casefold())
    if match:
        return [match.group(1)]
    return ["V"]


def normalize_signature(signature: str) -> str:
    return " ".join(part.strip() for part in signature.splitlines()).strip()


def collect_intrinsic_entries(rvv_root: Path) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    for path in sorted((rvv_root / "auto-generated").rglob("*.adoc")):
        if "api-testing" in path.parts:
            continue
        rel_path = path.relative_to(rvv_root)
        rank, category, subcategory = classify_intrinsic_path(rel_path)
        text = path.read_text()
        for match in INTRINSIC_SIGNATURE_RE.finditer(text):
            name = match.group("name")
            signature = normalize_signature(match.group("sig"))
            entry = entries.setdefault(
                name,
                {
                    "name": name,
                    "signatures": set(),
                    "best_rank": rank,
                    "category": category,
                    "subcategory": subcategory,
                    "path": rel_path.as_posix(),
                },
            )
            entry["signatures"].add(signature)
            if rank < int(entry["best_rank"]):
                entry["best_rank"] = rank
                entry["category"] = category
                entry["subcategory"] = subcategory
                entry["path"] = rel_path.as_posix()
    return entries


def build_instruction_lookup(
    base_instructions: dict[str, dict[str, object]],
) -> list[tuple[str, str]]:
    lookup = {(mnemonic.replace(".", "_"), mnemonic) for mnemonic in base_instructions}
    return sorted(lookup, key=lambda item: (-len(item[0]), item[0]))


def infer_instruction_mnemonic(name: str, lookup: list[tuple[str, str]]) -> str:
    stem = name.removeprefix("__riscv_").casefold()
    for suffix in POLICY_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for mangled, mnemonic in lookup:
        if stem == mangled or stem.startswith(mangled + "_"):
            return mnemonic
    return ""


def synthesize_instruction_records(
    base_instructions: dict[str, dict[str, object]],
    variant_requests: dict[str, set[tuple[str, str]]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for mnemonic in sorted(base_instructions):
        base = base_instructions[mnemonic]
        variants = {("agnostic", "unmasked")} | set(variant_requests.get(mnemonic, set()))
        for policy, masking in sorted(variants):
            form = instruction_variant_form(mnemonic, policy, masking)
            metadata = {
                "category": str(base["category"]),
                "extension": str(base["extension"]),
                "policy": policy,
                "masking": masking,
                "tail_policy": "undisturbed" if "tu" in policy else "agnostic",
            }
            if masking == "masked":
                metadata["mask_policy"] = "undisturbed" if "mu" in policy else "agnostic"
            records.append(
                {
                    "mnemonic": mnemonic,
                    "form": form,
                    "summary": base["summary"],
                    "isa": base["isa"],
                    "url": base["url"],
                    "description": {
                        "Description": base["description"],
                        "Operation": base["operation"],
                    },
                    "metadata": metadata,
                    "policy": policy,
                    "masking": masking,
                }
            )
    return records


def build_intrinsics_bundle(
    rvv_root: Path,
    base_instructions: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, set[tuple[str, str]]]]:
    lookup = build_instruction_lookup(base_instructions)
    entries = collect_intrinsic_entries(rvv_root)
    variant_requests: dict[str, set[tuple[str, str]]] = defaultdict(set)
    records: list[dict[str, object]] = []

    for name in sorted(entries):
        entry = entries[name]
        signatures = sorted(entry["signatures"])
        signature = signatures[0]
        policy, masking = infer_intrinsic_policy(name)
        mnemonic = infer_instruction_mnemonic(name, lookup)
        matched_instruction = base_instructions.get(mnemonic) if mnemonic else None
        isa = infer_intrinsic_isa(Path(str(entry["path"])), matched_instruction)
        refs: list[dict[str, str]] = []
        rendered_instructions: list[str] = []
        if matched_instruction is not None:
            form = instruction_variant_form(mnemonic, policy, masking)
            refs.append(
                {
                    "name": mnemonic,
                    "form": form,
                    "isa": "/".join(isa),
                    "policy": policy,
                    "masking": masking,
                    "tail_policy": "undisturbed" if "tu" in policy else "agnostic",
                    "mask_policy": ("undisturbed" if "mu" in policy else "agnostic")
                    if masking == "masked"
                    else "",
                }
            )
            rendered_instructions.append(form)
            variant_requests[mnemonic].add((policy, masking))

        prototype_section = "\n".join(signatures)
        description_target = (
            rendered_instructions[0] if rendered_instructions else name.removeprefix("__riscv_")
        )
        records.append(
            {
                "name": name,
                "signature": signature,
                "description": sentence(f"{entry['category']} intrinsic for {description_target}."),
                "header": "riscv_vector.h",
                "url": RVV_PROJECT_URL,
                "isa": isa,
                "category": entry["category"],
                "subcategory": entry["subcategory"],
                "instructions": rendered_instructions,
                "instruction_refs": refs,
                "metadata": {
                    "policy": policy,
                    "masking": masking,
                    "tail_policy": "undisturbed" if "tu" in policy else "agnostic",
                    **(
                        {"mask_policy": "undisturbed" if "mu" in policy else "agnostic"}
                        if masking == "masked"
                        else {}
                    ),
                    "source_path": str(entry["path"]),
                },
                "doc_sections": {
                    "Prototype": prototype_section,
                    "Semantics": sentence(
                        f"{entry['category']} intrinsic that maps to {description_target}."
                    ),
                },
            }
        )
    return records, variant_requests


def collect_base_instructions(udb_root: Path) -> dict[str, dict[str, object]]:
    instructions: dict[str, dict[str, object]] = {}
    blob = f"{UDB_PROJECT_URL}/blob/{git_rev(udb_root)}"
    for path in iter_vector_instruction_files(udb_root):
        parsed = parse_instruction_file(path, f"{blob}/{path.relative_to(udb_root).as_posix()}")
        if not parsed:
            continue
        mnemonic = str(parsed["mnemonic"])
        instructions.setdefault(mnemonic, parsed)
    return instructions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udb-root", type=Path, default=DEFAULT_UDB_ROOT)
    parser.add_argument("--rvv-root", type=Path, default=DEFAULT_RVV_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.udb_root.exists():
        raise SystemExit(f"missing unified-db clone at {args.udb_root}")
    if not args.rvv_root.exists():
        raise SystemExit(f"missing rvv-intrinsic-doc clone at {args.rvv_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_instructions = collect_base_instructions(args.udb_root)
    intrinsics, variant_requests = build_intrinsics_bundle(args.rvv_root, base_instructions)
    instructions = synthesize_instruction_records(base_instructions, variant_requests)

    instructions_payload = {
        "format": "riscv-unified-db-v1",
        "upstream_repo": UDB_PROJECT_URL,
        "upstream_commit": git_rev(args.udb_root),
        "instructions": instructions,
    }
    intrinsics_payload = {
        "format": "riscv-rvv-intrinsics-v1",
        "upstream_repo": RVV_PROJECT_URL,
        "upstream_commit": git_rev(args.rvv_root),
        "intrinsics": intrinsics,
    }

    (args.out_dir / "unified_db_bundle.json").write_text(
        json.dumps(instructions_payload, indent=2) + "\n"
    )
    (args.out_dir / "rvv_intrinsics_bundle.json").write_text(
        json.dumps(intrinsics_payload, indent=2) + "\n"
    )

    print(f"wrote {args.out_dir / 'unified_db_bundle.json'} instructions={len(instructions)}")
    print(f"wrote {args.out_dir / 'rvv_intrinsics_bundle.json'} intrinsics={len(intrinsics)}")


if __name__ == "__main__":
    main()
