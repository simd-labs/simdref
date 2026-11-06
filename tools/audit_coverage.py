"""Audit local simdref catalog against upstream source feeds.

Subcommands:
  fetch   Resolve upstream sources (live with offline fallback), extract
          canonical name sets, diff against the local catalog, and write
          ``docs/coverage/summary.json``.
  report  Pretty-print the existing summary.
  diff    Exit non-zero if any source falls below its threshold.

Upstream name extraction is deliberately tolerant: each upstream feed is a
JSON/XML payload with well-known name keys ("name", "mnemonic"). When a
payload is unparseable (network failure, schema drift) the audit marks
that source as ``unknown`` rather than failing the whole run.

Threshold resolution:
  per-source override in ``docs/coverage/thresholds.toml`` (optional)
  → ``--threshold`` CLI flag
  → ``SIMDREF_COVERAGE_THRESHOLD`` env var
  → 0.95 default
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# Repo-root relative import.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from simdref import ingest_sources as src  # noqa: E402
from simdref.ingest_catalog import parse_intel_payload  # noqa: E402
from simdref.storage import load_catalog  # noqa: E402

DEFAULT_THRESHOLD = 0.95
SUMMARY_PATH = REPO_ROOT / "docs" / "coverage" / "summary.json"
THRESHOLDS_PATH = REPO_ROOT / "docs" / "coverage" / "thresholds.toml"


# ---------------------------------------------------------------------------
# Upstream name extraction
# ---------------------------------------------------------------------------


def _extract_intel_names(text: str) -> set[str]:
    """Intrinsic names from one Intel payload, via the catalog ingest parser."""
    try:
        records = parse_intel_payload(text)
    except (ValueError, TypeError, KeyError, AttributeError, ET.ParseError):
        return set()
    return {record.name for record in records if record.name.startswith("_")}


def _extract_uops_mnemonics(text_or_path: Any) -> set[str]:
    source = Path(text_or_path).read_text() if isinstance(text_or_path, Path) else text_or_path
    names: set[str] = set()
    try:
        root = ET.fromstring(source)
    except ET.ParseError:
        return names
    for inst in root.iter():
        tag = inst.tag.split("}")[-1]
        if tag in {"instruction", "Instruction"}:
            m = inst.get("asm") or inst.get("mnemonic") or inst.get("iform")
            if m:
                names.add(m.split()[0].upper())
    return names


def _extract_arm_intrinsic_names(text: str) -> set[str]:
    try:
        payload = json.loads(text)
    except Exception:
        return set()

    if isinstance(payload, dict) and "intrinsics_json" in payload:
        inner = payload.get("intrinsics_json") or "[]"
        try:
            items = json.loads(inner) if isinstance(inner, str) else inner
        except Exception:
            return set()
    else:
        items = payload if isinstance(payload, list) else payload.get("intrinsics") or []

    names: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        n = item.get("name") or item.get("intrinsic") or item.get("intrinsic_name") or ""
        if not isinstance(n, str) or not n:
            continue
        # Arm inventory encodes optional prefix/suffix alternatives with
        # brackets (e.g. ``[__arm_]vddupq[_n]_u8``). Normalise by stripping
        # all bracket groups — the catalog always stores one canonical
        # entry per line of the upstream JSON.
        canonical = re.sub(r"\[[^\]]*\]", "", n)
        if canonical:
            names.add(canonical)
    return names


def _extract_arm_a64_mnemonics(text: str) -> set[str]:
    try:
        payload = json.loads(text)
    except Exception:
        return set()

    names: set[str] = set()

    def _is_aarchmrs(node: Any) -> bool:
        if not isinstance(node, dict):
            return False
        top = node.get("instructions")
        if not isinstance(top, list) or not top:
            return False
        return all(
            isinstance(n, dict) and str(n.get("_type", "")).endswith("InstructionSet") for n in top
        )

    def _first_literal_mnemonic(assembly: Any) -> str | None:
        """Pull the first Literal token out of an AARCHMRS assembly symbols
        list. That token is the mnemonic; everything after it (spaces,
        operands, rule references) is irrelevant here."""
        if not isinstance(assembly, dict):
            return None
        symbols = assembly.get("symbols")
        if not isinstance(symbols, list):
            return None
        for sym in symbols:
            if not isinstance(sym, dict):
                continue
            if not str(sym.get("_type", "")).endswith("Literal"):
                continue
            value = str(sym.get("value", "")).strip()
            if not value:
                continue
            token = value.split()[0].upper()
            if re.fullmatch(r"[A-Z][A-Z0-9]*", token):
                return token
            return None
        return None

    def _walk_aarchmrs(node: Any, in_a64: bool = False) -> None:
        """AARCHMRS FAT archives contain A64, A32 and T32 InstructionSets.
        The ``arm-a64`` source only covers AArch64, so scope the walk to the
        A64 subtree."""
        if isinstance(node, dict):
            node_type = str(node.get("_type", ""))
            if node_type.endswith("InstructionSet"):
                name = str(node.get("name") or "").strip().upper()
                in_a64 = name == "A64"
                for child in node.get("children") or []:
                    _walk_aarchmrs(child, in_a64)
                return
            if not in_a64:
                return
            if node_type.endswith("Instruction") and not node_type.endswith("InstructionAlias"):
                mnemonic = _first_literal_mnemonic(node.get("assembly"))
                if mnemonic:
                    names.add(mnemonic)
                return
            if node_type.endswith("InstructionAlias"):
                return
            for child in node.get("children") or []:
                _walk_aarchmrs(child, in_a64)
        elif isinstance(node, list):
            for item in node:
                _walk_aarchmrs(item, in_a64)

    def _walk_aarchmrs_root(payload: Any) -> None:
        if isinstance(payload, dict) and isinstance(payload.get("instructions"), list):
            for top in payload["instructions"]:
                _walk_aarchmrs(top, in_a64=False)

    def _walk_fallback(node: Any) -> None:
        """For non-AARCHMRS payloads (fixtures, ``instructions_json`` wrappers):
        trust explicit ``mnemonic`` / ``name`` fields on dict entries."""
        if isinstance(node, dict):
            for key in ("mnemonic", "name"):
                val = node.get(key)
                if isinstance(val, str) and val:
                    names.add(val.split()[0].upper())
            if isinstance(node.get("instructions_json"), str):
                try:
                    _walk_fallback(json.loads(node["instructions_json"]))
                except Exception:
                    pass
            for v in node.values():
                _walk_fallback(v)
        elif isinstance(node, list):
            for item in node:
                _walk_fallback(item)

    if _is_aarchmrs(payload):
        _walk_aarchmrs_root(payload)
    elif (
        isinstance(payload, dict)
        and payload.get("format") == "arm-aarchmrs-instructions-v1"
        and isinstance(payload.get("instructions_json"), str)
    ):
        try:
            inner = json.loads(payload["instructions_json"])
        except Exception:
            inner = None
        if _is_aarchmrs(inner):
            _walk_aarchmrs_root(inner)
        else:
            _walk_fallback(inner)
    else:
        _walk_fallback(payload)

    return {n for n in names if re.fullmatch(r"[A-Z][A-Z0-9]*", n)}


def _extract_riscv_rvv_names(text: str) -> set[str]:
    try:
        payload = json.loads(text)
    except Exception:
        return set()
    items = (
        payload
        if isinstance(payload, list)
        else payload.get("intrinsics") or payload.get("items") or []
    )
    names: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        n = item.get("name") or item.get("intrinsic") or ""
        if not isinstance(n, str) or not n:
            continue
        # Normalise to canonical `__riscv_`-prefixed form — catalog always
        # stores it that way.
        if not n.startswith("__riscv_"):
            n = f"__riscv_{n}"
        names.add(n)
    return names


def _extract_riscv_unified_db_mnemonics(text: str) -> set[str]:
    try:
        payload = json.loads(text)
    except Exception:
        return set()
    instructions = payload.get("instructions") or payload.get("records") or []
    names: set[str] = set()
    for item in instructions:
        if isinstance(item, dict):
            n = item.get("name") or item.get("mnemonic") or ""
            if n:
                names.add(n.lower().split()[0])
    return names


# ---------------------------------------------------------------------------
# Local catalog name extraction
# ---------------------------------------------------------------------------


def _local_intrinsic_names(catalog, arch: str) -> set[str]:
    return {r.name for r in catalog.intrinsics if r.architecture == arch}


def _local_instruction_mnemonics(catalog, arch: str, *, upper: bool = True) -> set[str]:
    out: set[str] = set()
    for r in catalog.instructions:
        if r.architecture != arch:
            continue
        m = getattr(r, "mnemonic", "") or r.key.split()[0]
        out.add(m.upper() if upper else m.lower())
    return out


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


def _source_specs(catalog):
    return [
        {
            "id": "intel-intrinsics",
            "label": "Intel Intrinsics Guide",
            "fetch": lambda: src.fetch_intel_data(),
            "upstream_fn": _extract_intel_names,
            "local_names": _local_intrinsic_names(catalog, "x86"),
        },
        {
            "id": "uops-info",
            "label": "uops.info instructions.xml",
            "fetch": lambda: src.fetch_uops_xml(),
            "upstream_fn": _extract_uops_mnemonics,
            "local_names": _local_instruction_mnemonics(catalog, "x86"),
        },
        {
            "id": "arm-intrinsics",
            "label": "Arm ACLE intrinsics",
            "fetch": lambda: src.fetch_arm_acle_data(),
            "upstream_fn": _extract_arm_intrinsic_names,
            "local_names": _local_intrinsic_names(catalog, "arm"),
        },
        {
            "id": "arm-a64",
            "label": "Arm AARCHMRS A64",
            "fetch": lambda: src.fetch_arm_a64_data(),
            "upstream_fn": _extract_arm_a64_mnemonics,
            "local_names": _local_instruction_mnemonics(catalog, "arm"),
        },
        {
            "id": "riscv-rvv",
            "label": "RISC-V RVV intrinsics",
            "fetch": lambda: src.fetch_riscv_rvv_intrinsics_data(),
            "upstream_fn": _extract_riscv_rvv_names,
            "local_names": _local_intrinsic_names(catalog, "riscv"),
        },
        {
            "id": "riscv-unified-db",
            "label": "RISC-V unified DB",
            "fetch": lambda: src.fetch_riscv_unified_db_data(),
            "upstream_fn": _extract_riscv_unified_db_mnemonics,
            "local_names": {
                m.lower() for m in _local_instruction_mnemonics(catalog, "riscv", upper=False)
            },
        },
    ]


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _diff_source(spec: dict) -> dict:
    try:
        text, version = spec["fetch"]()
    except Exception as exc:
        return {
            "label": spec["label"],
            "error": f"fetch failed: {exc}",
            "upstream": 0,
            "local": len(spec["local_names"]),
            "coverage": None,
        }
    upstream_names = spec["upstream_fn"](text)
    local = spec["local_names"]
    # Case-fold both sides for comparison (names like VADDPS vs vaddps).
    upstream_ci = {n.casefold() for n in upstream_names}
    local_ci = {n.casefold() for n in local}

    matched = upstream_ci & local_ci
    missing = sorted(upstream_names - {n for n in upstream_names if n.casefold() in matched})
    extra = sorted({n for n in local if n.casefold() not in upstream_ci})

    coverage = (len(matched) / len(upstream_ci)) if upstream_ci else None
    return {
        "label": spec["label"],
        "fetched_version": getattr(version, "version", ""),
        "fetched_url": getattr(version, "url", ""),
        "upstream": len(upstream_names),
        "local": len(local),
        "matched": len(matched),
        "coverage": coverage,
        "missing_sample": missing[:25],
        "extra_sample": extra[:25],
        "missing_total": len(missing),
        "extra_total": len(extra),
    }


def _load_thresholds() -> dict[str, float]:
    if not THRESHOLDS_PATH.exists():
        return {}
    with THRESHOLDS_PATH.open("rb") as fh:
        payload = tomllib.load(fh)
    return {k: float(v) for k, v in (payload.get("thresholds") or {}).items()}


def _default_threshold() -> float:
    raw = os.environ.get("SIMDREF_COVERAGE_THRESHOLD")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_fetch(args: argparse.Namespace) -> int:
    catalog = load_catalog()
    specs = _source_specs(catalog)
    summary = {
        "catalog": {
            "intrinsics": len(catalog.intrinsics),
            "instructions": len(catalog.instructions),
            "generated_at": catalog.generated_at,
        },
        "sources": {},
    }
    thresholds = _load_thresholds()
    default_th = args.threshold if args.threshold is not None else _default_threshold()
    for spec in specs:
        result = _diff_source(spec)
        result["threshold"] = thresholds.get(spec["id"], default_th)
        summary["sources"][spec["id"]] = result
        cov = result.get("coverage")
        cov_str = f"{cov:.3f}" if isinstance(cov, float) else "unknown"
        print(
            f"  {spec['id']:22s} upstream={result['upstream']:>6d} local={result['local']:>6d} cov={cov_str}"
        )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if not SUMMARY_PATH.exists():
        print(f"no summary at {SUMMARY_PATH}; run `audit_coverage fetch` first", file=sys.stderr)
        return 2
    summary = json.loads(SUMMARY_PATH.read_text())
    cat = summary.get("catalog", {})
    print(
        f"Catalog: {cat.get('intrinsics', '?')} intrinsics, {cat.get('instructions', '?')} instructions"
    )
    for sid, data in summary.get("sources", {}).items():
        cov = data.get("coverage")
        cov_str = f"{cov * 100:5.1f}%" if isinstance(cov, float) else "  ??? "
        print(
            f"  {sid:22s} {data.get('label', '')}  upstream={data['upstream']:>6d} "
            f"local={data['local']:>6d}  coverage={cov_str}"
        )
        if data.get("error"):
            print(f"    error: {data['error']}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    if not SUMMARY_PATH.exists():
        print(f"no summary at {SUMMARY_PATH}", file=sys.stderr)
        return 2
    summary = json.loads(SUMMARY_PATH.read_text())
    default_th = args.threshold if args.threshold is not None else _default_threshold()
    thresholds = _load_thresholds()
    failures: list[tuple[str, float, float]] = []
    for sid, data in summary.get("sources", {}).items():
        th = thresholds.get(sid, default_th)
        cov = data.get("coverage")
        if not isinstance(cov, float):
            continue  # Unknown coverage: cannot fail parity.
        if cov < th:
            failures.append((sid, cov, th))
    for sid, cov, th in failures:
        print(f"FAIL {sid}: coverage {cov:.3f} < threshold {th:.3f}", file=sys.stderr)
    return 1 if failures else 0


def _threshold_status(
    summary: dict, thresholds: dict[str, float], default: float
) -> dict[str, bool]:
    """Map each source -> whether it meets its threshold.

    Sources whose coverage is unknown (``None`` — e.g. a transient fetch
    failure) are omitted, so they never count as a threshold crossing.
    """
    status: dict[str, bool] = {}
    for sid, data in summary.get("sources", {}).items():
        cov = data.get("coverage")
        if isinstance(cov, float):
            status[sid] = cov >= thresholds.get(sid, default)
    return status


def cmd_gate(args: argparse.Namespace) -> int:
    """Decide whether coverage drift is *material* (a threshold crossing).

    Prints ``true`` if any source's pass/fail status against its threshold
    flipped relative to the committed baseline, else ``false``. Used by the
    coverage-audit workflow to avoid opening a PR for cosmetic count jitter
    that does not move any source across its floor.
    """
    if not SUMMARY_PATH.exists():
        print("false")
        print(f"no summary at {SUMMARY_PATH}", file=sys.stderr)
        return 0
    current = json.loads(SUMMARY_PATH.read_text())
    base_text = sys.stdin.read() if args.base == "-" else Path(args.base).read_text()
    base = json.loads(base_text) if base_text.strip() else {"sources": {}}

    default_th = args.threshold if args.threshold is not None else _default_threshold()
    thresholds = _load_thresholds()
    cur_status = _threshold_status(current, thresholds, default_th)
    base_status = _threshold_status(base, thresholds, default_th)

    crossings = [
        f"{sid}: {base_status.get(sid)} -> {cur_status[sid]}"
        for sid in cur_status
        if sid in base_status and base_status[sid] != cur_status[sid]
    ]
    print("true" if crossings else "false")
    for line in crossings:
        print(f"threshold crossing {line}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="simdref upstream coverage audit")
    parser.add_argument("--threshold", type=float, default=None, help="Coverage floor (0–1).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub_fetch = sub.add_parser("fetch", help="Fetch upstream, diff, write summary.")
    sub_fetch.set_defaults(func=cmd_fetch)

    sub_report = sub.add_parser("report", help="Pretty-print the current summary.")
    sub_report.set_defaults(func=cmd_report)

    sub_diff = sub.add_parser("diff", help="Exit non-zero if any source is below threshold.")
    sub_diff.set_defaults(func=cmd_diff)

    sub_gate = sub.add_parser(
        "gate",
        help="Print 'true' if drift crosses a threshold vs a baseline summary, else 'false'.",
    )
    sub_gate.add_argument(
        "--base",
        required=True,
        help="Baseline summary.json to compare against ('-' reads from stdin).",
    )
    sub_gate.set_defaults(func=cmd_gate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
