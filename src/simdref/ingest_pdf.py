"""PDF enrichment acquisition, caching, and merge helpers."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Callable

import msgpack

from simdref.models import InstructionRecord
from simdref.pdfrefs import apply_legacy_pdf_metadata, normalize_pdf_refs
from simdref.pdfparse.registry import get_pdf_source
from simdref.pdfparse.types import PdfEnrichmentResult, PdfSourceSpec
from simdref.storage import ensure_dir


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@lru_cache(maxsize=None)
def pdf_parser_signature(source_id: str) -> str:
    spec = get_pdf_source(source_id)
    hasher = hashlib.sha256()
    for path in spec.signature_paths:
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _load_cached_pdf_source(
    spec: PdfSourceSpec,
    pdf_path: Path,
    *,
    cache_path: Path | None = None,
    status: Callable[[str], None] | None = None,
) -> PdfEnrichmentResult | None:
    cache_path = cache_path or spec.cache_path
    if not cache_path.exists():
        return None
    try:
        payload = msgpack.unpackb(cache_path.read_bytes(), raw=False)
    except Exception:
        return None
    if payload.get("cache_version") != spec.cache_version:
        return None
    if payload.get("parser_signature") != pdf_parser_signature(spec.source_id):
        return None
    if payload.get("pdf_url") != spec.source_url:
        return None
    if payload.get("pdf_sha256") != _sha256_file(pdf_path):
        return None
    result_payload = payload.get("result")
    if not isinstance(result_payload, dict):
        return None
    result = PdfEnrichmentResult.from_dict(result_payload)
    if status is not None:
        status(
            f"Loaded cached {spec.display_name} descriptions for {len(result.descriptions)} mnemonic variants"
        )
    return result


def _save_cached_pdf_source(
    spec: PdfSourceSpec,
    pdf_path: Path,
    result: PdfEnrichmentResult,
    *,
    cache_path: Path | None = None,
) -> None:
    cache_path = cache_path or spec.cache_path
    ensure_dir(cache_path.parent)
    payload = {
        "cache_version": spec.cache_version,
        "parser_signature": pdf_parser_signature(spec.source_id),
        "pdf_url": spec.source_url,
        "pdf_sha256": _sha256_file(pdf_path),
        "result": result.to_dict(),
    }
    cache_path.write_bytes(msgpack.packb(payload, use_bin_type=True))


def load_or_parse_pdf_source(
    source_id: str,
    pdf_path: Path,
    *,
    cache_path: Path | None = None,
    status: Callable[[str], None] | None = None,
) -> PdfEnrichmentResult:
    spec = get_pdf_source(source_id)
    cached = _load_cached_pdf_source(spec, pdf_path, cache_path=cache_path, status=status)
    if cached is not None:
        return cached
    result = spec.parser(pdf_path, status=status)
    _save_cached_pdf_source(spec, pdf_path, result, cache_path=cache_path)
    if status is not None:
        status(
            f"Cached {spec.display_name} descriptions for {len(result.descriptions)} mnemonic variants"
        )
    return result


def find_pdf_source_path(source_id: str) -> Path | None:
    spec = get_pdf_source(source_id)
    return spec.find_source()


def merge_pdf_enrichment(
    instructions: list[InstructionRecord],
    source_id: str,
    result: PdfEnrichmentResult,
) -> None:
    # Build a list of candidate base mnemonics from a decorated mnemonic.
    _TYPE_SUFFIX_MAP = [
        ("PH", "PD"),
        ("PH", "PS"),
        ("BF16", "PS"),
        ("BF8", "PS"),
        ("BF8S", "PS"),
        ("HF8", "PS"),
        ("HF8S", "PS"),
        ("IBS", "DQ"),
        ("IUBS", "UDQ"),
    ]
    _GROUP_SUFFIXES = [
        "F32X8",
        "F32X4",
        "F32X2",
        "F64X4",
        "F64X2",
        "F128",
        "I32X8",
        "I32X4",
        "I32X2",
        "I64X4",
        "I64X2",
        "I128",
        "MB2Q",
        "MW2D",
        "BD",
        "BW",
        "BQ",
        "DQ",
        "WD",
        "WQ",
        "SD",
        "SS",
        "PD",
        "PS",
        "B",
        "W",
        "D",
        "Q",
        "64",
        "32",
        "16",
        "8",
    ]
    _MIN_GROUP_KEY_LEN = 5

    def _strip_prefix(mnemonic: str) -> str:
        value = mnemonic
        while value.startswith("{"):
            end = value.find("}")
            if end == -1:
                break
            value = value[end + 1 :].lstrip()
        for prefix in ("LOCK ", "REPE ", "REPNE ", "REP ", "REX64 "):
            if value.startswith(prefix):
                return value[len(prefix) :]
        return value

    def _base_candidates(mnemonic: str) -> list[str]:
        candidates: list[str] = []
        bare = _strip_prefix(mnemonic)
        if bare != mnemonic:
            candidates.append(bare)
        if bare.startswith("V") and len(bare) > 1:
            candidates.append(bare[1:])
        for candidate in list(candidates):
            if candidate.startswith("V") and len(candidate) > 1 and candidate[1:] not in candidates:
                candidates.append(candidate[1:])
        all_forms = [mnemonic, bare] + candidates
        for form in list(all_forms):
            if form.endswith("S") and len(form) > 3:
                candidates.append(form[:-1])
            for old_suffix, new_suffix in _TYPE_SUFFIX_MAP:
                if form.endswith(old_suffix):
                    candidates.append(form[: -len(old_suffix)] + new_suffix)
        for form in list(all_forms):
            for suffix in _GROUP_SUFFIXES:
                if form.endswith(suffix):
                    stem = form[: -len(suffix)]
                    if len(stem) >= _MIN_GROUP_KEY_LEN and stem not in candidates:
                        candidates.append(stem)
        return candidates

    for record in instructions:
        # The Intel SDM holds x86 mnemonics only; arm reuses names like ADC.
        if source_id == "intel-sdm" and record.architecture != "x86":
            continue
        mnemonic = record.mnemonic.upper()
        payload = result.descriptions.get(mnemonic)
        if payload is None:
            for candidate in _base_candidates(mnemonic):
                payload = result.descriptions.get(candidate)
                if payload is not None:
                    break
        if payload is None:
            continue
        record.description = dict(payload.sections)
        pdf_ref = {
            "source_id": source_id,
            "label": get_pdf_source(source_id).display_name,
            "url": f"{payload.source_url}#page={payload.page_start}"
            if payload.page_start
            else payload.source_url,
            "page_start": str(payload.page_start or ""),
            "page_end": str(payload.page_end or ""),
        }
        record.pdf_refs = normalize_pdf_refs([*record.pdf_refs, pdf_ref], record.metadata)
        record.metadata = apply_legacy_pdf_metadata(dict(record.metadata), record.pdf_refs)
