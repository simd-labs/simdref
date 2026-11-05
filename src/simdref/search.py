"""Fuzzy search and ranking for intrinsics and instructions.

Scores candidates through exact/prefix/substring matching, normalised token
overlap, rapidfuzz similarity, SIMD width bonuses, and intent-based biasing
(intrinsic vs instruction preference).  See ARCHITECTURE.md for details.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - fallback when dependency is unavailable
    from difflib import SequenceMatcher

    class _FallbackFuzz:
        @staticmethod
        def ratio(a: str, b: str) -> float:
            return SequenceMatcher(a=a, b=b).ratio() * 100.0

        @staticmethod
        def partial_ratio(a: str, b: str) -> float:
            return SequenceMatcher(a=a, b=b).ratio() * 100.0

        @staticmethod
        def token_set_ratio(a: str, b: str) -> float:
            return (
                SequenceMatcher(
                    a=" ".join(sorted(set(a.split()))), b=" ".join(sorted(set(b.split())))
                ).ratio()
                * 100.0
            )

    fuzz = _FallbackFuzz()

from simdref.models import Catalog


@dataclass(slots=True)
class SearchResult:
    kind: str
    key: str
    title: str
    subtitle: str
    score: float


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
WIDTH_TOKEN_RE = re.compile(r"^(mm|mm\d+|xmm|ymm|zmm)$")


def _normalize_tokens(value: str) -> list[str]:
    text = value.replace("_", " ").replace(",", " ").replace("{", " ").replace("}", " ")
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _normalize_text(value: str) -> str:
    return " ".join(_normalize_tokens(value))


def _normalized_instruction_query(value: str) -> str:
    return _normalize_text(value)


def _classify_query(query: str) -> str:
    lowered = query.casefold().strip()
    normalized = _normalize_text(query)
    if lowered.startswith("__riscv_"):
        return "intrinsic"
    if "." in lowered and not lowered.startswith("_"):
        return "instruction"
    if (
        lowered == "v"
        or lowered.startswith("zv")
        or lowered.startswith("zve")
        or lowered.startswith("rv")
    ):
        return "instruction"
    if (
        lowered.startswith("_mm")
        or lowered.startswith("__m")
        or normalized.startswith("mm ")
        or normalized == "mm"
    ):
        return "intrinsic"
    if "_" in lowered and "mm" in lowered:
        return "intrinsic"
    if normalized:
        tokens = normalized.split()
        if tokens and all(token.isalpha() or token.isalnum() for token in tokens):
            first = tokens[0]
            if first in {
                "add",
                "sub",
                "mul",
                "div",
                "mov",
                "cmp",
                "and",
                "or",
                "xor",
            } or first.startswith("v"):
                return "instruction"
    return "neutral"


def _token_prefix_score(query_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    matched = 0
    for q in query_tokens:
        if any(token.startswith(q) for token in candidate_tokens):
            matched += 1
    return 100.0 * matched / len(query_tokens)


def _token_overlap_count(query_tokens: list[str], candidate_tokens: list[str]) -> int:
    if not query_tokens or not candidate_tokens:
        return 0
    count = 0
    for q in query_tokens:
        if any(
            token == q or token.startswith(q) or q.startswith(token) for token in candidate_tokens
        ):
            count += 1
    return count


def _width_family_bonus(query: str, candidate: str) -> float:
    query_tokens = _normalize_tokens(query)
    candidate_tokens = _normalize_tokens(candidate)
    query_widths = {token for token in query_tokens if WIDTH_TOKEN_RE.match(token)}
    candidate_widths = {token for token in candidate_tokens if WIDTH_TOKEN_RE.match(token)}
    if not query_widths or not candidate_widths:
        return 0.0
    if query_widths & candidate_widths:
        return 22.0
    return -22.0


def _meaningful_query_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if not WIDTH_TOKEN_RE.match(token)]


def _has_structural_overlap(query: str, candidate: str) -> bool:
    q = query.casefold().strip()
    c = candidate.casefold().strip()
    if not q or not c:
        return False
    if q == c or c.startswith(q) or q in c:
        return True
    query_tokens = _normalize_tokens(query)
    candidate_tokens = _normalize_tokens(candidate)
    meaningful_query_tokens = _meaningful_query_tokens(query_tokens)
    if meaningful_query_tokens:
        return _token_overlap_count(meaningful_query_tokens, candidate_tokens) > 0
    return _token_overlap_count(query_tokens, candidate_tokens) > 0


def _base_score(query: str, candidate: str) -> float:
    q = query.casefold().strip()
    c = candidate.casefold().strip()
    if not q or not c:
        return 0.0
    if q == c:
        return 220.0
    if c.startswith(q):
        return 175.0
    if q in c:
        return 135.0
    qnorm = _normalize_text(query)
    cnorm = _normalize_text(candidate)
    if qnorm and qnorm == cnorm:
        return 190.0
    if qnorm and cnorm.startswith(qnorm):
        return 165.0
    query_tokens = qnorm.split()
    candidate_tokens = cnorm.split()
    token_prefix = _token_prefix_score(query_tokens, candidate_tokens)
    token_overlap = _token_overlap_count(query_tokens, candidate_tokens)
    if token_overlap == 0:
        return 0.0
    prefix_score = token_prefix + 40.0
    if prefix_score >= 155.0:
        return prefix_score
    token_set = fuzz.token_set_ratio(qnorm, cnorm) if qnorm and cnorm else 0.0
    partial = fuzz.partial_ratio(qnorm, cnorm) if qnorm and cnorm else 0.0
    ratio = fuzz.ratio(qnorm, cnorm) if qnorm and cnorm else 0.0
    return max(prefix_score, token_set + 20.0, partial + 10.0, ratio)


def _intent_bias(query_kind: str, result_kind: str) -> float:
    if query_kind == "intrinsic":
        return 45.0 if result_kind == "intrinsic" else -25.0
    if query_kind == "instruction":
        return 35.0 if result_kind == "instruction" else -10.0
    return 0.0


def _isa_match_bias(query: str, isa_values: list[str], result_kind: str) -> float:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return 0.0
    isa_tokens = {_normalize_text(value) for value in isa_values if _normalize_text(value)}
    if normalized_query not in isa_tokens:
        return 0.0
    return 30.0 if result_kind == "instruction" else -15.0


def _is_pure_isa_query(query: str, isa_values: list[str]) -> bool:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return False
    return normalized_query in {
        _normalize_text(value) for value in isa_values if _normalize_text(value)
    }


def _looks_like_isa_query(query: str) -> bool:
    lowered = query.casefold().strip()
    return (
        lowered == "v"
        or lowered.startswith("zv")
        or lowered.startswith("zve")
        or lowered.startswith("rv")
    )


def search_records(
    intrinsics: list, instructions: list, query: str, limit: int = 20
) -> list[SearchResult]:
    results: list[SearchResult] = []
    query_kind = _classify_query(query)
    for item in intrinsics:
        if (
            query_kind == "instruction"
            and _is_pure_isa_query(query, item.isa)
            and not _has_structural_overlap(query, item.name)
        ):
            continue
        if query_kind == "intrinsic" and not _has_structural_overlap(query, item.name):
            continue
        score = max(
            _base_score(query, item.name), _base_score(query, item.search_blob)
        ) + _intent_bias(query_kind, "intrinsic")
        score += _isa_match_bias(query, item.isa, "intrinsic")
        score += _width_family_bonus(query, item.name)
        if score >= 35:
            results.append(
                SearchResult(
                    kind="intrinsic",
                    key=item.name,
                    title=item.name,
                    subtitle=item.description,
                    score=score,
                )
            )
    for item in instructions:
        if _looks_like_isa_query(query) and not _is_pure_isa_query(query, item.isa):
            continue
        if (
            query_kind == "instruction"
            and not _looks_like_isa_query(query)
            and not _has_structural_overlap(query, item.key)
        ):
            continue
        score = max(
            _base_score(query, item.mnemonic),
            _base_score(query, item.key),
            _base_score(query, item.search_blob),
        )
        score += _intent_bias(query_kind, "instruction")
        score += _isa_match_bias(query, item.isa, "instruction")
        if score >= 35:
            results.append(
                SearchResult(
                    kind="instruction",
                    key=item.db_key,
                    title=item.key,
                    subtitle=(item.summary or "").strip(),
                    score=score,
                )
            )

    def sort_key(item: SearchResult):
        preferred = 0
        if query_kind == "intrinsic":
            preferred = 0 if item.kind == "intrinsic" else 1
        elif query_kind == "instruction":
            preferred = 0 if item.kind == "instruction" else 1
        return (-item.score, preferred, len(item.title), item.title)

    results.sort(key=sort_key)
    return results[:limit]


def search_catalog(catalog: Catalog, query: str, limit: int = 20) -> list[SearchResult]:
    return search_records(catalog.intrinsics, catalog.instructions, query, limit=limit)


def find_intrinsic(catalog: Catalog, name: str):
    target = name.casefold()
    for item in catalog.intrinsics:
        if item.name.casefold() == target:
            return item
    return None


def find_instruction(catalog: Catalog, query: str):
    target = query.casefold()
    normalized_target = _normalized_instruction_query(query)
    for item in catalog.instructions:
        if item.key.casefold() == target or item.mnemonic.casefold() == target:
            return item
        if normalized_target and (
            _normalized_instruction_query(item.key) == normalized_target
            or _normalized_instruction_query(item.mnemonic) == normalized_target
        ):
            return item
    return None


def find_instructions(catalog: Catalog, query: str) -> list:
    target = query.casefold()
    normalized_target = _normalized_instruction_query(query)
    matches = []
    for item in catalog.instructions:
        if item.key.casefold() == target or item.mnemonic.casefold() == target:
            matches.append(item)
            continue
        if normalized_target and (
            _normalized_instruction_query(item.key) == normalized_target
            or _normalized_instruction_query(item.mnemonic) == normalized_target
        ):
            matches.append(item)
    return matches
