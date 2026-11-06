"""Fuzzy search and ranking for intrinsics and instructions.

Scores candidates through exact/prefix/substring matching, normalised token
overlap, rapidfuzz similarity, SIMD width bonuses, and intent-based biasing
(intrinsic vs instruction preference).  See ARCHITECTURE.md for details.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from simdref.catkey import X86_VECTOR_WIDTHS
from simdref.models import Catalog


@dataclass(slots=True)
class SearchResult:
    kind: str
    key: str
    title: str
    subtitle: str
    score: float


def _normalize_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    start = -1
    for index, char in enumerate(value + " "):
        if char.isascii() and char.isalnum():
            if start < 0:
                start = index
        elif start >= 0:
            tokens.append(value[start:index].casefold())
            start = -1
    return tokens


def _normalize_text(value: str) -> str:
    return " ".join(_normalize_tokens(value))


def _is_width_token(token: str) -> bool:
    if token in X86_VECTOR_WIDTHS:
        return True
    return token[:2] == "mm" and (len(token) == 2 or token[2:].isdecimal())


def _classify_query(query: str) -> str:
    lowered = query.casefold().strip()
    normalized = _normalize_text(query)
    if lowered.startswith("__riscv_"):
        return "intrinsic"
    if "." in lowered and not lowered.startswith("_"):
        return "instruction"
    if _looks_like_isa_query(query):
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
    query_widths = {token for token in query_tokens if _is_width_token(token)}
    candidate_widths = {token for token in candidate_tokens if _is_width_token(token)}
    if not query_widths or not candidate_widths:
        return 0.0
    if query_widths & candidate_widths:
        return 22.0
    return -22.0


def _meaningful_query_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if not _is_width_token(token)]


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


def _normalized_isa_tokens(isa_values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in isa_values:
        text = _normalize_text(value)
        if text:
            tokens.add(text)
    return tokens


def _isa_match_bias(query: str, isa_values: list[str], result_kind: str) -> float:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return 0.0
    if normalized_query not in _normalized_isa_tokens(isa_values):
        return 0.0
    return 30.0 if result_kind == "instruction" else -15.0


def _is_pure_isa_query(query: str, isa_values: list[str]) -> bool:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return False
    return normalized_query in _normalized_isa_tokens(isa_values)


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


def find_instructions(catalog: Catalog, query: str) -> list:
    target = query.casefold()
    normalized_target = _normalize_text(query)
    matches = []
    for item in catalog.instructions:
        if item.key.casefold() == target or item.mnemonic.casefold() == target:
            matches.append(item)
            continue
        if normalized_target and (
            _normalize_text(item.key) == normalized_target
            or _normalize_text(item.mnemonic) == normalized_target
        ):
            matches.append(item)
    return matches
