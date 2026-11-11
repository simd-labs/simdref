"""A url is built only from a parsed base url and a fragment. A bad base gives no url."""

from __future__ import annotations

import json
from urllib.parse import urlsplit

from simdref.cuda import apply_cuda_docs
from simdref.models import IntrinsicRecord
from simdref.ptx import PTX_ISA_DOC_URL, parse_ptx_instructions_payload
from simdref.urls import canonical_url, resolve_url

FRAGMENT = "#parallel-synchronization-and-communication-instructions-match-sync"


def _full(url: str) -> bool:
    parts = urlsplit(url)
    return bool(parts.scheme and parts.netloc and parts.path)


def test_resolve_url_needs_a_full_base() -> None:
    assert resolve_url("", FRAGMENT) == ""
    assert resolve_url("docs.nvidia.com/x", FRAGMENT) == ""
    assert resolve_url("https://docs.nvidia.com", FRAGMENT) == ""
    assert resolve_url("https://docs.nvidia.com/cuda/", FRAGMENT) == (
        "https://docs.nvidia.com/cuda/" + FRAGMENT
    )
    assert resolve_url("https://a.test/x/y.html", "z.html#q") == "https://a.test/x/z.html#q"
    assert resolve_url("", "https://a.test/x") == "https://a.test/x"


def test_canonical_url_never_makes_a_half_url() -> None:
    assert canonical_url(FRAGMENT) == ""
    assert canonical_url("https://") == ""
    assert (
        canonical_url("uops.info/html-instr/ADC.html")
        == "https://www.uops.info/html-instr/ADC.html"
    )
    assert canonical_url("https://example.com/a") == "https://example.com/a"
    assert canonical_url("") == ""


def test_ptx_bundle_urls_are_full_even_for_a_bare_fragment() -> None:
    # An old bundle stores "#section" with no base.
    row = {
        "mnemonic": "match.any.sync.b32",
        "form": "d, a",
        "syntax": "match.any.sync.type d, a, membermask;",
        "url": FRAGMENT,
        "urls": [FRAGMENT],
    }
    record = parse_ptx_instructions_payload(json.dumps({"instructions": [row]}))[0]
    assert record.metadata["url"] == PTX_ISA_DOC_URL + FRAGMENT
    assert json.loads(record.metadata["urls"]) == [PTX_ISA_DOC_URL + FRAGMENT]


def test_cuda_row_with_an_empty_page_gets_no_url() -> None:
    rows = [IntrinsicRecord(name="__zyx_v1", signature="", description="", header="")]
    docs = {"math": {"__zyx_v1": {"anchor": "frag", "page": ""}}}
    apply_cuda_docs(rows, {}, docs)
    assert rows[0].url == ""
    docs = {"math": {"__zyx_v1": {"anchor": "frag", "page": "https://a.test/p.html"}}}
    apply_cuda_docs(rows, {}, docs)
    assert rows[0].url == "https://a.test/p.html#frag"
    assert _full(rows[0].url)


def test_display_never_shows_a_half_url() -> None:
    from simdref.display import canonical_url as shown

    assert shown(FRAGMENT) == ""
