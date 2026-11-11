"""Url building shared by the ingest and the display code.

A url comes from a parsed base url plus a reference, or from a stored host
path. A part that is not a full url gives "", never a url without a host.
"""

from __future__ import annotations

from urllib.parse import urljoin, urlsplit


def is_full_url(url: str) -> bool:
    """True when the url has a scheme, a host and a path."""
    parts = urlsplit(url)
    return bool(parts.scheme and parts.netloc and parts.path)


def resolve_url(base: str, ref: str = "") -> str:
    """The reference (a path, a ``#fragment`` or a full url) resolved against the base.

    An empty or partial base with a relative reference gives "".
    """
    ref = ref.strip()
    if is_full_url(ref):
        return ref
    base = base.strip()
    if not is_full_url(base):
        return ""
    joined = urljoin(base, ref)
    return joined if is_full_url(joined) else ""


def canonical_url(value: str) -> str:
    """A full url for a stored url value, or "" when the value names no host.

    A value with an http(s) scheme and a host passes through. A host path such
    as ``uops.info/x`` gets ``https://www.``. A bare ``#fragment`` gives "".
    """
    value = value.strip()
    parts = urlsplit(value)
    if parts.scheme in ("http", "https"):
        return value if parts.netloc else ""
    host = value.lstrip("/").partition("/")[0]
    if parts.scheme or not host or host.startswith(("#", "?")):
        return ""
    return f"https://www.{value.lstrip('/')}"
