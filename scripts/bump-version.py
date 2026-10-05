#!/usr/bin/env python3
"""Bump the release version across every file that carries it.

Usage::

    python scripts/bump-version.py 0.2.0

Edits ``pyproject.toml``'s ``[project].version``. The skill repo owns its
own three plugin-manifest version fields and its own sync check; this
script has nothing to do with them since the split.

Commit the result yourself.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def _rewrite_pyproject(version: str) -> None:
    text = PYPROJECT.read_text()
    new_text, n = re.subn(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError("could not locate version line in pyproject.toml")
    PYPROJECT.write_text(new_text)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} X.Y.Z", file=sys.stderr)
        return 2
    version = sys.argv[1].lstrip("v")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[.-][\w.]+)?", version):
        print(f"invalid version: {version!r}", file=sys.stderr)
        return 2

    _rewrite_pyproject(version)

    print(f"bumped pyproject.toml to {version}")
    print(f"next: git commit -am 'chore(release): bump version to {version}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
