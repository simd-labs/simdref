"""Guard against version drift between the runtime package and pyproject.toml.

``simdref.__version__`` derives from the installed wheel metadata (see
``src/simdref/__init__.py``), and ``pyproject.toml:[project].version`` is the
single source of truth that produces that metadata. For a real (non-dev)
install the two must agree; if they ever diverge this fails loudly.

Skipped for editable/source checkouts (``+source``) and stamped dev builds
(``.dev<N>``), where the installed metadata intentionally differs from the
plain release version in pyproject.toml.
"""

from __future__ import annotations

import pathlib

import tomllib

import pytest

import simdref


def _pyproject_version() -> str:
    root = pathlib.Path(__file__).resolve().parents[1]
    return tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]


def test_runtime_version_matches_pyproject() -> None:
    runtime = simdref.__version__
    if ".dev" in runtime or "+source" in runtime:
        pytest.skip(f"non-release build ({runtime}); version match not enforced")
    assert runtime == _pyproject_version()
