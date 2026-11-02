"""Regression tests for issue #60 class 1: PyPy packaging.

PyMuPDF ships no PyPy wheel and its sdist needs a C compiler, so PyPy
installs must not require it. A PDF parse must also work without it.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_pymupdf_dependency_is_cpython_only():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if "PyMuPDF>=" in ln]
    assert len(lines) == 1, lines
    assert 'PyMuPDF>=1.28,<2; platform_python_implementation == "CPython"' in lines[0]


def test_cli_imports_without_pymupdf():
    # PyMuPDF parses vendor PDFs at catalog build time only. Under PyPy it is
    # not installed, so no runtime import path may touch fitz/pymupdf.
    code = (
        "import sys; sys.modules['fitz'] = None; sys.modules['pymupdf'] = None; "
        "import simdref.cli, simdref.ingest, simdref.ingest_pdf, simdref.lsp; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
