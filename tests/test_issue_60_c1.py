"""Regression tests for issue #60 class 1: PyPy packaging.

pdfplumber and PyMuPDF live in the optional simdref[ingest] extra, so a
plain install must not require them. A PDF parse without the extra must
fail with an install hint that names the extra.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_pdf_deps_live_in_the_ingest_extra():
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = data["project"]["dependencies"]
    extra = data["project"]["optional-dependencies"]["ingest"]
    assert not [d for d in base if d.startswith(("pdfplumber", "PyMuPDF"))], base
    assert any(d.startswith("pdfplumber") for d in extra), extra
    assert any(d.startswith("PyMuPDF") for d in extra), extra


def test_parse_without_pdfplumber_names_the_ingest_extra():
    # The SDM parse entry is the only site that needs pdfplumber. Without
    # the extra it must name simdref[ingest] and its install command.
    code = (
        "import sys; sys.modules['pdfplumber'] = None; "
        "from pathlib import Path; "
        "from simdref.pdfparse.intel import parse_intel_sdm; "
        "parse_intel_sdm(Path('missing.pdf'))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "simdref[ingest]" in result.stderr
    assert "pip install 'simdref[ingest]'" in result.stderr


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
