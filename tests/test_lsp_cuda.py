"""CUDA device intrinsic support in the LSP: hover and completion.

CUDA device intrinsic rows (architecture "cuda") surface in hover and
completion only in CUDA documents (.cu/.cuh). In C/C++ documents the row is
treated as absent: hover returns None and completion drops the cuda rows
from the intrinsic pool before ranking. The f0e2f36 zone logic is untouched:
asm template strings and MSVC __asm blocks still offer mnemonics only.

This file starts python -m simdref.lsp. It reuses the stdio harness from
tests/test_lsp_ptx.py and builds a fixture catalog that also holds the CUDA
device intrinsics from tests/fixtures/cuda_intrinsics.json.

Each test carries its own negative control, so all three fail on commit
43c16bf, where the cuda gate does not exist and the rows would surface in
C/C++ documents.
"""

import atexit
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from conftest import FIXTURES_DIR, build_fixture_catalog
from simdref.cuda import parse_cuda_intrinsics_payload
from simdref.storage import build_sqlite
from test_lsp_ptx import _Server

_TMP = tempfile.TemporaryDirectory(prefix="simdref-lsp-cuda-catalog-")
atexit.register(_TMP.cleanup)

_CATALOG: Path | None = None

CU_URI = "file:///tmp/simdref_lsp_cuda.cu"
CPP_URI = "file:///tmp/simdref_lsp_cuda.cpp"

_CUDA_TEXT = (FIXTURES_DIR / "cuda_intrinsics.json").read_text()
_CUDA_ROWS = parse_cuda_intrinsics_payload(_CUDA_TEXT)
CUDA_NAMES = [row.name for row in _CUDA_ROWS]


def _build_cuda_catalog() -> Path:
    # The fixture catalog plus the CUDA device intrinsic rows.
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    base = build_fixture_catalog()
    catalog = replace(
        base,
        intrinsics=sorted(list(base.intrinsics) + _CUDA_ROWS, key=lambda item: item.name),
    )
    database = Path(_TMP.name) / "catalog.db"
    build_sqlite(catalog, database)
    _CATALOG = database
    return database


class LspCudaTests(unittest.TestCase):
    def setUp(self):
        database = _build_cuda_catalog()
        env = {
            **os.environ,
            "SIMDREF_CATALOG": str(database),
            "SIMDREF_SKIP_AUTOUPDATE": "1",
        }
        self.server = _Server(env)
        self.server.start()
        self.addCleanup(self.server.stop)

    def test_hover_cuda_intrinsic_in_cu_not_cpp(self):
        # (a) Hover __shfl_sync in a .cu file shows the signature and the
        # PTX Instructions line; the same hover in a .cpp file returns None.
        text = "unsigned v = __shfl_sync(mask, var, lane);\n"
        pos = text.index("__shfl_sync") + 2
        result = self.server.hover(CU_URI, "cuda", text, 0, pos)
        self.assertIsNotNone(result)
        value = result["contents"]["value"]
        self.assertIn("__shfl_sync", value)
        self.assertIn("shfl.sync", value)
        # The web link lands next to the source link.
        self.assertLess(value.index("[source]("), value.index("[simdref]("))
        result = self.server.hover(CPP_URI, "cpp", text, 0, pos)
        self.assertIsNone(result)

    def test_completion_cuda_prefix_in_cu_not_cpp(self):
        # (b) Completion of the prefix "__shfl" in plain .cu code offers
        # __shfl_sync; the same prefix in .cpp offers none of the cuda names.
        text = "void f() { __shfl\n}"
        pos = text.index("__shfl") + len("__shfl")
        items = self.server.completion(CU_URI, "cuda", text, 0, pos)
        self.assertIn("__shfl_sync", [item["label"] for item in items])
        items = self.server.completion(CPP_URI, "cpp", text, 0, pos)
        labels = [item["label"] for item in items]
        for name in CUDA_NAMES:
            self.assertNotIn(name, labels)

    def test_cpp_offers_no_cuda_and_hovers_none(self):
        # (c) A .cpp file offers no cuda intrinsic and hovers none, while the
        # same catalog's .cu file offers __shfl_sync (positive control).
        cu_text = "void f() { __shfl\n}"
        cu_pos = cu_text.index("__shfl") + len("__shfl")
        items = self.server.completion(CU_URI, "cuda", cu_text, 0, cu_pos)
        self.assertIn("__shfl_sync", [item["label"] for item in items])
        cpp_text = "void f() { __shfl\n}"
        cpp_pos = cpp_text.index("__shfl") + len("__shfl")
        items = self.server.completion(CPP_URI, "cpp", cpp_text, 0, cpp_pos)
        labels = [item["label"] for item in items]
        for name in CUDA_NAMES:
            self.assertNotIn(name, labels)
        for name in CUDA_NAMES:
            hover_text = f"void f() {name}();\n"
            hpos = hover_text.index(name) + 1
            result = self.server.hover(CPP_URI, "cpp", hover_text, 0, hpos)
            self.assertIsNone(result, msg=f".cpp hover must not show {name}")


if __name__ == "__main__":
    unittest.main()
