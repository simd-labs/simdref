"""CUDA wiring across catkey, filters, display, manpages and the CLI.

CUDA intrinsic rows must behave like the ACLE and RVV intrinsic rows in
every consumer: the catalog family maps, the filter facets, the display
labels, the man page and the bare ``isa <name>`` lookup.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from conftest import FIXTURES_DIR, build_fixture_catalog
from simdref import catkey, cli, display, filters
from simdref.cuda import parse_cuda_intrinsics_payload
from simdref.storage import build_sqlite, open_db


def _build_cuda_catalog(directory: Path) -> Path:
    base = build_fixture_catalog()
    rows = parse_cuda_intrinsics_payload((FIXTURES_DIR / "cuda_intrinsics.json").read_text())
    catalog = replace(
        base, intrinsics=sorted(list(base.intrinsics) + rows, key=lambda item: item.name)
    )
    database = directory / "catalog.db"
    build_sqlite(catalog, database)
    return database


class FamilyWiringTests(unittest.TestCase):
    def test_catkey_maps_cuda_architecture_to_cuda_family(self):
        self.assertEqual(catkey.isa_family("cuda"), "cuda")
        self.assertEqual(catkey.isa_family("CUDA"), "cuda")

    def test_filter_spec_orders_cuda_family(self):
        self.assertIn("CUDA", filters.ISA_FAMILY_ORDER)

    def test_display_labels_cuda(self):
        self.assertEqual(display.isa_family("CUDA"), "CUDA")
        self.assertEqual(display.display_architecture("cuda"), "CUDA")


class CliLookupTests(unittest.TestCase):
    def _run_lookup(self, database: Path, name: str) -> tuple[int, str]:
        buf = io.StringIO()
        with (
            unittest.mock.patch.object(cli, "open_db", lambda *a, **k: open_db(database)),
            unittest.mock.patch.object(cli, "SQLITE_PATH", database),
            unittest.mock.patch.object(cli, "sqlite_schema_is_current", lambda *a, **k: True),
            unittest.mock.patch.object(cli, "CATALOG_PATH", database.parent / "missing.msgpack"),
            unittest.mock.patch.object(sys, "argv", ["isa", name]),
            redirect_stdout(buf),
        ):
            code = cli.main()
        return code, buf.getvalue()

    def test_cli_lookup_exit_zero_names_shfl_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = _build_cuda_catalog(Path(tmp))
            code, out = self._run_lookup(database, "__shfl_sync")
        self.assertEqual(code, 0)
        self.assertIn("shfl.sync", out)

    def test_cli_preset_cuda_accepted_and_scopes_to_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = _build_cuda_catalog(Path(tmp))
            buf = io.StringIO()
            with (
                unittest.mock.patch.object(cli, "open_db", lambda *a, **k: open_db(database)),
                unittest.mock.patch.object(cli, "SQLITE_PATH", database),
                unittest.mock.patch.object(cli, "sqlite_schema_is_current", lambda *a, **k: True),
                unittest.mock.patch.object(
                    cli, "CATALOG_PATH", database.parent / "missing.msgpack"
                ),
                unittest.mock.patch.object(
                    sys, "argv", ["isa", "llm", "query", "__shfl_sync", "--preset", "cuda"]
                ),
                redirect_stdout(buf),
            ):
                try:
                    code = cli.main()
                except SystemExit as exc:  # typer standalone mode exits on success too
                    code = exc.code
        out = buf.getvalue()
        self.assertEqual(code, 0, out)
        self.assertIn("__shfl_sync", out)

    def test_manpage_names_shfl_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = _build_cuda_catalog(Path(tmp))
            with open_db(database) as conn:
                page = cli._render_manpage(conn, "__shfl_sync")
        assert page is not None
        self.assertIn("shfl.sync", page)
        self.assertIn("CUDA", page)


if __name__ == "__main__":
    unittest.main()
