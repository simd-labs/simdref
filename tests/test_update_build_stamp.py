"""``simdref update --build`` (now ``simdref build``) must write the version stamp.

Without a TTY the local build path used to skip ``write_installed_version_stamp``.
The next run then saw no stamp (or an old one), treated the fresh runtime as
stale and re-downloaded it. The test fakes the downloads and keeps every write
in a temp data dir; the stamp itself is the real one in the built ``catalog.db``.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console


class UpdateBuildStampTests(unittest.TestCase):
    def test_non_tty_build_writes_version_stamp(self):
        from typer.testing import CliRunner

        from conftest import build_fixture_catalog
        from simdref import cli, storage

        tmp = Path(tempfile.mkdtemp())
        db = tmp / "catalog.db"
        mp = tmp / "catalog.msgpack"
        cat = build_fixture_catalog()
        patches = [
            mock.patch.object(
                cli,
                "refresh_local_arm_intrinsics_bundle",
                lambda on_progress=None: [tmp / "arm.json"],
            ),
            mock.patch.object(
                cli,
                "refresh_local_arm_a64_archive",
                lambda on_progress=None: tmp / "AARCHMRS.tar",
            ),
            mock.patch.object(cli, "build_catalog", lambda include_sdm=False, status=None: cat),
            mock.patch.object(cli, "save_catalog", lambda c: storage.save_catalog(c, mp)),
            mock.patch.object(cli, "build_sqlite", lambda c: storage.build_sqlite(c, db)),
            mock.patch.object(cli, "WEB_DIR", tmp / "web"),
            mock.patch.object(cli, "_require_llvm_mca_or_hint", lambda: None),
            mock.patch.object(cli, "console", Console(file=io.StringIO(), force_terminal=False)),
            mock.patch.object(
                cli, "err_console", Console(file=io.StringIO(), force_terminal=False)
            ),
            mock.patch.object(storage, "SQLITE_PATH", db),
            mock.patch.object(storage, "CATALOG_PATH", mp),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(patches)])

        result = CliRunner().invoke(cli.app, ["build", "--man-dir", str(tmp / "man")])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(storage.read_installed_version_stamp(), cli.__version__)


if __name__ == "__main__":
    unittest.main()
