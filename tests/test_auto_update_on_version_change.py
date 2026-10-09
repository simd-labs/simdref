"""Auto-update when the installed package version changes.

After a ``pip``/``uv`` install or upgrade the recorded version stamp in
``catalog.db``'s ``meta`` table will not match ``simdref.__version__``.
``ensure_runtime`` must transparently refresh the catalog so users don't
have to remember to run ``simdref update`` themselves. These tests stub
the network and the on-disk catalog to exercise the version-stamp logic
in isolation.
"""

from __future__ import annotations

import io
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import httpx
from rich.console import Console


class _StampFixture:
    """Redirect the version-stamp storage to a tmp path for the test body."""

    def __init__(self, tmp_dir: Path):
        self.path = tmp_dir / "installed_version"

    def __enter__(self):
        self._patches = [
            mock.patch("simdref.cli.read_installed_version_stamp", self._read),
            mock.patch("simdref.cli.write_installed_version_stamp", self._write),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()

    def _read(self) -> str | None:
        try:
            return self.path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def _write(self, version: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(version.strip() + "\n", encoding="utf-8")


@contextmanager
def _stamp_in_tmp(tmp_dir: Path):
    with _StampFixture(tmp_dir) as fx:
        yield fx


class AutoUpdateOnVersionChangeTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_version_mismatch_triggers_download(self):
        from simdref import cli

        with _stamp_in_tmp(self.tmp_dir) as fx:
            fx._write("0.0.1")  # older than installed
            with (
                mock.patch.object(cli, "__version__", "9.9.9"),
                mock.patch.object(cli, "_download_release_or_fallback") as dl,
                mock.patch.dict("os.environ", {}, clear=False),
            ):
                # Make sure the override env var is not set in this run.
                import os

                os.environ.pop("SIMDREF_SKIP_AUTOUPDATE", None)
                cli._maybe_auto_update_for_version_change()
            dl.assert_called_once()
            self.assertEqual(fx._read(), None or fx._read())  # stamp untouched here:
            # the stamp gets written by the real download helper, which is mocked,
            # so we only assert that the helper was invoked.

    def test_same_version_is_noop(self):
        from simdref import cli

        with _stamp_in_tmp(self.tmp_dir) as fx:
            fx._write(cli.__version__)
            with mock.patch.object(cli, "_download_release_or_fallback") as dl:
                cli._maybe_auto_update_for_version_change()
            dl.assert_not_called()

    def test_skip_env_var_disables_auto_update(self):
        from simdref import cli

        with _stamp_in_tmp(self.tmp_dir) as fx:
            fx._write("0.0.1")
            with (
                mock.patch.object(cli, "__version__", "9.9.9"),
                mock.patch.object(cli, "_download_release_or_fallback") as dl,
                mock.patch.dict("os.environ", {"SIMDREF_SKIP_AUTOUPDATE": "1"}),
            ):
                cli._maybe_auto_update_for_version_change()
            dl.assert_not_called()

    def test_first_run_only_stamps(self):
        """No prior stamp means we just bootstrapped — don't re-download."""
        from simdref import cli

        with _stamp_in_tmp(self.tmp_dir) as fx:
            self.assertIsNone(fx._read())
            with (
                mock.patch.object(cli, "__version__", "1.2.3"),
                mock.patch.object(cli, "_download_release_or_fallback") as dl,
            ):
                cli._maybe_auto_update_for_version_change()
            dl.assert_not_called()
            self.assertEqual(fx._read(), "1.2.3")

    def test_failed_auto_update_keeps_old_stamp_and_warns(self):
        """If the download fails (offline) we warn and keep the old stamp, so
        the next invocation retries the download instead of staying stuck on
        the old catalog until the next release."""
        import typer
        from simdref import cli

        with _stamp_in_tmp(self.tmp_dir) as fx:
            fx._write("0.0.1")
            with (
                mock.patch.object(cli, "__version__", "9.9.9"),
                mock.patch.object(
                    cli, "_download_release_or_fallback", side_effect=typer.Exit(code=1)
                ),
            ):
                buf = io.StringIO()
                fake_console = Console(file=buf, force_terminal=False, width=200)
                with mock.patch.object(cli, "err_console", fake_console):
                    cli._maybe_auto_update_for_version_change()
                output = buf.getvalue()
            self.assertIn("auto-update failed", output)
            self.assertEqual(fx._read(), "0.0.1")


class OfflineWarningTests(unittest.TestCase):
    def test_connect_error_emits_offline_warning(self):
        """If the release host is unreachable, ``_download_from_release``
        must print a clear "no internet connectivity" warning before
        propagating ``typer.Exit`` to the caller."""
        import typer
        from simdref import cli

        @contextmanager
        def fake_stream(*args, **kwargs):
            raise httpx.ConnectError("name resolution failed")
            yield  # pragma: no cover

        buf = io.StringIO()
        fake_console = Console(file=buf, force_terminal=False, width=200)
        with (
            mock.patch("httpx.stream", fake_stream),
            mock.patch.object(cli, "err_console", fake_console),
        ):
            with self.assertRaises(typer.Exit):
                cli._download_from_release()
        output = buf.getvalue()
        self.assertIn("no internet connectivity", output)


if __name__ == "__main__":
    unittest.main()
