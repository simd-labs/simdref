"""The test session must never reach the catalog auto-update download.

``tests/conftest.py`` sets ``SIMDREF_SKIP_AUTOUPDATE`` for the whole
session, so subprocess CLI tests inherit it too. This test fails when
that setting is removed: a version-stamp mismatch then calls the
download path.
"""

from unittest import mock

from simdref import cli


def test_version_stamp_mismatch_does_not_download_in_test_session():
    with (
        mock.patch.object(cli, "read_installed_version_stamp", lambda: "0.0.0"),
        mock.patch.object(cli, "__version__", "9.9.9"),
        mock.patch.object(cli, "_bootstrap_interactive", lambda: None),
        mock.patch.object(cli, "sqlite_schema_is_current", lambda: True),
        mock.patch.object(cli, "_download_release_or_fallback") as dl,
    ):
        cli.ensure_runtime()
    dl.assert_not_called()
