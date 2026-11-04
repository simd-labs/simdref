"""Bootstrap banner + download progress reporting (issue #5).

A fresh install's first ``simdref annotate`` (or any command) triggers
``_bootstrap_interactive`` which downloads ``catalog.msgpack`` and
``catalog.db`` from a GitHub release. On the user's machine that took
~10 minutes with no output, indistinguishable from a hang in CI / agent
harnesses.

These tests exercise the plumbing — they do not hit the network; they
stub ``httpx.stream`` and inspect what gets printed.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from contextlib import contextmanager
from unittest import mock

import httpx


class _FakeResponse:
    def __init__(self, body: bytes, status_code: int = 200, content_length: bool = True):
        self._body = body
        self.status_code = status_code
        self.headers = {"content-length": str(len(body))} if content_length else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=mock.Mock(), response=mock.Mock(status_code=self.status_code)
            )

    def iter_bytes(self, chunk_size: int = 1024):
        # One chunk keeps the test deterministic; callers don't care.
        yield self._body


@contextmanager
def _stream_returning(body: bytes):
    @contextmanager
    def fake_stream(*args, **kwargs):
        yield _FakeResponse(body)

    with mock.patch("httpx.stream", fake_stream):
        yield


class _ChunkedResponse(_FakeResponse):
    """Yields the body in several chunks so the chunk loop runs repeatedly."""

    def __init__(self, chunks: list, content_length: bool = True):
        super().__init__(b"".join(chunks), content_length=content_length)
        self._chunks = chunks

    def iter_bytes(self, chunk_size: int = 1024):
        yield from self._chunks


@contextmanager
def _stream_returning_chunks(chunks: list, content_length: bool = True):
    @contextmanager
    def fake_stream(*args, **kwargs):
        yield _ChunkedResponse(chunks, content_length=content_length)

    with mock.patch("httpx.stream", fake_stream):
        yield


def _tick_seconds():
    # Each asset consumes one init read plus one read per chunk.
    # Prints happen at chunk 2 (t=6) and chunk 4 (t=12).
    while True:
        yield from (0.0, 1.0, 6.0, 7.0, 12.0, 13.0)


class DownloadProgressTests(unittest.TestCase):
    def test_non_tty_emits_per_asset_size_line(self):
        """In CI / non-TTY mode, each asset download should print a
        ``downloaded <asset>: N.N MB`` line so harnesses observe progress."""
        import tempfile
        from pathlib import Path
        from rich.console import Console
        from simdref import cli

        buf = io.StringIO()
        fake_console = Console(file=buf, force_terminal=False, width=200)

        tmp = Path(tempfile.mkdtemp())
        with (
            _stream_returning(b"x" * (2 * 1024 * 1024)),
            mock.patch.object(cli, "err_console", fake_console),
            mock.patch.object(cli, "DATA_DIR", tmp),
        ):
            cli._download_from_release()

        output = buf.getvalue()
        self.assertIn("catalog.msgpack", output)
        self.assertIn("catalog.db", output)
        self.assertIn("downloaded", output)
        self.assertIn("MB", output)
        self.assertIn("download complete", output)

    def test_non_tty_emits_periodic_progress_lines(self):
        """When the download spans more than 5 s of wall time, stderr should
        show periodic ``downloading X: M.M / T.T MB (NN%)`` lines so a slow
        link does not look hung; stdout stays clean."""
        import re
        import tempfile
        from pathlib import Path
        from rich.console import Console
        from simdref import cli

        buf = io.StringIO()
        out_buf = io.StringIO()
        fake_console = Console(file=buf, force_terminal=False, width=200)

        tmp = Path(tempfile.mkdtemp())
        chunks = [b"x" * (1024 * 1024)] * 5  # 5 MiB total per asset
        with (
            _stream_returning_chunks(chunks),
            mock.patch.object(cli, "err_console", fake_console),
            mock.patch.object(cli, "DATA_DIR", tmp),
            mock.patch("time.monotonic", side_effect=_tick_seconds()),
            contextlib.redirect_stdout(out_buf),
        ):
            cli._download_from_release()

        output = buf.getvalue()
        progress = re.findall(r"downloading catalog\.\w+: [\d.]+ / [\d.]+ MB \(\d+%\)", output)
        self.assertGreaterEqual(len(progress), 1)
        self.assertIn("(40%)", progress[0])
        # Existing start/end lines unchanged.
        self.assertIn("downloading catalog.msgpack from", output)
        self.assertIn("downloaded catalog.msgpack:", output)
        # Nothing download-related leaks to stdout.
        self.assertNotIn("downloading catalog", out_buf.getvalue())

    def test_non_tty_progress_line_without_content_length(self):
        """With no content-length header the periodic line shows only the
        received MB count, no total and no percentage."""
        import re
        import tempfile
        from pathlib import Path
        from rich.console import Console
        from simdref import cli

        buf = io.StringIO()
        fake_console = Console(file=buf, force_terminal=False, width=200)

        tmp = Path(tempfile.mkdtemp())
        chunks = [b"x" * (1024 * 1024)] * 5
        with (
            _stream_returning_chunks(chunks, content_length=False),
            mock.patch.object(cli, "err_console", fake_console),
            mock.patch.object(cli, "DATA_DIR", tmp),
            mock.patch("time.monotonic", side_effect=_tick_seconds()),
        ):
            cli._download_from_release()

        output = buf.getvalue()
        mb_only = re.findall(r"^downloading catalog\.\w+: [\d.]+ MB$", output, re.M)
        self.assertGreaterEqual(len(mb_only), 1)
        mid_lines = [line for line in output.splitlines() if line.startswith("downloading catalog")]
        self.assertFalse(any("%" in line for line in mid_lines))

    def test_bootstrap_banner_includes_hint(self):
        """The bootstrap banner must mention ``simdref update`` so users
        know how to pre-fetch explicitly next time."""
        from simdref import cli

        captured_err = []
        captured_out = []

        def _err_print(*args, **kwargs):
            captured_err.append(" ".join(str(a) for a in args))

        def _typer_echo(msg, *args, **kwargs):
            captured_out.append(str(msg))

        with (
            mock.patch.object(cli.err_console, "print", side_effect=_err_print),
            mock.patch("simdref.cli.typer.echo", side_effect=_typer_echo),
            mock.patch("sys.stdout.isatty", return_value=False),
            mock.patch.object(cli, "_download_release_or_fallback"),
        ):
            cli._bootstrap_interactive()

        banner = "\n".join(captured_err)
        self.assertIn("bootstrap", banner.lower())
        self.assertIn("simdref update", banner)
        # Non-TTY stdout markers bracket the download so log scrapers can
        # detect the bootstrap window.
        stdout = "\n".join(captured_out)
        self.assertIn("bootstrapping catalog", stdout)
        self.assertIn("bootstrap complete", stdout)


if __name__ == "__main__":
    unittest.main()
