"""Retry behavior of ``_download_from_release`` on transient CDN failures.

The GitHub release CDN bursts into 503/504 responses. ``_fetch_into`` must
retry 5xx, 429 and connection errors up to 4 attempts with backoff 2/4/8 s,
honour Retry-After, restart the stream from the beginning on each attempt,
and keep the 404 "next tag" behavior unchanged. Tests use
``httpx.MockTransport``; no network access.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import httpx

BODY = b"catalog-body-bytes"


def _fake_stream(handler):
    """Drop-in for ``httpx.stream`` backed by ``httpx.MockTransport``."""
    transport = httpx.MockTransport(handler)

    @contextmanager
    def fake_stream(*args, **kwargs):
        client = httpx.Client(transport=transport)
        with client.stream(*args, **kwargs) as resp:
            yield resp

    return fake_stream


def _ok(status: int = 200, **headers) -> httpx.Response:
    return httpx.Response(status, content=BODY, headers=headers)


def _interrupted_body():
    yield b"par"
    raise httpx.ReadError("stream broken")


def _run(spec: dict) -> tuple:
    """Drive ``_download_from_release`` against per-asset response queues.

    ``spec`` maps an asset name to the ordered list of responses that
    asset's requests receive. Returns ``(raised, stderr, counts, sleeps,
    data_dir)``.
    """
    from rich.console import Console

    from simdref import cli

    counts: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        asset = request.url.path.rsplit("/", 1)[-1]
        counts[asset] = counts.get(asset, 0) + 1
        return spec[asset][counts[asset] - 1]

    buf = io.StringIO()
    sleeps: list = []
    tmp = Path(tempfile.mkdtemp())
    with (
        mock.patch("httpx.stream", _fake_stream(handler)),
        mock.patch.object(cli, "err_console", Console(file=buf, force_terminal=False, width=200)),
        mock.patch.object(cli, "DATA_DIR", tmp),
        mock.patch("time.sleep", side_effect=sleeps.append),
        mock.patch.dict("os.environ", {"SIMDREF_RELEASE_TAG": "data-latest"}),
    ):
        raised = None
        try:
            cli._download_from_release()
        except BaseException as exc:
            raised = exc
    return raised, buf.getvalue(), counts, sleeps, tmp


class FetchRetryTests(unittest.TestCase):
    def test_503_twice_then_200_succeeds_after_three_requests(self):
        """503, 503, 200: the download succeeds after 3 requests with the
        2 s and 4 s backoff delays on stderr."""
        raised, out, counts, sleeps, tmp = _run(
            {
                "catalog.msgpack": [_ok(503), _ok(503), _ok()],
                "catalog.db": [_ok()],
            }
        )
        self.assertIsNone(raised)
        self.assertEqual(counts, {"catalog.msgpack": 3, "catalog.db": 1})
        self.assertEqual(sleeps, [2.0, 4.0])
        self.assertIn("retrying catalog.msgpack from data-latest in 2s (attempt 2 of 4)", out)
        self.assertIn("retrying catalog.msgpack from data-latest in 4s (attempt 3 of 4)", out)
        self.assertEqual((tmp / "catalog.msgpack").read_bytes(), BODY)
        self.assertEqual((tmp / "catalog.db").read_bytes(), BODY)
        self.assertIn("download complete", out)

    def test_404_fails_at_once_without_retry(self):
        """404 keeps its base behavior: one request, no sleep, no retry,
        and the "no compatible release asset found" failure."""
        import typer

        raised, out, counts, sleeps, tmp = _run(
            {
                "catalog.msgpack": [_ok(404)],
                "catalog.db": [_ok(404)],
            }
        )
        self.assertIsInstance(raised, typer.Exit)
        self.assertEqual(counts.get("catalog.msgpack"), 1)
        self.assertEqual(counts.get("catalog.db", 0), 0)
        self.assertEqual(sleeps, [])
        self.assertIn("no compatible release asset found", out)
        self.assertFalse((tmp / "catalog.msgpack").exists())

    def test_503_four_times_raises(self):
        """Four 503s exhaust the attempts: 4 requests, 2/4/8 s backoffs,
        then a failure naming the status."""
        import typer

        raised, out, counts, sleeps, tmp = _run(
            {
                "catalog.msgpack": [_ok(503)] * 4,
                "catalog.db": [_ok()],
            }
        )
        self.assertIsInstance(raised, typer.Exit)
        self.assertEqual(counts["catalog.msgpack"], 4)
        self.assertEqual(sleeps, [2.0, 4.0, 8.0])
        self.assertIn(
            "failed to download catalog.msgpack from data-latest after 4 attempts: 503",
            out,
        )
        self.assertFalse((tmp / "catalog.msgpack").exists())

    def test_429_honours_retry_after(self):
        """A 429 with Retry-After: 7 waits 7 s instead of the 2 s backoff."""
        raised, out, counts, sleeps, tmp = _run(
            {
                "catalog.msgpack": [_ok(429, **{"retry-after": "7"}), _ok()],
                "catalog.db": [_ok()],
            }
        )
        self.assertIsNone(raised)
        self.assertEqual(counts["catalog.msgpack"], 2)
        self.assertEqual(sleeps, [7.0])
        self.assertEqual((tmp / "catalog.msgpack").read_bytes(), BODY)

    def test_interrupted_stream_restarts_from_scratch(self):
        """A stream that breaks mid-body drops the partial bytes: the next
        attempt re-downloads from the beginning, and the final file holds
        only the full body."""
        raised, out, counts, sleeps, tmp = _run(
            {
                "catalog.msgpack": [
                    httpx.Response(
                        200,
                        content=_interrupted_body(),
                        headers={"content-length": str(len(BODY))},
                    ),
                    _ok(),
                ],
                "catalog.db": [_ok()],
            }
        )
        self.assertIsNone(raised)
        self.assertEqual(counts["catalog.msgpack"], 2)
        self.assertEqual((tmp / "catalog.msgpack").read_bytes(), BODY)


if __name__ == "__main__":
    unittest.main()
