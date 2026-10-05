"""Tests for bare-query dispatch (issue #2).

``simdref <mnemonic> [--arch ARCH] [--json]`` must:
  - resolve on exact mnemonic match and print to stdout without opening
    the TUI — previously a TTY invocation loaded the 92K-intrinsic TUI
    and hung at >1GB RSS.
  - accept ``--arch`` aliases (sapphirerapids → EMR).
  - emit a structured JSON record with ``latency_cycles`` / ``tput_cpi`` /
    ``ports`` per arch when ``--json`` is passed.
  - tolerate the leading verb ``show`` / ``lookup`` / ``info`` since users
    reach for one even though no such subcommand exists.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
import unittest.mock


def _run_cli(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"src{os.pathsep}{existing}" if existing else "src"
    env["COLUMNS"] = "200"
    return subprocess.run(
        [sys.executable, "-m", "simdref", *args],
        cwd=".",
        env=env,
        check=check,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


class BareQueryTests(unittest.TestCase):
    def test_bare_mnemonic_prints_without_tui(self):
        proc = _run_cli("vfmadd213pd")
        out = _strip_ansi(proc.stdout)
        self.assertRegex(out, r"variant")
        self.assertRegex(out, r"lat=\d")
        self.assertRegex(out, r"cpi=\d")

    def test_bare_mnemonic_with_sapphirerapids_alias_pins_to_emr(self):
        proc = _run_cli("vfmadd213pd", "--arch", "sapphirerapids")
        out = _strip_ansi(proc.stdout)
        self.assertIn("EMR", out)
        self.assertRegex(out, r"lat=\d")
        self.assertRegex(out, r"cpi=\d")

    def test_leading_show_verb_is_tolerated(self):
        """The original bug report used ``simdref show vgatherdpd --arch …``;
        there's no ``show`` subcommand, but the verb should be dropped
        rather than hanging in the TUI or reporting 'no match'."""
        proc = _run_cli("show", "vgatherdpd", "--arch", "sapphirerapids")
        out = _strip_ansi(proc.stdout)
        self.assertIn("EMR", out)
        self.assertRegex(out, r"lat=\d")

    def test_bare_query_json_emits_structured_lat_cpi(self):
        proc = _run_cli("vfmadd213pd", "--arch", "sapphirerapids", "--json")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["arch"], "EMR")
        self.assertGreaterEqual(len(payload["variants"]), 1)
        variant = payload["variants"][0]
        emr_rows = [r for r in variant["per_arch"] if r["arch"] == "EMR"]
        self.assertEqual(len(emr_rows), 1)
        self.assertIsNotNone(emr_rows[0]["latency_cycles"])
        self.assertIsNotNone(emr_rows[0]["tput_cpi"])

    def test_unknown_arch_exits_nonzero(self):
        proc = _run_cli("vfmadd213pd", "--arch", "not_a_core", check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not in the local catalog", _strip_ansi(proc.stderr))

    def test_unknown_mnemonic_in_non_tty_exits_2(self):
        proc = _run_cli("zzzqqq", check=False)
        self.assertEqual(proc.returncode, 2)


class BareIntrinsicQueryTests(unittest.TestCase):
    def test_bare_exact_intrinsic_prints_detail_non_interactive(self):
        proc = _run_cli("_mm_add_ps")
        out = _strip_ansi(proc.stdout)
        self.assertEqual(proc.returncode, 0)
        # Detail-only rows: the search table never prints these fields,
        # so a search-row hit cannot pass this test.
        self.assertIn("intrinsic: _mm_add_ps", out)
        self.assertIn("signature", out)
        self.assertIn("xmmintrin.h", out)

    def test_bare_fuzzy_query_prints_ranked_non_interactive(self):
        proc = _run_cli("_mm_add")
        out = _strip_ansi(proc.stdout)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("_mm_add_ps", out)
        self.assertNotIn("signature", out)

    def test_bare_fuzzy_query_json_parses(self):
        proc = _run_cli("_mm_add", "--json")
        self.assertEqual(proc.returncode, 0)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "search")
        self.assertTrue(payload["results"])

    def test_search_results_print_in_relevance_order(self):
        """_print_search_results_runtime must keep _search_runtime's
        relevance order; an alphabetical (ISA/title) re-sort is exactly
        the regression this test guards against."""
        from simdref import cli

        class FakeRecord:
            isa = ["SSE2"]
            arch_details = {}
            architecture = "x86"
            summary = "s"

        rows = [
            cli.SearchResult(kind="instruction", key="zz_add", title="zz", subtitle="s", score=0.9),
            cli.SearchResult(kind="instruction", key="aa_sub", title="aa", subtitle="s", score=0.1),
        ]
        inv_map = {r.key: FakeRecord() for r in rows}
        captured: list = []
        captured_json: list = []
        with (
            unittest.mock.patch.object(
                cli, "_search_runtime", lambda conn, q, limit=20: (rows, {}, inv_map)
            ),
            unittest.mock.patch.object(cli, "isa_visible", lambda isa, show_fp16=False: True),
            unittest.mock.patch.object(
                cli, "render_search_results", lambda rws: captured.append([r.key for r, *_ in rws])
            ),
            unittest.mock.patch.object(cli.typer, "echo", lambda s: captured_json.append(s)),
        ):
            cli._print_search_results_runtime(None, "q")
            self.assertEqual(captured[0], ["zz_add", "aa_sub"])
            cli._print_search_results_runtime(None, "q", as_json=True)
            keys = [r["key"] for r in json.loads(captured_json[0])["results"]]
            self.assertEqual(keys, ["zz_add", "aa_sub"])


class BareQueryEdgeTests(unittest.TestCase):
    def test_json_forces_non_interactive_even_on_tty(self):
        """--json must never open the TUI, even when stdio says TTY."""
        from simdref import cli

        with (
            unittest.mock.patch.object(cli.sys.stdin, "isatty", return_value=True),
            unittest.mock.patch.object(cli.sys.stdout, "isatty", return_value=True),
            unittest.mock.patch.object(
                cli, "_run_tui", side_effect=AssertionError("TUI must not launch with --json")
            ),
        ):
            rc = cli._smart_lookup("_mm_add", as_json=True)
            self.assertEqual(rc, 0)

    def test_unmatched_json_exits_nonzero_without_tui(self):
        from simdref import cli

        with (
            unittest.mock.patch.object(cli.sys.stdin, "isatty", return_value=True),
            unittest.mock.patch.object(cli.sys.stdout, "isatty", return_value=True),
            unittest.mock.patch.object(
                cli, "_run_tui", side_effect=AssertionError("TUI must not launch with --json")
            ),
        ):
            rc = cli._smart_lookup("zzzqqq", as_json=True)
            self.assertEqual(rc, 2)

    def test_invalid_arch_on_intrinsic_query_rejected(self):
        proc = _run_cli("_mm_add_ps", "--arch", "not_a_core", check=False)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--arch applies to instruction queries only", _strip_ansi(proc.stderr))

    def test_valid_arch_on_intrinsic_query_rejected(self):
        """--arch is meaningless for intrinsic detail even when the core
        name itself is valid; the rejection must use the documented exit
        code and message."""
        from simdref.perf_sources.cores import supported_core_ids

        core = supported_core_ids()[0]
        proc = _run_cli("_mm_add_ps", "--arch", core, check=False)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--arch applies to instruction queries only", _strip_ansi(proc.stderr))

    def test_valid_arch_on_fuzzy_result_rejected(self):
        """The same rule covers fuzzy hits: --arch cannot ride along on a
        ranked search list."""
        from simdref.perf_sources.cores import supported_core_ids

        core = supported_core_ids()[0]
        proc = _run_cli("_mm_add", "--arch", core, "--json", check=False)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertIn("--arch applies to instruction queries only", _strip_ansi(proc.stderr))

    def test_all_hidden_fuzzy_results_count_as_no_match(self):
        """When every search row is filtered by visibility (fp16/bf16),
        the command must exit 2, not print an empty table."""
        from simdref import cli

        # Stub the search to return a result whose ISA is hidden by default.
        class FakeRecord:
            isa = ["BF16"]
            arch_details = {}
            architecture = "x86"
            summary = "s"

        fake = cli.SearchResult(kind="instruction", key="k", title="t", subtitle="s", score=0.0)
        with (
            unittest.mock.patch.object(
                cli, "_search_runtime", lambda conn, q, limit=20: ([fake], {}, {"k": FakeRecord()})
            ),
            unittest.mock.patch.object(cli, "isa_visible", lambda isa, show_fp16=False: False),
            unittest.mock.patch.object(cli.sys.stdin, "isatty", return_value=False),
            unittest.mock.patch.object(cli.sys.stdout, "isatty", return_value=False),
        ):
            rc = cli._smart_lookup("_mm_add")
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
