"""PTX and CUDA support in the LSP: hover and completion.

This file starts python -m simdref.lsp. It uses the same setup as
tests/test_lsp_stdio.py. build_sqlite makes the fixture catalog (intel,
amd, riscv, ptx) one time per run. The catalog is read-only.

Cases:
  * hover on a .ptx line names the fixture row;
  * hover on a .cu asm template string names the row;
  * completion in the .cu asm template string gives only mnemonics
    (kind 14). The "frob." prefix gives the bare op row;
  * completion in the plain C++ code of the same .cu document gives only
    intrinsics (kind 3). It has no ptx rows;
  * a dotted completion prefix in a .s document keeps the old empty
    answer: the dot rule is ptx-only (review finding 1).

Three feature tests do not pass on origin/main. origin/main has no .ptx
recognition, no ptx flag, no PTX lookup path and no dot rule. The plain
code test passes on origin/main by design. origin/main already limits
intrinsics to the code zone. The test guards that split. The .s dot test
is a control: it passes on origin/main and fails when the dot gate drops.
"""

import atexit
import json
import os
import select
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from conftest import build_fixture_catalog, build_match_catalog
from simdref.storage import build_sqlite

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

_TMP = tempfile.TemporaryDirectory(prefix="simdref-lsp-ptx-catalog-")
atexit.register(_TMP.cleanup)

_CATALOG: Path | None = None
_MATCH_CATALOG: Path | None = None


def _build_ptx_catalog() -> Path:
    # The fixture catalog as a sqlite file. build_sqlite makes it one time
    # per run. The LSP subprocesses read it only.
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    database = Path(_TMP.name) / "catalog.db"
    build_sqlite(build_fixture_catalog(), database)
    _CATALOG = database
    return database


def _build_match_catalog() -> Path:
    # The fixture catalog plus the match-sample PTX rows (ld, add, vadd).
    global _MATCH_CATALOG
    if _MATCH_CATALOG is not None:
        return _MATCH_CATALOG
    database = Path(_TMP.name) / "match-catalog.db"
    build_sqlite(build_match_catalog(), database)
    _MATCH_CATALOG = database
    return database


def _write_msg(stream, obj) -> None:
    data = json.dumps(obj).encode("utf-8")
    stream.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data)
    stream.flush()


def _read_msg(stream):
    headers = {}
    while True:
        raw = stream.readline()
        if not raw:
            return None
        line = raw.decode("ascii", "replace").strip()
        if not line:
            break
        key, _sep, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    data = stream.read(length)
    return json.loads(data.decode("utf-8"))


class _Server:
    """One stdio LSP subprocess per test; requests answer by id."""

    def __init__(self, env):
        self._env = env
        self._next_id = 0
        self._process = None

    def start(self):
        self._process = subprocess.Popen(
            [sys.executable, "-m", "simdref.lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_ROOT),
            env=self._env,
        )

    def stop(self):
        if self._process is not None:
            self._process.stdin.close()
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
            self._process = None

    def request(self, method, params, timeout=60):
        self._next_id += 1
        msg_id = self._next_id
        _write_msg(
            self._process.stdin,
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": method,
                "params": params,
            },
        )
        deadline = timeout
        while True:
            ready, _w, _x = select.select([self._process.stdout], [], [], deadline)
            if not ready:
                raise TimeoutError(f"{method} timed out")
            msg = _read_msg(self._process.stdout)
            if msg is None:
                raise RuntimeError("server closed stdout")
            if msg.get("id") == msg_id:
                return msg

    def notify(self, method, params):
        _write_msg(
            self._process.stdin,
            {"jsonrpc": "2.0", "method": method, "params": params},
        )

    def open(self, uri, language_id, text):
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            },
        )

    def hover(self, uri, language_id, text, line, character):
        self.open(uri, language_id, text)
        msg = self.request(
            "textDocument/hover",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
        )
        if msg.get("error"):
            raise AssertionError(f"hover error: {msg['error']}")
        return msg.get("result")

    def completion(self, uri, language_id, text, line, character):
        self.open(uri, language_id, text)
        msg = self.request(
            "textDocument/completion",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
        )
        if msg.get("error"):
            raise AssertionError(f"completion error: {msg['error']}")
        result = msg.get("result")
        return (result or {}).get("items", [])


# .ptx document: the frob row, dotted modifiers, PTX operand shapes.
PTX_URI = "file:///tmp/simdref_lsp_ptx.ptx"
PTX_TEXT = "frob.global.v2.f32 {%f1, %f2}, [%rd1];\n"

# .cu document: line 0 is the asm template string, line 1 plain C++.
CU_URI = "file:///tmp/simdref_lsp_ptx.cu"
CU_TEXT = 'int f() { asm("frob.nc.f32 %f1, [%rd1];");\n    return _mm256_add_ps(a, b);\n}\n'


class LspPtxTests(unittest.TestCase):
    def setUp(self):
        database = _build_ptx_catalog()
        env = {**os.environ, "SIMDREF_CATALOG": str(database), "SIMDREF_SKIP_AUTOUPDATE": "1"}
        self.server = _Server(env)
        self.server.start()
        self.addCleanup(self.server.stop)

    def test_hover_ptx_document_names_row(self):
        # Column 5 is in "frob.global.v2.f32". The row mnemonic is frob.
        result = self.server.hover(PTX_URI, "ptx", PTX_TEXT, 0, 5)
        self.assertIsNotNone(result)
        self.assertIn("frob", result["contents"]["value"].casefold())

    def test_hover_cuda_asm_string_names_row(self):
        # Column 21 is inside "frob.nc.f32" in the asm template string.
        result = self.server.hover(CU_URI, "cuda", CU_TEXT, 0, 21)
        self.assertIsNotNone(result)
        self.assertIn("frob.nc", result["contents"]["value"].casefold())

    def test_completion_cuda_template_string_mnemonics_only(self):
        # Column 20 is right after "frob." in the template string.
        items = self.server.completion(CU_URI, "cuda", CU_TEXT, 0, 20)
        self.assertTrue(items)
        kinds = {item["kind"] for item in items}
        self.assertEqual(kinds, {14})
        labels = [item["label"] for item in items]
        # The dot rule: the "frob." prefix gives the bare op row.
        self.assertIn("frob d, [a]", labels)
        self.assertIn("frob.nc d, [a]", labels)

    def test_completion_cuda_plain_code_intrinsics_only(self):
        # Column 14 is after "_mm" in the plain C++ line. The zone gives
        # intrinsics only, no ptx rows, no kind 14.
        items = self.server.completion(CU_URI, "cuda", CU_TEXT, 1, 14)
        self.assertTrue(items)
        kinds = {item["kind"] for item in items}
        self.assertEqual(kinds, {3})
        self.assertNotIn("frob d, [a]", [item["label"] for item in items])

    def test_completion_ptx_dotted_name_still_completes(self):
        # A hand-typed "frob." and a "." trigger both complete in a .ptx file.
        text = "frob.\n"
        self.assertTrue(self.server.completion(PTX_URI, "ptx", text, 0, 5))
        msg = self.server.request(
            "textDocument/completion",
            {
                "textDocument": {"uri": PTX_URI},
                "position": {"line": 0, "character": 5},
                "context": {"triggerKind": 2, "triggerCharacter": "."},
            },
        )
        self.assertTrue(msg["result"]["items"])

    def test_completion_ptx_operand_and_comment_give_no_items(self):
        text = "frob.f32 %f1, frob\n// frob comment\n"
        self.assertEqual(self.server.completion(PTX_URI, "ptx", text, 0, 18), [])
        self.assertEqual(self.server.completion(PTX_URI, "ptx", text, 1, 7), [])

    def test_completion_dot_prefix_unchanged_outside_ptx(self):
        # Control: a dotted prefix in a .s document keeps origin/main's
        # empty answer. The dot rule is ptx-only (review finding 1).
        items = self.server.completion("file:///tmp/simdref_lsp_ptx_dot.s", "asm", "add.\n", 0, 4)
        self.assertEqual(items, [])


MATCH_CASES = (
    (
        "ld.global.nc.f32 %f1, [%rd1];\n",
        "ld.global.nc",
        "#data-movement-and-conversion-instructions-ld-global-nc",
    ),
    (
        "ld.global.cg.nc.f32 %f1, [%rd1];\n",
        "ld.global.nc",
        "#data-movement-and-conversion-instructions-ld-global-nc",
    ),
    ("add.f32 %f1, %f2, %f3;\n", "add", "#floating-point-instructions-add"),
    ("add.s32 %r1, %r2, %r3;\n", "add", "#integer-arithmetic-instructions-add"),
    (
        "vadd.u32.s32.sat %r1, %r2, %r3;\n",
        "vop.dtype.atype.btype",
        "#scalar-video-instructions-vadd-vsub-vabsdiff-vmin-vmax",
    ),
)


class LspPtxSpellingTests(unittest.TestCase):
    """Concrete spellings and a host asm line in a .cu file."""

    def setUp(self):
        database = _build_match_catalog()
        env = {**os.environ, "SIMDREF_CATALOG": str(database), "SIMDREF_SKIP_AUTOUPDATE": "1"}
        self.server = _Server(env)
        self.server.start()
        self.addCleanup(self.server.stop)

    def test_hover_concrete_spellings_name_the_documenting_section(self):
        # Each concrete spelling hovers the row of the template line that
        # documents it, with that line's section url. add.f32 is floating
        # point, add.s32 is integer: one merged `add` row, two sections.
        # The ld.global.cg.nc case only resolves through the qualifier-set
        # match: the chain lookup lands on the plain `ld` row instead.
        for text, name, url in MATCH_CASES:
            with self.subTest(text=text):
                result = self.server.hover(PTX_URI, "ptx", text, 0, 5)
                self.assertIsNotNone(result)
                value = result["contents"]["value"]
                self.assertIn(name.casefold(), value.casefold())
                self.assertIn(url, value)

    def test_hover_cuda_host_asm_stays_host_by_line(self):
        # A .cu file holds host asm and PTX side by side. The x86 line
        # hovers the x86 row, the dotted PTX line the PTX row: one rule
        # per line, never the document.
        host = 'int f() { __asm__("vaddps %ymm0, %ymm1, %ymm2"); }\n'
        result = self.server.hover(CU_URI, "cuda", host, 0, host.index("vaddps") + 2)
        self.assertIsNotNone(result)
        value = result["contents"]["value"]
        self.assertIn("vaddps", value.casefold())
        self.assertNotIn("#integer-arithmetic-instructions-add", value)
        ptx = 'int f() { __asm__("add.s32 %r1, %r2, %r3"); }\n'
        result = self.server.hover(CU_URI, "cuda", ptx, 0, ptx.index("add.s32") + 2)
        self.assertIsNotNone(result)
        self.assertIn("#integer-arithmetic-instructions-add", result["contents"]["value"])


if __name__ == "__main__":
    unittest.main()
