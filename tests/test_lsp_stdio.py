"""End-to-end LSP tests over stdio: inlay hints, inline asm, missing catalog."""

import json
import os
import select
import sqlite3
import subprocess
import sys
import time
import unittest
from pathlib import Path

from simdref.lsp import _mnemonic_from_asm_line
from simdref.storage import SQLITE_PATH

# The dev-install data dir has no catalog, so fall back to the user-level one.
# SIMDREF_CATALOG needs "isa update" output at that path, not a missing file:
# else a fresh CI skips every catalog test and proves nothing.
_USER_CATALOG = Path.home() / ".local" / "share" / "simdref" / "catalog.db"
_catalog_env = os.environ.get("SIMDREF_CATALOG", "")
_missing = Path(os.environ.get("TMPDIR", "/tmp")) / "simdref-test-no-catalog.db"
CATALOG = (
    Path(_catalog_env)
    if _catalog_env and Path(_catalog_env).is_file()
    else (_missing if _catalog_env else (SQLITE_PATH if SQLITE_PATH.is_file() else _USER_CATALOG))
)

ASM_TEXT = """# top comment
.text
.globl foo
foo:
    vfmadd231ps ymm0, ymm1, ymm2
    vaddps ymm0, ymm1, ymm2  # trailing comment
label2:
    ret
"""

CPP_TEXT = """#include <immintrin.h>
void f() {
    __asm__ volatile ("vfmadd231ps %ymm0, %ymm1, %ymm2\\n"
                      "vaddps %ymm0, %ymm1, %ymm2");
    int x = 0;  // vaddps in a comment stays silent
    asm volatile("\\tvaddps %1, %2, %0\\n\\tvmulps %1, %2, %0" : "=x"(x) : "x"(x), "x"(x));
    myasm("vaddps %xmm0, %xmm1, %xmm2");
    // asm("vaddps %xmm0, %xmm1, %xmm2");
}
"""

CPP_URI = "file:///tmp/simdref_lsp_test.cpp"
ASM_URI = "file:///tmp/simdref_lsp_test.s"
MISSING_CATALOG = "/tmp/simdref-nonexistent-catalog.db"


def _write_msg(proc, payload):
    body = json.dumps(payload).encode("utf-8")
    proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    proc.stdin.write(body)


def _read_msg(proc, timeout_s=10):
    # proc.stdout is unbuffered (bufsize=0), so select() sees every byte.
    if not select.select([proc.stdout], [], [], timeout_s)[0]:
        return None
    headers = {}
    while True:
        line = proc.stdout.readline().strip()
        if not line:
            break
        key, value = line.decode("ascii").split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers["content-length"])
    body = b""
    while len(body) < length:
        chunk = proc.stdout.read(length - len(body))
        if not chunk:
            return None
        body += chunk
    return json.loads(body)


class _Server:
    def __init__(self, catalog):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "simdref.lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env={**os.environ, "SIMDREF_CATALOG": str(catalog)},
        )
        self.next_id = 0
        self.notifications = []

    def request(self, method, params):
        self.next_id += 1
        _write_msg(
            self.proc, {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params}
        )
        while True:
            msg = _read_msg(self.proc)
            assert msg is not None, f"no response to {method}"
            if msg.get("id") == self.next_id:
                return msg
            self.notifications.append(msg)

    def notify(self, method, params):
        _write_msg(self.proc, {"jsonrpc": "2.0", "method": method, "params": params})

    def start(self):
        resp = self.request("initialize", {"processId": None, "capabilities": {}})
        self.notify("initialized", {})
        return resp["result"]["capabilities"]

    def open(self, uri, language_id, text):
        self.notify(
            "textDocument/didOpen",
            {"textDocument": {"uri": uri, "languageId": language_id, "version": 1, "text": text}},
        )

    def hints(self, uri, start=0, end=100):
        resp = self.request(
            "textDocument/inlayHint",
            {
                "textDocument": {"uri": uri},
                "range": {
                    "start": {"line": start, "character": 0},
                    "end": {"line": end, "character": 0},
                },
            },
        )
        return resp["result"]

    def hover(self, uri, line, character):
        params = {"textDocument": {"uri": uri}, "position": {"line": line, "character": character}}
        return self.request("textDocument/hover", params)["result"]

    def stop(self):
        try:
            self.request("shutdown", {})
            self.notify("exit", {})
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=2)


@unittest.skipUnless(CATALOG.is_file(), "needs isa update catalog")
class LspEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.server = _Server(CATALOG)
        self.addCleanup(self.server.stop)
        caps = self.server.start()
        self.assertTrue(caps.get("inlayHintProvider"), "inlayHintProvider not advertised")

    def test_asm_hints_one_per_instruction_line(self):
        self.server.open(ASM_URI, "asm", ASM_TEXT)
        hints = {h["position"]["line"]: h for h in self.server.hints(ASM_URI)}
        # comment, .text, .globl, "foo:" and "label2:" get nothing
        self.assertEqual(sorted(hints), [4, 5, 7])
        for line, hint in hints.items():
            self.assertTrue(hint["paddingLeft"])
            self.assertEqual(hint["position"]["character"], len(ASM_TEXT.split("\n")[line]))
        # vfmadd231ps has a 74-character summary, so it is cut. vaddps has 57, so it is not.
        self.assertEqual(len(hints[4]["label"]), 60)
        self.assertTrue(hints[4]["label"].endswith("..."))
        self.assertFalse(hints[5]["label"].endswith("..."))

    def test_range_limits_hints(self):
        self.server.open(ASM_URI, "asm", ASM_TEXT)
        self.assertEqual([h["position"]["line"] for h in self.server.hints(ASM_URI, 4, 4)], [4])

    def test_inlay_hint_answers_2000_lines_within_budget(self):
        # A few distinct mnemonics repeated over 2000 lines. The hint lookup must
        # ask the catalog once per distinct mnemonic, not once per line.
        text = "\n".join(
            f"    {'vaddps' if i % 2 else 'vfmadd231ps'} ymm0, ymm1, ymm2" for i in range(2000)
        )
        self.server.open(ASM_URI, "asm", text)
        started = time.monotonic()
        hints = self.server.hints(ASM_URI, 0, 2000)
        elapsed = time.monotonic() - started
        self.assertEqual(len(hints), 2000)
        self.assertLess(elapsed, 1.0, f"inlayHint on 2000 lines took {elapsed:.2f}s")

    def test_hint_label_cut_boundary(self):
        from simdref import lsp

        self.assertTrue(hasattr(lsp, "_cut_label"), "lsp has no shared label-cut helper")
        for length in (59, 60, 61):
            with self.subTest(length=length):
                label = lsp._cut_label("x" * length)
                self.assertEqual(len(label), length if length <= 60 else 60)
                self.assertEqual(label.endswith("..."), length > 60)

    def test_hover_bare_mnemonic(self):
        self.server.open(ASM_URI, "asm", ASM_TEXT)
        result = self.server.hover(ASM_URI, 4, 6)
        self.assertIsNotNone(result, "hover on vfmadd231ps returned null")
        self.assertIn("VFMADD231PS", result["contents"]["value"])

    def test_cpp_two_instructions_on_one_source_line_get_one_joined_hint(self):
        text = 'asm volatile("vaddps %ymm1, %ymm2, %ymm0\\n\\tvmulps %ymm0, %ymm0, %ymm3");\n'
        self.server.open(CPP_URI, "cpp", text)
        hints = self.server.hints(CPP_URI)
        self.assertEqual([h["position"]["line"] for h in hints], [0])
        self.assertIn("; ", hints[0]["label"])
        self.assertLessEqual(len(hints[0]["label"]), 60)

    def test_cpp_hints_only_inside_asm_strings(self):
        self.server.open(CPP_URI, "cpp", CPP_TEXT)
        # lines 2-3: two literals. line 5: extended asm, "\t" and "\n" splits give one hint per line.
        # Line 6 (myasm), line 7 (comment) and line 4 (// comment) stay silent.
        self.assertEqual([h["position"]["line"] for h in self.server.hints(CPP_URI)], [2, 3, 5])

    def test_cpp_hover_only_inside_asm_strings(self):
        self.server.open(CPP_URI, "cpp", CPP_TEXT)
        self.assertIsNotNone(self.server.hover(CPP_URI, 3, 28))  # vaddps in the literal
        self.assertIsNone(self.server.hover(CPP_URI, 4, 20))  # vaddps in a // comment
        self.assertIsNone(self.server.hover(CPP_URI, 1, 6))  # "f" in "void f()"

    def test_cpp_extension_without_language_id(self):
        self.server.open("file:///tmp/x.hxx", "", "int x;\n")
        self.assertEqual(self.server.hints("file:///tmp/x.hxx"), [])  # "int" is an x86 mnemonic

    def test_asm_semicolon_comment_in_dot_asm_file(self):
        uri = "file:///tmp/simdref_lsp_test.asm"
        self.server.open(uri, "nasm", "vaddps ymm0, ymm1, ymm2 ; add, then, more\n")
        hints = self.server.hints(uri)
        self.assertEqual([h["position"]["line"] for h in hints], [0])
        # Same hint as the comment-free line: the ";" text must not count as operands.
        self.assertEqual(hints[0]["label"], "Add Packed Single Precision Floating-Point Values.")

    def test_gas_semicolon_stays_a_statement_separator_in_dot_s_file(self):
        uri = "file:///tmp/simdref_lsp_test.s"
        self.server.open(uri, "nasm", "vaddps ymm0, ymm1, ymm2 ; vmulps ymm0, ymm0, ymm1\n")
        hints = self.server.hints(uri)
        # The ";" part is not a comment in a .s file, so its commas count: reads 5
        # operands. ORDER BY architecture,key picks an unmasked 3-operand form first.
        self.assertEqual(
            [h["label"] for h in hints],
            ["Add Packed Single Precision Floating-Point Values."],
        )
        # "adc eax, ebx ; j, k" parses as 4 operands in a .s file even when the client
        # sends languageId "nasm": the 4-operand form label differs from the 2-operand one.
        self.server.open(uri, "nasm", "adc eax, ebx ; j, k\n")
        self.assertEqual(self.server.hints(uri)[0]["label"], "Adc instruction.")

    def test_asm_semicolon_comment_in_dot_nasm_file(self):
        uri = "file:///tmp/simdref_lsp_test.nasm"
        self.server.open(uri, "asm", "adc eax, ebx ; j, k\n")
        hints = self.server.hints(uri)
        self.assertEqual([h["position"]["line"] for h in hints], [0])
        # With ";" a comment the line is the 2-operand form, not the 4-operand parse.
        self.assertEqual(hints[0]["label"], "Add With Carry.")

    def test_asm_semicolon_comment_from_language_id_masm(self):
        uri = "file:///tmp/simdref_lsp_test.txt"
        self.server.open(uri, "masm", "adc eax, ebx ; j, k\n")
        hints = self.server.hints(uri)
        self.assertEqual([h["position"]["line"] for h in hints], [0])
        self.assertEqual(hints[0]["label"], "Add With Carry.")

    def test_hint_avx512_mask_and_broadcast_operands(self):
        uri = "file:///tmp/simdref_lsp_test_zmm.s"
        good = "vaddps zmm0{k1}{z}, zmm1, zmm2\n"
        broadcast = "vaddps zmm0, zmm1, dword ptr [rbx]{1to16}\n"
        self.server.open(uri, "nasm", good + broadcast)
        hints = {h["position"]["line"]: h["label"] for h in self.server.hints(uri)}
        # Both lines read 3 operands and get the plain 3-operand vaddps hint. Without
        # the {} fix the "{1to16}" stays silent and the broadcast line reads 4 operands.
        self.assertIn("Add Packed Single Precision Floating-Point Values", hints[0])
        self.assertEqual(hints[0], hints[1], "a {...} group must not change the operand count")

    def test_hint_zero_operand_mnemonic_gets_the_zero_operand_summary(self):
        # MOVSD_XMM matches in no operand count for a bare "movsd" line. The hint must
        # not fall back to the XMM form with a different summary.
        uri = "file:///tmp/simdref_lsp_test_movsd.s"
        self.server.open(uri, "asm", "movsd\n")
        hints = self.server.hints(uri)
        self.assertEqual([h["position"]["line"] for h in hints], [0])
        self.assertEqual(hints[0]["label"], "Move Data From String to String.")


class MissingCatalogTests(unittest.TestCase):
    def test_missing_catalog_sends_show_message_once(self):
        server = _Server(MISSING_CATALOG)
        self.addCleanup(server.stop)
        server.start()
        server.open(ASM_URI, "asm", ASM_TEXT)
        self.assertEqual(server.hints(ASM_URI), [])
        self.assertIsNone(server.hover(ASM_URI, 4, 6))
        shown = [n for n in server.notifications if n.get("method") == "window/showMessage"]
        self.assertEqual(len(shown), 1)
        self.assertEqual(shown[0]["params"]["type"], 2)
        self.assertEqual(
            shown[0]["params"]["message"],
            "simdref catalog not found. Run: isa update, then restart the editor.",
        )

    def test_corrupt_catalog_does_not_crash(self):
        bad = Path(os.environ.get("TMPDIR", "/tmp")) / "simdref-corrupt-catalog.db"
        bad.write_bytes(b"not a database")
        self.addCleanup(bad.unlink)
        server = _Server(bad)
        self.addCleanup(server.stop)
        server.start()
        server.open(ASM_URI, "asm", ASM_TEXT)
        self.assertEqual(server.hints(ASM_URI), [])

    def test_stale_schema_catalog_does_not_crash(self):
        stale = Path(os.environ.get("TMPDIR", "/tmp")) / "simdref-stale-catalog.db"
        stale.unlink(missing_ok=True)
        conn = sqlite3.connect(stale)
        conn.execute("CREATE TABLE instructions_data (mnemonic TEXT, payload BLOB)")
        conn.execute("INSERT INTO instructions_data VALUES ('VADDPS', x'00')")
        conn.commit()
        conn.close()
        self.addCleanup(stale.unlink)
        server = _Server(stale)
        self.addCleanup(server.stop)
        server.start()
        server.open(ASM_URI, "asm", ASM_TEXT)
        self.assertEqual(server.hints(ASM_URI), [])
        self.assertEqual([n["method"] for n in server.notifications], ["window/showMessage"])


class AsmLineTests(unittest.TestCase):
    def test_label_plus_instruction_and_comment_commas(self):
        self.assertEqual(
            _mnemonic_from_asm_line("foo: vaddps ymm0, ymm1, ymm2 # a, b, c"), ("vaddps", 3)
        )

    def test_att_memory_operand_counts_once(self):
        self.assertEqual(
            _mnemonic_from_asm_line("vaddps (%rax,%rbx,4), %xmm1, %xmm2"), ("vaddps", 3)
        )

    def test_avx512_mask_and_broadcast_count_once(self):
        self.assertEqual(_mnemonic_from_asm_line("vaddps zmm0{k1}{z}, zmm1, zmm2"), ("vaddps", 3))
        self.assertEqual(
            _mnemonic_from_asm_line("vaddps zmm0, zmm1, dword ptr [rax]{1to16}"),
            ("vaddps", 3),
        )

    def test_semicolon_comment_only_when_flagged(self):
        self.assertEqual(
            _mnemonic_from_asm_line("vaddps ymm0, ymm1, ymm2 ; add, then, more", True),
            ("vaddps", 3),
        )
        # Without the flag ";" stays, as in GAS x86 statement separation.
        self.assertNotEqual(
            _mnemonic_from_asm_line("vaddps ymm0, ymm1, ymm2 ; add, then, more"),
            ("vaddps", 3),
        )


if __name__ == "__main__":
    unittest.main()
