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

from simdref import lsp
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
    def __init__(self, catalog, extra_env=None):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "simdref.lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env={**os.environ, "SIMDREF_CATALOG": str(catalog), **(extra_env or {})},
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

    def test_hover_matches_hint_on_second_instruction_of_asm_literal(self):
        # After the "\n\t" escape split, the raw line word is "tvmulps" but the
        # hover must name the same YMM form the inlay hint gives.
        text = 'asm("vaddps %ymm2, %ymm1, %ymm0\\n\\tvmulps %ymm3, %ymm0, %ymm0");\n'
        self.server.open(CPP_URI, "cpp", text)
        hints = self.server.hints(CPP_URI)
        self.assertEqual(len(hints), 1)
        self.assertEqual(
            hints[0]["label"],
            "Add Packed Single Precision Floating-Point Values.; Multi...",
        )
        col = text.index("vmulps")
        result = self.server.hover(CPP_URI, 0, col)
        self.assertIsNotNone(result, "hover on vmulps after the escape split returned null")
        self.assertIn("VMULPS (YMM, YMM, YMM)", result["contents"]["value"])

    def test_hover_after_astral_characters_in_comment(self):
        # LSP columns count UTF-16 code units: each emoji before "lfence" moves
        # the column by 2, so the hover must convert before indexing the line.
        prefix = "".join("\U0001f600" for _ in range(10))
        text = f"# {prefix} lfence\n"
        self.server.open(ASM_URI, "asm", text)
        col_lfence = len(("# " + prefix + " ").encode("utf-16-le")) // 2
        result = self.server.hover(ASM_URI, 0, col_lfence)
        self.assertIsNotNone(result, "hover on lfence after emoji returned null")
        self.assertIn("LFENCE", result["contents"]["value"])

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

    def test_inlay_hint_after_astral_characters_gets_utf16_column(self):
        # LSP columns count UTF-16 code units: each emoji adds 2 units to the
        # end-of-line column the hint returns.
        prefix = "".join("\U0001f600" for _ in range(10))
        text = f"# {prefix}\n    vaddps ymm0, ymm1, ymm2 # {prefix}\n"
        self.server.open(ASM_URI, "asm", text)
        hints = {h["position"]["line"]: h for h in self.server.hints(ASM_URI)}
        self.assertEqual(sorted(hints), [1])
        line = text.split("\n")[1]
        # The expected column comes from the codec, not from _utf16_column.
        # A wrong helper would otherwise agree with itself.
        utf16_length = len(line.encode("utf-16-le")) // 2
        self.assertNotEqual(utf16_length, len(line), "the line needs astral characters")
        self.assertEqual(hints[1]["position"]["character"], utf16_length)
        self.assertEqual(hints[1]["label"], "Add Packed Single Precision Floating-Point Values.")

    def test_utf16_column_is_the_reverse_of_code_point_index(self):
        line = "a \U0001f600 b"
        for index in range(len(line) + 1):
            with self.subTest(index=index):
                self.assertEqual(lsp._code_point_index(line, lsp._utf16_column(line, index)), index)

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

    def test_hover_instruction_full_page_operand_matched(self):
        self.server.open(ASM_URI, "asm", ASM_TEXT)
        result = self.server.hover(ASM_URI, 5, 6)
        self.assertIsNotNone(result, "hover on vaddps returned null")
        value = result["contents"]["value"]
        # The 3-operand YMM form, not the first masked (XMM, K, ...) row.
        self.assertIn("VADDPS (YMM, YMM, YMM)", value)
        self.assertNotIn("VADDPS (XMM, K, XMM, M128)", value)
        self.assertIn("## Description", value)
        self.assertIn("Add Packed Single Precision Floating-Point Values", value)
        self.assertIn("## SIMD Floating-Point Exceptions", value)
        self.assertIn("## instruction to intrinsic mapping", value)
        self.assertIn("_mm256_add_ps", value)
        # The per-uarch perf table as a Markdown table.
        self.assertIn("| microarch |", value)
        self.assertIn("| ZEN5 |", value)
        # Links at the end: web deep link, uops.info, felixcloutier.
        self.assertIn("https://simdref.diamondinoia.com/#VADDPS%20(YMM%2C%20YMM%2C%20YMM)", value)
        self.assertIn("https://www.uops.info/html-instr/VADDPS_YMM_YMM_YMM.html", value)
        self.assertIn("https://www.felixcloutier.com/x86/addps", value)

    def test_movsd_hover_matches_the_sse_form(self):
        # A bare "movsd" row is the string move, so the exact hit would show
        # the string form. A two-operand line must get the SSE form instead.
        # AT&T source order is memory->xmm, and the key lists destination first.
        self.server.open(ASM_URI, "asm", "movsd 0x10(%rsp), %xmm0\n")
        result = self.server.hover(ASM_URI, 0, 2)
        self.assertIsNotNone(result, "hover on movsd with operands returned null")
        value = result["contents"]["value"]
        self.assertIn("MOVSD_XMM (XMM, M64)", value)
        self.assertNotIn("MOVSD_XMM (M64, XMM)", value)
        self.assertNotIn("Move Data From String to String", value)

    def test_bare_string_movsd_keeps_the_string_form(self):
        # Zero operands: the exact "movsd" row IS the string move, so hover
        # keeps it instead of falling through to a matched form.
        self.server.open(ASM_URI, "asm", "movsd\n")
        result = self.server.hover(ASM_URI, 0, 2)
        self.assertIsNotNone(result, "hover on bare movsd returned null")
        self.assertIn("Move Data From String to String", result["contents"]["value"])
        self.assertNotIn("MOVSD_XMM (M64, XMM)", result["contents"]["value"])

    def test_hover_makes_no_network_call(self):
        # The server is a subprocess, so block sockets there via sitecustomize.
        import tempfile

        with tempfile.TemporaryDirectory() as sitepkg:
            Path(sitepkg, "sitecustomize.py").write_text(
                "import socket\n"
                "def _blocked(*a, **k): raise OSError('network blocked in test')\n"
                "socket.socket = _blocked\n"
                "socket.create_connection = _blocked\n"
            )
            server = _Server(
                CATALOG,
                {"PYTHONPATH": sitepkg + os.pathsep + os.environ.get("PYTHONPATH", "")},
            )
            self.addCleanup(server.stop)
            server.start()
            server.open(ASM_URI, "asm", ASM_TEXT)
            result = server.hover(ASM_URI, 5, 6)
        self.assertIsNotNone(result)
        self.assertIn("VADDPS (YMM, YMM, YMM)", result["contents"]["value"])

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

    def test_cpp_hover_inside_asm_literal_matches_the_ymm_form(self):
        text = 'asm volatile("vaddps %ymm2, %ymm1, %ymm0");\n'
        self.server.open(CPP_URI, "cpp", text)
        result = self.server.hover(CPP_URI, 0, 16)
        self.assertIsNotNone(result, "hover on vaddps inside the asm literal returned null")
        value = result["contents"]["value"]
        self.assertIn("VADDPS (YMM, YMM, YMM)", value)
        self.assertNotIn("VADDPS (XMM, K, XMM, M128)", value)

    def test_asm_hover_and_hint_agree_on_semicolon_comment(self):
        uri = "file:///tmp/simdref_lsp_test_hover.asm"
        self.server.open(uri, "nasm", "adc eax, ebx ; j, k\n")
        hint = self.server.hints(uri)[0]["label"]
        result = self.server.hover(uri, 0, 1)
        self.assertIsNotNone(result, "hover on adc returned null")
        value = result["contents"]["value"]
        # The hover must pick the same form as the hint on the same line. The
        # 4-operand parse lands on an ARM adc page, the 2-operand parse on the
        # x86 "Add With Carry." form the hint names.
        self.assertIn("Add With Carry.", value)
        self.assertEqual(hint, "Add With Carry.")

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
        # "adc eax, ebx ; j, k" parses as 3 operands in a .s file even when the client
        # sends languageId "nasm": the 3-operand form label differs from the 2-operand one.
        self.server.open(uri, "nasm", "adc eax, ebx ; j, k\n")
        # The 3-operand x86 form that names a 32-bit register, not the first ARM row.
        self.assertEqual(
            self.server.hints(uri)[0]["label"], "Add with carry 32-bit integer operands."
        )

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

    ATT_TEXT = (
        "xorl %eax, %eax\naddq $8, %rax\ncmpq %rsi, %rax\nadd $8, %rax\njmp .L2\n"
        "vpsrlq $1, %xmm0, %xmm1\nmovsb\nshl $1, %rax\nxor %eax, %eax\ncmp %rsi, %rax\n"
        "shl $2, %rax\n"
    )

    def _att_hover_and_hints(self):
        self.server.open(ASM_URI, "asm", self.ATT_TEXT)
        hints = {h["position"]["line"]: h["label"] for h in self.server.hints(ASM_URI)}
        return hints, lambda line: self.server.hover(ASM_URI, line, 1)["contents"]["value"]

    def test_att_size_suffixed_mnemonics_resolve_to_the_base_instruction(self):
        hints, hover = self._att_hover_and_hints()
        for line, base, key, hint_line in ((0, "XOR", "XOR (", 8), (1, "ADD", "ADD (R64", 3)):
            with self.subTest(base=base):
                self.assertIn(key, hover(line))
                self.assertEqual(hints[line], hints[hint_line])
        self.assertIn("CMP (", hover(2))
        self.assertEqual(hints[2], hints[9])
        self.assertEqual(hints[2], "Compare Two Operands.")

    def test_att_suffix_in_cpp_asm_string(self):
        self.server.open(CPP_URI, "cpp", 'asm("xorl %eax, %eax\\n\\taddq $8, %rax");\n')
        self.assertEqual(
            self.server.hints(CPP_URI)[0]["label"],
            "Logical Exclusive OR.; Add register and immediate operands.",
        )

    def test_real_mnemonics_ending_in_size_letters_keep_their_own_entry(self):
        _, hover = self._att_hover_and_hints()
        for line, name in ((5, "VPSRLQ ("), (6, "MOVSB"), (7, "SHL (")):
            with self.subTest(name=name):
                self.assertIn(name, hover(line))

    def test_shl_immediate_count_picks_the_i8_form(self):
        # Line 10 shifts by 2, so the fixed-1 form must not win the tie.
        _, hover = self._att_hover_and_hints()
        self.assertIn("SHL (R64, I8)", hover(10))
        self.assertNotIn("SHL (R64, 1)", hover(10))
        self.assertNotIn("SHL (R64, 0)", hover(10))

    def test_lea_picks_the_64_bit_form(self):
        # Every lea form has one operand, so the fallback picks. It must name
        # the width of %rdi, not the first form LEA_B (R16).
        self.server.open(ASM_URI, "asm", "lea 0x8(%rsp), %rdi\n")
        hover = self.server.hover(ASM_URI, 0, 1)["contents"]["value"]
        key = hover.splitlines()[1]
        self.assertIn("R64", key)
        self.assertNotIn("R16", key)

    def test_vpermilps_immediate_picks_the_i8_form(self):
        # The line names an immediate, so the I8 form must beat the
        # register form the width pre-pass took by name order.
        self.server.open(ASM_URI, "asm", "vpermilps $1, %ymm1, %ymm2\n")
        key = self.server.hover(ASM_URI, 0, 1)["contents"]["value"].splitlines()[1]
        self.assertIn("I8)", key)

    def test_pinsrw_register_picks_the_r32_form(self):
        # %eax is R32; the M16 form must not win on key order.
        self.server.open(ASM_URI, "asm", "pinsrw $1, %eax, %xmm0\n")
        key = self.server.hover(ASM_URI, 0, 1)["contents"]["value"].splitlines()[1]
        self.assertIn("R32", key)
        self.assertNotIn("M16", key)

    def test_percent_sp_is_the_16_bit_register_form(self):
        # "%sp" is the 16-bit x86 stack pointer: the chosen form must not
        # name an R64 or M64 form for it.
        self.server.open(ASM_URI, "asm", "mov %rax, %sp\n")
        hover = self.server.hover(ASM_URI, 0, 1)["contents"]["value"]
        self.assertNotIn("MOV (M64, R64)", hover)
        self.assertNotIn("R64)", hover)

    def test_intel_bare_sp_is_the_16_bit_register(self):
        # In an Intel file a bare "sp" is the 16-bit x86 register: the chosen
        # form must not name an R64 form for it.
        uri = "file:///tmp/simdref_lsp_test_sp.asm"
        self.server.open(uri, "nasm", "mov rax, sp\n")
        hover = self.server.hover(uri, 0, 1)["contents"]["value"]
        self.assertNotIn("MOV (M64, R64)", hover)
        self.assertNotIn("R64)", hover)

    def test_add_with_64_bit_register_is_not_the_8_bit_form(self):
        hints, hover = self._att_hover_and_hints()
        for line in (1, 3):
            self.assertNotEqual(hints[line], "Add 8-bit operands.")
            self.assertNotIn("ADD (AL", hover(line))
            self.assertNotIn("ADD (R8", hover(line))

    def test_jmp_to_a_label_picks_the_memory_form(self):
        # In AT&T a bare ".L2" is a symbol: a memory reference, not a rel8 immediate.
        _, hover = self._att_hover_and_hints()
        self.assertNotIn("Rel8", hover(4))
        self.assertIn("JMP (M64)", hover(4))

    def test_mov_of_a_bare_symbol_picks_a_memory_form(self):
        # "mov symbol, %eax" reads memory: the destination %eax leads the form
        # key and the bare symbol is a memory slot, not rel.
        self.server.open(ASM_URI, "asm", "mov symbol, %eax\n")
        hover = self.server.hover(ASM_URI, 0, 1)["contents"]["value"]
        self.assertIn("MOV (R32, M32)", hover)

    def test_att_operand_order_picks_store_form(self):
        # AT&T writes the destination last: "sub %rax, (%rbx)" stores, so the
        # memory operand leads the form key.
        self.server.open(ASM_URI, "asm", "sub %rax, (%rbx)\n")
        hover = self.server.hover(ASM_URI, 0, 1)["contents"]["value"]
        self.assertIn("SUB (M64, R64)", hover)
        self.assertNotIn("SUB (R64, M64)", hover)

    def test_att_operand_order_picks_load_form(self):
        # "sub (%rbx), %rax" loads: the register operand leads the form key.
        self.server.open(ASM_URI, "asm", "sub (%rbx), %rax\n")
        hover = self.server.hover(ASM_URI, 0, 1)["contents"]["value"]
        self.assertIn("SUB (R64, M64)", hover)
        self.assertNotIn("SUB (M64, R64)", hover)

    def test_jmp_star_register_picks_the_indirect_form(self):
        # "jmp *%rax" is the indirect jump, not the direct rel8 form.
        self.server.open(ASM_URI, "asm", "jmp *%rax\n")
        hover = self.server.hover(ASM_URI, 0, 1)["contents"]["value"]
        self.assertNotIn("Rel8", hover)
        self.assertIn("JMP (R64)", hover)

    def test_jmp_star_symbol_picks_the_indirect_memory_form(self):
        # "jmp *handler" reads the target through memory, not a rel8 immediate.
        self.server.open(ASM_URI, "asm", "jmp *handler\n")
        hover = self.server.hover(ASM_URI, 0, 1)["contents"]["value"]
        self.assertNotIn("Rel8", hover)
        self.assertIn("JMP (M64)", hover)

    def test_arm_scalar_lines_keep_the_arm_form(self):
        # The labels are the ones the base commit 94d7a53 gives. "r0" and a bare "sp"
        # are not x86 registers, and a "#16" immediate excludes x86 forms.
        lines = ("sub sp, sp, #16", "add sp, sp, #16", "add r0, r1, r2", "add r8, r9, #16")
        self.server.open(ASM_URI, "asm", "\n".join(lines) + "\n")
        self.assertEqual(
            [h["label"] for h in self.server.hints(ASM_URI)],
            ["Subtract operands.", "Add operands.", "Add operands.", "Add operands."],
        )

    def test_arm_three_register_add_keeps_the_arm_form(self):
        # x0/x1/x2 are AArch64 general registers, not x86, so the hover and
        # hint must not name an x86 form like ADD (R16, M16, I16).
        self.server.open(ASM_URI, "asm", "add x0, x1, x2\n")
        hover = self.server.hover(ASM_URI, 0, 1)
        self.assertIsNotNone(hover, "add x0, x1, x2 got no hover")
        value = hover["contents"]["value"]
        for x86_token in ("R16", "R64", "M16", "I16"):
            self.assertNotIn(x86_token, value, f"hover named an x86 form: {x86_token}")
        self.assertIn("Add operands.", value)
        hints = self.server.hints(ASM_URI)
        self.assertEqual([h["label"] for h in hints], ["Add operands."])

    def test_riscv_word_ops_do_not_resolve_to_x86(self):
        # "addw" has no catalog row, and its strip would land on the x86 "add".
        cases = [
            (op, "a0, a1, a2")
            for op in (
                "addw",
                "subw",
                "sllw",
                "srlw",
                "sraw",
                "mulw",
                "divw",
                "divuw",
                "remw",
                "remuw",
            )
        ]
        cases += [(op, "a0, a1, 5") for op in ("addiw", "slliw", "srliw", "sraiw")]
        cases += [("negw", "a0, a1"), ("addw", "a0, a1, %pcrel_hi(sym)")]
        for op, operands in cases:
            with self.subTest(op=op, operands=operands):
                self.server.open(ASM_URI, "asm", f"{op} {operands}\n")
                self.assertEqual(self.server.hints(ASM_URI), [])
                self.assertIsNone(self.server.hover(ASM_URI, 0, 1))

    def test_att_x86_word_ops_still_resolve(self):
        # "%ax" and a bare symbol are not RISC-V registers, so the guard must not fire.
        self.server.open(ASM_URI, "asm", "addw %ax, %bx\nmulw %ax\nmulw sym\n")
        add = self.server.hover(ASM_URI, 0, 1)
        self.assertIsNotNone(add, "addw %ax, %bx got no hover")
        self.assertIn("ADD (", add["contents"]["value"])
        mul = self.server.hover(ASM_URI, 1, 1)
        self.assertIsNotNone(mul, "mulw %ax got no hover")
        self.assertIn("MUL (R16)", mul["contents"]["value"])
        mem = self.server.hover(ASM_URI, 2, 1)
        self.assertIsNotNone(mem, "mulw sym got no hover")
        self.assertIn("MUL (", mem["contents"]["value"])

    def test_intel_word_ops_still_resolve(self):
        # Intel syntax writes registers bare: the guard is AT&T-only, so a bare
        # "ax" must not read as RISC-V.
        uri = "file:///tmp/simdref_lsp_test_word.asm"
        self.server.open(uri, "nasm", "add ax, bx\naddw ax, bx\n")
        add = self.server.hover(uri, 0, 1)
        self.assertIsNotNone(add, "add ax, bx in nasm got no hover")
        self.assertIn("ADD (", add["contents"]["value"])
        addw = self.server.hover(uri, 1, 1)
        self.assertIsNotNone(addw, "addw ax, bx in nasm got no hover")
        self.assertIn("ADD (", addw["contents"]["value"])

    def test_movsbl_is_not_the_string_move(self):
        # movsbl is a sign-extend. The suffix strip must not land on the string op movsb.
        self.server.open(ASM_URI, "asm", "movsbl %al, %eax\n")
        self.assertEqual(self.server.hints(ASM_URI), [])
        self.assertIsNone(self.server.hover(ASM_URI, 0, 1))

    def test_vinsertf128_register_line_picks_the_register_form(self):
        # Intel: xmm2 is a v128 register, not M128. The vector classes must rank
        # the register form above the memory form.
        uri = "file:///tmp/simdref_lsp_vinsertf128.asm"
        self.server.open(uri, "nasm", "vinsertf128 ymm0, ymm1, xmm2, 0\n")
        hover = self.server.hover(uri, 0, 1)["contents"]["value"]
        self.assertIn("VINSERTF128 (YMM, YMM, XMM, I8)", hover)
        self.assertNotIn("VINSERTF128 (YMM, YMM, M128, I8)", hover)

    def test_vpmovzxbw_picks_the_ymm_destination(self):
        # AT&T: %ymm0 is a v256 destination, so the (YMM, XMM) form outranks the
        # (XMM, XMM) form by width agreement.
        self.server.open(ASM_URI, "asm", "vpmovzxbw %xmm1, %ymm0\n")
        hover = self.server.hover(ASM_URI, 0, 1)["contents"]["value"]
        self.assertIn("VPMOVZXBW (YMM, XMM)", hover)
        self.assertNotIn("VPMOVZXBW (XMM, XMM)", hover)

    def test_vpsllq_register_count_picks_the_xmm_count_form(self):
        # AT&T: three register operands (xmm, ymm, ymm). The vector classes must
        # pick the (YMM, YMM, XMM) form, not an all-XMM variant.
        self.server.open(ASM_URI, "asm", "vpsllq %xmm1, %ymm2, %ymm3\n")
        hover = self.server.hover(ASM_URI, 0, 1)["contents"]["value"]
        self.assertIn("VPSLLQ (YMM, YMM, XMM)", hover)
        self.assertNotIn("VPSLLQ (XMM, XMM, XMM)", hover)

    def test_cmpsd_xmm_pair_is_not_the_string_form(self):
        # AT&T: two xmm operands. The bare string-op CMPSD key must not shadow the
        # SSE forms on an operand-matched line.
        self.server.open(ASM_URI, "asm", "cmpsd %xmm1, %xmm0\n")
        result = self.server.hover(ASM_URI, 0, 1)
        self.assertIsNotNone(result, "hover on a cmpsd xmm pair returned null")
        value = result["contents"]["value"]
        self.assertIn("CMPSD_XMM", value)
        self.assertNotIn("String", value)

    def test_arm_vector_add_keeps_the_arm_form(self):
        # AArch64: the .4s arrangement is an ARM vector signal. The line must rank
        # the AArch64 vector form, not an x86 ADD form.
        uri = "file:///tmp/simdref_lsp_arm_vec.s"
        self.server.open(uri, "asm", "add v0.4s, v1.4s, v2.4s\n")
        result = self.server.hover(uri, 0, 1)
        self.assertIsNotNone(result, "hover on an arm vector add returned null")
        value = result["contents"]["value"]
        self.assertIn("V0.<T_OPTION>", value)
        self.assertNotIn("ADD (R16", value)

    def test_arm_fp_register_keeps_the_arm_form(self):
        # AArch64: fp is x29, not the stack pointer. The line must rank an
        # X-register (or XZR) ADD form, not an x86 ADD form.
        uri = "file:///tmp/simdref_lsp_arm_fp.s"
        self.server.open(uri, "asm", "add fp, x1, x2\n")
        result = self.server.hover(uri, 0, 1)
        self.assertIsNotNone(result, "hover on an arm fp add returned null")
        value = result["contents"]["value"]
        self.assertIn("XDORXZR", value)
        self.assertNotIn("R16", value)
        self.assertNotIn("R64", value)
        self.assertNotIn("M16", value)

    def test_arm_x_registers_pick_the_x_form(self):
        # AArch64: x0-x30 are 64-bit and w0-w30 are 32-bit. The width classes must
        # separate the X and W forms.
        uri = "file:///tmp/simdref_lsp_arm_xw.s"
        self.server.open(uri, "asm", "add x0, x1, x2\nadd w0, w1, w2\n")
        hover_x = self.server.hover(uri, 0, 1)["contents"]["value"]
        hover_w = self.server.hover(uri, 1, 1)["contents"]["value"]
        self.assertIn("XD", hover_x)
        self.assertNotIn("WD", hover_x)
        self.assertIn("WD", hover_w)
        self.assertNotIn("XD", hover_w)

    def test_arm32_addw_does_not_show_x86(self):
        # ARM32: addw is not in the catalog. The x86-gated suffix strip must not
        # resolve it to an x86 ADD form on a non-x86 line.
        uri = "file:///tmp/simdref_lsp_addw.s"
        self.server.open(uri, "asm", "addw r0, r1, r2\n")
        self.assertIsNone(self.server.hover(uri, 0, 1), "addw on an ARM32 line has no hover")

    def test_att_bare_register_names_are_symbols(self):
        # AT&T: a bare sp and a bare x1 (no %) are symbols. "mov sp, %eax"
        # loads the symbol into %eax, so the Intel-ordered form is
        # (R32, M32), never Moffs.
        uri = "file:///tmp/simdref_lsp_bare_sym.s"
        self.server.open(uri, "asm", "mov sp, %eax\nmov x1, %eax\n")
        hover_sp = self.server.hover(uri, 0, 1)["contents"]["value"]
        self.assertIn("MOV (R32, M32)", hover_sp)
        self.assertNotIn("Moffs", hover_sp)
        hover_x1 = self.server.hover(uri, 1, 1)["contents"]["value"]
        self.assertIn("MOV (R32, M32)", hover_x1)
        self.assertNotIn("Moffs", hover_x1)

    def test_intel_nop_hover_matches_the_hint(self):
        # In an Intel .asm file "nop" must hover the x86 NOP, not AArch64.
        # Hover and hint must name the same entry.
        uri = "file:///tmp/simdref_lsp_nop.asm"
        self.server.open(uri, "nasm", "nop\n")
        hover = self.server.hover(uri, 0, 1)
        self.assertIsNotNone(hover, "hover on nop returned null")
        value = hover["contents"]["value"]
        self.assertNotIn("A64", value)
        self.assertIn("No Operation", value)
        # The inlay hint must agree.
        hint = self.server.hints(uri, 0, 1)
        self.assertTrue(hint, "inlay hint for nop is empty")
        hint_label = hint[0]["label"]
        self.assertNotIn("A64", hint_label)
        self.assertIn("No Operation", hint_label)

    def test_att_bare_16_bit_register_names_are_symbols(self):
        # In AT&T bare si/di/bp are symbols, same rule as bare sp.
        uri = "file:///tmp/simdref_lsp_bare16.s"
        self.server.open(uri, "asm", "mov si, %eax\nmov di, %eax\nmov bp, %eax\n")
        for i in range(3):
            hover = self.server.hover(uri, i, 1)["contents"]["value"]
            self.assertIn("MOV (R32, M32)", hover, msg=f"line {i}: {hover}")
            self.assertNotIn("Moffs", hover, msg=f"line {i}: {hover}")

    def test_riscv_vector_lines_do_not_land_on_x86_or_arm(self):
        # RISC-V vector: these lines must stay on RISC-V forms, not fall onto x86
        # or AArch64 ADD/MOV forms.
        uri = "file:///tmp/simdref_lsp_riscv.s"
        self.server.open(uri, "asm", "vadd.vv v1, v2, v3\nvle32.v v1, (a0)\n")
        hover_vadd = self.server.hover(uri, 0, 1)["contents"]["value"]
        self.assertIn("vadd.vv", hover_vadd)
        self.assertNotIn("ADD (R16", hover_vadd)
        self.assertNotIn("ADD (R32", hover_vadd)
        hover_vle = self.server.hover(uri, 1, 1)["contents"]["value"]
        self.assertIn("vle32.v", hover_vle)
        self.assertNotIn("ADD (R16", hover_vle)
        self.assertNotIn("ADD (R32", hover_vle)

    def test_arm_single_precision_scalar_lines_keep_their_arm_form(self):
        # s0-s31 are AArch64 scalar FP registers. A bare s-register also reads
        # as a RISC-V saved register. That is a weak vote, so the mnemonic's
        # catalog rows arbitrate. An ARM-only mnemonic keeps its ARM form.
        uri = "file:///tmp/simdref_lsp_arm_s.s"
        lines = ("ldr s1, [x11], #4", "fneg s0, s1", "fmov s0, w1", "fmadd s0, s1, s2, s3")
        self.server.open(uri, "asm", "\n".join(lines) + "\n")
        for i, line in enumerate(lines):
            with self.subTest(line=line):
                self.assertIsNotNone(self.server.hover(uri, i, 1), f"{line} got no hover")
        self.assertEqual(len(self.server.hints(uri)), len(lines))

    def test_arm_double_precision_scalar_picks_the_scalar_form(self):
        # d0-d31 are AArch64 64-bit scalar FP registers, not vectors. The
        # numbered catalog placeholders D0 and D1 accept the same class.
        uri = "file:///tmp/simdref_lsp_arm_d.s"
        self.server.open(uri, "asm", "fneg d0, d1\n")
        hover = self.server.hover(uri, 0, 1)
        self.assertIsNotNone(hover, "fneg d0, d1 got no hover")
        value = hover["contents"]["value"]
        self.assertIn("FNEG (D0, D1)", value)
        self.assertNotIn("T_OPTION", value)

    def test_riscv_saved_registers_keep_the_riscv_form(self):
        # The catalog has no SD or ADDI row, so these lines keep their current
        # behaviour: no hover and no hint.
        uri = "file:///tmp/simdref_lsp_riscv_saved.s"
        text = "addi s0, sp, 16\nsd s1, 8(sp)\n"
        self.server.open(uri, "asm", text)
        for line in range(2):
            self.assertIsNone(self.server.hover(uri, line, 1))
        self.assertEqual(self.server.hints(uri), [])


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
            _mnemonic_from_asm_line("foo: vaddps ymm0, ymm1, ymm2 # a, b, c")[:2], ("vaddps", 3)
        )

    def test_att_memory_operand_counts_once(self):
        self.assertEqual(
            _mnemonic_from_asm_line("vaddps (%rax,%rbx,4), %xmm1, %xmm2")[:2], ("vaddps", 3)
        )

    def test_avx512_mask_and_broadcast_count_once(self):
        self.assertEqual(
            _mnemonic_from_asm_line("vaddps zmm0{k1}{z}, zmm1, zmm2")[:2], ("vaddps", 3)
        )
        self.assertEqual(
            _mnemonic_from_asm_line("vaddps zmm0, zmm1, dword ptr [rax]{1to16}")[:2],
            ("vaddps", 3),
        )

    def test_semicolon_comment_only_when_flagged(self):
        self.assertEqual(
            _mnemonic_from_asm_line("vaddps ymm0, ymm1, ymm2 ; add, then, more", True)[:2],
            ("vaddps", 3),
        )
        # Without the flag ";" stays, as in GAS x86 statement separation.
        self.assertNotEqual(
            _mnemonic_from_asm_line("vaddps ymm0, ymm1, ymm2 ; add, then, more"),
            ("vaddps", 3),
        )


if __name__ == "__main__":
    unittest.main()
