#!/usr/bin/env python3
"""Generate vendored CUDA intrinsics source bundle from LLVM/clang at llvmorg-23.1.4.

Fetches Apache-2.0 licensed headers and .td files from GitHub, parses the four
clang CUDA headers to extract ~582 distinct API names with signatures and headers,
classifies them, and resolves PTX instruction links BY PARSING LLVM TABLEGEN
.td FILES (BuiltinsNVPTX.td, IntrinsicsNVVM.td, NVPTXIntrinsics.td, NVPTXInstrInfo.td).

NO NAME PATTERNS ARE USED. This is the key difference from commit 4149c02.

Classification:
- full_chain: chains through __nvvm_* -> BuiltinsNVPTX.td -> int_nvvm_*
  (IntrinsicsNVVM.td, aliases from comment block) -> PTX asm string in
  NVPTXIntrinsics.td/NVPTXInstrInfo.td
- libdevice: calls __nv_* (bitcode functions, the ~420-name group)
- atomic: lowered to atom.* templates via atomicrmw (no named int_nvvm_*)
- plain: plain C/C++ body

No NVIDIA material is vendored. All source files are Apache-2.0 WITH LLVM-exception.

Usage: python tools/vendor_cuda_sources.py [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "vendor" / "cuda"
LLVM_ORG_REF = "llvmorg-23.1.4"
RAW_BASE = f"https://raw.githubusercontent.com/llvm/llvm-project/{LLVM_ORG_REF}"

CLANG_HEADER_PATHS = [
    "clang/lib/Headers/__clang_cuda_intrinsics.h",
    "clang/lib/Headers/__clang_cuda_device_functions.h",
    "clang/lib/Headers/__clang_cuda_math.h",
    "clang/lib/Headers/__clang_cuda_builtin_vars.h",
]

TD_PATHS = {
    "BuiltinsNVPTX.td": "clang/include/clang/Basic/BuiltinsNVPTX.td",
    "IntrinsicsNVVM.td": "llvm/include/llvm/IR/IntrinsicsNVVM.td",
    "NVPTXIntrinsics.td": "llvm/lib/Target/NVPTX/NVPTXIntrinsics.td",
    "NVPTXInstrInfo.td": "llvm/lib/Target/NVPTX/NVPTXInstrInfo.td",
}

MATH_DOC_URL = "https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__FUNC.html"


def _fetch(url: str) -> str:
    try:
        import httpx

        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text
    except ImportError:
        import urllib.request

        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read().decode("utf-8", "replace")


# ----------------------------------------------------------------
# Structural parsing helpers (no regex) - from llvm-src/analyze.py
# ----------------------------------------------------------------


def logical_lines(text: str) -> list[str]:
    out, buf = [], ""
    for ln in text.split("\n"):
        s = ln.rstrip()
        if buf:
            buf += " " + s.strip()
        else:
            buf = s
        if s.endswith("\\") and not s.endswith("\\\\"):
            buf = buf[:-1]
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


def strip_comments(text: str) -> str:
    out = []
    in_block = False
    for ln in text.split("\n"):
        r = ""
        i = 0
        while i < len(ln):
            if in_block:
                j = ln.find("*/", i)
                if j < 0:
                    break
                in_block = False
                i = j + 2
            else:
                j2 = ln.find("//", i)
                j3 = ln.find("/*", i)
                if j2 >= 0 and (j3 < 0 or j2 < j3):
                    r += ln[i:j2]
                    break
                if j3 >= 0:
                    r += ln[i:j3]
                    in_block = True
                    i = j3 + 2
                else:
                    r += ln[i:]
                    break
        out.append(r)
    return "\n".join(out)


def find_body(text: str, brace_pos: int) -> str | None:
    depth = 0
    for i in range(brace_pos, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
            if depth == 1:
                start = i
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _skip_paren_group(text: str, start: int) -> int:
    """Return the index just past the balanced ``( ... )`` group at *start*."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def drop_attribute_groups(text: str) -> str:
    """Remove every ``__attribute__(( ... ))`` group from the text."""
    out = []
    i = 0
    while True:
        a = text.find("__attribute__(", i)
        if a < 0:
            return "".join(out) + text[i:]
        out.append(text[i:a])
        i = _skip_paren_group(text, a + len("__attribute__"))


def name_of(fn_head: str) -> tuple[str, str]:
    s = fn_head.strip()
    i = len(s)
    while i > 0 and (s[i - 1].isalnum() or s[i - 1] == "_"):
        i -= 1
    name = s[i:]
    rest = s[:i].strip()
    return name, rest


# ----------------------------------------------------------------
# Parse headers
# ----------------------------------------------------------------


def parse_functions_from_text(text: str, header: str) -> list[dict]:
    text = strip_comments(text)
    lls = logical_lines(text)
    joined = "\n".join(lls)
    prefix_tokens = (
        "__DEVICE__",
        "inline __device__",
        "__inline__ __device__",
        "static __device__",
        "static inline __attribute__((always_inline))",
    )
    functions: list[dict] = []
    i = 0
    while True:
        best = None
        raw = None
        for tok in prefix_tokens:
            j = joined.find(tok + " ", i)
            if j == -1:
                j = joined.find(tok + "{", i)
            if j >= 0 and (raw is None or j < raw):
                raw = j
            if j >= 0 and (best is None or j < best[0]):
                if j == 0 or joined[j - 1] == "\n":
                    best = (j, tok)
        if best is None:
            if raw is None:
                break
            i = raw + 1
            continue
        j, tok = best
        line_start = joined.rfind("\n", 0, j) + 1
        if joined[line_start:j].strip().startswith("#"):
            i = j + len(tok)
            continue
        br = joined.find("{", j + len(tok))
        sc = joined.find(";", j + len(tok))
        if br < 0 or (sc >= 0 and sc < br):
            i = j + len(tok)
            continue
        # Drop every __attribute__(( ... )) group, wherever it sits in the
        # declaration, before the name is taken: an attribute before or
        # inside the declarator otherwise parses as the name "__attribute__".
        head = drop_attribute_groups(joined[j + len(tok) : br])
        p = head.find("(")
        if p < 0:
            i = br + 1
            continue
        name, _ = name_of(head[:p])
        body = find_body(joined, br)
        if body is None:
            i = br + 1
            continue
        decl = joined[j + len(tok) : br].strip()
        if name and name[0] not in "0123456789":
            functions.append(
                {
                    "name": name,
                    "signature": decl,
                    "header": header,
                    "body": body[1:-1].strip(),
                }
            )
        i = br + len(body)
    return functions


def parse_macro_shuffles(text: str, header: str) -> list[dict]:
    lls = logical_lines(text)
    macrogen = {}
    for k, ln in enumerate(lls):
        s = ln.strip()
        if s.startswith("#define __MAKE_SHUFFLES("):
            # Look for continuation lines in the definition
            full_def = s
            j = k + 1
            while j < len(lls) and not lls[j].strip().startswith(
                ("__MAKE", "#pragma pop_macro", "#define")
            ):
                full_def += " " + lls[j].strip()
                j += 1
            macrogen["__MAKE_SHUFFLES"] = full_def
        elif s.startswith("#define __MAKE_SYNC_SHUFFLES("):
            full_def = s
            j = k + 1
            while j < len(lls) and not lls[j].strip().startswith(
                ("__MAKE", "#pragma pop_macro", "#define")
            ):
                full_def += " " + lls[j].strip()
                j += 1
            macrogen["__MAKE_SYNC_SHUFFLES"] = full_def

    invocations: list[dict] = []
    i = 0
    while i < len(lls):
        ln = lls[i]
        s = ln.strip()
        if s.startswith("__MAKE_SHUFFLES(") or s.startswith("__MAKE_SYNC_SHUFFLES("):
            # Merge continuation lines for multi-line macro invocations
            full_line = s
            j = i + 1
            depth = s.count("(") - s.count(")")
            while depth > 0 and j < len(lls):
                next_line = lls[j].strip()
                full_line += " " + next_line
                depth += next_line.count("(") - next_line.count(")")
                j += 1
            i = j - 1

            # Parse the complete invocation
            mn = "__MAKE_SYNC_SHUFFLES" if "SYNC" in s else "__MAKE_SHUFFLES"
            argstr = full_line[full_line.find("(") + 1 :]
            if argstr.rstrip().endswith(");"):
                argstr = argstr[:-2].rstrip()
            depth = 0
            out = ""
            for c in argstr:
                if c == "(":
                    depth += 1
                elif c == ")":
                    if depth == 0:
                        break
                    depth -= 1
                out += c
            args = [a.strip() for a in out.split(",") if a.strip()]
            body = macrogen.get(mn, "")
            ints = [a for a in args if a.startswith("__nvvm_")]
            invocations.append(
                {
                    "name": args[0] if args else "",
                    "header": header,
                    "macro": mn,
                    "n_overloads": len(ints),
                    "intrinsics": ints,
                }
            )
        i += 1
    return invocations


def parse_builtin_vars(text: str) -> list[dict]:
    llv = logical_lines(text)
    accessors: list[dict] = []
    for ln in llv:
        s = ln.strip()
        if s.startswith("__CUDA_DEVICE_BUILTIN("):
            argstr = s[s.find("(") + 1 :]
            argstr = argstr[: argstr.rfind(")")]
            args = [a.strip() for a in argstr.split(",")]
            if len(args) == 2:
                accessors.append({"field": args[0], "call": args[1]})
    return accessors


# ----------------------------------------------------------------
# Parse .td files
# ----------------------------------------------------------------


def parse_builtins_nvptx(text: str) -> set[str]:
    """Return set of __nvvm_* names defined in BuiltinsNVPTX.td."""
    out = set()
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith("def ") and " : NVPTXBuiltin" in s:
            nm_start = s.find("def ") + 4
            nm_end = s.find(" :", nm_start)
            if nm_end > nm_start:
                out.add(s[nm_start:nm_end].strip())
    return out


def _def_statements(text: str) -> list[str]:
    """Join def statements. A statement can span several physical lines."""
    out: list[str] = []
    cur = ""
    in_str = False
    for ln in strip_comments(text).split("\n"):
        s = ln.strip()
        if not cur:
            if not s.startswith("def "):
                continue
        else:
            cur += " "
        for c in s:
            cur += c
            if c == '"':
                in_str = not in_str
        if not in_str and s.endswith(";"):
            out.append(cur)
            cur = ""
    return out


def _split_top_commas(s: str) -> list[str]:
    """Split on commas outside quotes and brackets."""
    out: list[str] = []
    cur = ""
    depth = 0
    in_str = False
    for c in s:
        if c == '"':
            in_str = not in_str
        if not in_str:
            if c in "<([{":
                depth += 1
            elif c in ">)]}":
                depth -= 1
            elif c == "," and depth == 0:
                out.append(cur)
                cur = ""
                continue
        cur += c
    out.append(cur)
    return [a.strip() for a in out]


def builtin_feature_floors(text: str) -> dict[str, tuple[str, str]]:
    """Map builtin name to (min_sm, min_ptx). Only defs with a floor.

    A def looks like ``def __nvvm_foo : NVPTXBuiltinSM<"int()", SM_70>;``.
    The first template arg is the prototype and has no floor. The other
    args carry tokens like ``SM_70`` or ``PTX78``. PTX78 means PTX 7.8.
    """
    out: dict[str, tuple[str, str]] = {}
    for stmt in _def_statements(text):
        head, sep, tail = stmt.partition(" : ")
        if not sep or "NVPTXBuiltin" not in tail:
            continue
        name = head[4:].strip()
        lt = tail.find("<")
        gt = tail.rfind(">")
        if lt < 0 or gt <= lt:
            continue
        min_sm = ""
        min_ptx = ""
        for tok in _split_top_commas(tail[lt + 1 : gt])[1:]:
            if tok.startswith("SM_"):
                min_sm = tok.lower()
            elif tok.startswith("PTX") and tok[3:].isdigit():
                n = int(tok[3:])
                min_ptx = f"{n // 10}.{n % 10}"
        if min_sm or min_ptx:
            out[name] = (min_sm, min_ptx)
    return out


def parse_alias_block(text: str) -> dict[str, tuple[str, str]]:
    """Parse comment block: //   * llvm.nvvm.X --> target"""
    out = {}
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith("//   * llvm.nvvm.") and "-->" in s:
            left_part = s.split("//   * ", 1)[1].split("-->", 1)[0].strip()
            nm = left_part.replace("llvm.nvvm.", "").strip()
            right = s.split("-->", 1)[1].strip().split("(")[0].strip()
            if right.startswith("llvm.nvvm."):
                out[nm] = ("nvvm", right[10:])
            elif right.startswith("llvm."):
                out[nm] = ("corellvm", right[5:])
            else:
                out[nm] = ("plainir", right)
    return out


def tokens_with_prefix(text: str, prefix: str) -> set[str]:
    """Find all tokens starting with prefix."""
    out = set()
    i = 0
    while True:
        j = text.find(prefix, i)
        if j < 0:
            break
        k = j + len(prefix)
        nm = prefix
        while k < len(text) and (text[k].isalnum() or text[k] == "_"):
            nm += text[k]
            k += 1
        out.add(nm)
        i = k
    return out


def _looks_like_ptx_opcode(s: str) -> bool:
    """Check if a string looks like a PTX opcode: xxx.yyy.zzz format."""
    if not s or s.startswith(".") or s.endswith("."):
        return False
    if "." not in s:
        return False
    first = s.split(".")[0]
    if not first or not all(c.isalnum() or c == "_" for c in first):
        return False
    if any(c in s for c in " \t\n$%,;{}[]<>"):
        return False
    return True


# ----------------------------------------------------------------
# Structural TableGen expansion (multiclass/defm/foreach by structure)
#
# Expands def/defm/multiclass/foreach/let/defvar statements and class
# template bodies so each generated record contributes its own
# evaluated values. PTX asm strings and int_nvvm_* references are read
# from those record values, never from name-keyed tables.
# ----------------------------------------------------------------


def td_tokens(text: str) -> list[tuple[str, str]]:
    """Tokenize TableGen source. Comments drop; strings keep their text."""
    toks: list[tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == '"':
            j = i + 1
            out = ""
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    out += text[j + 1]
                    j += 2
                    continue
                out += text[j]
                j += 1
            toks.append(("str", out))
            i = j + 1
            continue
        if c.isdigit():
            j = i
            while j < n and (text[j].isdigit() or text[j] == "."):
                j += 1
            toks.append(("int", text[i:j]))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            toks.append(("id", text[i:j]))
            i = j
            continue
        toks.append(("op", c))
        i += 1
    return toks


class _TDParser:
    """Statement and expression parser over TableGen tokens."""

    def __init__(self, toks: list[tuple[str, str]]):
        self.toks = toks
        self.i = 0

    def peek(self, offset: int = 0) -> tuple[str, str]:
        j = self.i + offset
        return self.toks[j] if j < len(self.toks) else ("eof", "")

    def at_op(self, ch: str) -> bool:
        return self.peek() == ("op", ch)

    def eat_op(self, ch: str) -> bool:
        if self.at_op(ch):
            self.i += 1
            return True
        return False

    def expect_op(self, ch: str) -> None:
        self.eat_op(ch)

    # ---- expressions ----

    def parse_expr(self):
        node = self.parse_atom()
        while self.at_op("#"):
            self.i += 1
            node = ("cat", node, self.parse_atom())
        return node

    def parse_atom(self):
        kind, val = self.peek()
        if kind == "str":
            self.i += 1
            return ("str", val)
        if kind == "int":
            self.i += 1
            return ("int", val)
        if kind == "id":
            self.i += 1
            if val in ("true", "false"):
                return ("bool", val == "true")
            node = ("ref", val)
        elif kind == "op" and val == "!":
            self.i += 1
            nxt = self.next()
            fn = nxt[1] if nxt[0] == "id" else ""
            if self.at_op("<"):
                self._skip_angles()
            self.expect_op("(")
            args = self.parse_call_args()
            self.expect_op(")")
            node = ("call", fn, args)
        elif kind == "op" and val == "(":
            self.i += 1
            head = self.parse_expr()
            args: list = []
            names: list = []
            while not self.at_op(")") and self.peek()[0] != "eof":
                arg = self.parse_expr()
                name = None
                if self.at_op(":"):
                    self.i += 1
                    self.eat_op("$")
                    nxt = self.next()
                    name = nxt[1] if nxt[0] == "id" else None
                args.append(arg)
                names.append(name)
            self.expect_op(")")
            node = ("dag", head, args, names)
        elif kind == "op" and val == "[":
            self.i += 1
            items = []
            while not self.at_op("]") and self.peek()[0] != "eof":
                items.append(self.parse_expr())
                self.eat_op(",")
            self.expect_op("]")
            node = ("list", items)
        elif kind == "op" and val == "{":
            body = self.parse_stmts(True)
            node = ("block", body)
        else:
            self.i += 1
            return ("bad", val)
        while self.at_op(".") and self.peek(1)[0] == "id":
            self.i += 1
            field = self.next()[1]
            node = ("field", node, field)
        return node

    def next(self) -> tuple[str, str]:
        tok = self.peek()
        self.i += 1
        return tok

    def _skip_angles(self) -> None:
        depth = 0
        while self.peek()[0] != "eof":
            tok = self.next()
            if tok == ("op", "<"):
                depth += 1
            elif tok == ("op", ">"):
                depth -= 1
                if depth == 0:
                    return

    def parse_call_args(self) -> list:
        args: list = []
        while not self.at_op(")") and self.peek()[0] != "eof":
            expr = self.parse_expr()
            if self.at_op(":"):
                self.i += 1
                expr = ("pair", expr, self.parse_expr())
            args.append(expr)
            self.eat_op(",")
        return args

    def parse_args(self) -> list[list[tuple[str, str]]]:
        """Split the inside of <...> into per-argument token lists."""
        groups: list[list[tuple[str, str]]] = []
        cur: list[tuple[str, str]] = []
        angles = 1
        rounds = 0
        squares = 0
        while self.peek()[0] != "eof":
            tok = self.next()
            if tok[0] == "op":
                ch = tok[1]
                if ch == "<":
                    angles += 1
                elif ch == ">":
                    angles -= 1
                    if angles == 0:
                        break
                elif ch in "([{":
                    if ch == "(":
                        rounds += 1
                    elif ch == "[":
                        squares += 1
                    else:
                        angles += 1
                elif ch in ")]}":
                    if ch == ")":
                        rounds -= 1
                    elif ch == "]":
                        squares -= 1
                    else:
                        angles -= 1
                elif ch == "," and angles == 1 and rounds == 0 and squares == 0:
                    groups.append(cur)
                    cur = []
                    continue
            cur.append(tok)
        if cur:
            groups.append(cur)
        return groups

    def parse_params(self) -> list[tuple[str, object]]:
        if not self.eat_op("<"):
            return []
        params: list[tuple[str, object]] = []
        for group in self.parse_args():
            params.append(self._param_from(group))
        return params

    def _param_from(self, toks: list[tuple[str, str]]) -> tuple[str, object]:
        eq = next((idx for idx, tok in enumerate(toks) if tok == ("op", "=")), None)
        decl_end = eq if eq is not None else len(toks)
        name = None
        for tok in reversed(toks[:decl_end]):
            if tok[0] == "id":
                name = tok[1]
                break
        default = None
        if eq is not None and toks[eq + 1 :]:
            default = _TDParser(toks[eq + 1 :]).parse_expr()
        return (name, default)

    def parse_tmpl_list(self) -> list[tuple[str, list]]:
        items: list[tuple[str, list]] = []
        while self.peek()[0] != "eof":
            if self.peek()[0] != "id":
                break
            name = self.next()[1]
            args: list = []
            if self.at_op("<"):
                self.i += 1
                args = [self._arg_expr(group) for group in self.parse_args()]
            items.append((name, args))
            if not self.eat_op(","):
                break
        return items

    def _arg_expr(self, toks: list[tuple[str, str]]):
        eq = next((idx for idx, tok in enumerate(toks) if tok == ("op", "=")), None)
        if eq is not None and eq > 0 and toks[eq - 1][0] == "id":
            return ("named", toks[eq - 1][1], _TDParser(toks[eq + 1 :]).parse_expr())
        return _TDParser(toks).parse_expr()

    # ---- statements ----

    def parse_stmts(self, in_block: bool) -> list:
        stmts: list = []
        while self.peek()[0] != "eof":
            if in_block and self.at_op("}"):
                self.i += 1
                break
            stmt = self.parse_stmt()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    def parse_stmt(self):
        kind, val = self.peek()
        if kind == "op" and val == ";":
            self.i += 1
            return None
        if kind == "op" and val == "{":
            self.i += 1
            return ("block", self.parse_stmts(True))
        if kind == "id" and val in ("class", "multiclass"):
            self.i += 1
            nxt = self.next()
            name = nxt[1] if nxt[0] == "id" else None
            params = self.parse_params()
            parents: list[tuple[str, list]] = []
            if self.eat_op(":"):
                parents = self.parse_tmpl_list()
            body = []
            if self.at_op("{"):
                self.i += 1
                body = self.parse_stmts(True)
            self.eat_op(";")
            return ("classdef", val, name, params, parents, body)
        if kind == "id" and val == "def":
            self.i += 1
            name_expr = None
            if not self.at_op(":"):
                name_expr = self.parse_expr()
            tmpls: list[tuple[str, list]] = []
            if self.eat_op(":"):
                tmpls = self.parse_tmpl_list()
            body = []
            if self.at_op("{"):
                self.i += 1
                body = self.parse_stmts(True)
            self.eat_op(";")
            return ("def", name_expr, tmpls, body)
        if kind == "id" and val == "defm":
            self.i += 1
            name_expr = self.parse_expr()
            self.expect_op(":")
            tmpls = self.parse_tmpl_list()
            self.eat_op(";")
            return ("defm", name_expr, tmpls)
        if kind == "id" and val in ("defvar", "defset"):
            self.i += 1
            nxt = self.next()
            name = nxt[1] if nxt[0] == "id" else None
            self.expect_op("=")
            expr = self.parse_expr()
            self.eat_op(";")
            return ("defvar", name, expr)
        if kind == "id" and val == "foreach":
            self.i += 1
            nxt = self.next()
            var = nxt[1] if nxt[0] == "id" else None
            self.expect_op("=")
            list_expr = self.parse_expr()
            if self.peek() == ("id", "in"):
                self.i += 1
            stmt = self.parse_stmt()
            return ("foreach", var, list_expr, stmt)
        if kind == "id" and val == "let":
            self.i += 1
            assigns = []
            while True:
                nxt = self.next()
                fname = nxt[1] if nxt[0] == "id" else None
                self.expect_op("=")
                assigns.append((fname, self.parse_expr()))
                if not self.eat_op(","):
                    break
            if self.peek() == ("id", "in"):
                self.i += 1
                return ("let", assigns, self.parse_stmt())
            self.eat_op(";")
            return ("fieldlet", assigns)
        # Field assignment (with optional type prefix) or an unrecognized
        # statement: find the statement end and classify by a top-level '='.
        start = self.i
        depth = 0
        eq_at = None
        end = None
        while self.peek()[0] != "eof":
            tok = self.next()
            if tok[0] == "op" and tok[1] in "(<[{":
                depth += 1
            elif tok[0] == "op" and tok[1] in ")>]}":
                depth -= 1
                if depth < 0:
                    end = self.i - 1
                    break
            elif depth == 0 and tok == ("op", "=") and eq_at is None:
                eq_at = self.i - 1
            elif depth == 0 and tok == ("op", ";"):
                end = self.i - 1
                break
        if eq_at is not None and end is not None and end > eq_at:
            name = None
            for tok in reversed(self.toks[start:eq_at]):
                if tok[0] == "id":
                    name = tok[1]
                    break
            expr = _TDParser(self.toks[eq_at + 1 : end]).parse_expr()
            return ("fieldlet", [(name, expr)])
        return None


def _td_parse(text: str) -> list:
    return _TDParser(td_tokens(text)).parse_stmts(False)


def _eval(node, env, classes, defs, rec):
    kind = node[0]
    if kind == "str":
        return node[1]
    if kind == "int":
        return node[1]
    if kind == "bool":
        return node[1]
    if kind == "ref":
        name = node[1]
        if name in env:
            return env[name]
        return ("rec", name)
    if kind == "field":
        base = _eval(node[1], env, classes, defs, rec)
        target = None
        if isinstance(base, tuple) and base and base[0] == "rec":
            target = defs.get(base[1])
        if target and node[2] in target:
            return target[node[2]]
        return None
    if kind == "cat":
        out = ""
        for part in _cat_parts(node, env, classes, defs, rec):
            if part is None:
                return None
            out += part
        return out
    if kind == "call":
        return _eval_call(node, env, classes, defs, rec)
    if kind == "dag":
        head = _eval(node[1], env, classes, defs, rec)
        args = [_eval(a, env, classes, defs, rec) for a in node[2]]
        return ("dag", head, args, node[3])
    if kind == "list":
        return [_eval(item, env, classes, defs, rec) for item in node[1]]
    return None


def _cat_parts(node, env, classes, defs, rec) -> list:
    if node[0] == "cat":
        return _cat_parts(node[1], env, classes, defs, rec) + _cat_parts(
            node[2], env, classes, defs, rec
        )
    value = _eval(node, env, classes, defs, rec)
    return [_value_str(value)]


def _value_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, tuple) and value and value[0] == "rec":
        return value[1]
    return str(value)


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _eval_call(node, env, classes, defs, rec):
    fn = node[1]
    args = node[2]

    def ev(idx):
        if idx >= len(args):
            return None
        return _eval(args[idx], env, classes, defs, rec)

    if fn == "if":
        if len(args) < 3:
            return None
        return ev(1) if _truthy(ev(0)) else ev(2)
    if fn == "cast":
        return ev(0)
    if fn == "eq":
        return len(args) >= 2 and ev(0) == ev(1)
    if fn == "ne":
        return len(args) >= 2 and ev(0) != ev(1)
    if fn == "not":
        return not _truthy(ev(0))
    if fn == "and":
        return all(_truthy(_eval(a, env, classes, defs, rec)) for a in args)
    if fn == "or":
        return any(_truthy(_eval(a, env, classes, defs, rec)) for a in args)
    if fn == "cond":
        for arg in args:
            if arg[0] == "pair":
                if _truthy(_eval(arg[1], env, classes, defs, rec)):
                    return _eval(arg[2], env, classes, defs, rec)
        return None
    if fn == "strconcat":
        out = ""
        for arg in args:
            piece = _value_str(_eval(arg, env, classes, defs, rec))
            if piece is None:
                return None
            out += piece
        return out
    if fn == "size":
        value = ev(0)
        return len(value) if value is not None else None
    if fn == "substr":
        value = _value_str(ev(0))
        start = _as_int(_value_str(ev(1)))
        if value is None or start is None:
            return None
        if len(args) > 2:
            stop = _as_int(_value_str(ev(2)))
            return value[start:] if stop is None else value[start:stop]
        return value[start:]
    if fn == "con":
        return ("bag", [_eval(a, env, classes, defs, rec) for a in args])
    if fn == "dag":
        return ("bag", [_eval(a, env, classes, defs, rec) for a in args])
    if fn == "listconcat":
        out: list = []
        for arg in args:
            value = _eval(arg, env, classes, defs, rec)
            if isinstance(value, list):
                out.extend(value)
            elif value is not None:
                out.append(value)
        return out
    if fn == "listsplat":
        value = ev(0)
        count = _as_int(_value_str(ev(1)))
        return [value] * count if count is not None else None
    if fn == "foreach":
        # !foreach(x, list, expr) maps expr over the list.
        if len(args) < 3:
            return None
        items = _eval(args[1], env, classes, defs, rec)
        if not isinstance(items, list):
            return None
        var = args[0][1] if args[0][0] == "ref" else None
        out = []
        for item in items:
            env2 = dict(env)
            if var:
                env2[var] = item
            out.append(_eval(args[2], env2, classes, defs, rec))
        return out
    if fn in ("add", "sub", "mul", "div", "shl", "srl", "sra"):
        a = _as_int(_value_str(ev(0)))
        b = _as_int(_value_str(ev(1)))
        if a is None or b is None:
            return None
        return str(
            {
                "add": a + b,
                "sub": a - b,
                "mul": a * b,
                "div": a // b if b else 0,
                "shl": a << b,
                "srl": (a % (1 << 32)) >> b,
                "sra": a >> b,
            }[fn]
        )
    if fn in ("le", "lt", "gt", "ge"):
        a = _as_int(_value_str(ev(0)))
        b = _as_int(_value_str(ev(1)))
        if a is None or b is None:
            return False
        return {"le": a <= b, "lt": a < b, "gt": a > b, "ge": a >= b}[fn]
    return None


def _truthy(value) -> bool:
    return value not in (None, "", 0, "0", "false", False)


def _exec_stmts(stmts, env, classes, defs, records, rec, prefix):
    for stmt in stmts:
        _exec_stmt(stmt, env, classes, defs, records, rec, prefix)


def _exec_stmt(stmt, env, classes, defs, records, rec, prefix):
    kind = stmt[0]
    if kind == "block":
        _exec_stmts(stmt[1], env, classes, defs, records, rec, prefix)
        return
    if kind == "classdef":
        classes[stmt[2]] = (stmt[1], stmt[3], stmt[4], stmt[5])
        return
    if kind == "defvar":
        env[stmt[1]] = _eval(stmt[2], env, classes, defs, rec)
        return
    if kind in ("fieldlet", "field"):
        assigns = stmt[1] if kind == "fieldlet" else [(stmt[1], stmt[2])]
        for name, expr in assigns:
            value = _eval(expr, env, classes, defs, rec)
            if rec is not None and name:
                rec["fields"][name] = value
                rec["values"].append(value)
        return
    if kind == "let":
        env2 = dict(env)
        for name, expr in stmt[1]:
            env2[name] = _eval(expr, env, classes, defs, rec)
        if stmt[2] is not None:
            _exec_stmt(stmt[2], env2, classes, defs, records, rec, prefix)
        return
    if kind == "foreach":
        values = _eval(stmt[2], env, classes, defs, rec)
        if not isinstance(values, list):
            values = [values]
        for value in values:
            env2 = dict(env)
            if stmt[1]:
                env2[stmt[1]] = value
            if stmt[3] is not None:
                _exec_stmt(stmt[3], env2, classes, defs, records, rec, prefix)
        return
    if kind == "def":
        name = prefix + _name_of(_eval(stmt[1], env, classes, defs, rec) if stmt[1] else None)
        record = {"name": name, "values": [], "fields": {}}
        env2 = dict(env)
        env2["NAME"] = name
        for clsname, argnodes in stmt[2]:
            _instantiate(clsname, argnodes, env2, classes, defs, records, record, "")
        _exec_stmts(stmt[3], env2, classes, defs, records, record, "")
        records.append(record)
        if name:
            defs[name] = record["fields"]
        return
    if kind == "defm":
        name = _name_of(_eval(stmt[1], env, classes, defs, rec))
        env2 = dict(env)
        env2["NAME"] = name
        for clsname, argnodes in stmt[2]:
            _instantiate(clsname, argnodes, env2, classes, defs, records, rec, name)
        return


def _name_of(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, tuple) and value and value[0] == "rec":
        return value[1]
    return str(value)


def _instantiate(clsname, argnodes, env, classes, defs, records, rec, prefix):
    argvals = [_eval(a, env, classes, defs, rec) for a in argnodes]
    if rec is not None:
        rec["values"].extend(argvals)
    entry = classes.get(clsname)
    if entry is None:
        return
    _, params, parents, body = entry
    env2 = dict(env)
    pos = 0
    for argnode, argval in zip(argnodes, argvals):
        if argnode[0] == "named":
            env2[argnode[1]] = argval
            if rec is not None:
                rec["fields"][argnode[1]] = argval
            continue
        if pos < len(params):
            pname, default = params[pos]
            if pname:
                env2[pname] = argval
                if rec is not None:
                    rec["fields"][pname] = argval
            pos += 1
    for pname, default in params[pos:]:
        if pname and pname not in env2:
            value = _eval(default, env2, classes, defs, rec) if default is not None else None
            env2[pname] = value
            if rec is not None:
                rec["fields"][pname] = value
    for parent, argnodes2 in parents:
        _instantiate(parent, argnodes2, env2, classes, defs, records, rec, prefix)
    _exec_stmts(body, env2, classes, defs, records, rec, prefix)


def _asm_opcode(text: str) -> str | None:
    """The PTX opcode at the head of one record asm string.

    TableGen-free template artifacts drop: ${...} substitutions and
    $operand references. The opcode is the leading dotted word.
    """
    out = ""
    i = 0
    while i < len(text):
        if text.startswith("${", i):
            j = text.find("}", i)
            i = len(text) if j < 0 else j + 1
            continue
        c = text[i]
        if c in "$ \t\n\\;,{}[]<>()":
            break
        out += c
        i += 1
    out = out.rstrip(".")
    return out if _looks_like_ptx_opcode(out) else None


def td_record_links(text: str) -> tuple[dict[str, set[str]], int]:
    """Expand one .td document set into (int_nvvm_* -> PTX opcodes, records)."""
    records: list[dict] = []
    classes: dict = {}
    defs: dict = {}
    _exec_stmts(_td_parse(text), {}, classes, defs, records, None, "")
    table: dict[str, set[str]] = {}
    for record in records:
        strings: list[str] = []
        names: list[str] = []
        for value in record["values"]:
            _collect(value, strings, names)
        intrinsics = {n for n in names if n.startswith("int_nvvm_")} | {
            s for s in strings if s.startswith("int_nvvm_")
        }
        asm = {op for s in strings if (op := _asm_opcode(s))}
        for name in intrinsics:
            table.setdefault(name, set()).update(asm)
    return table, len(records)


def _collect(value, strings: list, names: list) -> None:
    if value is None:
        return
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, list):
        for item in value:
            _collect(item, strings, names)
    elif isinstance(value, tuple) and value:
        if value[0] == "rec":
            names.append(value[1])
        elif value[0] == "dag":
            _collect(value[1], strings, names)
            for item in value[2]:
                _collect(item, strings, names)
        elif value[0] == "bag":
            for item in value[1]:
                _collect(item, strings, names)


def build_builtin_to_ptx(
    builtins_text: str,
    intrinsics_text: str,
    instr_info_text: str,
    nvptx_intr_text: str,
) -> dict[str, list[str]]:
    """Map __nvvm_* builtins to PTX opcodes through the .td record chains.

    Chain: the BuiltinsNVPTX.td def names the builtin; the NVVMBuiltin rule
    in IntrinsicsNVVM.td binds it to intrinsic ``int_nvvm_<suffix>``; the
    expanded NVPTX records that reference that intrinsic carry the asm
    string. The IntrinsicsNVVM.td auto-upgrade comment table renames
    retired intrinsic names. A builtin with no such record chain stays
    unlinked and is counted by ``classify``.
    """
    builtins = parse_builtins_nvptx(builtins_text)
    aliases = parse_alias_block(intrinsics_text)

    table, record_count = td_record_links(instr_info_text + "\n" + nvptx_intr_text)
    print(f"  Expanded PTX records: {record_count}, intrinsic links: {len(table)}")

    mapping: dict[str, list[str]] = {}
    for bn in sorted(builtins):
        if not bn.startswith("__nvvm_"):
            continue
        # NVVMBuiltin (IntrinsicsNVVM.td) binds builtin "__nvvm_X" to
        # intrinsic "int_nvvm_X".
        target = "int_nvvm_" + bn[len("__nvvm_") :]
        seen: set[str] = set()
        while target and target not in seen:
            seen.add(target)
            ops = table.get(target)
            if ops:
                mapping[bn] = sorted(ops)
                break
            alias = aliases.get(target[len("int_nvvm_") :].replace("_", "."))
            if alias and alias[0] == "nvvm":
                target = "int_nvvm_" + alias[1].replace(".", "_")
            else:
                break
    return mapping


# ----------------------------------------------------------------
# Classification
# ----------------------------------------------------------------


def calls_in(body: str) -> tuple[set[str], set[str]]:
    nvvm, nv = set(), set()
    for prefix, target_set in [("__nvvm_", nvvm), ("__nv_", nv)]:
        i = 0
        while True:
            j = body.find(prefix, i)
            if j < 0:
                break
            k = j + len(prefix)
            nm = prefix
            while k < len(body) and (body[k].isalnum() or body[k] == "_"):
                nm += body[k]
                k += 1
            target_set.add(nm)
            i = k
    return nvvm, nv


def classify(
    name: str, body: str, builtin_to_ptx: dict[str, list[str]], intr: dict | None = None
) -> tuple[str, list[str]]:
    """Classify intrinsic and return (class, ptx_ops)."""
    intr = intr or {}
    calls = tokens_with_prefix(body, "__nvvm_")
    calls.update(intr.get("_nvvm_intrinsics") or [])
    # Atomics lower straight to atomicrmw: clang names no int_nvvm_*
    # intrinsic for the __nvvm_atom_*_gen_* builtins, so no record chain
    # exists in LLVM and the class carries no PTX claim.
    if any(bn.startswith("__nvvm_atom") for bn in calls):
        return ("atomic", [])
    if tokens_with_prefix(body, "__nv_"):
        return ("libdevice", [])
    if calls:
        ptx_ops: list[str] = []
        resolved = 0
        for bn in sorted(calls):
            ops = builtin_to_ptx.get(bn)
            if ops:
                resolved += 1
                ptx_ops.extend(ops)
        if not ptx_ops:
            return ("unchainable_nvvm", [])
        if resolved == len(calls):
            return ("full_chain", sorted(set(ptx_ops)))
        return ("partial_chain", sorted(set(ptx_ops)))
    if "__asm__" in body.lower() or "asm(" in body.lower():
        return ("asm", [])
    return ("plain", [])


# ----------------------------------------------------------------
# Per-row enrichment: calls, arch guards, feature floors
# ----------------------------------------------------------------


def identifier_tokens(text: str) -> set[str]:
    """Split text on every char outside [A-Za-z0-9_]."""
    out: set[str] = set()
    cur = ""
    for c in text + " ":
        if c.isalnum() or c == "_":
            cur += c
        elif cur:
            out.add(cur)
            cur = ""
    return out


def extract_calls(body: str, function_names: set[str], self_name: str) -> list[str]:
    """Names the body uses: sibling functions and __nvvm_/__nv_ builtins."""
    calls = {
        tok
        for tok in identifier_tokens(body)
        if tok != self_name
        and (tok in function_names or tok.startswith("__nvvm_") or tok.startswith("__nv_"))
    }
    return sorted(calls)


def _arch_guard_of(directive: str) -> str:
    """NNN of ``__CUDA_ARCH__ >= NNN`` or ``__CUDA_ARCH__ > NNN`` in one directive."""
    toks = directive.split()
    for i, tok in enumerate(toks):
        if tok != "__CUDA_ARCH__" or i + 2 >= len(toks):
            continue
        if toks[i + 1] not in (">=", ">"):
            continue
        num = ""
        for c in toks[i + 2]:
            if not c.isdigit():
                break
            num += c
        return num
    return ""


def arch_guard_lines(text: str) -> list[str]:
    """Innermost open __CUDA_ARCH__ guard per logical line. '' when none."""
    stack: list[str] = []
    out: list[str] = []
    for ln in logical_lines(strip_comments(text)):
        s = ln.strip()
        if s.startswith("#if"):
            # shortcut: an #elif/#else branch keeps the guard. No function
            # definition sits in such a branch today. Upgrade when one does.
            stack.append(_arch_guard_of(s))
        elif s.startswith("#endif"):
            if stack:
                stack.pop()
        out.append(next((g for g in reversed(stack) if g), ""))
    return out


def arch_guard_for_name(texts: list[str], name: str) -> str:
    """Guard at the first ``name(`` use in the texts. '' when none.

    For a plain function this is its definition line. A name can also be
    used first as a call in a guarded block. That call site then carries
    the guard of the row.
    """
    for text in texts:
        lls = logical_lines(strip_comments(text))
        guards = arch_guard_lines(text)
        for i, ln in enumerate(lls):
            j = ln.find(name)
            while j >= 0:
                before = ln[j - 1] if j > 0 else " "
                after = ln[j + len(name) :].lstrip()
                if not (before.isalnum() or before == "_") and after.startswith("("):
                    return guards[i]
                j = ln.find(name, j + 1)
    return ""


def _sm_num(value: str) -> int:
    digits = ""
    for c in value:
        if c.isdigit():
            digits += c
    return int(digits) if digits else 0


def _ptx_num(value: str) -> tuple[int, ...]:
    return tuple(int(p) for p in value.split("."))


def min_floors(
    calls: list[str],
    calls_by_name: dict[str, list[str]],
    builtin_floors: dict[str, tuple[str, str]],
    arch_guard: str,
) -> tuple[str, str]:
    """(min_sm, min_ptx) from the builtins reachable over the calls graph.

    The sm floor is the highest one over all reachable __nvvm_ builtins.
    When no reachable builtin has an sm floor, the __CUDA_ARCH__ guard is
    the floor. The ptx floor is the highest reachable one.
    """
    seen_funcs: set[str] = set()
    sm_vals: list[str] = []
    ptx_vals: list[str] = []
    stack = list(calls)
    while stack:
        tok = stack.pop()
        if tok in calls_by_name:
            if tok not in seen_funcs:
                seen_funcs.add(tok)
                stack.extend(calls_by_name[tok])
            continue
        if tok.startswith("__nvvm_"):
            floor = builtin_floors.get(tok)
            if floor:
                if floor[0]:
                    sm_vals.append(floor[0])
                if floor[1]:
                    ptx_vals.append(floor[1])
    if sm_vals:
        min_sm = max(sm_vals, key=_sm_num)
    elif arch_guard:
        min_sm = f"sm_{int(arch_guard) // 10}"
    else:
        min_sm = ""
    min_ptx = max(ptx_vals, key=_ptx_num) if ptx_vals else ""
    return min_sm, min_ptx


DIM_SUFFIXES = ("_x", "_y", "_z", "_w")


def special_register_tokens(builtin_names: set[str]) -> set[str]:
    """PTX special register tokens from the ``__nvvm_read_ptx_sreg_*`` defs."""
    out: set[str] = set()
    for nm in builtin_names:
        if not nm.startswith("__nvvm_read_ptx_sreg_"):
            continue
        tok = nm[len("__nvvm_read_ptx_sreg_") :]
        for suf in DIM_SUFFIXES:
            if tok.endswith(suf):
                tok = tok[: -len(suf)]
                break
        if tok:
            out.add(tok)
    return out


def drop_special_register_refs(
    instructions: list[str], sreg_tokens: set[str]
) -> tuple[list[str], list[str]]:
    """Split refs into (kept, dropped). A special register read is no
    instruction. Its first dot token names the register."""
    kept: list[str] = []
    dropped: list[str] = []
    for op in instructions:
        if op.split(".")[0] in sreg_tokens:
            dropped.append(op)
        else:
            kept.append(op)
    return kept, dropped


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Fetch headers
    print("Fetching clang CUDA headers...")
    all_intr: list[dict] = []
    header_texts: list[str] = []
    function_names: set[str] = set()
    for hpath in CLANG_HEADER_PATHS:
        header_name = Path(hpath).name
        print(f"  {header_name}...")
        try:
            text = _fetch(f"{RAW_BASE}/{hpath}")
        except Exception as exc:
            print(f"    ERROR: {exc}", file=sys.stderr)
            continue
        header_texts.append(text)
        funcs = parse_functions_from_text(text, header_name)
        function_names.update(f["name"] for f in funcs)
        all_intr.extend(funcs)
        print(f"    -> {len(funcs)} definitions")
        if header_name == "__clang_cuda_intrinsics.h":
            macros = parse_macro_shuffles(text, header_name)
            for m in macros:
                # Store the nvvm intrinsics list as a proper list in the dict
                # so classify can find them
                all_intr.append(
                    {
                        "name": m["name"],
                        "signature": f"{m['name']}(...)  // {m['n_overloads']} overloads",
                        "header": header_name,
                        "body": " ".join(m["intrinsics"]),  # Space-separated list of __nvvm_* names
                        "_nvvm_intrinsics": m["intrinsics"],  # Also store as list
                        "_is_macro": True,
                    }
                )
            print(f"    -> {len(macros)} macro invocations")
        if header_name == "__clang_cuda_builtin_vars.h":
            accessors = parse_builtin_vars(text)
            for a in accessors:
                all_intr.append(
                    {
                        "name": a["field"],
                        "signature": f"{a['field']}  // builtin var",
                        "header": header_name,
                        "body": f"accessor: {a['call']}",
                        "_is_builtin_var": True,
                    }
                )
            print(f"    -> {len(accessors)} builtin vars")

    # Calls per row: macros use their intrinsic lists, builtin vars none
    for intr in all_intr:
        if intr.get("_is_macro"):
            intr["calls"] = sorted(intr.get("_nvvm_intrinsics") or [])
        elif intr.get("_is_builtin_var"):
            intr["calls"] = []
        else:
            intr["calls"] = extract_calls(intr.get("body", ""), function_names, intr["name"])

    # Fetch .td files
    print("\nFetching .td files...")
    td_texts = {}
    for local, upstream in TD_PATHS.items():
        print(f"  {local}...")
        try:
            td_texts[local] = _fetch(f"{RAW_BASE}/{upstream}")
        except Exception as exc:
            print(f"    ERROR: {exc}", file=sys.stderr)
            td_texts[local] = ""

    # Parse .td files and build mapping
    print("\nParsing .td files to build builtin->PTX mapping...")
    builtin_to_ptx = build_builtin_to_ptx(
        td_texts["BuiltinsNVPTX.td"],
        td_texts["IntrinsicsNVVM.td"],
        td_texts["NVPTXInstrInfo.td"],
        td_texts["NVPTXIntrinsics.td"],
    )
    print(f"  Builtin->PTX mappings: {len(builtin_to_ptx)}")

    # Feature floors and special register tokens from BuiltinsNVPTX.td
    builtin_floors = builtin_feature_floors(td_texts["BuiltinsNVPTX.td"])
    sreg_tokens = special_register_tokens(parse_builtins_nvptx(td_texts["BuiltinsNVPTX.td"]))
    print(f"  Builtin feature floors: {len(builtin_floors)}")
    print(f"  Special register tokens: {len(sreg_tokens)}")

    # Classify
    print("\nClassifying intrinsics...")
    seen: dict[str, dict] = {}
    counts = defaultdict(int)
    ptx_linked = 0

    for intr in all_intr:
        name = intr["name"]
        body = intr.get("body", "")

        cls, ptx_ops = classify(name, body, builtin_to_ptx, intr)
        counts[cls] += 1
        if ptx_ops:
            ptx_linked += 1

        if intr["header"] == "__clang_cuda_math.h":
            doc_url = f"{MATH_DOC_URL}#{name}"
        else:
            doc_url = f"https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#{name}"

        rec = {
            "name": name,
            "signature": intr["signature"],
            "description": f"CUDA device intrinsic from {intr['header']}.",
            "header": intr["header"],
            "url": doc_url,
            "architecture": "cuda",
            "isa": ["CUDA"],
            "category": "",
            "subcategory": "",
            "instructions": ptx_ops,
            "instruction_refs": [{"name": op, "form": "", "architecture": "ptx"} for op in ptx_ops],
            "calls": intr.get("calls", []),
            "arch_guard": "",
            "min_sm": "",
            "min_ptx": "",
            "metadata": {"classification": cls},
            "source": "clang-cuda-headers",
            "doc_sections": {},
            "notes": [],
            "aliases": [],
        }

        if name in seen:
            existing = seen[name]
            if rec["signature"] not in existing["aliases"]:
                existing["aliases"].append(rec["signature"])
            for op in rec["instructions"]:
                if op not in existing["instructions"]:
                    existing["instructions"].append(op)
                    existing["instruction_refs"].append(
                        {"name": op, "form": "", "architecture": "ptx"}
                    )
            for call in rec["calls"]:
                if call not in existing["calls"]:
                    existing["calls"].append(call)
        else:
            seen[name] = rec

    unique = list(seen.values())

    # Guards, floors and special register refs per row
    calls_by_name = {r["name"]: r["calls"] for r in unique}
    dropped_refs: set[str] = set()
    dropped_rows = 0
    guarded_rows = 0
    floored_rows = 0
    for r in unique:
        r["calls"] = sorted(r["calls"])
        r["arch_guard"] = arch_guard_for_name(header_texts, r["name"])
        r["min_sm"], r["min_ptx"] = min_floors(
            r["calls"], calls_by_name, builtin_floors, r["arch_guard"]
        )
        # Floor provenance for the sm coverage counts.
        td_sm, td_ptx = min_floors(r["calls"], calls_by_name, builtin_floors, "")
        r["min_sm_from"] = "nvvm-td" if td_sm else ("cuda-arch-guard" if r["min_sm"] else "")
        r["min_ptx_from"] = "nvvm-td" if td_ptx else ""
        # The design doc stores the two floors in row metadata.
        r["metadata"]["min_sm"] = r["min_sm"]
        r["metadata"]["min_ptx"] = r["min_ptx"]
        kept, dropped = drop_special_register_refs(r["instructions"], sreg_tokens)
        r["instructions"] = kept
        r["instruction_refs"] = [{"name": op, "form": "", "architecture": "ptx"} for op in kept]
        if dropped:
            dropped_rows += 1
            dropped_refs.update(dropped)
        if r["arch_guard"]:
            guarded_rows += 1
        if r["min_sm"]:
            floored_rows += 1

    # Build bundle
    bundle = {
        "format": "clang-cuda-headers-v2",
        "upstream": {
            "repo": "https://github.com/llvm/llvm-project",
            "ref": LLVM_ORG_REF,
            "license": "Apache-2.0 WITH LLVM-exception",
        },
        "intrinsics": unique,
        "metadata": {
            "source_file_count": len(CLANG_HEADER_PATHS),
            "td_file_count": len(TD_PATHS),
            "ptx_linked_count": ptx_linked,
            "classification": dict(counts),
            "total_intrinsics": len(unique),
            "builtin_floors": len(builtin_floors),
            "guarded_rows": guarded_rows,
            "floored_rows": floored_rows,
            "dropped_special_register_refs": sorted(dropped_refs),
            "dropped_special_register_ref_rows": dropped_rows,
        },
    }

    outpath = args.out_dir / "cuda_intrinsics.json"
    outpath.write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"\nWrote {outpath}")
    print(f"Total intrinsics: {len(unique)}")
    print(f"Classifications: {dict(counts)}")
    print(f"PTX-linked: {ptx_linked}")

    # Doc pages, fetched here and parsed at catalog build time. The
    # bundle is gitignored data, like the intrinsics bundle. The catalog
    # build fetches the pages itself when this file is missing.
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from simdref.cuda_docs import build_docs

        docs = build_docs(_fetch, {r["name"] for r in unique})
        docs_path = args.out_dir / "cuda_docs.json"
        docs_path.write_text(json.dumps(docs))
        print(f"Wrote {docs_path}")
        print(f"Doc pages: {docs['metadata']}")
    except Exception as exc:
        print(f"Doc page fetch failed: {exc}", file=sys.stderr)
    print(f"Builtin feature floors: {len(builtin_floors)}")
    print(f"Rows with arch guard: {guarded_rows}")
    print(f"Rows with min_sm: {floored_rows}")
    print(f"Rows with min_ptx: {sum(1 for r in unique if r['min_ptx'])}")
    kept_refs = sum(len(r["instructions"]) for r in unique)
    print(
        f"Refs kept: {kept_refs}, dropped special-register refs: "
        f"{len(dropped_refs)} in {dropped_rows} rows"
    )


if __name__ == "__main__":
    main()
