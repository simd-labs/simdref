from __future__ import annotations

import functools
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from simdref.perf import best_cpi, best_latency
from simdref.queries import linked_instruction_records
from simdref.search import search_records
from simdref.storage import (
    SQLITE_PATH,
    load_intrinsic_from_db,
    load_instruction_from_db,
    load_instructions_by_mnemonic_from_db,
    open_db,
    search_intrinsic_candidates_from_db,
    search_instruction_candidates_from_db,
    sqlite_schema_is_current,
)


WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
MNEMONIC_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.]*$")
ASM_COMMENT_RE = re.compile(r"#.*$|//.*$|/\*.*?\*/")
# In a .asm file (NASM/MASM) ";" starts a comment. A .s/.S file in GAS syntax
# uses ";" as a statement separator, so it stays.
ASM_SEMICOLON_COMMENT_RE = re.compile(r";.*$")
# Matches a string literal or a comment. The scanner blanks comments and keeps strings.
# ponytail: character literals ('"') and raw strings are not handled.
C_TOKEN_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"|//[^\n]*|/\*.*?\*/', re.DOTALL)
# asm("..."), __asm__("..."), __asm("..."), with optional qualifiers. Adjacent
# string literals concatenate. Extended asm (": outputs : inputs") ends at ':'.
ASM_STRING_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:__asm__|__asm|asm)"
    r"(?:\s+(?:volatile|__volatile__|goto|inline))*\s*\(\s*"
    r'("(?:[^"\\]|\\.)*"(?:\s*"(?:[^"\\]|\\.)*")*)'
    r"(?=\s*[:)])"
)
STRING_PART_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
ASM_SPLIT_RE = re.compile(r"\\n|;")
C_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx", ".c++", ".h", ".hpp", ".hh", ".hxx", ".cu", ".cuh")
CATALOG_MISSING = "simdref catalog not found. Run: isa update, then restart the editor."


@dataclass
class Session:
    documents: dict[str, str]
    languages: dict[str, str] = field(default_factory=dict)


def _jsonrpc_write(payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _jsonrpc_read() -> dict | None:
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode("ascii").split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body)


def _word_at(text: str, line: int, character: int) -> str | None:
    lines = text.splitlines()
    if line >= len(lines):
        return None
    current = lines[line]
    for match in WORD_RE.finditer(current):
        if match.start() <= character <= match.end():
            return match.group(0)
    return None


def _line_prefix(text: str, line: int, character: int) -> str:
    lines = text.splitlines()
    if line >= len(lines):
        return ""
    current = lines[line][:character]
    match = re.search(WORD_RE.pattern + r"$", current)
    return match.group(0) if match else ""


def _hover_markdown(conn, word: str, allow_instruction: bool = True) -> str | None:
    intrinsic = load_intrinsic_from_db(conn, word)
    if intrinsic is not None:
        lines = [f"```c\n{intrinsic.signature}\n```"]
        if intrinsic.description:
            lines.append(intrinsic.description)
        meta = []
        if intrinsic.header:
            meta.append(f"header `{intrinsic.header}`")
        if intrinsic.isa:
            meta.append(f"ISA {', '.join(intrinsic.isa)}")
        if intrinsic.category:
            meta.append(f"category {intrinsic.category}")
        if intrinsic.url:
            meta.append(f"[source]({intrinsic.url})")
        if meta:
            lines.append(" | ".join(meta))
        if intrinsic.instructions:
            lines.append(f"Instructions: {', '.join(intrinsic.instructions[:6])}")
        linked = linked_instruction_records(None, intrinsic, conn=conn)
        if linked:
            latencies = [
                best_latency(item.arch_details)
                for item in linked
                if best_latency(item.arch_details) != "-"
            ]
            throughputs = [
                best_cpi(item.arch_details) for item in linked if best_cpi(item.arch_details) != "-"
            ]
            perf = []
            if latencies:
                perf.append(f"best latency {min(latencies, key=lambda value: float(value))} cycles")
            if throughputs:
                perf.append(f"best cycle/instr {min(throughputs, key=lambda value: float(value))}")
            if perf:
                lines.append("Performance: " + ", ".join(perf))
        return "\n\n".join(lines)

    if not allow_instruction:
        return None
    instruction = load_instruction_from_db(conn, word)
    if instruction is None:
        # Catalog keys look like "VADDPS (XMM, XMM, XMM)", so a bare mnemonic needs this lookup.
        matches = load_instructions_by_mnemonic_from_db(conn, word)
        instruction = matches[0] if matches else None
    if instruction is not None:
        lines = [f"```asm\n{instruction.key}\n```"]
        if instruction.summary:
            lines.append(instruction.summary)
        meta = []
        if instruction.isa:
            meta.append(f"ISA {', '.join(instruction.isa)}")
        if instruction.metadata.get("category"):
            meta.append(f"category {instruction.metadata['category']}")
        if meta:
            lines.append(" | ".join(meta))
        if instruction.linked_intrinsics:
            lines.append(f"Intrinsics: {', '.join(instruction.linked_intrinsics[:6])}")
        perf = []
        lat = best_latency(instruction.arch_details)
        cpi = best_cpi(instruction.arch_details)
        if lat != "-":
            perf.append(f"best latency {lat} cycles")
        if cpi != "-":
            perf.append(f"best cycle/instr {cpi}")
        if perf:
            lines.append("Performance: " + ", ".join(perf))
        return "\n\n".join(lines)
    return None


def _completion_candidates(conn, prefix: str, limit: int = 50) -> list[dict]:
    prefix_folded = prefix.casefold()
    emitted: set[tuple[str, str]] = set()
    items: list[dict] = []
    candidate_limit = max(limit * 3, 100)
    intrinsics = search_intrinsic_candidates_from_db(conn, prefix or "_mm", limit=candidate_limit)
    instructions = search_instruction_candidates_from_db(
        conn, prefix or "_mm", limit=candidate_limit
    )
    for result in search_records(intrinsics, instructions, prefix or "_mm", limit=candidate_limit):
        label = result.title
        if prefix_folded and not label.casefold().startswith(prefix_folded):
            continue
        key = (result.kind, label)
        if key in emitted:
            continue
        emitted.add(key)
        kind = 3 if result.kind == "intrinsic" else 14
        items.append({"label": label, "kind": kind, "detail": result.subtitle, "insertText": label})
        if len(items) >= limit:
            return items
    return items


def _is_c_doc(language_id: str, uri: str) -> bool:
    return language_id in ("c", "cpp") or uri.lower().endswith(C_EXTENSIONS)


def _asm_literals(text: str) -> list[tuple[int, str]]:
    """Return (offset, body) for each string literal inside an asm(...) call in C source."""
    masked = C_TOKEN_RE.sub(
        lambda m: m.group(0) if m.group(0)[0] == '"' else re.sub(r"[^\n]", " ", m.group(0)),
        text,
    )
    return [
        (match.start(1) + part.start(1), part.group(1))
        for match in ASM_STRING_RE.finditer(masked)
        for part in STRING_PART_RE.finditer(match.group(1))
    ]


def _in_asm_string(text: str, line: int, character: int) -> bool:
    offset = sum(len(item) + 1 for item in text.split("\n")[:line]) + character
    return any(start <= offset <= start + len(body) for start, body in _asm_literals(text))


def _operand_count(operands: str) -> int:
    """Count top-level commas: "(%rax,%rbx,4)" and "{1to16}" count as one operand."""
    if not operands.strip():
        return 0
    depth, count = 0, 1
    for char in operands:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            count += 1
    return count


def _mnemonic_from_asm_line(line: str, semicolon_is_comment: bool = False):
    """Return (mnemonic, operand_count) for an asm line, or None."""
    line = ASM_COMMENT_RE.sub("", line)
    if semicolon_is_comment:
        line = ASM_SEMICOLON_COMMENT_RE.sub("", line)
    line = line.strip()
    if not line or line.startswith("."):
        return None
    parts = line.split(None, 1)
    if parts[0].endswith(":"):
        line = parts[1].strip() if len(parts) > 1 else ""
        if not line or line.startswith("."):
            return None
        parts = line.split(None, 1)
    if not MNEMONIC_RE.match(parts[0]):
        return None
    return parts[0], _operand_count(parts[1] if len(parts) > 1 else "")


def _operands_of_key(key: str) -> list[str]:
    start, end = key.find("("), key.rfind(")")
    inner = key[start + 1 : end].strip() if 0 <= start < end else ""
    return [piece.strip() for piece in inner.split(",")] if inner else []


def _cut_label(text: str) -> str:
    return text[:57] + "..." if len(text) > 60 else text


@functools.lru_cache(maxsize=1024)
def _mnemonic_matches(db_path: str, mnemonic: str) -> tuple:
    # The inlay-hint loop asks the same mnemonics for every line, so query once
    # per (catalog, mnemonic) and reuse. The key carries the path, not the
    # connection, so a reloaded catalog at the same path would keep stale rows;
    # clear the cache if catalog reload lands.
    conn = open_db(path=Path(db_path))
    try:
        return tuple(load_instructions_by_mnemonic_from_db(conn, mnemonic))
    finally:
        conn.close()


def _hint_label(db_path: str, mnemonic: str, operand_count: int) -> str | None:
    matches = _mnemonic_matches(db_path, mnemonic)
    if not matches:
        return None
    # ponytail: a line has no architecture, so take the first form with the same operand count.
    # The fallback takes the first form with fewest operands, never a form with more
    # operands than the line: a zero-operand "movsd" must not get the MOVSD_XMM hint.
    chosen = next(
        (m for m in matches if len(_operands_of_key(m.key)) == operand_count),
        min(matches, key=lambda m: len(_operands_of_key(m.key))),
    )
    return _cut_label(chosen.summary or "")


def _inlay_hints(conn, text: str, language_id: str, uri: str, start: int, end: int) -> list[dict]:
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    source_lines = [line.rstrip("\r") for line in text.split("\n")]
    is_c_doc = _is_c_doc(language_id, uri)
    semicolon_is_comment = not is_c_doc and uri.lower().endswith(".asm")
    if is_c_doc:
        candidates = [
            (text.count("\n", 0, offset), segment)
            for offset, body in _asm_literals(text)
            for segment in ASM_SPLIT_RE.split(body.replace("\\t", " "))
        ]
    else:
        candidates = list(enumerate(source_lines))
    hints: dict[int, list[str]] = {}  # every instruction on the source line feeds its hint
    for line, segment in candidates:
        if not start <= line <= end:
            continue
        parsed = _mnemonic_from_asm_line(segment, semicolon_is_comment)
        label = _hint_label(db_path, *parsed) if parsed else None
        if label is not None:
            hints.setdefault(line, []).append(label)
    return [
        {
            # LSP columns count UTF-16 code units.
            "position": {
                "line": line,
                "character": len(source_lines[line].encode("utf-16-le")) // 2,
            },
            "label": _cut_label("; ".join(hints[line])),
            "paddingLeft": True,
        }
        for line in sorted(hints)
    ]


def _open_catalog() -> sqlite3.Connection | None:
    path = Path(os.environ.get("SIMDREF_CATALOG") or SQLITE_PATH)
    try:
        # A stale schema opens fine but fails at the first payload read.
        return open_db(path=path) if sqlite_schema_is_current(path) else None
    except sqlite3.Error:
        return None


def main() -> int:
    conn = _open_catalog()
    session = Session(documents={})
    while True:
        message = _jsonrpc_read()
        if message is None:
            return 0
        method = message.get("method")
        if method == "initialize":
            _jsonrpc_write(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "capabilities": {
                            "hoverProvider": True,
                            "textDocumentSync": 1,
                            "inlayHintProvider": True,
                            "completionProvider": {
                                "resolveProvider": False,
                                "triggerCharacters": ["_", ".", "m", "v"],
                            },
                        }
                    },
                }
            )
        elif method == "initialized":
            if conn is None:
                _jsonrpc_write(
                    {
                        "jsonrpc": "2.0",
                        "method": "window/showMessage",
                        "params": {"type": 2, "message": CATALOG_MISSING},
                    }
                )
        elif method == "shutdown":
            _jsonrpc_write({"jsonrpc": "2.0", "id": message["id"], "result": None})
        elif method == "exit":
            return 0
        elif method == "textDocument/didOpen":
            doc = message["params"]["textDocument"]
            session.documents[doc["uri"]] = doc["text"]
            session.languages[doc["uri"]] = doc.get("languageId", "")
        elif method == "textDocument/didChange":
            params = message["params"]
            session.documents[params["textDocument"]["uri"]] = params["contentChanges"][-1]["text"]
        elif method == "textDocument/didClose":
            uri = message["params"]["textDocument"]["uri"]
            session.documents.pop(uri, None)
            session.languages.pop(uri, None)
        elif method == "textDocument/hover":
            params = message["params"]
            uri = params["textDocument"]["uri"]
            text = session.documents.get(uri, "")
            line, character = params["position"]["line"], params["position"]["character"]
            word = _word_at(text, line, character)
            # In C/C++ only the asm strings hold instructions; intrinsics hover everywhere.
            allow_instruction = not _is_c_doc(session.languages.get(uri, ""), uri) or (
                _in_asm_string(text, line, character)
            )
            body = _hover_markdown(conn, word, allow_instruction) if conn and word else None
            _jsonrpc_write(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"contents": {"kind": "markdown", "value": body}} if body else None,
                }
            )
        elif method == "textDocument/completion":
            params = message["params"]
            uri = params["textDocument"]["uri"]
            text = session.documents.get(uri, "")
            prefix = _line_prefix(text, params["position"]["line"], params["position"]["character"])
            items = _completion_candidates(conn, prefix) if conn else []
            _jsonrpc_write(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"isIncomplete": False, "items": items},
                }
            )
        elif method == "textDocument/inlayHint":
            params = message["params"]
            uri = params["textDocument"]["uri"]
            rng = params["range"]
            hints = (
                _inlay_hints(
                    conn,
                    session.documents.get(uri, ""),
                    session.languages.get(uri, ""),
                    uri,
                    rng["start"]["line"],
                    rng["end"]["line"],
                )
                if conn
                else []
            )
            _jsonrpc_write({"jsonrpc": "2.0", "id": message["id"], "result": hints})
    return 0


if __name__ == "__main__":
    sys.exit(main())
