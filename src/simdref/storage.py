"""Msgpack and SQLite persistence for the simdref catalog.

Stores a compact msgpack snapshot for reuse and a SQLite database with
FTS5 virtual tables for fast full-text search with BM25 ranking.
"""

from __future__ import annotations

import contextlib
import os
import re
import secrets
import sqlite3
import stat
import sys
import zlib
from itertools import islice
from pathlib import Path

import msgpack

from simdref.models import Catalog, InstructionRecord, IntrinsicRecord, SourceVersion


def derive_arm_arch(isa: list[str] | None, metadata: dict[str, str] | None) -> str | None:
    """Classify an Arm intrinsic as A32/A64/BOTH from its supported_architectures.

    Rules (matching the Part A preset design):
    * contains A64 and (A32 or v7 or MVE) -> "BOTH"
    * contains A64 only -> "A64"
    * contains A32 or v7 or MVE only -> "A32"
    * non-Arm rows or missing metadata -> None
    """
    supported = str((metadata or {}).get("supported_architectures") or "").strip()
    if not supported:
        # MVE-only Arm intrinsics may not carry supported_architectures;
        # infer A32 from the ISA list in that case.
        if isa and any(tok.upper() == "MVE" for tok in isa):
            return "A32"
        return None
    upper = supported.upper()
    has_a64 = "A64" in upper
    has_a32 = ("A32" in upper) or ("V7" in upper) or ("MVE" in upper)
    if has_a64 and has_a32:
        return "BOTH"
    if has_a64:
        return "A64"
    if has_a32:
        return "A32"
    return None


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]


def _default_data_dir() -> Path:
    """Return the platform-appropriate data directory for simdref.

    Uses repo-relative paths for editable/dev installs, and a
    platform-appropriate user data directory for wheel installs.
    """
    # Dev install: pyproject.toml next to src/simdref/
    if (REPO_ROOT / "pyproject.toml").exists() and (REPO_ROOT / "src" / "simdref").is_dir():
        return REPO_ROOT / "data" / "derived"

    # Installed: use platform-appropriate data dir
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "simdref"


DATA_DIR = _default_data_dir()
_is_dev_install = DATA_DIR == REPO_ROOT / "data" / "derived"

if _is_dev_install:
    WEB_DIR = REPO_ROOT / "web"
    DEFAULT_MAN_DIR = REPO_ROOT / "share" / "man"
else:
    WEB_DIR = DATA_DIR / "web"
    # Target the XDG data-root man dir: man-db auto-discovers
    # ~/.local/share/man (or $XDG_DATA_HOME/man) with no MANPATH edits,
    # so plain `man vpaddd` works after `simdref install-manpages`.
    DEFAULT_MAN_DIR = DATA_DIR.parent / "man"

CATALOG_PATH = DATA_DIR / "catalog.msgpack"
SQLITE_PATH = DATA_DIR / "catalog.db"
SQLITE_SCHEMA_VERSION = "13"


def _pack_payload(obj: dict) -> bytes:
    """Msgpack + zlib-9. Payloads compress to ~20-40% of their raw size."""
    return zlib.compress(msgpack.packb(obj, use_bin_type=True), 9)


def _unpack_payload(data: bytes):
    """Inverse of :func:`_pack_payload`; also reads raw-msgpack rows from
    pre-v13 databases. The 0x78 gate is exact: zlib streams start with 0x78,
    while a raw msgpack payload can never do so — our payloads are dicts
    (fixmap 0x80-0x8f, map16 0xde, map32 0xdf) and 0x78 would decode as the
    fixint 120. The try/except is belt-and-braces for anything unforeseen."""
    if data[:1] == b"\x78":
        try:
            data = zlib.decompress(data)
        except zlib.error:
            pass  # raw msgpack row from an older database
    return msgpack.unpackb(data, raw=False)


def read_installed_version_stamp() -> str | None:
    """Return the package version that last refreshed catalog.db's meta table, or None."""
    if not SQLITE_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'installed_version'").fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return None
    return (row[0].strip() or None) if row else None


def write_installed_version_stamp(version: str) -> None:
    """Record the package version that just refreshed catalog.db's meta table."""
    if not SQLITE_PATH.exists():
        return  # best-effort — stamp persistence must not break commands
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        try:
            conn.execute(
                "INSERT INTO meta VALUES ('installed_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (version.strip(),),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        pass  # best-effort — stamp persistence must not break commands


FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SQLITE_INSERT_BATCH_SIZE = 512


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_catalog(path: Path = CATALOG_PATH) -> Catalog:
    payload = msgpack.unpackb(path.read_bytes(), raw=False)
    return Catalog.from_dict(payload)


def load_catalog_from_db(path: Path = SQLITE_PATH) -> Catalog:
    """Rebuild the in-memory catalog from the SQLite runtime alone.

    The msgpack snapshot is optional (pruned after install/update); this is
    the fallback used by ``simdref export``/``install-manpages`` and by
    offline schema rebuilds when the snapshot is absent.
    """
    with open_db(path) as conn:
        intrinsics = [
            IntrinsicRecord(**_unpack_payload(row["payload"]))
            for row in conn.execute("SELECT payload FROM intrinsics_data ORDER BY name")
        ]
        instructions = [
            InstructionRecord(**_unpack_payload(row["payload"]))
            for row in conn.execute("SELECT payload FROM instructions_data ORDER BY db_key")
        ]
        sources = load_sources_from_db(conn)
        generated_at = generated_at_from_db(conn)
    return Catalog(
        intrinsics=intrinsics,
        instructions=instructions,
        sources=sources,
        generated_at=generated_at,
    )


def _write_atomic(path, write_fn) -> None:
    """Publish ``path`` atomically.

    Writes via ``write_fn(fh)`` to a random sibling temp file, then
    ``os.replace``. A symlinked destination keeps its link; the real file
    is replaced. An existing target's mode is set on the temp before the
    first byte is written; a new file gets ``0o666 & ~umask``, the same
    mode as ``path.open("wb")``. Any failure unlinks the temp and
    re-raises.
    """
    # ponytail: keeps mode bits only; owner, group and ACLs are not copied
    # (a user cache file).
    target = Path(os.path.realpath(path))
    old_mode = stat.S_IMODE(os.stat(target).st_mode) if target.exists() else None
    while True:
        tmp = f"{target}.{secrets.token_hex(8)}.tmp"
        try:
            fh = os.fdopen(
                os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, old_mode or 0o666), "wb"
            )
            break
        except FileExistsError:
            continue
    try:
        with fh:
            fh.atomic_path = Path(tmp)
            if old_mode is not None:
                os.fchmod(fh.fileno(), old_mode)  # tighten before any byte
            write_fn(fh)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            fh.close()
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def save_catalog(catalog: Catalog, path: Path = CATALOG_PATH) -> None:
    ensure_dir(path.parent)
    packer = msgpack.Packer(use_bin_type=True)

    def write(fh) -> None:
        fh.write(packer.pack_map_header(4))
        fh.write(packer.pack("intrinsics"))
        fh.write(packer.pack_array_header(len(catalog.intrinsics)))
        for record in catalog.intrinsics:
            fh.write(packer.pack(record.to_dict()))
        fh.write(packer.pack("instructions"))
        fh.write(packer.pack_array_header(len(catalog.instructions)))
        for record in catalog.instructions:
            fh.write(packer.pack(record.to_dict()))
        fh.write(packer.pack("sources"))
        fh.write(packer.pack_array_header(len(catalog.sources)))
        for source in catalog.sources:
            fh.write(packer.pack(source.to_dict()))
        fh.write(packer.pack("generated_at"))
        fh.write(packer.pack(catalog.generated_at))

    _write_atomic(path, write)


def open_db(path: Path = SQLITE_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def sqlite_schema_is_current(path: Path = SQLITE_PATH) -> bool:
    if not path.exists():
        return False
    conn = sqlite3.connect(path)
    try:
        meta = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if meta is None:
            return False
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None or row[0] != SQLITE_SCHEMA_VERSION:
            return False
        expected_columns = {
            "id",
            "name",
            "architecture",
            "signature",
            "description",
            "header",
            "isa",
            "category",
            "subcategory",
            "arm_arch",
            "payload",
        }
        actual_columns = {
            item[1] for item in conn.execute("PRAGMA table_info(intrinsics_data)").fetchall()
        }
        if expected_columns != actual_columns:
            return False
        expected_instruction_columns = {
            "db_key",
            "key",
            "architecture",
            "mnemonic",
            "form",
            "summary",
            "isa",
            "category",
            "payload",
        }
        actual_instruction_columns = {
            item[1] for item in conn.execute("PRAGMA table_info(instructions_data)").fetchall()
        }
        if expected_instruction_columns != actual_instruction_columns:
            return False
        instr_indexes = {
            item[1] for item in conn.execute("PRAGMA index_list(instructions_data)").fetchall()
        }
        if "idx_instruction_category" not in instr_indexes:
            return False
        intr_indexes = {
            item[1] for item in conn.execute("PRAGMA index_list(intrinsics_data)").fetchall()
        }
        return "idx_intrinsic_arm_arch" in intr_indexes
    except sqlite3.Error:
        return False
    finally:
        conn.close()


_ALPHA_NUM_SPLIT = re.compile(r"[a-zA-Z]+|[0-9]+")


def _tokenize_name(name: str) -> str:
    """Split alpha/numeric boundaries for better FTS matching.

    _mm256_add_epi32 → mm 256 add epi 32
    VADDPS (YMM, YMM, YMM) → vaddps ymm ymm ymm
    """
    return " ".join(_ALPHA_NUM_SPLIT.findall(name)).lower()


def _batched(items, size: int = SQLITE_INSERT_BATCH_SIZE):
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch


def build_sqlite(catalog: Catalog, path: Path = SQLITE_PATH) -> None:
    ensure_dir(path.parent)
    _db_tmp = path.with_name(path.name + ".tmp")
    _db_tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(_db_tmp)
    cur = conn.cursor()
    cur.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE sources (
            source TEXT PRIMARY KEY,
            payload BLOB NOT NULL
        );
        CREATE TABLE intrinsics_data (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL COLLATE NOCASE,
            architecture TEXT NOT NULL,
            signature TEXT NOT NULL,
            description TEXT NOT NULL,
            header TEXT NOT NULL,
            isa TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT NOT NULL DEFAULT '',
            arm_arch TEXT,
            payload BLOB NOT NULL
        );
        CREATE INDEX idx_intrinsic_name ON intrinsics_data (name);
        CREATE INDEX idx_intrinsic_arm_arch ON intrinsics_data (arm_arch);
        CREATE TABLE instructions_data (
            db_key TEXT PRIMARY KEY COLLATE NOCASE,
            key TEXT NOT NULL COLLATE NOCASE,
            architecture TEXT NOT NULL,
            mnemonic TEXT NOT NULL COLLATE NOCASE,
            form TEXT NOT NULL,
            summary TEXT NOT NULL,
            isa TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT '',
            payload BLOB NOT NULL
        );
        CREATE INDEX idx_instruction_key ON instructions_data (key);
        CREATE INDEX idx_instruction_mnemonic ON instructions_data (mnemonic);
        CREATE INDEX idx_instruction_category ON instructions_data (category);
        CREATE VIRTUAL TABLE intrinsics_fts USING fts5(name, signature, description, header, isa, category, instructions, notes, aliases, summary, name_tokens);
        CREATE VIRTUAL TABLE instructions_fts USING fts5(key, mnemonic, form, summary, isa, linked_intrinsics, aliases, key_tokens);
        """
    )
    cur.execute("INSERT INTO meta VALUES (?, ?)", ("schema_version", SQLITE_SCHEMA_VERSION))
    cur.execute("INSERT INTO meta VALUES (?, ?)", ("generated_at", catalog.generated_at))

    # Sources
    source_rows = ((source.source, _pack_payload(source.to_dict())) for source in catalog.sources)
    for batch in _batched(source_rows):
        cur.executemany("INSERT INTO sources VALUES (?, ?)", batch)

    # Build a mnemonic -> summary lookup from instructions for fast access
    _instr_summary: dict[str, str] = {}
    for irec in catalog.instructions:
        if irec.mnemonic and irec.summary and irec.mnemonic not in _instr_summary:
            _instr_summary[irec.mnemonic] = irec.summary

    # Intrinsics data + FTS
    intrinsics_data_batch = []
    intrinsics_fts_batch = []
    for record in catalog.intrinsics:
        payload = _pack_payload(record.to_dict())
        intrinsics_data_batch.append(
            (
                record.name,
                record.architecture,
                record.signature,
                record.description,
                record.header,
                " ".join(record.isa),
                record.category,
                record.subcategory,
                derive_arm_arch(record.isa, record.metadata),
                payload,
            )
        )
        instr_summary = ""
        if record.instructions:
            mnemonic = record.instructions[0].split("(")[0].split()[0].strip()
            instr_summary = _instr_summary.get(mnemonic, "")
        if not instr_summary and record.description:
            instr_summary = record.description.split(".")[0] + "."
        intrinsics_fts_batch.append(
            (
                record.name,
                record.signature,
                record.description,
                record.header,
                " ".join(record.isa),
                record.category,
                " ".join(record.instructions),
                " ".join(record.notes),
                " ".join(record.aliases),
                instr_summary,
                _tokenize_name(record.name),
            )
        )
        if len(intrinsics_data_batch) >= SQLITE_INSERT_BATCH_SIZE:
            cur.executemany(
                "INSERT INTO intrinsics_data (name, architecture, signature, description, header, isa, category, subcategory, arm_arch, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                intrinsics_data_batch,
            )
            cur.executemany(
                "INSERT INTO intrinsics_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                intrinsics_fts_batch,
            )
            intrinsics_data_batch.clear()
            intrinsics_fts_batch.clear()
    if intrinsics_data_batch:
        cur.executemany(
            "INSERT INTO intrinsics_data (name, architecture, signature, description, header, isa, category, subcategory, arm_arch, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            intrinsics_data_batch,
        )
        cur.executemany(
            "INSERT INTO intrinsics_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            intrinsics_fts_batch,
        )

    # Instructions data + FTS
    instructions_data_batch = []
    instructions_fts_batch = []
    for record in catalog.instructions:
        payload = _pack_payload(record.to_dict())
        instructions_data_batch.append(
            (
                record.db_key,
                record.key,
                record.architecture,
                record.mnemonic,
                record.form,
                record.summary,
                " ".join(record.isa),
                record.metadata.get("category", "") if isinstance(record.metadata, dict) else "",
                payload,
            )
        )
        instructions_fts_batch.append(
            (
                record.key,
                record.mnemonic,
                record.form,
                record.summary,
                " ".join(record.isa),
                " ".join(record.linked_intrinsics),
                " ".join(record.aliases),
                _tokenize_name(record.key),
            )
        )
        if len(instructions_data_batch) >= SQLITE_INSERT_BATCH_SIZE:
            cur.executemany(
                "INSERT INTO instructions_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                instructions_data_batch,
            )
            cur.executemany(
                "INSERT INTO instructions_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                instructions_fts_batch,
            )
            instructions_data_batch.clear()
            instructions_fts_batch.clear()
    if instructions_data_batch:
        cur.executemany(
            "INSERT INTO instructions_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            instructions_data_batch,
        )
        cur.executemany(
            "INSERT INTO instructions_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            instructions_fts_batch,
        )

    conn.commit()
    # Switch back to DELETE before publishing: a reader that only opens the
    # published catalog.db must never create -wal/-shm sidecars next to it.
    # Switching away from WAL forces SQLite to checkpoint every WAL page into
    # the main file first, so the rename below never orphans content in
    # `<tmp>-wal` the way a bare close under WAL used to (Python 3.11 does
    # not checkpoint on close while a cursor is alive).
    cur.close()
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    # Atomic publish: build into a sibling .tmp so a crash mid-build never
    # leaves the runtime without a database (matters when refreshing from a
    # DB that was just read as the fallback source).
    _db_tmp.replace(path)
    # `replace()` only swaps the main file: a WAL-mode catalog.db published
    # before this fix left `-wal`/`-shm` siblings at `path` that a plain
    # rename does not touch. Drop them so a re-publish over an old WAL-mode
    # catalog.db doesn't leave that debris next to the new DELETE-mode one.
    for suffix in ("-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)


def load_sources_from_db(conn: sqlite3.Connection) -> list[SourceVersion]:
    rows = conn.execute("SELECT payload FROM sources ORDER BY source").fetchall()
    return [SourceVersion(**_unpack_payload(row["payload"])) for row in rows]


def generated_at_from_db(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key = 'generated_at'").fetchone()
    return row["value"] if row else ""


def load_intrinsic_from_db(conn: sqlite3.Connection, name: str) -> IntrinsicRecord | None:
    row = conn.execute(
        "SELECT payload FROM intrinsics_data WHERE name = ? ORDER BY id LIMIT 1",
        (name,),
    ).fetchone()
    if not row:
        return None
    return IntrinsicRecord(**_unpack_payload(row["payload"]))


def load_instruction_from_db(conn: sqlite3.Connection, key: str) -> InstructionRecord | None:
    row = conn.execute(
        """
        SELECT payload
        FROM instructions_data
        WHERE db_key = ? OR key = ?
        ORDER BY CASE WHEN db_key = ? THEN 0 ELSE 1 END, architecture, key
        LIMIT 1
        """,
        (key, key, key),
    ).fetchone()
    if not row:
        return None
    return InstructionRecord(**_unpack_payload(row["payload"]))


def load_instructions_by_mnemonic_from_db(
    conn: sqlite3.Connection, mnemonic: str
) -> list[InstructionRecord]:
    rows = conn.execute(
        "SELECT payload FROM instructions_data WHERE mnemonic = ? ORDER BY architecture, key",
        (mnemonic,),
    ).fetchall()
    return [InstructionRecord(**_unpack_payload(row["payload"])) for row in rows]


def load_instructions_by_mnemonic_prefix_from_db(
    conn: sqlite3.Connection, prefix: str, limit: int = 400
) -> list[InstructionRecord]:
    rows = conn.execute(
        """
        SELECT payload
        FROM instructions_data
        WHERE mnemonic LIKE ? || '%'
        ORDER BY mnemonic, architecture, key
        LIMIT ?
        """,
        (prefix, limit),
    ).fetchall()
    return [InstructionRecord(**_unpack_payload(row["payload"])) for row in rows]


def _fts_match_query(query: str) -> str:
    tokens = [token.casefold() for token in FTS_TOKEN_RE.findall(query)]
    return " AND ".join(f'"{token}"*' for token in tokens if token)


def _append_filter_clause(
    base_sql: str,
    table: str,
    filter_spec,
    enabled_families,
    enabled_categories,
    binds: list,
    match_marker: str,
    enabled_arm_arch=None,
) -> str:
    """Splice a FilterSpec WHERE fragment into the SQL query after the
    FTS MATCH placeholder, so bind ordering stays correct.

    ``match_marker`` must be the exact string that appears immediately after
    the MATCH ``?`` placeholder in ``base_sql`` (e.g. a newline-preserving
    pattern) — the helper inserts ``AND <clause>`` just after it.
    """
    if filter_spec is None:
        return base_sql
    clause, extra_binds = filter_spec.sql_predicate(
        table,
        enabled_families=enabled_families,
        enabled_categories=enabled_categories,
        enabled_arm_arch=enabled_arm_arch,
    )
    if not clause:
        return base_sql
    binds.extend(extra_binds)
    if match_marker not in base_sql:
        return base_sql
    return base_sql.replace(match_marker, f"{match_marker} AND {clause} ", 1)


def search_intrinsic_candidates_from_db(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 200,
    *,
    filter_spec=None,
    enabled_families=None,
    enabled_categories=None,
    enabled_arm_arch=None,
) -> list[IntrinsicRecord]:
    match_query = _fts_match_query(query)
    if not match_query:
        return []
    binds: list = [match_query]
    sql = """
        SELECT intrinsics_data.payload
        FROM intrinsics_fts
        JOIN intrinsics_data ON intrinsics_data.id = intrinsics_fts.rowid
        WHERE intrinsics_fts MATCH ?
        ORDER BY bm25(intrinsics_fts), length(intrinsics_data.name), intrinsics_data.name
        LIMIT ?
        """
    sql = _append_filter_clause(
        sql,
        "intrinsics_data",
        filter_spec,
        enabled_families,
        enabled_categories,
        binds,
        match_marker="intrinsics_fts MATCH ?",
        enabled_arm_arch=enabled_arm_arch,
    )
    binds.append(limit)
    rows = conn.execute(sql, binds).fetchall()
    return [IntrinsicRecord(**_unpack_payload(row["payload"])) for row in rows]


def search_instruction_candidates_from_db(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 200,
    *,
    filter_spec=None,
    enabled_families=None,
    enabled_categories=None,
) -> list[InstructionRecord]:
    match_query = _fts_match_query(query)
    if not match_query:
        return []
    binds: list = [match_query]
    sql = """
        SELECT instructions_data.payload
        FROM instructions_fts
        JOIN instructions_data ON instructions_data.rowid = instructions_fts.rowid
        WHERE instructions_fts MATCH ?
        ORDER BY bm25(instructions_fts), length(instructions_data.key), instructions_data.key
        LIMIT ?
        """
    sql = _append_filter_clause(
        sql,
        "instructions_data",
        filter_spec,
        enabled_families,
        enabled_categories,
        binds,
        match_marker="instructions_fts MATCH ?",
    )
    binds.append(limit)
    rows = conn.execute(sql, binds).fetchall()
    return [InstructionRecord(**_unpack_payload(row["payload"])) for row in rows]
