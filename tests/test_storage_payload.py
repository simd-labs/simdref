"""Compressed-payload helpers (schema v13) and the DB-only catalog rebuild."""

from __future__ import annotations

import os

from pathlib import Path

import msgpack
import pytest

from conftest import build_fixture_catalog
from simdref.storage import (
    _pack_payload,
    _unpack_payload,
    build_sqlite,
    load_catalog,
    load_catalog_from_db,
    save_catalog,
)


def test_unpack_payload_reads_legacy_raw_msgpack_rows():
    """Pre-v13 databases stored raw msgpack; the reader must keep working."""
    obj = {"memory": "dst", "ops": [1, {"x": None}]}
    raw = msgpack.packb(obj, use_bin_type=True)
    assert raw[0] != 0x78  # otherwise the sniff could not exist
    assert _unpack_payload(raw) == obj


def test_pack_unpack_payload_roundtrip_is_compressed():
    obj = {"name": "vaddps", "description": {f"k{i}": "x" * 50 for i in range(20)}}
    blob = _pack_payload(obj)
    assert blob[0] == 0x78  # zlib stream marker
    assert len(blob) < len(msgpack.packb(obj, use_bin_type=True))
    assert _unpack_payload(blob) == obj


def test_load_catalog_from_db_matches_msgpack_snapshot(tmp_path: Path):
    """After the msgpack snapshot is pruned, the DB alone must rebuild the
    in-memory catalog."""
    catalog = build_fixture_catalog()
    db = tmp_path / "catalog.db"
    build_sqlite(catalog, db)

    rebuilt = load_catalog_from_db(db)
    assert len(rebuilt.intrinsics) == len(catalog.intrinsics)
    assert len(rebuilt.instructions) == len(catalog.instructions)
    # The sources table keys on `source`, so read-back order is alphabetical,
    # not ingest order — compare order-insensitively.
    assert sorted(s.source for s in rebuilt.sources) == sorted(s.source for s in catalog.sources)
    want = {i.name: i for i in catalog.intrinsics}["_mm256_add_ps"]
    got = next(i for i in rebuilt.intrinsics if i.name == "_mm256_add_ps")
    # _search_blob is a derived cache computed in __post_init__; it can be
    # stale w.r.t. post-construction linking on either side, so exclude it.
    from dataclasses import asdict

    got_d, want_d = asdict(got), asdict(want)
    got_d.pop("_search_blob", None)
    want_d.pop("_search_blob", None)
    assert got_d == want_d


def test_save_catalog_failed_write_leaves_old_snapshot(tmp_path: Path):
    """A failure mid-serialisation must not clobber the existing
    catalog.msgpack: the old file survives and no temp file remains.
    Injected in write_fn itself, so it also works when tests run as root."""
    catalog = build_fixture_catalog()
    path = tmp_path / "catalog.msgpack"
    sentinel = b"old snapshot, must survive"
    path.write_bytes(sentinel)

    class ExplodingRecord:
        def to_dict(self):
            raise RuntimeError("disk full")

    catalog.intrinsics[1] = ExplodingRecord()
    with pytest.raises(RuntimeError):
        save_catalog(catalog, path=path)
    assert path.read_bytes() == sentinel
    assert [p for p in tmp_path.iterdir() if p.name != path.name] == []  # no temp left


def test_write_atomic_fchmod_failure_leaves_no_fd_and_no_temp(tmp_path: Path, monkeypatch):
    from simdref import storage

    path = tmp_path / "out.bin"
    path.write_bytes(b"old")
    monkeypatch.setattr(os, "fchmod", lambda *a: (_ for _ in ()).throw(OSError("fchmod")))
    nfd_before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(OSError):
        storage._write_atomic(path, lambda fh: fh.write(b"new"))
    assert path.read_bytes() == b"old"
    assert len(os.listdir("/proc/self/fd")) == nfd_before
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_write_atomic_target_appears_between_stat_and_open(tmp_path: Path, monkeypatch):
    """Target absent at stat, another writer creates it (mode 0200) before
    our open: umask semantics win, the late target's mode is not inherited."""
    from simdref import storage

    path = tmp_path / "out.bin"
    real_open = os.open

    def open_with_race(p, flags, mode=0o777, *a, **kw):
        if str(p).startswith(str(path)) and not path.exists():
            path.write_bytes(b"")
            path.chmod(0o200)
        return real_open(p, flags, mode, *a, **kw)

    old = os.umask(0o077)
    try:
        monkeypatch.setattr(os, "open", open_with_race)
        storage._write_atomic(path, lambda fh: fh.write(b"x"))
        assert (path.stat().st_mode & 0o777) == 0o600
    finally:
        os.umask(old)


def test_write_atomic_replace_failure_leaves_no_temp(tmp_path: Path, monkeypatch):
    from simdref import storage

    path = tmp_path / "out.bin"
    path.write_bytes(b"old")
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("replace")))
    with pytest.raises(OSError):
        storage._write_atomic(path, lambda fh: fh.write(b"new"))
    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_write_atomic_keeps_private_mode(tmp_path: Path):
    from simdref import storage

    path = tmp_path / "out.bin"
    path.write_bytes(b"old")
    path.chmod(0o600)
    storage._write_atomic(path, lambda fh: fh.write(b"new"))
    assert (path.stat().st_mode & 0o777) == 0o600


def test_write_atomic_keeps_mode_zero(tmp_path: Path):
    """An existing 0000 file stays 0000: mode zero must survive the
    ``old_mode is not None`` path, not collapse to a falsy default."""
    from simdref import storage

    path = tmp_path / "out.bin"
    path.write_bytes(b"old")
    path.chmod(0o000)
    storage._write_atomic(path, lambda fh: fh.write(b"new"))
    assert (path.stat().st_mode & 0o777) == 0o000


def test_write_atomic_new_file_respects_umask(tmp_path: Path):
    from simdref import storage

    old = os.umask(0o077)
    try:
        path = tmp_path / "new.bin"
        storage._write_atomic(path, lambda fh: fh.write(b"x"))
        assert (path.stat().st_mode & 0o777) == 0o600
    finally:
        os.umask(old)


def test_save_catalog_overlapping_writers_leave_a_valid_file(tmp_path: Path, monkeypatch):
    """Invariant: each writer gets an independent temp file, both
    succeed, the final file is a valid complete catalog, and no temp
    files remain."""
    import threading

    catalog = build_fixture_catalog()
    path = tmp_path / "catalog.msgpack"
    barrier = threading.Barrier(2)
    orig_replace = os.replace

    def sync_replace(src, dst):
        barrier.wait(timeout=10)
        return orig_replace(src, dst)

    monkeypatch.setattr(os, "replace", sync_replace)

    results = {}

    def write(tag):
        try:
            save_catalog(catalog, path=path)
            results[tag] = "ok"
        except Exception as exc:  # noqa: BLE001 - record, then assert below
            results[tag] = f"error:{exc.__class__.__name__}"

    threads = [threading.Thread(target=write, args=(tag,)) for tag in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == {"a": "ok", "b": "ok"}, results
    loaded = load_catalog(path)
    assert len(loaded.intrinsics) == len(catalog.intrinsics)
    assert len(loaded.instructions) == len(catalog.instructions)
    assert [p for p in tmp_path.iterdir() if p.name != path.name] == []


def test_save_catalog_preserves_destination_mode(tmp_path: Path):
    """os.replace must not clobber the published file's mode. A
    pre-existing file keeps its mode; a new file gets 0o666 & ~umask."""
    catalog = build_fixture_catalog()
    path = tmp_path / "catalog.msgpack"
    path.write_bytes(b"x")
    path.chmod(0o644)
    save_catalog(catalog, path=path)
    assert (path.stat().st_mode & 0o777) == 0o644

    new = tmp_path / "fresh.msgpack"
    save_catalog(catalog, path=new)
    umask = os.umask(0o022)
    os.umask(umask)
    assert (new.stat().st_mode & 0o777) == (0o666 & ~umask)


def test_save_catalog_through_symlink_replaces_target_keeps_link(tmp_path: Path):
    """A symlinked destination keeps its link; the real file gets the new
    content."""
    catalog = build_fixture_catalog()
    real = tmp_path / "real.msgpack"
    link = tmp_path / "catalog.msgpack"
    os.symlink(real, link)
    save_catalog(catalog, path=link)
    assert link.is_symlink()
    assert not real.is_symlink()
    loaded = load_catalog(real)
    assert len(loaded.intrinsics) == len(catalog.intrinsics)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["catalog.msgpack", "real.msgpack"]


def test_download_from_release_replaces_readonly_target(tmp_path: Path, monkeypatch):
    """A download writes through the file object ``_write_atomic`` gives
    ``write_fn`` and replaces an existing 0400 target."""
    import httpx
    import typer

    from simdref import cli

    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)

    class _Resp:
        status_code = 200
        headers: dict = {}

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size=1024 * 64):
            yield b"new-payload"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "stream", lambda *a, **kw: _Resp())

    dest = tmp_path / "catalog.msgpack"
    dest.write_bytes(b"old")
    dest.chmod(0o400)
    try:
        cli._download_from_release()
    except typer.Exit:
        pass  # second asset (catalog.db) may exit on progress/console; content check stands
    assert dest.read_bytes() == b"new-payload"
    assert (dest.stat().st_mode & 0o777) == 0o400
