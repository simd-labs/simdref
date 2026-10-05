"""Tests for fixes to simd-labs/simdref#2.

Covers:
- JSON annotate emits structured latency/cpi/ports.
- A pinned but-unpopulated arch produces a ``[missing:<arch>]`` tag
  instead of a silent ``lat=- cpi=-``.
- The bare-query dispatcher does not launch the TUI in a non-TTY context
  (it prints a plain summary and returns 0/2 without blocking).
- ``supported_core_ids()`` lists canonical cores used for error messages.
"""

from __future__ import annotations

import json

from simdref.annotate import (
    AnnotateOptions,
    annotate_stream,
    format_annotation,
)
from simdref.models import InstructionRecord
from simdref.perf_sources.cores import canonical_core_id, supported_core_ids


def _make_record(
    *,
    mnemonic: str = "vaddps",
    summary: str = "Add packed single-precision floats.",
    arch_details: dict | None = None,
) -> InstructionRecord:
    return InstructionRecord(
        mnemonic=mnemonic,
        form=f"{mnemonic} ymm, ymm, ymm",
        summary=summary,
        arch_details=arch_details or {},
    )


def _arch_entry(lat: str, cpi: str, *, kind: str = "measured", ports: str | None = None) -> dict:
    measurement: dict = {"TP": cpi}
    if ports is not None:
        measurement["ports"] = ports
    return {
        "latencies": [{"cycles": lat}],
        "measurement": measurement,
        "source_kind": kind,
    }


class _FakeConn:
    def close(self):
        pass


def _fake_lookup(mapping):
    def _fake(mnemonic, conn):
        return mapping.get(mnemonic.lower(), [])

    return _fake


def test_format_annotation_tags_missing_arch():
    rec = _make_record(arch_details={"cortex-a72": _arch_entry("3", "0.5")})
    out = format_annotation(
        rec,
        performance=True,
        docs=False,
        arch="sapphirerapids",
        agg="avg",
        include_modeled=False,
    )
    assert "[missing:sapphirerapids]" in out
    assert "lat=-" in out


def test_json_record_has_structured_latency_cpi_ports(monkeypatch):
    from simdref import annotate as annotate_mod

    rec = _make_record(
        arch_details={
            "SKX": _arch_entry("3", "0.5", ports="1*p015"),
            "ZEN4": _arch_entry("5", "1.0", ports="1*FP0"),
        }
    )
    monkeypatch.setattr(annotate_mod, "lookup", _fake_lookup({"vaddps": [rec]}))
    src = ["    vaddps %ymm2, %ymm1, %ymm0\n"]
    opts = AnnotateOptions(fmt="json")
    out = "".join(annotate_stream(src, opts=opts, conn=_FakeConn()))
    records = json.loads(out)
    assert records and records[0]["mnemonic"] == "vaddps"
    assert records[0]["latency"] == 4.0
    assert records[0]["cpi"] == 0.75
    # Ports collected across contributing arches.
    assert set(records[0]["ports"]) == {"1*p015", "1*FP0"}


def test_json_record_none_when_perf_missing(monkeypatch):
    from simdref import annotate as annotate_mod

    rec = _make_record(arch_details={"cortex-a72": _arch_entry("3", "0.5")})
    monkeypatch.setattr(annotate_mod, "lookup", _fake_lookup({"vaddps": [rec]}))
    src = ["    vaddps %ymm2, %ymm1, %ymm0\n"]
    opts = AnnotateOptions(fmt="json", arch="sapphirerapids")
    out = "".join(annotate_stream(src, opts=opts, conn=_FakeConn()))
    records = json.loads(out)
    assert records[0]["latency"] is None
    assert records[0]["cpi"] is None
    assert records[0]["ports"] is None
    assert "[missing:sapphirerapids]" in records[0]["annotation"]


def test_supported_core_ids_contains_aarch64_riscv_x86():
    ids = supported_core_ids()
    assert "cortex-a72" in ids
    assert "sifive-u74" in ids
    # x86 cores are carried under uops.info short codes (EMR, SKX, ZEN4, ...).
    # Sapphire Rapids uses the Golden Cove P-core microarchitecture and maps
    # to EMR (Emerald Rapids) since uops.info has no dedicated SPR row.
    assert "EMR" in ids
    assert "ZEN4" in ids
    assert canonical_core_id("sapphirerapids") == "EMR"
    assert canonical_core_id("skylake-x") == "SKX"
    assert canonical_core_id("zen4") == "ZEN4"


def test_smart_lookup_does_not_launch_tui_on_non_tty(monkeypatch, capsys):
    from simdref import cli

    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(cli, "ensure_runtime", lambda: None)
    fake_conn = _FakeConn()
    monkeypatch.setattr(cli, "open_db", lambda: fake_conn)
    monkeypatch.setattr(cli, "load_intrinsic_from_db", lambda conn, name: None)
    monkeypatch.setattr(cli, "_find_instructions_fast", lambda q: [])

    monkeypatch.setattr(cli, "_search_runtime", lambda *a, **kw: ([], {}, {}))

    def _boom(*a, **kw):
        raise AssertionError("TUI must not launch in non-TTY context")

    monkeypatch.setattr(cli, "_run_tui", _boom)

    rc = cli._smart_lookup("vgatherdpd")
    assert rc == 2


def test_smart_lookup_prints_summary_on_non_tty_match(monkeypatch, capsys):
    from simdref import cli

    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(cli, "ensure_runtime", lambda: None)
    fake_conn = _FakeConn()
    monkeypatch.setattr(cli, "open_db", lambda: fake_conn)
    monkeypatch.setattr(cli, "load_intrinsic_from_db", lambda conn, name: None)

    rec = _make_record(arch_details={"cortex-a72": _arch_entry("3", "0.5")})
    monkeypatch.setattr(cli, "_find_instructions_fast", lambda q: [rec])
    monkeypatch.setattr(
        cli, "_run_tui", lambda **kw: (_ for _ in ()).throw(AssertionError("no TUI"))
    )

    rc = cli._smart_lookup("vaddps")
    out = capsys.readouterr().out
    assert rc == 0
    assert "vaddps" in out
    assert "lat=" in out


def _patch_db(monkeypatch, cli):
    monkeypatch.setattr(cli, "ensure_runtime", lambda: None)
    monkeypatch.setattr(cli, "open_db", lambda: _FakeConn())
    monkeypatch.setattr(cli, "load_intrinsic_from_db", lambda conn, name: None)


def test_smart_lookup_fuzzy_match_on_tty_opens_tui_with_preset(monkeypatch):
    """A fuzzy hit on a TTY opens the TUI with the given preset (as on
    origin/main); the passed preset wins over the default."""
    from simdref import cli

    seen = {}

    class _Result:
        kind = "instruction"
        key = "vaddps"
        title = "vaddps"
        subtitle = "Add packed single-precision floats."

    rec = _make_record(arch_details={"SKX": _arch_entry("3", "0.5")})
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    _patch_db(monkeypatch, cli)
    monkeypatch.setattr(cli, "_find_instructions_fast", lambda q: [])
    monkeypatch.setattr(cli, "_search_runtime", lambda *a, **kw: ([_Result()], {}, {"vaddps": rec}))
    monkeypatch.setattr(cli, "_run_tui", lambda **kw: seen.update(kw) or 0)

    rc = cli._smart_lookup("vaddp", preset="arm64")
    assert rc == 0
    assert seen == {"initial_query": "vaddp", "initial_preset": "arm64"}


def test_smart_lookup_fuzzy_match_non_tty_prints_and_skips_tui(monkeypatch):
    """Non-TTY keeps printing the search list and never opens the TUI,
    even with a non-default preset."""
    from simdref import cli

    class _Result:
        kind = "instruction"
        key = "vaddps"
        title = "vaddps"
        subtitle = "Add packed single-precision floats."

    rec = _make_record(arch_details={"SKX": _arch_entry("3", "0.5")})
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    _patch_db(monkeypatch, cli)
    monkeypatch.setattr(cli, "_find_instructions_fast", lambda q: [])
    monkeypatch.setattr(cli, "_search_runtime", lambda *a, **kw: ([_Result()], {}, {"vaddps": rec}))
    monkeypatch.setattr(
        cli, "_run_tui", lambda **kw: (_ for _ in ()).throw(AssertionError("no TUI"))
    )

    rc = cli._smart_lookup("vaddp", preset="arm64")
    assert rc == 0


def test_smart_lookup_fuzzy_match_json_tty_prints_and_skips_tui(monkeypatch):
    """--json prints even on a TTY (no interactive UI under --json)."""
    from simdref import cli

    class _Result:
        kind = "instruction"
        key = "vaddps"
        title = "vaddps"
        subtitle = "Add packed single-precision floats."

    rec = _make_record(arch_details={"SKX": _arch_entry("3", "0.5")})
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    _patch_db(monkeypatch, cli)
    monkeypatch.setattr(cli, "_find_instructions_fast", lambda q: [])
    monkeypatch.setattr(cli, "_search_runtime", lambda *a, **kw: ([_Result()], {}, {"vaddps": rec}))
    monkeypatch.setattr(
        cli, "_run_tui", lambda **kw: (_ for _ in ()).throw(AssertionError("no TUI"))
    )

    rc = cli._smart_lookup("vaddp", preset="arm64", as_json=True)
    assert rc == 0
