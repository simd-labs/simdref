"""Tests for simd-labs/simdref#57 part b.

The ZEN4 VPCOMPRESSD store form is microcoded (144 uops). The uops.info
row still measures it: TP 72.50 cycles. The annotate output must keep
that CPI. The RET call/ret loop harness rows must stay filtered, and
their ports string must go away together with the CPI.
"""

from __future__ import annotations

import json
from pathlib import Path

from simdref.annotate import (
    AnnotateOptions,
    annotate_stream,
)
from simdref.ingest_catalog import parse_uops_xml

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "issue_57" / "uops_measurements.xml"


def _records() -> dict[str, object]:
    return {rec.form: rec for rec in parse_uops_xml(_FIXTURE.read_text())}


class _FakeConn:
    def close(self):
        pass


def _fake_lookup(mapping):
    def _fake(mnemonic, conn):
        return mapping.get(mnemonic.lower(), [])

    return _fake


def _annotate(monkeypatch, rec, line: str, arch: str) -> dict:
    from simdref import annotate as annotate_mod

    monkeypatch.setattr(annotate_mod, "lookup", _fake_lookup({rec.mnemonic.lower(): [rec]}))
    opts = AnnotateOptions(fmt="json", arch=arch)
    out = "".join(annotate_stream([line + "\n"], opts=opts, conn=_FakeConn()))
    records = json.loads(out)
    assert records, out
    return records[0]


def test_zen4_vpcompressd_memory_keeps_measured_cpi(monkeypatch):
    # Issue #57 repro: the ZEN4 row drops the measured 72.50 and the 144 uops.
    rec = _records()["VPCOMPRESSD (M512, K, ZMM)"]
    row = _annotate(monkeypatch, rec, "vpcompressd ZMMWORD PTR [rsi+rcx*4]{k1},zmm0", "ZEN4")
    assert row["latency"] == 52.0
    assert row["cpi"] == 72.5
    assert row["ports"] == ["1*FP01+1*FP12+2*FP23+18*FP45"]
    assert "cpi=72.50" in row["annotation"]


def test_zen4_microcoded_form_keeps_cpi_when_tp_figures_differ(monkeypatch):
    # VPCOMPRESSB is the same microcoded class: uops=480, TP_unrolled and
    # TP_loop differ by measurement noise (167.50 vs 168.47).
    rec = _records()["VPCOMPRESSB (M512, K, ZMM)"]
    row = _annotate(monkeypatch, rec, "vpcompressb ZMMWORD PTR [rsi+rcx*4]{k1},zmm0", "ZEN4")
    assert row["cpi"] == 167.5


def test_zen4_vpcompressd_reports_uops_and_microcoded(monkeypatch):
    # Issue #57 "Expected": keep the uop count of the ZEN4 row and mark the
    # form microcoded. The real row has no uops_MS: AMD cores count macro-
    # ops only, and 144 macro-ops is far over the AMD microcode bound of
    # two macro ops (AMD 57647).
    rec = _records()["VPCOMPRESSD (M512, K, ZMM)"]
    row = _annotate(monkeypatch, rec, "vpcompressd ZMMWORD PTR [rsi+rcx*4]{k1},zmm0", "ZEN4")
    assert row["uops"] == 144
    assert row["microcoded"] is True
    assert "uops=144" in row["annotation"]
    assert "microcoded" in row["annotation"]


def test_icl_vpcompressd_keeps_uops_ms_rule(monkeypatch):
    # The ICL row of the same form has uops=4 but uops_MS=0: Intel rows
    # stay on the uops_MS rule, so the form is not microcoded on ICL.
    rec = _records()["VPCOMPRESSD (M512, K, ZMM)"]
    row = _annotate(monkeypatch, rec, "vpcompressd ZMMWORD PTR [rsi+rcx*4]{k1},zmm0", "ICL")
    assert row["uops"] == 4
    assert row["microcoded"] is False
    assert "microcoded" not in row["annotation"]


def test_zen4_vaddps_reports_not_microcoded(monkeypatch):
    # A plain 1-macro-op ZEN4 form is fastpath single, not microcoded.
    rec = _records()["VADDPS (ZMM, ZMM, ZMM)"]
    row = _annotate(monkeypatch, rec, "vaddps %zmm1, %zmm0, %zmm2", "ZEN4")
    assert row["uops"] == 1
    assert row["microcoded"] is False
    assert "microcoded" not in row["annotation"]


def test_skx_vaddps_reports_uops_and_not_microcoded(monkeypatch):
    # A plain 1-uop form: uops_MS is 0, so microcoded is False, not omitted.
    rec = _records()["VADDPS (ZMM, ZMM, ZMM)"]
    row = _annotate(monkeypatch, rec, "vaddps %zmm1, %zmm0, %zmm2", "SKX")
    assert row["uops"] == 1
    assert row["microcoded"] is False
    assert "uops=1" in row["annotation"]
    assert "microcoded" not in row["annotation"]


def test_ret_harness_row_drops_cpi_and_ports(monkeypatch):
    # The RET call/ret loop harness reports uops=20 and TP 33.35 for the
    # whole loop. The row must drop both the CPI and the harness ports.
    rec = _records()["RET_NEAR"]
    row = _annotate(monkeypatch, rec, "ret", "HSW")
    assert row["cpi"] is None
    assert row["ports"] is None
    assert "cpi=-" in row["annotation"]
