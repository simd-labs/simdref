"""Issue #65 perf-data defects: M12, M11 and M8.

M12: an empty top-ranked latency slot must not block a lower source from
filling latency. M11: mnemonic-level timing must not spread onto ZA tile
and ZA array forms. M8: pinning an arch with zero perf rows must warn on
stderr. Fixture entry shapes copy rows of the shipped catalog
(catalog.db, read only).
"""

from __future__ import annotations

import unittest.mock

from typer.testing import CliRunner

from simdref import cli as cli_mod
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.perf_sources.merge import PerfRow, merge_perf_rows
from simdref.storage import build_sqlite

# Real uops.info slot: KMOVQ (K, M64) on ICL has measurement but no latency.
ICL_KMOVQ_EMPTY = {
    "doc": {},
    "iaca": [],
    "latencies": [],
    "measurement": {
        "TP_loop": "1.00",
        "TP_ports": "1.00",
        "TP_unrolled": "1.00",
        "available_simple_decoders": "2",
        "complex_decoder": "1",
        "ports": "1*p23+1*p5",
        "uops": "2",
        "uops_MITE": "3",
        "uops_MS": "0",
        "uops_retire_slots": "3",
    },
}

# Real uops.info slot: KMOVW (K, R32) on ICL carries its latency.
ICL_KMOVW_FILLED = {
    "doc": {},
    "iaca": [],
    "latencies": [
        {
            "cycles": "3",
            "cycles_is_upper_bound": "1",
            "start_op": "2",
            "target_op": "1",
        }
    ],
    "measurement": {
        "TP_loop": "1.00",
        "TP_ports": "1.00",
        "TP_unrolled": "1.00",
        "ports": "1*p5",
        "uops": "1",
        "uops_MITE": "1",
        "uops_MS": "0",
        "uops_retire_slots": "1",
    },
}

INTEL_GUIDE_URL = "https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html"
ARM_SWOPT_URL = "https://documentation-service.arm.com/static/668bc0a369e89f01e39c4668"

FMLA_ZA_FORM = (
    "FMLA (Z3.4S[<WV>, #0<OPTIONAL_COMMA_VGX2>], {<OPT_SPACE><ZN1>.4S-<ZN2>.4S<OPT_SPACE>}, Z2.4S)"
)
FMLA_PLAIN_FORM = "FMLA (V0.4S, V1.4S, V2.4S)"
LD1W_ZA_FORM = (
    "LD1W ({<OPT_SPACE><ZAT__0TO3><HV>.S[W1, #0]<OPT_SPACE>}, P0/Z, [<XNSP__3><OPTIONAL_XM__7>])"
)
LD1W_PLAIN_FORM = "LD1W ({<OPT_SPACE>Z0.S<OPT_SPACE>}, P0/Z, [<XNSP__3>#0])"


def _record(mnemonic: str, form: str, arch: str, arch_details=None) -> InstructionRecord:
    return InstructionRecord(
        mnemonic=mnemonic,
        form=form,
        summary="test",
        architecture=arch,
        arch_details=dict(arch_details or {}),
    )


def _intel_guide_row(mnemonic: str, form: str, latency: str) -> PerfRow:
    # Shape of the shipped intel-guide KMOVQ (K, R64) SKL row.
    return PerfRow(
        mnemonic=mnemonic,
        core="ICL",
        source="intel-guide",
        source_kind="modeled",
        source_version="3.6.9",
        architecture="x86",
        form=form,
        latency=latency,
        cpi="1",
        applies_to="form",
        citation_url=INTEL_GUIDE_URL,
    )


def _arm_swopt_row(mnemonic: str, latency: str, cpi: str, group: str, extra) -> PerfRow:
    # Shape of the shipped arm-swopt neoverse-v2 rows.
    return PerfRow(
        mnemonic=mnemonic,
        core="neoverse-v2",
        source="arm-swopt",
        source_kind="modeled",
        source_version="Issue 3.0 (r0p2)",
        architecture="arm",
        latency=latency,
        cpi=cpi,
        applies_to="mnemonic",
        citation_url=ARM_SWOPT_URL,
        extra_measurement={
            "pipelines": extra,
            "throughput_ipc": str(round(1 / float(cpi))),
            "instruction_group": group,
        },
    )


class TestM12EmptySlotFill:
    def test_lower_source_fills_empty_latency_slot(self):
        record = _record("KMOVQ", "KMOVQ (K, M64)", "x86", {"ICL": dict(ICL_KMOVQ_EMPTY)})
        merge_perf_rows([record], [_intel_guide_row("KMOVQ", "", "3")])
        entry = record.arch_details["ICL"]
        assert entry["latencies"] == [{"cycles": "3"}]
        # The top source keeps its measurement; the slot is not displaced.
        assert entry["measurement"]["ports"] == "1*p23+1*p5"
        assert "source" not in entry

    def test_populated_latency_is_not_replaced(self):
        record = _record("KMOVW", "KMOVW (K, R32)", "x86", {"ICL": dict(ICL_KMOVW_FILLED)})
        merge_perf_rows([record], [_intel_guide_row("KMOVW", "", "9")])
        assert record.arch_details["ICL"]["latencies"] == [ICL_KMOVW_FILLED["latencies"][0]]

    def test_single_source_core_stays_empty(self):
        record = _record("KMOVQ", "KMOVQ (K, M64)", "x86", {"ICL": dict(ICL_KMOVQ_EMPTY)})
        merge_perf_rows([record], [_intel_guide_row("KMOVQ", "", "")])
        assert record.arch_details["ICL"]["latencies"] == []


class TestM11NoMnemonicTimingOnZaForms:
    def _records(self):
        return [
            _record("FMLA", FMLA_PLAIN_FORM, "arm"),
            _record("FMLA", FMLA_ZA_FORM, "arm"),
            _record("LD1W", LD1W_PLAIN_FORM, "arm"),
            _record("LD1W", LD1W_ZA_FORM, "arm"),
        ]

    def test_mnemonic_rows_land_on_non_za_forms_only(self):
        rows = [
            _arm_swopt_row("FMLA", "4", "0.25", "ASIMD FP multiply accumulate", "V"),
            _arm_swopt_row("LD1W", "6", "0.333", "Contiguous load, scalar + imm", "L"),
        ]
        records = self._records()
        merge_perf_rows(records, rows)
        assert records[0].arch_details["neoverse-v2"]["latencies"] == [{"cycles": "4"}]
        assert records[2].arch_details["neoverse-v2"]["latencies"] == [{"cycles": "6"}]
        assert records[1].arch_details == {}
        assert records[3].arch_details == {}

    def test_form_row_for_a_za_form_still_attaches(self):
        za_fmla = _record("FMLA", FMLA_ZA_FORM, "arm")
        row = PerfRow(
            mnemonic="FMLA",
            core="neoverse-v2",
            source="arm-swopt",
            source_kind="modeled",
            source_version="test",
            architecture="arm",
            form=FMLA_ZA_FORM,
            latency="7",
        )
        merge_perf_rows([za_fmla], [row])
        assert za_fmla.arch_details["neoverse-v2"]["latencies"] == [{"cycles": "7"}]


def test_m8_warns_when_pinned_arch_has_no_rows(tmp_path, monkeypatch):
    add = _record(
        "ADD",
        "ADD (R64, R64)",
        "x86",
        {
            "SKL": {
                "doc": {},
                "iaca": [],
                "latencies": [{"cycles": "1", "start_op": "1", "target_op": "1"}],
                "measurement": {"TP_loop": "0.25", "ports": "1*p0156", "uops": "1"},
            }
        },
    )
    db = tmp_path / "issue-65.db"
    build_sqlite(
        Catalog(
            intrinsics=[],
            instructions=[add],
            sources=[
                SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")
            ],
            generated_at="2025-01-01",
        ),
        db,
    )
    monkeypatch.setattr("simdref.asm_identity._tool_path", lambda *args: None)
    monkeypatch.setattr(cli_mod, "ensure_runtime", lambda: None)
    monkeypatch.setattr(cli_mod, "SQLITE_PATH", db)
    runner = CliRunner()
    warned = runner.invoke(
        cli_mod.app,
        ["annotate", "-", "--arch", "gnr", "--format", "json"],
        input="addq %rax, %rbx\n",
    )
    assert warned.exit_code == 0, warned.output
    assert "warning: no timing data for GNR in this catalog" in warned.output
    quiet = runner.invoke(
        cli_mod.app,
        ["annotate", "-", "--arch", "skylake", "--format", "json"],
        input="addq %rax, %rbx\n",
    )
    assert quiet.exit_code == 0, quiet.output
    assert "warning: no timing data" not in quiet.output
