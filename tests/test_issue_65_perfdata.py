"""Issue #65 perf-data defects: M12, M11 and M8.

M12: perf fields merge one by one and each carried field names its own
source, so a borrowed latency never rides along with the donor's ports or
uops. M11: a mnemonic-level row never spreads onto SME-only forms (ZA
tiles, ZA arrays, ZT0 tables). M8: pinning an arch with zero perf rows
must warn on stderr. Fixture entry shapes copy rows of the shipped
catalog (catalog.db, read only).
"""

from __future__ import annotations

import unittest.mock

from typer.testing import CliRunner

from simdref import cli as cli_mod
from simdref import perf
from simdref.annotate import arch_perf, arch_perf_tag
from simdref.display import measurement_rows, provenance_lines
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.perf_sources.merge import PerfRow, merge_perf_rows
from simdref.queries import filter_arch_details_by_source_kind
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

# Real uops.info slot: KMOVQ (K, M64) on ZEN4 carries uops and ports but
# no latency. The shipped instlatx64 ZEN4 row supplies the latency (#65).
ZEN4_KMOVQ_EMPTY = {
    "doc": {"TP": "1.00", "latency": "2", "ports": "FP0/1 ", "uops": "2 "},
    "iaca": [],
    "latencies": [],
    "measurement": {
        "TP_loop": "1.00",
        "TP_ports": "0.50",
        "TP_unrolled": "1.00",
        "ports": "1*FP01",
        "uops": "2",
    },
}

# The ZEN4 slot shape with its CPI key dropped: a slot missing two fields
# borrows each from its own donor.
ZEN4_KMOVQ_NO_CPI = {
    **ZEN4_KMOVQ_EMPTY,
    "measurement": {k: v for k, v in ZEN4_KMOVQ_EMPTY["measurement"].items() if k != "TP_loop"},
}

INTEL_GUIDE_URL = "https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html"
ARM_SWOPT_URL = "https://documentation-service.arm.com/static/668bc0a369e89f01e39c4668"
INSTLATX64_URL = "https://github.com/InstLatx64/InstLatX64"

FMLA_ZA_FORM = (
    "FMLA (Z3.4S[<WV>, #0<OPTIONAL_COMMA_VGX2>], {<OPT_SPACE><ZN1>.4S-<ZN2>.4S<OPT_SPACE>}, Z2.4S)"
)
FMLA_PLAIN_FORM = "FMLA (V0.4S, V1.4S, V2.4S)"
LD1W_ZA_FORM = (
    "LD1W ({<OPT_SPACE><ZAT__0TO3><HV>.S[W1, #0]<OPT_SPACE>}, P0/Z, [<XNSP__3><OPTIONAL_XM__7>])"
)
LD1W_PLAIN_FORM = "LD1W ({<OPT_SPACE>Z0.S<OPT_SPACE>}, P0/Z, [<XNSP__3>#0])"
# Real SME-only shapes: ZT0 table forms carry no ZA operand at all, so the
# removed ZA regex never caught them (#65 M11).
LDR_ZT0_FORM = "LDR (ZT0, [<XNSP__3>])"
LUTI2_ZT0_FORM = "LUTI2 (Z0.4S, ZT0, Z1[<INDEX__28>])"
# Real SME+SVE shape: not SME-only, so mnemonic rows still land on it.
ADD_SME_SVE_FORM = "ADD ({<OPT_SPACE><ZDN1>.4S-<ZDN2>.4S<OPT_SPACE>}, {<OPT_SPACE><ZDN1>.4S-<ZDN2>.4S<OPT_SPACE>}, Z2.4S)"


def _record(
    mnemonic: str, form: str, arch: str, arch_details=None, isa: tuple[str, ...] = ()
) -> InstructionRecord:
    return InstructionRecord(
        mnemonic=mnemonic,
        form=form,
        summary="test",
        architecture=arch,
        arch_details=dict(arch_details or {}),
        isa=list(isa),
    )


def _intel_guide_row(
    mnemonic: str, form: str, latency: str, core: str = "ICL", cpi: str = "1"
) -> PerfRow:
    # Shape of the shipped intel-guide KMOVQ (K, R64) SKL row.
    return PerfRow(
        mnemonic=mnemonic,
        core=core,
        source="intel-guide",
        source_kind="modeled",
        source_version="3.6.9",
        architecture="x86",
        form=form,
        latency=latency,
        cpi=cpi,
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


def _instlatx64_row(mnemonic: str, core: str, latency: str) -> PerfRow:
    # Shape of the shipped instlatx64 ZEN4 rows.
    return PerfRow(
        mnemonic=mnemonic,
        core=core,
        source="instlatx64",
        source_kind="measured",
        source_version="aida_bench64.dll build: 4.3.758.0 Jun 9 2017",
        architecture="x86",
        form="",
        latency=latency,
        applies_to="mnemonic",
        citation_url=INSTLATX64_URL,
    )


def _amd_swopt_row(mnemonic: str, core: str, latency: str, cpi: str) -> PerfRow:
    # Shape of the shipped amd-swopt Zen rows.
    return PerfRow(
        mnemonic=mnemonic,
        core=core,
        source="amd-swopt",
        source_kind="modeled",
        source_version="47180",
        architecture="x86",
        form="",
        latency=latency,
        cpi=cpi,
        applies_to="mnemonic",
    )


class TestFieldWiseMerge:
    def test_borrowed_latency_keeps_the_owners_ports_and_uops(self):
        # Real shape: the ICL KMOVQ slot has uops, ports and CPI but no
        # latency; the intel-guide row lends its latency alone.
        record = _record("KMOVQ", "KMOVQ (K, M64)", "x86", {"ICL": dict(ICL_KMOVQ_EMPTY)})
        merge_perf_rows([record], [_intel_guide_row("KMOVQ", "", "3")])
        entry = record.arch_details["ICL"]
        assert entry["latencies"] == [{"cycles": "3"}]
        assert entry["latency_source"] == "intel-guide"
        # The owner keeps the fields it has; no donor label on them.
        assert entry["measurement"]["TP_loop"] == "1.00"
        assert entry["measurement"]["ports"] == "1*p23+1*p5"
        assert entry["measurement"]["uops"] == "2"
        assert "cpi_source" not in entry
        assert "ports_source" not in entry
        assert "uops_source" not in entry

    def test_each_borrowed_field_names_its_own_donor(self):
        # Real shape: the ZEN4 KMOVQ slot without its CPI. Two donors fill
        # two fields, and each field keeps its own label.
        record = _record("KMOVQ", "KMOVQ (K, M64)", "x86", {"ZEN4": dict(ZEN4_KMOVQ_NO_CPI)})
        merge_perf_rows(
            [record],
            [
                _intel_guide_row("KMOVQ", "", "3", core="ZEN4", cpi=""),
                _amd_swopt_row("KMOVQ", "ZEN4", "2", "0.5"),
            ],
        )
        entry = record.arch_details["ZEN4"]
        assert entry["latencies"] == [{"cycles": "3"}]
        assert entry["latency_source"] == "intel-guide"
        assert entry["measurement"]["TP_loop"] == "0.5"
        assert entry["cpi_source"] == "amd-swopt"
        # Ports and uops never gain a donor label.
        assert "ports_source" not in entry
        assert "uops_source" not in entry

    def test_zen4_kmovq_latency_labels_the_donor_and_ports_the_owner(self):
        # The probe shape: a ZEN4 kmovq line whose latency comes from
        # instlatx64 while uops and ports stay uops.info.
        record = _record("KMOVQ", "KMOVQ (K, M64)", "x86", {"ZEN4": dict(ZEN4_KMOVQ_EMPTY)})
        merge_perf_rows([record], [_instlatx64_row("KMOVQ", "ZEN4", "1.0")])
        entry = record.arch_details["ZEN4"]
        assert entry["latencies"] == [{"cycles": "1.0"}]
        assert entry["latency_source"] == "instlatx64"
        assert entry["measurement"]["ports"] == "1*FP01"
        assert entry["measurement"]["uops"] == "2"

        lat, cpi, kind = arch_perf(record, "ZEN4")
        assert lat == 1.0 and cpi == 1.0
        tag = arch_perf_tag("ZEN4", lat, cpi, kind)
        assert "latency from instlatx64" in tag
        assert "owner uops.info" in tag

        rows = measurement_rows(record)
        assert len(rows) == 1
        assert rows[0]["latency"] == "1.0"
        assert rows[0]["latency_source"] == "instlatx64"
        assert rows[0]["source_tool"] == "uops.info"
        assert rows[0]["ports"] == "1*FP01"
        assert "instlatx64" in " ".join(provenance_lines(rows))

    def test_llm_and_filter_match_both_kinds_of_a_borrowed_slot(self):
        record = _record("KMOVQ", "KMOVQ (K, M64)", "x86", {"ICL": dict(ICL_KMOVQ_EMPTY)})
        merge_perf_rows([record], [_intel_guide_row("KMOVQ", "", "3")])
        entry = record.arch_details["ICL"]

        timing = cli_mod._llm_timing(record.arch_details, perf.record_iclass(record))
        assert timing["ICL"]["latency_source"] == "intel-guide"
        assert timing["ICL"]["source_kind"] == "mixed"

        payload = {"arch_details": {"ICL": entry}}
        assert cli_mod._record_has_source_kind(payload, "measured")
        assert cli_mod._record_has_source_kind(payload, "modeled")
        assert (
            filter_arch_details_by_source_kind({"ICL": entry}, "modeled") == payload["arch_details"]
        )
        assert (
            filter_arch_details_by_source_kind({"ICL": entry}, "measured")
            == payload["arch_details"]
        )


class TestM12EmptySlotFill:
    def test_lower_source_fills_empty_latency_slot(self):
        record = _record("KMOVQ", "KMOVQ (K, M64)", "x86", {"ICL": dict(ICL_KMOVQ_EMPTY)})
        merge_perf_rows([record], [_intel_guide_row("KMOVQ", "", "3")])
        entry = record.arch_details["ICL"]
        assert entry["latencies"] == [{"cycles": "3"}]
        # The filled latency names its donor; the top source keeps its
        # measurement and the slot is not displaced.
        assert entry["latency_source"] == "intel-guide"
        assert entry["measurement"]["ports"] == "1*p23+1*p5"
        assert "source" not in entry

    def test_populated_latency_is_not_replaced(self):
        record = _record("KMOVW", "KMOVW (K, R32)", "x86", {"ICL": dict(ICL_KMOVW_FILLED)})
        merge_perf_rows([record], [_intel_guide_row("KMOVW", "", "9")])
        assert record.arch_details["ICL"]["latencies"] == [ICL_KMOVW_FILLED["latencies"][0]]
        # The owner's latency keeps the owner's label.
        assert "latency_source" not in record.arch_details["ICL"]

    def test_single_source_core_stays_empty(self):
        record = _record("KMOVQ", "KMOVQ (K, M64)", "x86", {"ICL": dict(ICL_KMOVQ_EMPTY)})
        merge_perf_rows([record], [_intel_guide_row("KMOVQ", "", "")])
        assert record.arch_details["ICL"]["latencies"] == []


class TestM11NoMnemonicTimingOnSmeOnlyForms:
    def _records(self):
        return [
            _record("FMLA", FMLA_PLAIN_FORM, "arm", isa=("SVE", "ASIMD")),
            _record("FMLA", FMLA_ZA_FORM, "arm", isa=("SME",)),
            _record("LD1W", LD1W_PLAIN_FORM, "arm", isa=("SVE",)),
            _record("LD1W", LD1W_ZA_FORM, "arm", isa=("SME",)),
            _record("LDR", LDR_ZT0_FORM, "arm", isa=("SME",)),
            _record("LUTI2", LUTI2_ZT0_FORM, "arm", isa=("SME",)),
        ]

    def test_mnemonic_rows_never_land_on_sme_only_forms(self):
        rows = [
            _arm_swopt_row("FMLA", "4", "0.25", "ASIMD FP multiply accumulate", "V"),
            _arm_swopt_row("LD1W", "6", "0.333", "Contiguous load, scalar + imm", "L"),
            _arm_swopt_row("LDR", "5", "0.5", "Load/store", "L"),
            _arm_swopt_row("LUTI2", "3", "0.333", "SVE lookup table", "L"),
        ]
        records = self._records()
        merge_perf_rows(records, rows)
        assert records[0].arch_details["neoverse-v2"]["latencies"] == [{"cycles": "4"}]
        assert records[2].arch_details["neoverse-v2"]["latencies"] == [{"cycles": "6"}]
        assert records[1].arch_details == {}
        assert records[3].arch_details == {}
        assert records[4].arch_details == {}
        assert records[5].arch_details == {}

    def test_mnemonic_rows_still_land_on_sme_sve_forms(self):
        # SME+SVE forms carry an SVE tag, so they are not SME-only.
        record = _record("ADD", ADD_SME_SVE_FORM, "arm", isa=("SME", "SVE"))
        merge_perf_rows([record], [_arm_swopt_row("ADD", "1", "0.5", "SVE arithmetic", "V")])
        assert record.arch_details["neoverse-v2"]["latencies"] == [{"cycles": "1"}]

    def test_form_row_for_a_za_form_still_attaches(self):
        za_fmla = _record("FMLA", FMLA_ZA_FORM, "arm", isa=("SME",))
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
