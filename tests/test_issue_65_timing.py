"""simdref#65: the runtime latency pick (M6, M7, M15).

Fixture rows copy the stored uops.info latency pairs and operand flags of
the catalog rows the issue names. llvm-mca-23 gives the expected figure.
"""

from __future__ import annotations

import json

from simdref import annotate as annotate_mod
from simdref.annotate import AnnotateOptions, annotate_stream, arch_latency, format_annotation
from simdref.display import measurement_rows
from simdref.models import InstructionRecord


class _FakeConn:
    pass


def _record(key, operands, arch, latencies, measurement=None):
    mnemonic = key.split(" ", 1)[0]
    return InstructionRecord(
        mnemonic=mnemonic,
        form=key,
        summary="",
        architecture="x86",
        isa=["BASE"],
        operand_details=operands,
        arch_details={
            arch: {"measurement": measurement or {"TP_loop": "0.50"}, "latencies": latencies}
        },
    )


def _reg(idx, r=True, w=False):
    op = {"idx": str(idx), "type": "reg"}
    if r:
        op["r"] = "1"
    if w:
        op["w"] = "1"
    return op


def _annotate_json(monkeypatch, record, line, arch):
    monkeypatch.setattr(annotate_mod, "lookup", lambda mnemonic, conn, isa=None: [record])
    opts = AnnotateOptions(arch=arch, isa="x86", fmt="json", docs=False)
    out = "".join(annotate_stream([f"\t{line}\n"], opts=opts, conn=_FakeConn()))
    return json.loads(out)[0]


# ADD (R64, M64) on SKL: reg->reg 1, mem->reg cycles_mem 3 (upper bound) and
# cycles_addr 6. llvm-mca-23 -mcpu=skylake: addq (%rax), %rbx latency 6.
ADD_R64_M64_SKL = _record(
    "ADD (R64, M64)",
    [
        _reg(1, w=True),
        {"idx": "2", "type": "mem", "r": "1"},
        {"idx": "3", "type": "flags", "w": "1"},
    ],
    "SKL",
    [
        {"start_op": "1", "target_op": "1", "cycles": "1"},
        {"start_op": "1", "target_op": "3", "cycles": "1"},
        {
            "start_op": "2",
            "target_op": "1",
            "cycles_mem": "3",
            "cycles_mem_is_upper_bound": "1",
            "cycles_addr": "6",
        },
    ],
)


def test_m6_memory_line_shows_the_memory_path(monkeypatch):
    rec = _annotate_json(monkeypatch, ADD_R64_M64_SKL, "add (%rax),%rbx", "SKL")
    assert rec["latency"] == 6.0
    assert "lat=6.0c" in rec["annotation"]
    assert "latency_min" not in rec


# POPCNT (R64, R64) on ICL: the pair from the write-only destination stores
# 0. llvm-mca-23 -mcpu=icelake-client: popcntq latency 3.
POPCNT_R64_R64_ICL = _record(
    "POPCNT (R64, R64)",
    [_reg(1, r=False, w=True), _reg(2), {"idx": "3", "type": "flags", "w": "1"}],
    "ICL",
    [
        {"start_op": "1", "target_op": "1", "cycles": "0"},
        {"start_op": "1", "target_op": "3", "cycles": "0"},
        {"start_op": "2", "target_op": "1", "cycles": "3"},
    ],
)


def test_m7_write_only_destination_pair_is_skipped():
    assert arch_latency(POPCNT_R64_R64_ICL, "ICL") == 3.0


# VCVTSI2SS (XMM, XMM, R32) on ZEN4: xmm pair 0, r32 pair 8 (upper bound).
def test_m7_zero_pair_beside_a_real_pair_does_not_win():
    rec = _record(
        "VCVTSI2SS (XMM, XMM, R32)",
        [_reg(1, r=False, w=True), _reg(2), _reg(3)],
        "ZEN4",
        [
            {"start_op": "2", "target_op": "1", "cycles": "0"},
            {"start_op": "3", "target_op": "1", "cycles": "8", "cycles_is_upper_bound": "1"},
        ],
    )
    assert arch_latency(rec, "ZEN4") == 8.0


# ADD (R64, I8) on ADL-P: pairs 0 and 1, uops 0 in the source.
# llvm-mca-23 -mcpu=alderlake: addq $7, %rbx uops 0, latency 1.
def test_m7_add_immediate_shows_one_cycle():
    rec = _record(
        "ADD (R64, I8)",
        [
            _reg(1, w=True),
            {"idx": "2", "type": "imm", "r": "1"},
            {"idx": "3", "type": "flags", "w": "1"},
        ],
        "ADL-P",
        [
            {"start_op": "1", "target_op": "1", "cycles": "0"},
            {"start_op": "1", "target_op": "3", "cycles": "1"},
        ],
        measurement={"TP_loop": "0.20", "uops": "0"},
    )
    out = format_annotation(
        rec, performance=True, docs=False, arch="ADL-P", agg="avg", include_modeled=False
    )
    assert out.startswith("lat=1.0c cpi=0.20 uops=0 ")


# DIV (R64) on SKL stores five min-max pairs: 35-90, 34-88, 32-87, 5-75, 5-73.
DIV_R64_SKL = _record(
    "DIV (R64)",
    [_reg(1), _reg(2, w=True), _reg(3, w=True), {"idx": "4", "type": "flags", "w": "1"}],
    "SKL",
    [
        {"start_op": "1", "target_op": "2", "min_cycles": "35", "max_cycles": "90"},
        {"start_op": "1", "target_op": "3", "min_cycles": "34", "max_cycles": "88"},
        {"start_op": "2", "target_op": "2", "min_cycles": "32", "max_cycles": "87"},
        {"start_op": "3", "target_op": "2", "min_cycles": "5", "max_cycles": "75"},
        {"start_op": "3", "target_op": "3", "min_cycles": "5", "max_cycles": "73"},
    ],
    measurement={"TP_loop": "21.06", "uops": "33", "uops_MS": "32"},
)


def test_m15_stored_range_is_shown_and_kept_in_json(monkeypatch):
    rec = _annotate_json(monkeypatch, DIV_R64_SKL, "div %rbx", "SKL")
    assert "lat=5.0-90.0c" in rec["annotation"]
    assert rec["latency"] == 90.0
    assert (rec["latency_min"], rec["latency_max"]) == (5.0, 90.0)


def test_hover_row_and_annotate_agree():
    # The hover perf table reads measurement_rows; annotate reads arch_latency.
    for record, arch, text in (
        (ADD_R64_M64_SKL, "SKL", "6"),
        (POPCNT_R64_R64_ICL, "ICL", "3"),
        (DIV_R64_SKL, "SKL", "5-90"),
    ):
        row = measurement_rows(record)[0]
        assert row["latency"] == text
        assert arch_latency(record, arch) == float(text.split("-")[-1])


# KMOVQ (M64, K): the only pair starts at the write-only memory operand.
# No read-start pair has data, so the stored value stays (never 0 or None).
def test_no_read_start_pair_keeps_the_stored_value():
    rec = _record(
        "KMOVQ (M64, K)",
        [{"idx": "1", "type": "mem", "w": "1"}, _reg(2)],
        "SKL",
        [{"start_op": "1", "target_op": "1", "cycles_addr": "5"}],
    )
    assert arch_latency(rec, "SKL") == 5.0
    assert measurement_rows(rec)[0]["latency"] == "5"
