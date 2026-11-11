"""Test for the interned ``sm`` column in the intrinsics search shard.

The shard carries one ``sm`` value per row: the row's floor, or null when the
row has no floor. The test decodes the column from a tiny catalog with a CUDA
row (a floor), a PTX row (an ``sm`` list), and an x86 row (none).
"""

from __future__ import annotations

from simdref.export import _columnar_intrinsics, _columnar_instructions
from simdref.filters import record_sm_floor


def _entry(name, isa, arch, desc, ins, sm=None):
    entry = {
        "name": name,
        "subtitle": desc[:80],
        "architecture": arch,
        "isa": isa,
        "lat": "-",
        "cpi": "-",
        "display_architecture": arch,
        "display_isa": isa,
        "isa_families": [],
        "isa_subs": [],
        "search_fields": [name, desc, isa, ins],
    }
    if sm:
        entry["sm"] = sm
    return entry


def _decode(res, key):
    values = res[key]
    return [None if i < 0 else values[i] for i in res["cols"][key]]


def test_intrinsics_shard_decodes_sm_floors():
    cuda_sm = record_sm_floor({"metadata": {"min_sm": "sm_70"}})
    ptx_sm = record_sm_floor({"metadata": {"sm": "sm_53, sm_80"}})
    entries = [
        _entry("__zyx_sync_v1", "CUDA", "cuda", "Sync the warp.", "sync", cuda_sm),
        _entry("zyx.sync.b32", "PTX", "ptx", "Sync the lanes.", "d, a, b", ptx_sm),
        _entry("zyx", "SSE2", "x86", "Do the work.", "y, x"),
    ]
    res = _columnar_intrinsics(entries)
    assert _decode(res, "sm") == ["sm_70", "sm_53", None]
    assert _decode(res, "cat") == [None, None, None]


def _instr_entry(key, mnemonic, form, arch, summary, isa, sm=None):
    entry = {
        "key": key,
        "mnemonic": mnemonic,
        "form": form,
        "architecture": arch,
        "summary": summary[:80],
        "isa": isa,
        "linked_intrinsics": [],
        "lat": "-",
        "cpi": "-",
        "display_architecture": arch,
        "display_key": form,
        "display_form": form,
        "display_mnemonic": mnemonic,
        "display_isa": isa,
        "isa_families": [],
        "isa_subs": [],
        "search_fields": [mnemonic, form, form, summary, isa],
    }
    if sm:
        entry["sm"] = sm
    return entry


def test_instructions_shard_decodes_sm_floors():
    ptx_sm = record_sm_floor({"metadata": {"sm": "sm_53, sm_80"}})
    entries = [
        _instr_entry(
            "ptx:sync.b32", "sync.b32", "sync.b32", "ptx", "Sync the lanes.", "PTX", ptx_sm
        ),
        _instr_entry("x86:add", "add", "add (r, r)", "x86", "Add two regs.", "SSE2"),
        _instr_entry("arm:add", "add", "add (r, r, r)", "arm", "Add three regs.", "NEON"),
    ]
    res = _columnar_instructions(entries)
    assert _decode(res, "sm") == ["sm_53", None, None]
