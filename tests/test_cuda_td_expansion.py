"""Tests for the structural TableGen expansion in tools/vendor_cuda_sources.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "vendor_cuda_sources.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("vendor_cuda_sources", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Small .td slice in the shapes of NVPTXIntrinsics.td: a foreach with a
# computed !cast name and computed asm string, and a defm over a multiclass
# whose intrinsic comes in as a class argument.
TD_SNIPPET = """
class RegTyInfo<string ptx_type> {
  string PtxType = ptx_type;
}
def I1RT : RegTyInfo<"pred">;

foreach sync = [false, true] in {
  foreach mode = ["idx", "bfly"] in {
    defvar Intr = !cast<Intrinsic>("int_nvvm_shfl_" # !if(sync, "sync_", "") # mode # "_i32");
    def : Inst<"shfl." # !if(sync, "sync.", "") # mode # ".b32", [Intr]>;
  }
}

multiclass VOTE<string mode, RegTyInfo t, Intrinsic op> {
  def : Inst<"vote." # mode # "." # t.PtxType, [(op i1:$pred)]>;
}
defm VOTE_ALL : VOTE<"all", I1RT, int_nvvm_vote_all>;

multiclass SYNC<string mode, Intrinsic op> {
  def : Inst<"barrier.sync." # mode, [op]>;
}
defm SYNC_ALL : SYNC<"all", int_nvvm_barrier_cta_sync_all>;

class WRAP<string variant, Intrinsic intr>
  : Inst<"wrap." # variant, [intr]>;
def wrapped : WRAP<"go", int_nvvm_wrapped>;
"""

BUILTINS_SNIPPET = """
def __nvvm_shfl_sync_idx_i32 : NVPTXBuiltin<"int(int)">;
def __nvvm_vote_all : NVPTXBuiltin<"int(int)">;
def __nvvm_wrapped : NVPTXBuiltin<"int(int)">;
def __nvvm_bar0_and : NVPTXBuiltin<"int(int)">;
def __nvvm_barrier_sync : NVPTXBuiltin<"void(unsigned int)">;
"""

ALIAS_SNIPPET = """
//   * llvm.nvvm.barrier.sync        --> llvm.nvvm.barrier.cta.sync.all(x)
"""


class TestRecordExpansion:
    def test_foreach_builds_names_and_asm_strings(self):
        tool = _load_tool()
        table, _ = tool.td_record_links(TD_SNIPPET)
        assert table["int_nvvm_shfl_idx_i32"] == {"shfl.idx.b32"}
        assert table["int_nvvm_shfl_sync_idx_i32"] == {"shfl.sync.idx.b32"}
        assert table["int_nvvm_shfl_sync_bfly_i32"] == {"shfl.sync.bfly.b32"}

    def test_defm_binds_intrinsic_argument(self):
        tool = _load_tool()
        table, _ = tool.td_record_links(TD_SNIPPET)
        assert table["int_nvvm_vote_all"] == {"vote.all.pred"}

    def test_class_body_assembly_string_resolves(self):
        tool = _load_tool()
        table, _ = tool.td_record_links(TD_SNIPPET)
        assert table["int_nvvm_wrapped"] == {"wrap.go"}


class TestBuiltinMapping:
    def test_records_resolve_without_name_tables(self):
        tool = _load_tool()
        mapping = tool.build_builtin_to_ptx(BUILTINS_SNIPPET, ALIAS_SNIPPET, TD_SNIPPET, "")
        assert mapping["__nvvm_shfl_sync_idx_i32"] == ["shfl.sync.idx.b32"]
        assert mapping["__nvvm_vote_all"] == ["vote.all.pred"]

    def test_unresolved_builtin_stays_unlinked(self):
        # __nvvm_bar0_and resembles bar.red.and.pred by name only; no record
        # chain names it, so it must not be mapped.
        tool = _load_tool()
        mapping = tool.build_builtin_to_ptx(BUILTINS_SNIPPET, ALIAS_SNIPPET, TD_SNIPPET, "")
        assert "__nvvm_bar0_and" not in mapping

    def test_alias_table_renames_retired_intrinsics(self):
        tool = _load_tool()
        mapping = tool.build_builtin_to_ptx(BUILTINS_SNIPPET, ALIAS_SNIPPET, TD_SNIPPET, "")
        assert mapping["__nvvm_barrier_sync"] == ["barrier.sync.all"]

    def test_asm_opcode_extraction(self):
        tool = _load_tool()
        assert tool._asm_opcode("atom${sem:sem}${scope:scope}.add") == "atom.add"
        assert tool._asm_opcode("mov.u32 \\t$d, %tid.x;") == "mov.u32"
        assert (
            tool._asm_opcode("ld.global.nc.${Sign:sign}$fromWidth \\t$d, [$s];") == "ld.global.nc"
        )
        assert tool._asm_opcode("shfl.sync.idx.b32") == "shfl.sync.idx.b32"
        assert tool._asm_opcode("int_nvvm_shfl_sync_idx_i32") is None
