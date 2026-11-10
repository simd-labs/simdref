"""Tests for the attribute-group handling in the CUDA header declaration parser."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "vendor_cuda_sources.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("vendor_cuda_sources", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Both reviewer lines, verbatim from __clang_cuda_device_functions.h.
ATTRIBUTED_HEADER = """
#define __DEVICE__ static __attribute__((always_inline, nothrow))

__DEVICE__ void __attribute__((overloadable)) __brkpt(void) {
  return __brkpt();
}

__DEVICE__ __attribute__((noreturn)) void __trap(void) { __builtin_trap(); }

__DEVICE__ unsigned __ballot_sync(unsigned member_mask, int predicate) {
  return __nvvm_vote_sync_ballot_pred(member_mask, predicate);
}
"""


class TestAttributeGroupsInDeclarations:
    def test_attribute_between_return_type_and_name(self):
        tool = _load_tool()
        funcs = tool.parse_functions_from_text(ATTRIBUTED_HEADER, "t.h")
        names = [f["name"] for f in funcs]
        assert "__brkpt" in names
        assert "__attribute__" not in names

    def test_attribute_before_return_type(self):
        tool = _load_tool()
        funcs = tool.parse_functions_from_text(ATTRIBUTED_HEADER, "t.h")
        names = [f["name"] for f in funcs]
        assert "__trap" in names
        trap = next(f for f in funcs if f["name"] == "__trap")
        assert trap["body"] == "__builtin_trap();"

    def test_plain_declaration_control(self):
        # A declaration with no attribute must keep parsing unchanged.
        tool = _load_tool()
        funcs = tool.parse_functions_from_text(ATTRIBUTED_HEADER, "t.h")
        names = [f["name"] for f in funcs]
        assert "__ballot_sync" in names

    def test_drop_attribute_groups_is_structural(self):
        tool = _load_tool()
        s = "static __attribute__((a, b(c))) void __attribute__((d(1))) f"
        assert tool.drop_attribute_groups(s) == "static  void  f"
