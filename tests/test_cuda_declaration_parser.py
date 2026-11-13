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
        from simdref.cuda_docs import drop_attribute_groups

        s = "static __attribute__((a, b(c))) void __attribute__((d(1))) f"
        assert drop_attribute_groups(s) == "static  void  f"


NORMALIZE_HEADER = """
#define __ZYX_DEVICE__ static __device__ __forceinline__

static inline __device__ float __zyx_norm_v1(float x) { return x; }
__device__ static inline float __zyx_norm_v1(float x) { return x; }
inline __device__ static float __zyx_norm_v1(float x) { return x; }
__ZYX_DEVICE__ float __zyx_norm_v1(float x) { return x; }
"""


class TestNormalizeDeclaration:
    """One normal form per declaration, whatever the qualifier order."""

    def test_three_orders_and_a_macro_qualifier_agree(self):
        from simdref.cuda_docs import declaration_qualifiers, normalize_declaration

        qualifiers = declaration_qualifiers(NORMALIZE_HEADER)
        want = (["float"], "__zyx_norm_v1", "float x")
        lines = [line for line in NORMALIZE_HEADER.splitlines() if "__zyx_norm_v1" in line]
        assert len(lines) == 4
        for line in lines:
            assert normalize_declaration(line, qualifiers) == want

    def test_no_return_type_is_not_a_declaration(self):
        from simdref.cuda_docs import declaration_qualifiers, normalize_declaration

        qualifiers = declaration_qualifiers(NORMALIZE_HEADER)
        assert normalize_declaration("__zyx_norm_v1(x)", qualifiers) == ([], "__zyx_norm_v1", "x")

    def test_statement_shapes_declare_nothing(self):
        from simdref.cuda_docs import declaration_qualifiers, normalize_declaration

        qualifiers = declaration_qualifiers(NORMALIZE_HEADER)
        for line in ("if (x) {", "return f(x);", "block.sync();", "x = f(a);"):
            assert normalize_declaration(line, qualifiers) is None, line
