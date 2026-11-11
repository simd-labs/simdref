"""Tests for feature floors, calls, arch guards and special register drops.

All builtin and function names in the fixtures are invented.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "vendor_cuda_sources.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("vendor_cuda_sources", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Invented builtin defs in the four NVPTXBuiltin class shapes. One def
# wraps its floor args across two lines.
TD_FLOORS = """
def __nvvm_zq_plain : NVPTXBuiltin<"int(int)">;
def __nvvm_zq_smonly : NVPTXBuiltinSM<"int(unsigned int)", SM_70>;
def __nvvm_zq_ptxonly : NVPTXBuiltinPTX<"int(int, int)", PTX60>;
def __nvvm_zq_both : NVPTXBuiltinSMAndPTX<"float(float)", SM_90, PTX78>;
def __nvvm_zq_wrapped : NVPTXBuiltinSMAndPTX<"int(int, int)",
                       SM_86, PTX72>;
def __nvvm_zq_v100 : NVPTXBuiltinSM<"int(int)", SM_100a>;
"""

# Invented functions under nested __CUDA_ARCH__ guards. Each name is
# defined before any other use, so the definition site carries the guard.
GUARDED_HEADER = """
#if __CUDA_ARCH__ >= 300
inline int __zq_alpha(int v) { return v; }
#if defined(FOO) && __CUDA_ARCH__ >= 700
inline int __zq_beta(int v) { return v; }
#endif
inline int __zq_gamma(int v) { return v; }
#endif
#if __CUDA_ARCH__ > 320
inline int __zq_delta(int v) { return v; }
#endif
inline int __zq_epsilon(int v) { return v; }
"""


class TestBuiltinFeatureFloors:
    def test_all_four_class_shapes(self):
        tool = _load_tool()
        floors = tool.builtin_feature_floors(TD_FLOORS)
        # No floor on the plain shape.
        assert "__nvvm_zq_plain" not in floors
        assert floors["__nvvm_zq_smonly"] == ("sm_70", "")
        assert floors["__nvvm_zq_ptxonly"] == ("", "6.0")
        assert floors["__nvvm_zq_both"] == ("sm_90", "7.8")

    def test_wrapped_defs_and_long_sm(self):
        tool = _load_tool()
        floors = tool.builtin_feature_floors(TD_FLOORS)
        assert floors["__nvvm_zq_wrapped"] == ("sm_86", "7.2")
        assert floors["__nvvm_zq_v100"] == ("sm_100a", "")
        assert len(floors) == 5


class TestCalls:
    def test_body_calls(self):
        tool = _load_tool()
        body = (
            "int t = __zq_wrap_b(v); if (t) t = __zq_wrap_a(t); "
            "return __nvvm_zq_thing(t) + __nv_zq_helper(t) + my_local + 1;"
        )
        names = {"__zq_wrap_a", "__zq_wrap_b"}
        calls = tool.extract_calls(body, names, "__zq_wrap_a")
        # Self drops out. A sibling function and the two builtin
        # prefixes stay. Plain locals do not.
        assert "__zq_wrap_a" not in calls
        assert "my_local" not in calls
        assert calls == ["__nv_zq_helper", "__nvvm_zq_thing", "__zq_wrap_b"]


class TestArchGuard:
    def test_nested_if_uses_innermost_guard(self):
        tool = _load_tool()
        texts = [GUARDED_HEADER]
        assert tool.arch_guard_for_name(texts, "__zq_alpha") == "300"
        assert tool.arch_guard_for_name(texts, "__zq_beta") == "700"
        assert tool.arch_guard_for_name(texts, "__zq_gamma") == "300"

    def test_gt_form_and_no_guard(self):
        tool = _load_tool()
        texts = [GUARDED_HEADER]
        assert tool.arch_guard_for_name(texts, "__zq_delta") == "320"
        assert tool.arch_guard_for_name(texts, "__zq_epsilon") == ""


class TestMinFloors:
    FLOORS = {
        "__nvvm_zq_lo": ("sm_60", ""),
        "__nvvm_zq_hi": ("sm_75", "7.0"),
        "__nvvm_zq_ptx": ("", "6.3"),
    }

    def test_transitive_max_wins(self):
        tool = _load_tool()
        calls_by_name = {
            "__zq_mid": ["__nvvm_zq_lo", "__zq_leaf"],
            "__zq_leaf": ["__nvvm_zq_hi", "__nvvm_zq_ptx"],
        }
        got = tool.min_floors(["__zq_mid"], calls_by_name, self.FLOORS, "")
        assert got == ("sm_75", "7.0")

    def test_guard_fallback(self):
        tool = _load_tool()
        # No reachable builtin has an sm floor. The guard is the floor.
        got = tool.min_floors(["__nvvm_zq_ptx"], {}, self.FLOORS, "320")
        assert got == ("sm_32", "6.3")
        assert tool.min_floors([], {}, self.FLOORS, "700") == ("sm_70", "")


class TestSpecialRegisters:
    SREG_BUILTINS = {
        "__nvvm_read_ptx_sreg_zbble_x",
        "__nvvm_read_ptx_sreg_zbble_y",
        "__nvvm_read_ptx_sreg_gnarl",
        "__nvvm_read_ptx_sreg_qux_w",
        "__nvvm_unrelated_thing",
    }

    def test_tokens_from_def_names(self):
        tool = _load_tool()
        tokens = tool.special_register_tokens(self.SREG_BUILTINS)
        # The _x/_y/_w dimension suffixes drop. The plain name stays.
        assert tokens == {"zbble", "gnarl", "qux"}

    def test_drop_sreg_refs_keeps_real_instruction(self):
        tool = _load_tool()
        tokens = tool.special_register_tokens(self.SREG_BUILTINS)
        kept, dropped = tool.drop_special_register_refs(
            ["zbble.thing.b32", "shfl.sync.idx.b32", "gnarl", "qux.7"], tokens
        )
        assert kept == ["shfl.sync.idx.b32"]
        assert sorted(dropped) == ["gnarl", "qux.7", "zbble.thing.b32"]
