import contextlib
import dataclasses
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATE_SOURCES_PATH = REPO_ROOT / "tools" / "validate_sources.py"
UOPS_VARIANT_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "uops-variant-assignment.json"

UOPS_FIXTURE = """<?xml version="1.0"?>
<root>
  <extension name="RAO_INT">
    <instruction name="AADD" asm="{evex} AADD" string="AADD_EVEX (M32, R32)" iform="AADD_MEMi32_GPR32i32_APX" iclass="AADD" category="ATOMIC" extension="APX_F" summary="Atomic add 32-bit.">
      <operand idx="0" type="mem" width="32"/>
      <operand idx="1" type="reg" width="32" name="r32"/>
    </instruction>
    <instruction name="AADD" asm="{evex} AADD" string="AADD_EVEX (M32, R32)" iform="AADD_MEMi32_GPR32i32_APX_NF" iclass="AADD" category="ATOMIC" extension="APX_F" summary="Atomic add 32-bit.">
      <operand idx="0" type="mem" width="32"/>
      <operand idx="1" type="reg" width="32" name="r32"/>
    </instruction>
  </extension>
</root>
"""

UOPS_MEASUREMENT_FIXTURE = """<?xml version="1.0"?>
<root>
  <extension name="AVX">
    <instruction name="VFOO" asm="VFOO" string="VFOO (YMM, YMM)" iform="VFOO_YMMqq_YMMqq" iclass="VFOO" category="XFER" extension="AVX" summary="Foo.">
      <operand idx="0" type="reg" width="256" name="ymm"/>
      <operand idx="1" type="reg" width="256" name="ymm"/>
      <architecture name="MTL-E">
        <measurement TP_loop="0.17" uops="1" ports="1*p1">
          <latency cycles="1" start_op="2" target_op="1"/>
        </measurement>
      </architecture>
    </instruction>
  </extension>
</root>
"""

UOPS_SIBLING_IFORM_FIXTURE = """<?xml version="1.0"?>
<root>
  <extension name="AVX512">
    <instruction name="VBAR" asm="VBAR" string="VBAR (XMM, K)" iform="VBAR_XMMdq_MASKmskw_AVX512" iclass="VBAR" category="XFER" extension="AVX512VL" summary="Bar.">
      <operand idx="0" type="reg" width="128" name="xmm"/>
      <operand idx="1" type="reg" width="8" name="k"/>
      <architecture name="MTL-E">
        <measurement TP_loop="0.17" uops="1" ports="1*p1">
          <latency cycles="1" start_op="2" target_op="1"/>
        </measurement>
      </architecture>
    </instruction>
    <instruction name="VBAR" asm="VBAR" string="VBAR (XMM, K, M128)" iform="VBAR_XMMdq_MASKmskw_AVX512" iclass="VBAR" category="XFER" extension="AVX512VL" summary="Bar.">
      <operand idx="0" type="reg" width="128" name="xmm"/>
      <operand idx="1" type="reg" width="8" name="k"/>
      <operand idx="2" type="mem" width="128"/>
      <architecture name="MTL-E">
        <measurement TP_loop="0.17" uops="1" ports="1*p1">
          <latency cycles="1" start_op="2" target_op="1"/>
        </measurement>
      </architecture>
    </instruction>
  </extension>
</root>
"""


UOPS_VARIANT_SIBLING_FIXTURE = """<?xml version="1.0"?>
<root>
  <extension name="AVX">
    <instruction name="VMOVQ" asm="VMOVQ" string="VMOVQ (XMM, XMM)" iform="VMOVQ_XMMdq_XMMdq_1" iclass="VMOVQ" category="XFER" extension="AVX" summary="Move Qword.">
      <operand idx="0" type="reg" width="128" name="xmm"/>
      <operand idx="1" type="reg" width="128" name="xmm"/>
      <architecture name="MTL-E">
        <measurement TP_loop="0.17" uops="1" ports="1*p1">
          <latency cycles="1" start_op="2" target_op="1"/>
        </measurement>
      </architecture>
    </instruction>
    <instruction name="VMOVQ" asm="VMOVQ" string="VMOVQ_D6 (XMM, XMM)" iform="VMOVQ_XMMdq_XMMdq_2" iclass="VMOVQ" category="XFER" extension="AVX" summary="Move Qword.">
      <operand idx="0" type="reg" width="128" name="xmm"/>
      <operand idx="1" type="reg" width="128" name="xmm"/>
      <architecture name="MTL-E">
        <measurement TP_loop="0.17" uops="1" ports="1*p1">
          <latency cycles="1" start_op="2" target_op="1"/>
        </measurement>
      </architecture>
    </instruction>
  </extension>
</root>
"""

UOPS_SWAP_FIXTURE = """<?xml version="1.0"?>
<root>
  <extension name="AVX">
    <instruction name="VMOVAPD" asm="VMOVAPD" string="VMOVAPD_28 (YMM, YMM)" iform="VMOVAPD_28_YMMqq_YMMqq" iclass="VMOVAPD" category="XFER" extension="AVX" summary="Move packed doubles.">
      <operand idx="0" type="reg" width="256" name="ymm"/>
      <operand idx="1" type="reg" width="256" name="ymm"/>
      <architecture name="MTL-E">
        <measurement TP_loop="0.17" uops="1" ports="1*p1">
          <latency cycles="1" start_op="2" target_op="1"/>
        </measurement>
      </architecture>
    </instruction>
    <instruction name="VMOVAPD" asm="VMOVAPD" string="VMOVAPD_29 (YMM, YMM)" iform="VMOVAPD_29_YMMqq_YMMqq" iclass="VMOVAPD" category="XFER" extension="AVX" summary="Move packed doubles.">
      <operand idx="0" type="reg" width="256" name="ymm"/>
      <operand idx="1" type="reg" width="256" name="ymm"/>
      <architecture name="MTL-E">
        <measurement TP_loop="0.50" uops="2" ports="1*p0">
          <latency cycles="2" start_op="2" target_op="1"/>
        </measurement>
      </architecture>
    </instruction>
  </extension>
</root>
"""

UOPS_COLLISION_FIXTURE = """<?xml version="1.0"?>
<root>
  <extension name="AVX">
    <instruction name="VFOO" asm="VFOO" string="VFOO (YMM, YMM)" iform="VFOO_YMMqq_YMMqq_A" iclass="VFOO" category="XFER" extension="AVX" summary="Foo.">
      <operand idx="0" type="reg" width="256" name="ymm"/>
      <operand idx="1" type="reg" width="256" name="ymm"/>
      <architecture name="MTL-E">
        <measurement TP_loop="0.17" uops="1" ports="1*p1">
          <latency cycles="1" start_op="2" target_op="1"/>
        </measurement>
      </architecture>
    </instruction>
    <instruction name="VFOO" asm="VFOO" string="VFOO (YMM, YMM)" iform="VFOO_YMMqq_YMMqq_B" iclass="VFOO" category="XFER" extension="AVX" summary="Foo.">
      <operand idx="0" type="reg" width="256" name="ymm"/>
      <operand idx="1" type="reg" width="256" name="ymm"/>
      <architecture name="MTL-E">
        <measurement TP_loop="0.50" uops="2" ports="1*p0">
          <latency cycles="2" start_op="2" target_op="1"/>
        </measurement>
      </architecture>
    </instruction>
  </extension>
</root>
"""


def _load_rows(module: Any, xml_path: Path) -> Any:
    return module.parse_uops_xml(xml_path)


def _variant_name_assignment(rows: Any, assignments: dict[int, str]) -> dict[str, list]:
    """Pin which raw record each variant row name and its plain sibling carries.

    A variant row name carries a raw-string suffix such as VMOVAPD_29. The
    plain-named sibling is the row named by the normalized form. Each entry
    maps the row name to its iform and the ordinals of the raw records the
    parser assigned to that row.
    """
    from simdref.ingest_catalog import _normalize_uops_form

    by_form = {row.form: row for row in rows}
    ordinals: dict[str, list[int]] = {}
    for node_index, key in assignments.items():
        ordinals.setdefault(key, []).append(node_index)
    pinned: dict[str, list] = {}
    for row in rows:
        plain = _normalize_uops_form(row.form)
        if plain == row.form:
            continue
        for form in (row.form, plain):
            pinned_row = by_form.get(form)
            if pinned_row is None:
                continue
            pinned[form] = [
                pinned_row.metadata.get("iform", ""),
                sorted(ordinals.get(pinned_row.db_key, [])),
            ]
    return pinned


def _rebuild_variant_assignment() -> dict[str, list]:
    from simdref.ingest_catalog import _parse_uops_xml_keyed
    from simdref.ingest_sources import fetch_uops_xml

    source, _source_version = fetch_uops_xml()
    rows, assignments = _parse_uops_xml_keyed(source)
    return _variant_name_assignment(rows, assignments)


def _load_validate_sources() -> Any:
    spec = importlib.util.spec_from_file_location(
        "validate_sources_under_test", VALIDATE_SOURCES_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _has_local_sdm_material() -> bool:
    return (
        Path("vendor/intel/intel-sdm.pdf").exists()
        or Path("data/derived/intel-sdm-descriptions.msgpack").exists()
    )


class UopsRowValidationTests(unittest.TestCase):
    def _run_validator(self, module: Any, xml_path: Path, rows: Any = None) -> str:
        module.fetch_uops_xml = lambda: (xml_path, None)
        if rows is not None:
            module.parse_uops_xml = lambda _source: rows
        with contextlib.redirect_stdout(buffer := io.StringIO()):
            _checked, failures = module.validate_uops_instructions()
        self.last_failures = failures
        return buffer.getvalue()

    def test_decorated_and_duplicate_uops_rows_validate(self):
        # One row with a leading '{evex}' tag and one variant with the same
        # normalized key: parse_uops_xml keeps both on rows of their own,
        # and the validator has to compare rows in that normalized space.
        module = _load_validate_sources()
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "instructions.xml"
            xml_path.write_text(UOPS_FIXTURE)
            text = self._run_validator(module, xml_path)
        self.assertEqual(self.last_failures, 0, text)

    def test_wrong_iform_is_flagged(self):
        module = _load_validate_sources()
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "instructions.xml"
            xml_path.write_text(UOPS_FIXTURE)
            rows = _load_rows(module, xml_path)
            target = next(
                row for row in rows if row.metadata.get("iform") == "AADD_MEMi32_GPR32i32_APX"
            )
            mutated = dataclasses.replace(
                target, metadata={**target.metadata, "iform": "__REVIEW_WRONG_IFORM__"}
            )
            rows = [mutated if row is target else row for row in rows]
            text = self._run_validator(module, xml_path, rows)
        self.assertGreater(self.last_failures, 0, text)
        self.assertIn("iform mismatch for AADD_MEMi32_GPR32i32_APX", text)

    def test_dropped_measurement_uarch_is_flagged(self):
        module = _load_validate_sources()
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "instructions.xml"
            xml_path.write_text(UOPS_MEASUREMENT_FIXTURE)
            rows = _load_rows(module, xml_path)
            target = next(row for row in rows if row.arch_details)
            mutated = dataclasses.replace(target, arch_details={})
            rows = [mutated if row is target else row for row in rows]
            text = self._run_validator(module, xml_path, rows)
        self.assertGreater(self.last_failures, 0, text)
        self.assertIn("missing measurements for VFOO", text)

    def test_same_iform_sibling_cannot_hide_measurement_loss(self):
        # Two parsed rows share one iform; clearing one row's measurements
        # must fail, even though the sibling row still covers the data.
        module = _load_validate_sources()
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "instructions.xml"
            xml_path.write_text(UOPS_SIBLING_IFORM_FIXTURE)
            rows = _load_rows(module, xml_path)
            target = next(row for row in rows if row.form == "VBAR (XMM, K, M128)")
            mutated = dataclasses.replace(target, arch_details={})
            rows = [mutated if row is target else row for row in rows]
            text = self._run_validator(module, xml_path, rows)
        self.assertGreater(self.last_failures, 0, text)
        self.assertIn("missing measurements for VBAR", text)

    def test_cleared_variant_row_cannot_hide_behind_sibling(self):
        # Two same-key variants keep rows of their own with identical
        # measurements. Clearing one row must fail even though the sibling
        # row still holds the same data.
        module = _load_validate_sources()
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "instructions.xml"
            xml_path.write_text(UOPS_VARIANT_SIBLING_FIXTURE)
            rows = _load_rows(module, xml_path)
            target = next(row for row in rows if row.form == "VMOVQ (XMM, XMM)")
            mutated = dataclasses.replace(target, arch_details={})
            rows = [mutated if row is target else row for row in rows]
            text = self._run_validator(module, xml_path, rows)
        self.assertGreater(self.last_failures, 0, text)
        self.assertIn("missing measurements for VMOVQ_XMMdq_XMMdq_1", text)

    def test_extra_uarch_on_row_is_flagged(self):
        # A row that carries uarch data no mapped raw record has must fail,
        # so unsupported extra uarch data cannot ride along unnoticed.
        module = _load_validate_sources()
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "instructions.xml"
            xml_path.write_text(UOPS_MEASUREMENT_FIXTURE)
            rows = _load_rows(module, xml_path)
            target = next(row for row in rows if row.arch_details)
            fake = {"measurement": {"TP_loop": "9.99"}, "latencies": []}
            mutated = dataclasses.replace(
                target,
                arch_details={**target.arch_details, "ZZZFAKE": fake},
            )
            rows = [mutated if row is target else row for row in rows]
            text = self._run_validator(module, xml_path, rows)
        self.assertGreater(self.last_failures, 0, text)
        self.assertIn("uarch set mismatch for", text)

    def test_same_group_sibling_data_swap_is_flagged(self):
        # Swapping two same-group variants' measurements keeps the
        # group-wide data intact. Each raw record must still find the
        # measurements of its own row.
        module = _load_validate_sources()
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "instructions.xml"
            xml_path.write_text(UOPS_SWAP_FIXTURE)
            rows = _load_rows(module, xml_path)
            first = next(row for row in rows if row.form == "VMOVAPD (YMM, YMM)")
            second = next(row for row in rows if row.form == "VMOVAPD_29 (YMM, YMM)")
            rows = [
                dataclasses.replace(first, arch_details=second.arch_details)
                if row is first
                else dataclasses.replace(second, arch_details=first.arch_details)
                if row is second
                else row
                for row in rows
            ]
            text = self._run_validator(module, xml_path, rows)
        self.assertGreater(self.last_failures, 0, text)
        self.assertIn("missing measurements for", text)

    def test_collision_fallback_rows_validate(self):
        # Two same-raw-form records with different iforms land on a
        # canonical row and a collision-fallback row. Both rows hold their
        # own data and must validate.
        module = _load_validate_sources()
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "instructions.xml"
            xml_path.write_text(UOPS_COLLISION_FIXTURE)
            text = self._run_validator(module, xml_path)
        self.assertEqual(self.last_failures, 0, text)


class UopsVariantAssignmentTests(unittest.TestCase):
    def test_variant_name_assignment_matches_golden_fixture(self):
        # A parser mutation can swap row names and data while its map
        # follows the swap. The golden fixture pins the assignment, so a
        # swap changes the rebuilt map and fails here.
        from simdref.ingest_sources import SourceUnavailableError

        try:
            rebuilt = _rebuild_variant_assignment()
        except SourceUnavailableError as exc:
            self.skipTest(f"uops.info XML unavailable: {exc}")
        golden = json.loads(UOPS_VARIANT_FIXTURE_PATH.read_text())
        if rebuilt != golden:
            changed = sorted(
                set(rebuilt) | set(golden),
                key=lambda form: (rebuilt.get(form) != golden.get(form), form),
            )
            detail = "\n".join(
                f"  {form}: golden={golden.get(form)} rebuilt={rebuilt.get(form)}"
                for form in changed
                if rebuilt.get(form) != golden.get(form)
            )
            self.fail(
                "uops variant-name assignment changed. If the change is "
                "intended, regenerate the golden fixture with one command:\n"
                "python3 tests/test_source_validation.py --regenerate-uops-variants\n"
                f"changed rows:\n{detail}"
            )


@unittest.skipUnless(
    os.environ.get("SIMDREF_LIVE") == "1",
    "live-network source validation disabled (set SIMDREF_LIVE=1 to run)",
)
class SourceValidationTests(unittest.TestCase):
    def test_live_source_validation_script_passes(self):
        result = subprocess.run(
            [sys.executable, "tools/validate_sources.py"],
            cwd=".",
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("source validation passed", result.stdout)
        self.assertIn("validated x86 intrinsic links", result.stdout)
        self.assertIn("validated RISC-V intrinsic links", result.stdout)
        self.assertIn("validated RISC-V coverage summary", result.stdout)

    @unittest.skipUnless(_has_local_sdm_material(), "Intel SDM source/cache not available locally")
    def test_live_source_validation_with_required_sdm_passes(self):
        result = subprocess.run(
            [sys.executable, "tools/validate_sources.py", "--require-sdm"],
            cwd=".",
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("validated Intel SDM semantics", result.stdout)
        self.assertIn("validated Intel SDM coverage", result.stdout)
        self.assertIn("source validation passed", result.stdout)


if __name__ == "__main__":
    if "--regenerate-uops-variants" in sys.argv:
        rebuilt = _rebuild_variant_assignment()
        UOPS_VARIANT_FIXTURE_PATH.write_text(json.dumps(rebuilt, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(rebuilt)} rows to {UOPS_VARIANT_FIXTURE_PATH}")
    else:
        unittest.main()
