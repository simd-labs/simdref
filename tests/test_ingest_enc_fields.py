"""Ingest tests for the byte-identity encoding fields (spec K1-K4).

Each behavioral test names the mutation that must make it fail: reverting the
walker's mnemonic fix, the sibling-row rule, the kept encoding block or the
split-attribute identity must each break its own test here.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from simdref.arm_instructions import parse_arm_instruction_payload
from simdref.ingest_catalog import parse_uops_xml
from simdref.models import Catalog, InstructionRecord, SourceVersion
from simdref.riscv import parse_riscv_instruction_payload
from simdref.storage import (
    SQLITE_SCHEMA_VERSION,
    build_sqlite,
    sqlite_schema_is_current,
)
from test_asm_identity import _keyless_db


def _enc(entries: list[tuple[int, int, str]], width: int = 32) -> dict:
    return {
        "_type": "Instruction.Encodeset",
        "width": width,
        "values": [
            {
                "_type": "Instruction.Encodeset.Bits",
                "range": {"_type": "Range", "start": start, "width": span},
                "value": {"_type": "Values.Value", "meaning": None, "value": f"'{value}'"},
            }
            for start, span, value in entries
        ],
    }


def _lit(value: str) -> dict:
    return {"_type": "Instruction.Symbols.Literal", "value": value}


def _rule(rule_id: str) -> dict:
    return {"_type": "Instruction.Symbols.RuleReference", "rule_id": rule_id}


def _node(name: str, symbols: list[dict], entries: list[tuple[int, int, str]]) -> dict:
    return {
        "_type": "Instruction.Instruction",
        "name": name,
        "operation_id": name.split("_")[0],
        "assembly": {"symbols": symbols},
        "encoding": _enc(entries),
    }


def _tree(nodes: list[dict], set_name: str = "A64") -> str:
    payload = {
        "instructions": [
            {"_type": "Instruction.InstructionSet", "name": set_name, "children": nodes}
        ]
    }
    return json.dumps(payload)


class AarchmrsEncodingKeysTest(unittest.TestCase):
    """Rows carry the encodeset (mask, match) of their Instruction."""

    def test_rows_carry_enc_fields(self) -> None:
        # Mutation: revert the walker's enc fields -> no enc_mask.
        records = parse_arm_instruction_payload(
            _tree([_node("RET_64R_branch_reg", [_lit("RET"), _rule("Xd")], [(24, 8, "11010110")])])
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.enc_bits, 32)
        self.assertEqual(record.enc_mask, 0xFF000000)
        self.assertEqual(record.enc_match, 0xD6000000)

    def test_x_bits_add_no_mask(self) -> None:
        records = parse_arm_instruction_payload(
            _tree([_node("FOO_x", [_lit("FOO")], [(24, 8, "11010110"), (0, 5, "xxxxx")])])
        )
        self.assertEqual(records[0].enc_mask, 0xFF000000)

    def test_unparsable_encodeset_keeps_row_without_key(self) -> None:
        records = parse_arm_instruction_payload(
            _tree([_node("FOO_x", [_lit("FOO")], [(24, 8, "11")])])  # width mismatch
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].enc_mask, 0)


class AarchmrsMissingRowsTest(unittest.TestCase):
    """Spec K4 rows: renders that glue a placeholder onto the mnemonic."""

    def test_retoption_render_yields_ret_row(self) -> None:
        # Mutation: revert the leading-run mnemonic parse -> no RET row.
        records = parse_arm_instruction_payload(
            _tree(
                [_node("RET_64R_branch_reg", [_lit("RET"), _rule("Rn_option")], [(0, 5, "11111")])]
            )
        )
        self.assertEqual([(r.mnemonic, r.form) for r in records], [("RET", "RET")])

    def test_dotted_cond_render_yields_b_row(self) -> None:
        records = parse_arm_instruction_payload(
            _tree(
                [
                    _node(
                        "B_only_condbranch",
                        [_lit("B"), _lit("."), _rule("cond_option"), _rule("SPACE"), _rule("imm")],
                        [(0, 5, "00000")],
                    )
                ]
            )
        )
        self.assertEqual(records[0].mnemonic, "B")

    def test_non_a64_sets_stay_out(self) -> None:
        # A32/T32 encodings cannot be scoped from a bare word (spec K2/K4).
        records = parse_arm_instruction_payload(
            _tree([_node("MOV_r_A1", [_lit("MOV")], [(0, 5, "00000")])], set_name="A32")
        )
        self.assertEqual(records, [])


class AarchmrsSiblingRowsTest(unittest.TestCase):
    """Spec K4: sibling encodings of one rendered form are never merged."""

    def test_same_form_different_encoding_stays_two_rows(self) -> None:
        # Mutation: revert the sibling rule -> one row, keys merged.
        records = parse_arm_instruction_payload(
            _tree(
                [
                    _node(
                        "SQXTN_scalar",
                        [_lit("SQXTN"), _rule("SPACE"), _rule("Vd")],
                        [(24, 8, "01011110")],
                    ),
                    _node(
                        "SQXTN_vector",
                        [_lit("SQXTN"), _rule("SPACE"), _rule("Vd")],
                        [(24, 8, "00101110")],
                    ),
                ]
            )
        )
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0].db_key, records[1].db_key)
        self.assertNotEqual(records[0].enc_match, records[1].enc_match)
        self.assertTrue(records[1].form.endswith("[SQXTN_vector]"))

    def test_identical_encoding_duplicates_collapse(self) -> None:
        records = parse_arm_instruction_payload(
            _tree(
                [
                    _node("FOO_a", [_lit("FOO")], [(0, 5, "11111")]),
                    _node("FOO_b", [_lit("FOO")], [(0, 5, "11111")]),
                ]
            )
        )
        self.assertEqual(len(records), 1)


class RiscvEncodingBlocksTest(unittest.TestCase):
    """Spec K3: the vendor keeps the YAML encoding block, the reader keys rows."""

    YAML_KEYED = """\
$schema: "inst_schema.json#"
kind: instruction
name: slli
long_name: Shift left logical
description: |
  Shift xs1 left by shamt.
definedBy:
  extension:
    name: I
assembly: xd, xs1, shamt
encoding:
  RV32:
    match: 0000000----------001-----0010011
    variables:
      - name: shamt
        location: 24-20
  RV64:
    match: 000000-----------001-----0010011
    variables:
      - name: shamt
        location: 25-20
"""

    YAML_FLAT = """\
$schema: "inst_schema.json#"
kind: instruction
name: add
long_name: Integer add
description: |
  Add xs1 to xs2.
definedBy:
  extension:
    name: I
assembly: xd, xs1, xs2
encoding:
  match: 0000000----------000-----0110011
"""

    def _vendor(self):
        path = Path(__file__).resolve().parents[1] / "tools" / "vendor_riscv_sources.py"
        spec = importlib.util.spec_from_file_location("vendor_riscv_sources", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.git_rev = lambda _repo: "c0ffee"
        return module

    def _records(self, yamls: dict[tuple[str, str], str]) -> list[dict]:
        tool = self._vendor()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inst = root / "spec" / "std" / "isa" / "inst"
            for subdir, name in yamls:
                (inst / subdir).mkdir(parents=True, exist_ok=True)
                (inst / subdir / f"{name}.yaml").write_text(yamls[(subdir, name)])
            base = tool.collect_base_instructions(root)
            self.assertIn("encoding", base["slli"])
            return tool.synthesize_instruction_records(
                base, {name: {("agnostic", "masked")} for name in base}
            )

    def test_vendor_keeps_encoding_block(self) -> None:
        # Mutation: drop "encoding" in parse_instruction_file -> KeyError here.
        records = self._records({("I", "slli"): self.YAML_KEYED, ("I", "add"): self.YAML_FLAT})
        by_form = {record["form"]: record for record in records}
        self.assertIn("encoding", by_form["slli"])
        self.assertIn("RV32:", by_form["slli"]["encoding"])
        self.assertIn("match: 0000000", by_form["add"]["encoding"])

    def test_reader_keys_base_row_with_per_base_patterns(self) -> None:
        records = self._records({("I", "slli"): self.YAML_KEYED, ("I", "add"): self.YAML_FLAT})
        parsed = parse_riscv_instruction_payload(
            json.dumps({"format": "riscv-unified-db-v1", "instructions": records})
        )
        keyed = {record.form: record for record in parsed}
        slli = keyed["slli"]
        self.assertEqual(slli.enc_bits, 32)
        bases = {entry[0] for entry in slli.enc_extra}
        self.assertEqual(bases, {"RV32", "RV64"})
        self.assertNotEqual(slli.enc_extra[0][2], slli.enc_extra[1][2])
        add = keyed["add"]
        self.assertEqual(add.enc_extra, [["", 32, 0xFE00707F, 0x33]])
        # The variant shares the encoding but keeps no key of its own
        # ("store the key on the base-instruction row", DESIGN.md).
        self.assertEqual(keyed["slli [masked]"].enc_mask, 0)
        self.assertEqual(keyed["add [masked]"].enc_mask, 0)


class UopsSplitAttrsTest(unittest.TestCase):
    """Spec K1: the uops.info variant attributes ride on the row."""

    XML = """\
<root>
  <instruction asm="ADD" string="ADD (R32, R32)" iclass="ADD" iform="ADD_GPRv_GPRv_0F" eosz="2" category="Arith" />
  <instruction asm="ADD" string="ADD (R32, R32)" iclass="ADD" iform="ADD_GPRv_GPRv_0F" eosz="3" category="Arith" />
</root>
"""

    def test_split_attrs_stored_and_variants_stay_separate(self) -> None:
        # Mutation: drop split_attrs from the merge identity -> one row and
        # one attrs dict survives, so the lookup cannot split the pool.
        records = parse_uops_xml(self.XML)
        self.assertEqual(len(records), 2)
        attrs = [json.loads(r.metadata["split_attrs"]) for r in records]
        self.assertEqual(sorted(a["eosz"] for a in attrs), ["2", "3"])
        self.assertNotEqual(records[0].db_key, records[1].db_key)

    def test_no_attrs_no_key(self) -> None:
        records = parse_uops_xml('<root><instruction asm="NOP" string="NOP" /></root>')
        self.assertNotIn("split_attrs", records[0].metadata)


class StorageEncColumnsTest(unittest.TestCase):
    """The lookup reads its key from columns, without unpacking payloads."""

    def test_columns_and_index_present(self) -> None:
        record = InstructionRecord(
            mnemonic="SQXTN",
            form="SQXTN (D0, V1.8B)",
            summary="Saturating extract.",
            architecture="arm",
            metadata={"category": "SIMD", "iform": "", "split_attrs": '{"eosz": "2"}'},
            enc_bits=32,
            enc_mask=0xFF3FFC00,
            enc_match=0x5E214800,
            enc_extra=[["RV32", 32, 1, 1]],
        )
        catalog = Catalog(
            intrinsics=[],
            instructions=[record],
            sources=[
                SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")
            ],
            generated_at="2025-01-01",
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "catalog.db"
            build_sqlite(catalog, db)
            self.assertTrue(sqlite_schema_is_current(db))
            conn = sqlite3.connect(db)
            try:
                row = conn.execute(
                    "SELECT enc_bits, enc_mask, enc_match, enc_extra, iform, split_attrs"
                    " FROM instructions_data"
                ).fetchone()
                indexes = {item[1] for item in conn.execute("PRAGMA index_list(instructions_data)")}
            finally:
                conn.close()
        self.assertEqual(row[:3], (32, 0xFF3FFC00, 0x5E214800))
        self.assertEqual(json.loads(row[3]), [["RV32", 32, 1, 1]])
        self.assertEqual(json.loads(row[5]), {"eosz": "2"})
        self.assertIn("idx_instruction_enc", indexes)
        self.assertEqual(SQLITE_SCHEMA_VERSION, "14")

    def test_schema_13_still_current(self) -> None:
        # C4: storage accepts the last released schema; every installed
        # catalog is 13 until a release ships a rebuilt one.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(sqlite_schema_is_current(_keyless_db(Path(tmp))))

    def test_old_column_set_is_stale(self) -> None:
        record = InstructionRecord(mnemonic="ADD", form="ADD", summary="Add.")
        catalog = Catalog(
            intrinsics=[],
            instructions=[record],
            sources=[
                SourceVersion(source="t", version="t", fetched_at="2025-01-01", url="test://")
            ],
            generated_at="2025-01-01",
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "catalog.db"
            build_sqlite(catalog, db)
            conn = sqlite3.connect(db)
            try:
                conn.execute("ALTER TABLE instructions_data DROP COLUMN enc_extra")
                conn.commit()
            finally:
                conn.close()
            self.assertFalse(sqlite_schema_is_current(db))

    def test_model_round_trip_keeps_enc_fields(self) -> None:
        record = InstructionRecord(
            mnemonic="ADD",
            form="ADD",
            summary="Add.",
            enc_bits=32,
            enc_mask=1,
            enc_match=1,
            enc_extra=[["RV32", 32, 1, 1]],
        )
        clone = InstructionRecord.from_dict(record.to_dict())
        self.assertEqual(
            (clone.enc_bits, clone.enc_mask, clone.enc_match, clone.enc_extra),
            (32, 1, 1, [["RV32", 32, 1, 1]]),
        )


if __name__ == "__main__":
    unittest.main()
