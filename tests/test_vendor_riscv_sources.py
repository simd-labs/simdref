"""Docs links that tools/vendor_riscv_sources.py gives RISC-V vector instructions."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import vendor_riscv_sources as vendor  # noqa: E402

YAML = "https://github.com/riscv/riscv-unified-db/blob/c0ffee/spec/std/isa/inst/X/x.yaml"

ADD_YAML = """\
$schema: "inst_schema.json#"
kind: instruction
name: add
long_name: Integer add
description: Add xs1 to xs2.
definedBy:
  extension:
    name: I
assembly: xd, xs1, xs2
encoding:
  match: 0000000----------000-----0110011
operation(): X[xd] = X[xs1] + X[xs2];
"""


def test_spec_extensions_link_their_docs_page():
    assert vendor.instruction_doc_url(["V"], YAML) == vendor.V_DOC_URL
    assert vendor.instruction_doc_url(["Zvkned"], YAML) == vendor.VECTOR_CRYPTO_DOC_URL
    assert vendor.instruction_doc_url(["Zvbb"], YAML) == vendor.VECTOR_CRYPTO_DOC_URL
    assert vendor.instruction_doc_url(["Zvfbfwma"], YAML) == vendor.BFLOAT16_DOC_URL


def test_extension_without_docs_page_links_upstream_yaml(tmp_path):
    # Zvdot4a8i is on no page of vendor/riscv/docs_pages.json.
    path = tmp_path / "vdot4a.vv.yaml"
    path.write_text(
        "name: vdot4a.vv\n"
        "long_name: Vector 4-element signed-signed dot-product accumulate (vector-vector)\n"
        "description: |\n  Computes the dot product.\n"
        "definedBy:\n  extension:\n    name: Zvdot4a8i\n"
        "operation(): |\n"
    )
    parsed = vendor.parse_instruction_file(path, YAML)
    assert parsed["isa"] == ["Zvdot4a8i"]
    assert parsed["url"] == YAML


def test_yaml_fields_are_parsed_not_line_scraped(tmp_path):
    """Spec 33 (U8): the YAML unit reads the fields as parsed values.

    Inline ``operation():`` text and plain scalar descriptions land on the
    record, and the encoding rows are computed at vendor time.
    """
    # Mutation: the old block-only line scrape drops the text -> asserts fail.
    path = tmp_path / "add.yaml"
    path.write_text(ADD_YAML)
    parsed = vendor.parse_instruction_file(path, YAML)
    assert parsed["operation"] == "X[xd] = X[xs1] + X[xs2];"
    assert parsed["description"] == "Add xs1 to xs2."
    assert parsed["summary"] == "Integer add."
    assert parsed["encoding"] == [["", 32, 0xFE00707F, 0x33]]
