"""Core data models for simdref.

Defines the four dataclasses that represent the catalog:
:class:`SourceVersion`, :class:`IntrinsicRecord`, :class:`InstructionRecord`,
and :class:`Catalog`.  All use ``slots=True`` for memory efficiency when
holding thousands of records.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from simdref.pdfrefs import apply_legacy_pdf_metadata, normalize_pdf_refs


@dataclass(slots=True)
class SourceVersion:
    source: str
    version: str
    fetched_at: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "version": self.version,
            "fetched_at": self.fetched_at,
            "url": self.url,
        }


@dataclass(slots=True)
class IntrinsicRecord:
    name: str
    signature: str
    description: str
    header: str
    url: str = ""
    architecture: str = "x86"
    isa: list[str] = field(default_factory=list)
    category: str = ""
    subcategory: str = ""
    instructions: list[str] = field(default_factory=list)
    instruction_refs: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    doc_sections: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    source: str = "intel"
    _search_blob: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        fields = [
            self.name,
            self.signature,
            self.description,
            self.header,
            self.url,
            self.architecture,
            self.category,
            " ".join(self.isa),
            " ".join(self.instructions),
            " ".join(self.aliases),
            " ".join(self.metadata.values()),
            " ".join(self.doc_sections.values()),
        ]
        self._search_blob = " ".join(x for x in fields if x)

    @property
    def search_blob(self) -> str:
        return self._search_blob

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.signature,
            "description": self.description,
            "header": self.header,
            "url": self.url,
            "architecture": self.architecture,
            "isa": self.isa,
            "category": self.category,
            "subcategory": self.subcategory,
            "instructions": self.instructions,
            "instruction_refs": self.instruction_refs,
            "metadata": self.metadata,
            "doc_sections": self.doc_sections,
            "notes": self.notes,
            "aliases": self.aliases,
            "source": self.source,
        }


@dataclass(slots=True)
class InstructionRecord:
    mnemonic: str
    form: str
    summary: str
    architecture: str = "x86"
    isa: list[str] = field(default_factory=list)
    operand_details: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    arch_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    linked_intrinsics: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    description: dict[str, str] = field(default_factory=dict)
    pdf_refs: list[dict[str, str]] = field(default_factory=list)
    source: str = "uops.info"
    # Byte-identity encoding key (spec K2/K3): fixed bits of the upstream
    # encoding pattern. enc_extra holds alternate patterns for other bases,
    # e.g. a RISC-V instruction whose RV32 and RV64 match strings differ:
    # [[base, bits, mask, match], ...].
    enc_bits: int = 0
    enc_mask: int = 0
    enc_match: int = 0
    enc_extra: list[list[Any]] = field(default_factory=list)
    _search_blob: str = field(default="", repr=False)
    _key: str = field(default="", init=False, repr=False)
    _db_key: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self.pdf_refs = normalize_pdf_refs(self.pdf_refs, self.metadata)
        self.metadata = apply_legacy_pdf_metadata(dict(self.metadata), self.pdf_refs)
        self._key = self.form.strip()
        if not self._key:
            self._key = self.mnemonic
        elif not self._key.casefold().startswith(self.mnemonic.casefold()):
            self._key = f"{self.mnemonic} {self._key}".strip()
        self._db_key = f"{self.architecture}:{self._key.casefold()}"
        fields = [
            self.mnemonic,
            self.form,
            self.summary,
            self.architecture,
            " ".join(self.isa),
            " ".join(self.operands),
            " ".join(self.linked_intrinsics),
            " ".join(self.aliases),
        ]
        self._search_blob = " ".join(x for x in fields if x)

    @property
    def key(self) -> str:
        return self._key

    @property
    def db_key(self) -> str:
        return self._db_key

    @property
    def operands(self) -> list[str]:
        rendered: list[str] = []
        for operand in self.operand_details:
            rendered_rw = "".join(flag for flag in ("r", "w") if operand.get(flag) == "1")
            idx = operand.get("idx", "")
            text_parts = [
                f"idx={idx}" if idx else "",
                rendered_rw,
                operand.get("type", ""),
                operand.get("width", ""),
                operand.get("xtype", ""),
                operand.get("name", ""),
            ]
            text = " ".join(part for part in text_parts if part)
            if text:
                rendered.append(text)
        return rendered

    @property
    def search_blob(self) -> str:
        return self._search_blob

    @property
    def metrics(self) -> dict[str, dict[str, str]]:
        return {
            arch: measurement
            for arch, details in self.arch_details.items()
            if (measurement := details.get("measurement"))
        }

    def to_dict(self) -> dict[str, Any]:
        metadata = apply_legacy_pdf_metadata(dict(self.metadata), self.pdf_refs)
        return {
            "mnemonic": self.mnemonic,
            "form": self.form,
            "summary": self.summary,
            "architecture": self.architecture,
            "isa": self.isa,
            "operand_details": self.operand_details,
            "metadata": metadata,
            "arch_details": self.arch_details,
            "linked_intrinsics": self.linked_intrinsics,
            "aliases": self.aliases,
            "description": self.description,
            "pdf_refs": self.pdf_refs,
            "source": self.source,
            "enc_bits": self.enc_bits,
            "enc_mask": self.enc_mask,
            "enc_match": self.enc_match,
            "enc_extra": self.enc_extra,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InstructionRecord":
        data = dict(payload)
        data.setdefault("architecture", "x86")
        data.setdefault("description", {})
        metadata = dict(data.get("metadata") or {})
        data["metadata"] = metadata
        data["pdf_refs"] = normalize_pdf_refs(data.get("pdf_refs"), metadata)
        # Keep only known fields. A catalog a newer simdref built can carry
        # fields this version does not know (enc_id); so did legacy payloads
        # (metrics, operands). Otherwise the load dies with a TypeError.
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(slots=True)
class Catalog:
    intrinsics: list[IntrinsicRecord]
    instructions: list[InstructionRecord]
    sources: list[SourceVersion]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "intrinsics": [item.to_dict() for item in self.intrinsics],
            "instructions": [item.to_dict() for item in self.instructions],
            "sources": [item.to_dict() for item in self.sources],
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Catalog":
        return cls(
            intrinsics=[
                IntrinsicRecord(architecture="x86", **item)
                if "architecture" not in item
                else IntrinsicRecord(**item)
                for item in payload.get("intrinsics", [])
            ],
            instructions=[
                InstructionRecord.from_dict(item) for item in payload.get("instructions", [])
            ],
            sources=[
                SourceVersion(
                    source=item.get("source", ""),
                    version=item.get("version", ""),
                    fetched_at=item.get("fetched_at", ""),
                    url=item.get("url", ""),
                )
                for item in payload.get("sources", [])
            ],
            generated_at=payload["generated_at"],
        )
