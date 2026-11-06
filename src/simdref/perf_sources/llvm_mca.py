"""Drive the llvm-exegesis → llvm-mc → llvm-mca pipeline per core.

The ingester no longer synthesises per-mnemonic asm snippets. It
enumerates every LLVM-schedulable opcode per target ``(triple, cpu)``
via ``llvm-exegesis``, disassembles the captured bytes, and feeds the
result through ``llvm-mca --instruction-tables=full --json``. Three
subprocess calls per core replace the ~55,000 calls of the former
regex-based synthesiser.

Failure modes:

- ``llvm-mca`` / ``llvm-exegesis`` / ``llvm-mc`` absent or too old:
  :class:`LLVMMcaUnavailable`, which the CLI surfaces with an install
  hint.
- any subprocess emits non-zero status or unparseable output:
  :class:`LLVMMcaError` with the offending core's triple + cpu.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable

from simdref.models import InstructionRecord
from simdref.perf_sources.cores import CANONICAL_CORES, CoreSpec
from simdref.perf_sources.merge import PerfRow, fmt_cycles

LLVM_MCA_MIN_VERSION: int = 18  # JSON output stabilised in LLVM 18.
LLVM_MCA_CITATION = "https://llvm.org/docs/CommandGuide/llvm-mca.html"


class LLVMMcaUnavailable(RuntimeError):
    """Raised when a required LLVM binary is missing or too old."""

    install_hint: str = (
        "Install the LLVM toolchain (llvm-mca, llvm-exegesis, llvm-mc) to "
        "build the modeled perf catalog:\n"
        "  apt:    sudo apt install llvm\n"
        "  brew:   brew install llvm\n"
        "  conda:  conda install -c conda-forge llvm-tools\n"
        "Or download the pre-built release artifact: simdref update"
    )


class LLVMMcaError(RuntimeError):
    """Raised when one of the pipeline stages emits unusable output."""


@dataclass(frozen=True)
class LLVMMcaVersion:
    major: int
    raw: str


def detect_llvm_mca_version(executable: str = "llvm-mca") -> LLVMMcaVersion:
    """Return the LLVM major version, or raise :class:`LLVMMcaUnavailable`."""
    if shutil.which(executable) is None:
        raise LLVMMcaUnavailable(f"{executable!r} not found on PATH")
    try:
        proc = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise LLVMMcaUnavailable(f"failed to run {executable} --version: {exc}") from exc
    for line in (proc.stdout + proc.stderr).splitlines():
        token = line.lower()
        if "llvm version" in token:
            tail = token.split("llvm version", 1)[1].strip().split()[0]
            head = tail.split(".", 1)[0]
            if head.isdigit():
                major = int(head)
                if major < LLVM_MCA_MIN_VERSION:
                    raise LLVMMcaUnavailable(
                        f"llvm-mca {major} is older than required {LLVM_MCA_MIN_VERSION}"
                    )
                return LLVMMcaVersion(major=major, raw=tail)
    raise LLVMMcaUnavailable(f"could not parse version from {executable} --version")


def parse_llvm_mca_json(
    payload: dict[str, Any],
    *,
    core: CoreSpec,
    mnemonic: str,
    mca_version: str,
) -> PerfRow | None:
    """Extract one :class:`PerfRow` from a single-region ``llvm-mca --json`` payload.

    Kept as a small utility for callers that want to run llvm-mca on a
    standalone snippet (outside the scheduling pipeline). It tolerates
    both the modern ``InstructionInfoView`` schema (LLVM 18+) and the
    legacy flat ``Instructions`` + ``SummaryView.IPC`` schema used by
    older fixtures.
    """
    regions = payload.get("CodeRegions") or []
    if not regions:
        return None
    region = regions[0]
    info_view = region.get("InstructionInfoView") or {}
    info_list = info_view.get("InstructionList") or []
    latency: Any = None
    rthroughput: Any = None
    if info_list and isinstance(info_list[0], dict):
        latency = info_list[0].get("Latency")
        rthroughput = info_list[0].get("RThroughput")
    if latency is None and rthroughput is None:
        insts = region.get("Instructions") or []
        if insts and isinstance(insts[0], dict):
            latency = insts[0].get("Latency")
        summary = region.get("SummaryView") or {}
        ipc = summary.get("IPC")
        if isinstance(ipc, (int, float)) and ipc > 0:
            rthroughput = 1.0 / float(ipc)
    cpi = ""
    if isinstance(rthroughput, (int, float)):
        cpi = fmt_cycles(float(rthroughput))
    return PerfRow(
        mnemonic=mnemonic,
        core=core.canonical_id,
        source="llvm-mca",
        source_kind="modeled",
        source_version=mca_version,
        architecture="arm" if core.architecture == "aarch64" else core.architecture,
        latency=str(latency) if latency is not None else "",
        cpi=cpi,
        applies_to="mnemonic",
        citation_url=LLVM_MCA_CITATION,
    )


# RISC-V compressed rows carry no operand syntax (their catalog form is the
# mnemonic), so their llvm-mca input line is the mnemonic plus the first
# operand shape llvm-mca accepts. The feature string enables the compressed
# families on top of the row's -mcpu, in the same "+ext,..." style as the
# riscv entry of asm_identity._ARCH_CFG.
_RISCV_MATTR = "+m,+a,+f,+d,+c,+v,+zba,+zbb,+zbs,+zca,+zcb,+zcmop"
_RISCV_ASM_FORMS: tuple[str, ...] = (
    "",
    " a0",
    " ra",
    " f0",
    " 4",
    " 0",
    " 8",
    " a0, 4",
    " a0, 8",
    " a0, a1",
    " f0, 4",
    " f0, 8",
    " a0, 0(a1)",
    " a0, 8(a1)",
    " f0, 0(a1)",
    " f0, 8(a1)",
    " fs0, 0(a1)",
    " fs0, 8(a1)",
    " a0, 4(sp)",
    " a0, 8(sp)",
    " f0, 4(sp)",
    " f0, 8(sp)",
    " fs0, 4(sp)",
    " fs0, 8(sp)",
    " a0, sp, 4",
    " a0, a1, 4",
    " sp, 16",
    " {ra}, -16",
    " {ra, s0}, -16",
    " a0, a1, 0(a2)",
    " a0, 4, a1",
    " a0, a1, a2",
)


def mca_command(core: CoreSpec, executable: str, *, mattr: str | None = None) -> list[str]:
    """Base llvm-mca flags for one asm run; *mattr* adds an opt-in -mattr."""
    cmd = [
        executable,
        "--instruction-tables=full",
        "--json",
        # llvm-exegesis legitimately emits encodings that are architecturally
        # "unpredictable" (LDP with Rt2==Rt, writeback with the base in the
        # destination, etc.). llvm-mca refuses to schedule these by default
        # — skip them so a small number of special corner cases don't fail
        # the whole core.
        "--skip-unsupported-instructions=any",
        f"--mtriple={core.llvm_triple}",
        f"--mcpu={core.llvm_cpu}",
    ]
    if mattr is not None:
        cmd.append(f"-mattr={mattr}")
    return cmd


def _mca_json(
    asm: str, core: CoreSpec, *, executable: str, timeout: float = 15.0
) -> dict[str, Any] | None:
    """Run llvm-mca on one asm line. ``None`` when it rejects or crashes."""
    cmd = mca_command(core, executable, mattr=_RISCV_MATTR)
    try:
        proc = subprocess.run(
            cmd,
            input=asm + "\n",
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _scheduled_any(payload: dict[str, Any]) -> bool:
    """True when the payload carries at least one scheduled instruction."""
    for region in payload.get("CodeRegions") or []:
        if (region.get("InstructionInfoView") or {}).get("InstructionList"):
            return True
        if region.get("Instructions"):
            return True
    return False


def _compressed_payload(
    mnemonic: str, core: CoreSpec, *, executable: str
) -> tuple[str, dict[str, Any]] | None:
    """First ``mnemonic + operands`` text llvm-mca schedules, with its payload."""
    for form in _RISCV_ASM_FORMS:
        line = mnemonic + form
        payload = _mca_json(line, core, executable=executable)
        if payload is not None and _scheduled_any(payload):
            return line, payload
    return None


def ingest_compressed_rows(
    records: Iterable[InstructionRecord],
    core: CoreSpec,
    *,
    mca_version: str,
    executable: str = "llvm-mca",
) -> list[PerfRow]:
    """Feed the riscv ``c.*`` / ``cm.*`` rows to llvm-mca as their own asm.

    The row named stays the compressed encoding; the timing comes from
    llvm-mca's model of that instruction itself (issue #65 item 27). Any
    row llvm-mca rejects stays without timing.
    """
    if shutil.which(executable) is None:
        raise LLVMMcaUnavailable(f"{executable!r} not found on PATH")
    rows: list[PerfRow] = []
    for record in records:
        if record.architecture.casefold() != "riscv":
            continue
        mnemonic = record.mnemonic.strip()
        if not mnemonic.casefold().startswith(("c.", "cm.")):
            continue
        found = _compressed_payload(mnemonic, core, executable=executable)
        if found is None:
            continue
        _, payload = found
        row = parse_llvm_mca_json(payload, core=core, mnemonic=mnemonic, mca_version=mca_version)
        if row is not None and (row.latency or row.cpi):
            rows.append(row)
    return rows


def ingest_llvm_mca(
    records: Iterable[InstructionRecord],
    *,
    cores: Iterable[CoreSpec] | None = None,
    executable: str = "llvm-mca",
    cache_root: Any = None,
) -> tuple[list[PerfRow], str]:
    """Enumerate LLVM-schedulable opcodes across cores, returning rows + version.

    Each aarch64 / riscv core triggers one llvm-exegesis +
    llvm-mc + llvm-mca pipeline, cached under ``vendor/perf-cache/``.
    x86 cores are skipped (scheduling models come from uops.info).

    *records* feeds the riscv compressed rows (``c.*`` / ``cm.*``) to
    llvm-mca per core via :func:`ingest_compressed_rows`; the pipeline
    labels those rows as their uncompressed aliases, so they need their
    own pass.

    Raises :class:`LLVMMcaUnavailable` when a required binary is absent
    or too old. Raises :class:`LLVMMcaError` on any pipeline failure —
    no silent fallback.
    """
    # Lazy import to break the circular dependency (llvm_scheduling
    # imports the error classes + citation URL from this module).
    from simdref.perf_sources import llvm_scheduling  # noqa: PLC0415

    version = detect_llvm_mca_version(executable)
    records = list(records)
    target_cores = list(cores) if cores is not None else list(CANONICAL_CORES)
    rows: list[PerfRow] = []
    for core in target_cores:
        if core.architecture == "x86":
            continue
        rows.extend(
            llvm_scheduling.collect_core_schedule(
                core,
                mca_version=version.raw,
                cache_root=cache_root,
            )
        )
        rows.extend(
            ingest_compressed_rows(records, core, mca_version=version.raw, executable=executable)
        )
    return rows, version.raw
