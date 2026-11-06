"""Per-core LLVM scheduling pipeline (llvm-exegesis → llvm-mc → llvm-mca).

Three subprocess calls per core enumerate every LLVM-schedulable opcode,
disassemble its canonical asm form, and measure via ``llvm-mca
--instruction-tables=full --json``. The result is a list of
:class:`~simdref.perf_sources.merge.PerfRow` ready for
:func:`~simdref.perf_sources.merge.merge_perf_rows`.

No regex. Input is YAML (llvm-exegesis) and JSON (llvm-mca); output is
structured data.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from simdref.perf_sources.cores import CoreSpec, perf_architecture
from simdref.perf_sources.llvm_mca import (
    LLVM_MCA_CITATION,
    LLVMMcaError,
    LLVMMcaUnavailable,
    mca_command,
)
from simdref.perf_sources.merge import PerfRow, fmt_cycles

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_ROOT = _REPO_ROOT / "vendor" / "perf-cache"

# Minimum number of times a fixed-width chunk must appear inside a
# ``prepare-and-assemble-snippet`` buffer before we trust it as a real
# instruction (as opposed to a random slice of prologue / epilogue).
# llvm-exegesis always emits ≥ 4 repeats for target opcodes.
_MIN_REPEAT_RUN = 3


class LLVMSchedulingError(LLVMMcaError):
    """Raised when the scheduling pipeline produces unusable output.

    Subclasses :class:`LLVMMcaError` so existing ``except LLVMMcaError``
    handlers keep working.
    """


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise LLVMMcaUnavailable(f"{binary!r} not found on PATH")
    return path


def _parse_exegesis_yaml(text: str) -> list[dict[str, str]]:
    """Reduce llvm-exegesis multi-document YAML to ``[{opcode, snippet}]``.

    The ``opcode`` is the first whitespace-delimited token of
    ``key.instructions[0]`` (LLVM's MC opcode name, e.g. ``FADDv4f32``).
    The ``snippet`` is the ``assembled_snippet`` hex string verbatim.
    Docs missing either field are dropped.
    """
    entries: list[dict[str, str]] = []
    for doc in yaml.safe_load_all(text):
        if not isinstance(doc, dict):
            continue
        key_block = doc.get("key")
        if not isinstance(key_block, dict):
            continue
        instructions = key_block.get("instructions")
        if not isinstance(instructions, list) or not instructions:
            continue
        first = str(instructions[0] or "").strip()
        if not first:
            continue
        opcode = first.split(None, 1)[0]
        snippet = str(doc.get("assembled_snippet") or "").strip()
        if not opcode or not snippet:
            continue
        entries.append({"opcode": opcode, "snippet": snippet})
    return entries


def _parse_hex_snippet(hex_str: str) -> bytes:
    """Decode a hex-encoded snippet to bytes (uppercase/lowercase agnostic)."""
    try:
        return bytes.fromhex(hex_str)
    except ValueError as exc:
        raise LLVMSchedulingError(f"malformed hex snippet: {hex_str[:40]}...") from exc


def _extract_repeated_chunks(snippet_hex: str, architecture: str) -> list[bytes]:
    """Return every fixed-width chunk that appears ``≥ _MIN_REPEAT_RUN`` times.

    llvm-exegesis emits one of two snippet shapes:

    - ``prologue || opcode × N || epilogue`` — the common case;
    - ``prologue || (opcode, breaker) × N || epilogue`` — when the
      opcode depends on its own output, exegesis interleaves a breaker
      instruction to keep the latency-chain isolated.

    In the second shape neither chunk repeats *consecutively* but both
    repeat by total count, so we count chunk frequency at every valid
    byte alignment and keep everything that repeats often enough. The
    breaker is a real ISA instruction and its scheduling data is just
    as legitimate as the target's, so the caller can disassemble and
    measure both without any additional bookkeeping.

    AArch64 has 4-byte-aligned, fixed 4-byte instructions. RISC-V mixes
    2- and 4-byte instructions on 2-byte alignment, so we sweep both
    widths and both starting byte offsets.
    """
    try:
        data = _parse_hex_snippet(snippet_hex)
    except LLVMSchedulingError:
        return []
    # AArch64 is fixed 4-byte on 4-byte boundaries → only offset 0 is
    # meaningful. RISC-V is 2-byte aligned, so 4-byte windows must also
    # be probed at offset 2. Sweeping every byte offset (as the prior
    # revision did) picks up rotated views of the real pattern.
    configurations: tuple[tuple[int, tuple[int, ...]], ...]
    if architecture == "riscv":
        configurations = ((4, (0, 2)), (2, (0,)))
    else:
        configurations = ((4, (0,)),)
    kept: set[bytes] = set()
    for width, byte_offsets in configurations:
        for start_byte in byte_offsets:
            counter: dict[bytes, int] = {}
            pos = start_byte
            while pos + width <= len(data):
                chunk = data[pos : pos + width]
                counter[chunk] = counter.get(chunk, 0) + 1
                pos += width
            for chunk, count in counter.items():
                if count >= _MIN_REPEAT_RUN:
                    kept.add(chunk)
    return sorted(kept)


def _hex_to_byte_line(data: bytes) -> str:
    """Turn ``b'\\xef\\xb9 \\x20\\x4e'`` into ``'0xEF 0xB9 0x20 0x4E'``."""
    return " ".join(f"0x{b:02X}" for b in data)


def build_byte_lines(entries: list[dict[str, str]], architecture: str) -> list[str]:
    """Map exegesis entries to disassembly-ready hex-byte lines.

    Dedupes identical byte sequences across opcodes — different MC
    opcodes occasionally encode to the same bytes, and feeding duplicates
    to llvm-mc wastes work in the later stages.
    """
    seen: set[bytes] = set()
    lines: list[str] = []
    for entry in entries:
        chunks = _extract_repeated_chunks(entry.get("snippet", ""), architecture)
        for chunk in chunks:
            if chunk in seen:
                continue
            seen.add(chunk)
            lines.append(_hex_to_byte_line(chunk))
    return lines


def _keep_disassembly_line(line: str) -> bool:
    if not line.startswith("\t"):
        return False
    body = line.lstrip("\t").strip()
    if not body:
        return False
    if body.startswith(".") or body.startswith("#"):
        return False
    return True


def _filter_disassembly(text: str) -> str:
    """Drop ``.text`` directives, comments, and blank lines so llvm-mca
    receives a clean asm stream."""
    body = "\n".join(line for line in text.splitlines() if _keep_disassembly_line(line))
    return body + "\n" if body else ""


def _run_exegesis(core: CoreSpec, output: Path, *, executable: str) -> None:
    _require(executable)
    cmd = [
        executable,
        "--mode=latency",
        "--opcode-index=-1",
        "--benchmark-phase=prepare-and-assemble-snippet",
        f"--mtriple={core.llvm_triple}",
        f"--mcpu={core.llvm_cpu}",
        f"--benchmarks-file={output}",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as exc:
        raise LLVMMcaUnavailable(f"{executable} failed: {exc}") from exc
    if proc.returncode != 0 or not output.exists():
        raise LLVMSchedulingError(
            f"{executable} failed for {core.canonical_id} "
            f"(triple={core.llvm_triple}, cpu={core.llvm_cpu}): "
            f"exit={proc.returncode}; stderr={proc.stderr.strip()[:500]}"
        )


def _run_disassemble(hex_lines: list[str], core: CoreSpec, *, executable: str) -> str:
    _require(executable)
    stdin = "\n".join(hex_lines) + "\n"
    cmd = [
        executable,
        f"--triple={core.llvm_triple}",
        f"--mcpu={core.llvm_cpu}",
        "--disassemble",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=stdin,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise LLVMMcaUnavailable(f"{executable} failed: {exc}") from exc
    if proc.returncode != 0:
        raise LLVMSchedulingError(
            f"{executable} --disassemble failed for {core.canonical_id}: "
            f"exit={proc.returncode}; stderr={proc.stderr.strip()[:500]}"
        )
    return proc.stdout


def _try_mca_once(
    asm_text: str, core: CoreSpec, executable: str, timeout: float
) -> tuple[int, str, str]:
    """Run llvm-mca once and return ``(returncode, stdout, stderr)``."""
    _require(executable)
    try:
        proc = subprocess.run(
            mca_command(core, executable),
            input=asm_text,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise LLVMMcaUnavailable(f"{executable} failed: {exc}") from exc
    return proc.returncode, proc.stdout, proc.stderr


def _merge_mca_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate ``Instructions`` + ``InstructionInfoView`` across payloads.

    The scheduling numbers in ``--instruction-tables=full`` are
    per-instruction (no cross-instruction simulation state), so
    concatenating is safe.
    """
    if not payloads:
        return {"CodeRegions": []}
    if len(payloads) == 1:
        return payloads[0]
    merged_instructions: list[Any] = []
    merged_info: list[Any] = []
    merged_pressure: list[dict[str, Any]] = []
    template = payloads[0]
    for payload in payloads:
        regions = payload.get("CodeRegions") or []
        if not regions:
            continue
        region = regions[0]
        offset = len(merged_info)
        merged_instructions.extend(region.get("Instructions") or [])
        merged_info.extend((region.get("InstructionInfoView") or {}).get("InstructionList") or [])
        # ``ResourcePressureInfo`` entries carry a per-payload
        # ``InstructionIndex``. Re-index into the merged space so the
        # join in :func:`_pressure_by_index` stays consistent.
        for entry in (region.get("ResourcePressureView") or {}).get("ResourcePressureInfo") or []:
            if not isinstance(entry, dict):
                continue
            shifted = dict(entry)
            if isinstance(shifted.get("InstructionIndex"), int):
                shifted["InstructionIndex"] = shifted["InstructionIndex"] + offset
            merged_pressure.append(shifted)
    first_region = (template.get("CodeRegions") or [{}])[0]
    return {
        **template,
        "CodeRegions": [
            {
                **first_region,
                "Instructions": merged_instructions,
                "InstructionInfoView": {
                    **(first_region.get("InstructionInfoView") or {}),
                    "InstructionList": merged_info,
                },
                "ResourcePressureView": {
                    **(first_region.get("ResourcePressureView") or {}),
                    "ResourcePressureInfo": merged_pressure,
                },
            }
        ],
    }


def _run_mca(
    asm_text: str,
    core: CoreSpec,
    *,
    executable: str,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Run llvm-mca, recursively quarantining lines that crash it.

    llvm-mca (22.x) segfaults on a handful of weird RVV encodings even
    under ``--skip-unsupported-instructions=any`` — a known upstream
    bug. Rather than failing the whole core, we bisect the input on
    each crash, drop the single offending line, and merge the surviving
    halves' payloads.
    """
    lines = [line for line in asm_text.splitlines() if line.strip()]
    if not lines:
        return {"CodeRegions": []}

    def run_block(block: list[str]) -> dict[str, Any]:
        if not block:
            return {"CodeRegions": []}
        text = "\n".join(block) + "\n"
        rc, stdout, _ = _try_mca_once(text, core, executable, timeout)
        if rc == 0:
            try:
                return json.loads(stdout)
            except json.JSONDecodeError as exc:
                raise LLVMSchedulingError(
                    f"{executable} emitted non-JSON for {core.canonical_id}: {exc}"
                ) from exc
        if len(block) == 1:
            # Single line crashed llvm-mca — quarantine and move on. This
            # is where the upstream crash bug lives; we cannot do better.
            return {"CodeRegions": []}
        mid = len(block) // 2
        left = run_block(block[:mid])
        right = run_block(block[mid:])
        # If a clean rerun is possible (both halves survived individually
        # but something about their combination wasn't the problem),
        # prefer the merged payload. Either way we merge what we have.
        return _merge_mca_payloads([left, right])

    return run_block(lines)


def _asm_mnemonic(line: str) -> str:
    """First whitespace-delimited token of an llvm-mc asm line, uppercased."""
    normalized = line.replace("\t", " ").strip()
    if not normalized:
        return ""
    return normalized.split(None, 1)[0].upper()


def _format_port_name(name: str) -> str:
    """Normalise raw LLVM resource names for a readable ``ports`` column.

    llvm-mca emits resources shaped like ``N1UnitV0`` or ``N1UnitD.\\x00``
    (the trailing bytes disambiguate sub-units of a replicated resource).
    We strip a short ``<vendor>Unit`` prefix when present, and then drop
    any non-printable / ``.`` suffix so ``N1UnitD.\\x00`` → ``D0``.

    Names without a recognised prefix (e.g. RISC-V ``SiFive7VA1``) are
    returned unchanged — a slightly longer label is still readable.
    """
    prefix, sep, suffix = name.partition("Unit")
    core = (
        suffix if sep and suffix and prefix and len(prefix) <= 5 and prefix[0].isupper() else name
    )
    if "." in core:
        head, _, tail = core.partition(".")
        # Map ``D.\x00`` / ``D.\x01`` → ``D0`` / ``D1`` so the column stays
        # printable and distinguishes the replicated sub-units.
        index_bytes = [b for b in tail.encode("utf-8", errors="ignore") if b < 32]
        if index_bytes:
            return f"{head}{index_bytes[0]}"
        printable_tail = "".join(ch for ch in tail if ch.isprintable() and ch != "\x00")
        return f"{head}{printable_tail}" if printable_tail else head
    return core


def _pressure_by_index(
    payload: dict[str, Any],
    payload_region: dict[str, Any],
) -> dict[int, list[tuple[str, float]]]:
    """Build ``{InstructionIndex: [(port, usage), ...]}`` from llvm-mca.

    ``TargetInfo.Resources`` lives at the payload top level, while
    ``ResourcePressureView.ResourcePressureInfo`` is per-region. Joins
    the two tables and drops entries whose ``ResourceUsage`` rounds to
    zero at two decimal places.
    """
    target_info = payload.get("TargetInfo") or payload_region.get("TargetInfo") or {}
    resources = target_info.get("Resources") or []
    view = payload_region.get("ResourcePressureView") or {}
    pressure_info = view.get("ResourcePressureInfo") or []
    result: dict[int, list[tuple[str, float]]] = {}
    for entry in pressure_info:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("InstructionIndex")
        res_idx = entry.get("ResourceIndex")
        usage = entry.get("ResourceUsage")
        if not isinstance(idx, int) or not isinstance(res_idx, int):
            continue
        if not isinstance(usage, (int, float)) or round(float(usage), 2) <= 0.0:
            continue
        if res_idx < 0 or res_idx >= len(resources):
            continue
        raw_name = str(resources[res_idx] or "")
        if not raw_name:
            continue
        port = _format_port_name(raw_name)
        result.setdefault(idx, []).append((port, float(usage)))
    return result


def _format_ports(entries: list[tuple[str, float]]) -> str:
    """``[('V0', 0.5), ('V1', 0.5)]`` → ``'0.50*V0 0.50*V1'``.

    Uses uops.info's ``count*port`` convention so the ARM / RISC-V modeled
    ``ports`` column lines up with the format readers already expect from
    the x86 measured side.
    """
    return " ".join(f"{usage:.2f}*{name}" for name, usage in entries)


def _kind_label(info: dict[str, Any]) -> str:
    """Map ``mayLoad`` / ``mayStore`` flags to a compact label or ``''``."""
    may_load = bool(info.get("mayLoad"))
    may_store = bool(info.get("mayStore"))
    if may_load and may_store:
        return "load+store"
    if may_load:
        return "load"
    if may_store:
        return "store"
    return ""


def _build_perf_rows(
    payload: dict[str, Any],
    asm_lines: list[str],
    *,
    core: CoreSpec,
    mca_version: str,
) -> list[PerfRow]:
    """Join an ``llvm-mca --json`` payload to the asm lines that produced it.

    ``payload`` is the parsed ``--instruction-tables=full --json``
    output. The payload's ``InstructionInfoView.InstructionList`` is
    parallel to the payload's own ``Instructions`` array, which reflects
    any lines that llvm-mca skipped via
    ``--skip-unsupported-instructions``. When the two lists' lengths
    match we prefer the payload's labels as the ground truth; otherwise
    we fall back to the caller-supplied ``asm_lines``.

    Emits one :class:`PerfRow` per unique mnemonic (first-seen wins —
    same-mnemonic form variants contend for the same
    ``arch_details[core]`` slot anyway in
    :func:`merge_perf_rows`).
    """
    regions = payload.get("CodeRegions") or []
    if not regions:
        return []
    region = regions[0]
    info_list = (region.get("InstructionInfoView") or {}).get("InstructionList") or []
    payload_insts = region.get("Instructions") or []
    labels: list[str]
    if len(payload_insts) == len(info_list) and all(
        isinstance(item, str) for item in payload_insts
    ):
        labels = list(payload_insts)
    else:
        labels = list(asm_lines)
    arch_label = perf_architecture(core.architecture)
    pressure_by_index = _pressure_by_index(payload, region)

    rows: list[PerfRow] = []
    seen: set[str] = set()
    for idx, (asm_line, info) in enumerate(zip(labels, info_list)):
        if not isinstance(info, dict):
            continue
        mnemonic = _asm_mnemonic(asm_line)
        if not mnemonic or mnemonic in seen:
            continue
        seen.add(mnemonic)
        latency = info.get("Latency")
        rthroughput = info.get("RThroughput")
        cpi = ""
        if isinstance(rthroughput, (int, float)):
            cpi = fmt_cycles(rthroughput)
        extra: dict[str, str] = {}
        num_uops = info.get("NumMicroOpcodes")
        if isinstance(num_uops, (int, float)) and num_uops > 0:
            extra["uops"] = str(int(num_uops))
        ports_str = _format_ports(pressure_by_index.get(idx, []))
        if ports_str:
            extra["ports"] = ports_str
        kind = _kind_label(info)
        if kind:
            extra["kind"] = kind
        rows.append(
            PerfRow(
                mnemonic=mnemonic,
                core=core.canonical_id,
                source="llvm-mca",
                source_kind="modeled",
                source_version=mca_version,
                architecture=arch_label,
                latency=str(latency) if latency is not None else "",
                cpi=cpi,
                applies_to="mnemonic",
                citation_url=LLVM_MCA_CITATION,
                extra_measurement=extra,
            )
        )
    return rows


def _disassembly_to_asm_lines(disasm_text: str) -> list[str]:
    """Turn ``_filter_disassembly`` output back into a list of asm lines."""
    return [line for line in disasm_text.splitlines() if line.strip()]


def collect_core_schedule(
    core: CoreSpec,
    *,
    mca_version: str,
    cache_root: Path | None = None,
    executable_exegesis: str = "llvm-exegesis",
    executable_mc: str = "llvm-mc",
    executable_mca: str = "llvm-mca",
) -> list[PerfRow]:
    """Run the llvm-exegesis → llvm-mc → llvm-mca pipeline for one core.

    Returns one :class:`PerfRow` per unique assembly mnemonic that
    llvm-mca's scheduling model produces latency / throughput data for.

    Cache layout: ``<cache_root>/<triple>/<cpu>/{exegesis.yaml,
    disassembly.s, mca.json}``. Callers pin ``mca_version`` into
    ``cache_root`` themselves if they want LLVM-version isolation.

    Raises :class:`LLVMMcaUnavailable` when a required binary is missing
    and :class:`LLVMSchedulingError` (a ``LLVMMcaError`` subclass) on
    any subprocess failure, empty result, or malformed output — no
    silent fallback.
    """
    if core.architecture not in {"aarch64", "riscv"}:
        return []

    root = cache_root if cache_root is not None else DEFAULT_CACHE_ROOT
    cache_dir = root / core.llvm_triple / core.llvm_cpu
    exegesis_path = cache_dir / "exegesis.yaml"
    disasm_path = cache_dir / "disassembly.s"
    mca_path = cache_dir / "mca.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not exegesis_path.exists():
        _run_exegesis(core, exegesis_path, executable=executable_exegesis)

    entries = _parse_exegesis_yaml(exegesis_path.read_text())
    if not entries:
        raise LLVMSchedulingError(
            f"{executable_exegesis} produced no entries for {core.canonical_id}"
        )

    if not disasm_path.exists():
        hex_lines = build_byte_lines(entries, core.architecture)
        if not hex_lines:
            raise LLVMSchedulingError(f"could not extract any opcode bytes from {exegesis_path}")
        raw = _run_disassemble(hex_lines, core, executable=executable_mc)
        disasm_path.write_text(_filter_disassembly(raw))

    disasm_text = disasm_path.read_text()
    asm_lines = _disassembly_to_asm_lines(disasm_text)
    if not asm_lines:
        raise LLVMSchedulingError(
            f"{executable_mc} produced empty disassembly for {core.canonical_id}"
        )

    if not mca_path.exists():
        payload = _run_mca(disasm_text, core, executable=executable_mca)
        mca_path.write_text(json.dumps(payload))
    else:
        payload = json.loads(mca_path.read_text())

    rows = _build_perf_rows(payload, asm_lines, core=core, mca_version=mca_version)
    if not rows:
        raise LLVMSchedulingError(
            f"{executable_mca} produced no schedule rows for {core.canonical_id}"
        )
    return rows
