"""Intel Intrinsics Guide offline perf tables (latency and throughput).

Source: the ``Intel-Intrinsics-Guide-Offline`` bundle Intel publishes for
the Intrinsics Guide. ``perf2.js`` holds per-form latency (``l``) and
throughput in cycles per instruction (``t``) for five Intel cores.
``notes.json`` holds the rounding and suppress-all-exceptions note text.

Download links (the offline bundle is the only one that ships perf2.js):

- https://cdrdv2.intel.com/v1/dl/getContent/764289?fileName=Intel-Intrinsics-Guide-Offline-3.6.4.zip
- https://www.intel.com/content/dam/develop/public/us/en/include/intrinsics-guide/files/data.js
- https://www.intel.com/content/dam/develop/public/us/en/include/intrinsics-guide/files/intrinsics.json

The values come from Intel's Software Developer Manuals, so they are
vendor-guide numbers, not measurements: ``source_kind`` is ``"modeled"``.

The bundle is used under Intel's removal-request policy stated in the
project README. Vendor only the derived JSON, never the raw bundle.

Refresh: ``python -m simdref.perf_sources.intel_guide refresh RAWDIR``
rebuilds ``vendor/intel_guide/*.json`` from a raw extract of the bundle
(the directory holding ``perf2.js`` and ``notes.json``).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from simdref.perf_sources.cores import canonical_core_id
from simdref.perf_sources.merge import SOURCE_INTEL_GUIDE, PerfRow, write_json

SOURCE_VERSION = "3.6.9"
CITATION_URL = "https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html"

# perf2.js labels have spaces and do not match any cores.py alias, so this
# map bridges them to canonical ids without touching cores.py. Intel does
# not say whether "Skylake" is client or server and whether "Alderlake" is
# the P-core or E-core; the client ids (SKL, ADL-P) follow the uops.info
# convention. cores.py aliases map Sapphire Rapids to EMR (same Golden
# Cove P-core microarchitecture). "Icelake Xeon" maps to the ICX server
# id, so the client and server rows no longer collide.
_CPU_CORE: dict[str, str] = {
    "Alderlake": "ADL-P",
    "Skylake": "SKL",
    "Sapphire Rapids": "EMR",
    "Icelake Intel Core": "ICL",
    "Icelake Xeon": "ICX",
}

# When two perf2.js labels map to one canonical core, keep the row whose
# label sorts first here. The Icelake labels now map to distinct cores,
# so no label pair collides today; the guard stays for future additions.
_CPU_PREFERENCE = (
    "Alderlake",
    "Skylake",
    "Sapphire Rapids",
    "Icelake Intel Core",
    "Icelake Xeon",
)
_CPU_RANK = {name: rank for rank, name in enumerate(_CPU_PREFERENCE)}

# Guard: every mapped id must stay canonical. Canonical ids live in
# cores.py; an alias that loses its entry would vent wrong cores here.
if any(canonical_core_id(core) != core for core in _CPU_CORE.values()):
    raise RuntimeError("_CPU_CORE holds a non-canonical core id")


def _perf_entries(text: str):
    """Yield ``(form_key, cpu, latency, throughput)`` for each perf2.js row.

    shortcut: perf2.js is machine-generated, one row per line with fixed
    quoting (no escapes, fields always l then t), so a line scan replaces
    the two regexes this used to be; upgrade to a real JS parser if a
    future release adds escaping or reorders the fields.
    """
    current: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.endswith('":['):
            current = s[1:-3]
            continue
        if current is None or not s.startswith('{"'):
            continue
        cp_close = s.find('"', 2)
        if cp_close < 2:
            continue
        cpu = s[2:cp_close]
        l_at = s.find('l:"', cp_close)
        if l_at < 0:
            continue
        l_open = l_at + 3
        l_close = s.find('"', l_open)
        t_at = s.find('t:"', l_close)
        if t_at < 0:
            continue
        t_open = t_at + 3
        t_close = s.find('"', t_open)
        if t_close < 0:
            continue
        yield current, cpu, s[l_open:l_close], s[t_open:t_close]


def _row(
    mnemonic: str, form: str, core: str, source_version: str, latency: str, cpi: str
) -> PerfRow:
    """One form-scoped perf row from the guide's per-core table."""
    return PerfRow(
        mnemonic=mnemonic,
        form=form,
        core=core,
        source=SOURCE_INTEL_GUIDE,
        source_kind="modeled",
        source_version=source_version,
        architecture="x86",
        latency=latency,
        cpi=cpi,
        applies_to="form",
        citation_url=CITATION_URL,
    )


@dataclass(frozen=True)
class IntelGuideParse:
    """Parser output: the kept ``PerfRow``s plus drop statistics."""

    rows: list[PerfRow]
    # perf2.js CPU label -> count of rows dropped (unknown CPU, or the
    # losing row of an Icelake client/server collision).
    dropped: dict[str, int] = field(default_factory=dict)


def _mnemonic(key: str) -> str:
    return key.split("_", 1)[0]


def parse_perf_js(text: str) -> IntelGuideParse:
    """Parse the ``perf2.js`` object literal into :class:`PerfRow`s.

    Pure: no IO. Empty latency or throughput strings stay empty, never
    coerced to ``"0"``.
    """
    kept: list[PerfRow] = []
    seen: Counter[str] = Counter()
    kept_label: Counter[str] = Counter()
    # form key -> canonical core -> (cpu label, latency, throughput); the
    # label that sorts first in _CPU_PREFERENCE wins a two-label collision.
    per_form: dict[str, dict[str, tuple[str, str, str]]] = {}
    for key, cpu, latency, cpi in _perf_entries(text):
        seen[cpu] += 1
        core = _CPU_CORE.get(cpu)
        if core is None:
            continue
        per_core = per_form.setdefault(key, {})
        existing = per_core.get(core)
        if existing is None or _CPU_RANK[cpu] < _CPU_RANK[existing[0]]:
            per_core[core] = (cpu, latency, cpi)
    for key, per_core in per_form.items():
        mnemonic = _mnemonic(key)
        for core, (cpu, latency, cpi) in per_core.items():
            kept_label[cpu] += 1
            kept.append(_row(mnemonic, key, core, SOURCE_VERSION, latency, cpi))
    dropped = {label: count - kept_label[label] for label, count in seen.items()}
    return IntelGuideParse(rows=kept, dropped=dropped)


def parse_notes_json(text: str) -> dict[str, str]:
    """Parse ``notes.json`` into a ``note_name -> note_value`` mapping."""
    payload = json.loads(text)
    return {note["note_name"]: note["note_value"] for note in payload["notes"]}


def default_vendor_dir() -> Path:
    """Return the repo ``vendor/intel_guide`` directory."""
    return Path(__file__).resolve().parents[3] / "vendor" / "intel_guide"


def load_vendored(path: str | Path | None = None) -> list[PerfRow]:
    """Read the vendored ``vendor/intel_guide/*.json`` files into rows."""
    vendor = Path(path) if path is not None else default_vendor_dir()
    rows: list[PerfRow] = []
    for file in sorted(vendor.glob("*.json")):
        if file.name == "notes.json":
            continue  # note text, not perf rows
        payload = json.loads(file.read_text())
        for entry in payload["rows"]:
            rows.append(
                _row(
                    entry["mnemonic"],
                    entry["form"],
                    payload["core"],
                    payload["source_version"],
                    entry["latency"],
                    entry["cpi"],
                )
            )
    return rows


def refresh(rawdir: str | Path, outdir: str | Path | None = None) -> IntelGuideParse:
    """Parse raw ``perf2.js`` and ``notes.json`` and rewrite the vendored JSON.

    Writes one JSON file per core plus ``notes.json``. Returns the parse
    result so the caller can report kept and dropped rows.
    """
    rawdir = Path(rawdir)
    outdir = Path(outdir) if outdir is not None else default_vendor_dir()
    result = parse_perf_js((rawdir / "perf2.js").read_text())
    outdir.mkdir(parents=True, exist_ok=True)
    by_core: dict[str, list[dict[str, str]]] = {}
    for row in result.rows:
        by_core.setdefault(row.core, []).append(
            {
                "mnemonic": row.mnemonic,
                "form": row.form,
                "latency": row.latency,
                "cpi": row.cpi,
            }
        )
    for core, entries in sorted(by_core.items()):
        payload = {
            "source": SOURCE_INTEL_GUIDE,
            "source_version": SOURCE_VERSION,
            "citation_url": CITATION_URL,
            "core": core,
            "rows": sorted(entries, key=lambda e: e["form"]),
        }
        write_json(payload, outdir / f"{core}.json")
    notes = parse_notes_json((rawdir / "notes.json").read_text())
    write_json(notes, outdir / "notes.json")
    return result


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] != "refresh" or len(args) != 2:
        print("usage: python -m simdref.perf_sources.intel_guide refresh RAWDIR", file=sys.stderr)
        return 2
    result = refresh(args[1])
    counts = Counter(row.core for row in result.rows)
    for core in sorted(counts):
        print(f"{core}: {counts[core]} rows vendored")
    for label, count in sorted(result.dropped.items()):
        if count:
            print(f"dropped {count} rows for unknown or colliding CPU label {label!r}")
    print(f"total: {len(result.rows)} rows vendored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
