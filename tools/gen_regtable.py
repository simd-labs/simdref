"""Generate src/simdref/regtable.py from the pinned LLVM's register data.

Usage: python3 tools/gen_regtable.py TBLGEN LLVM_ROOT [OUT]

Runs llvm-tblgen --dump-json over the four targets, reads the register and
register-class records (names, classes, widths, superclasses), and writes the
checked-in table. simdref reads no LLVM data at runtime.

The class map below names the register classes the matcher classifies; the
name sets come from LLVM, never from a hand list. The first RegAltName of a
member gives its alias spelling (Q0 -> v0, X8 -> s0). The ARM32 operand names
are the numbered spellings plus pc and the GAS alias ip (R12; the ARM
assembler's regname table spells it). The AArch64 ZA tile slices are not
register definitions; lsp.py keeps the code rule for them. sp, bp, si, di
and s1 edge cases are code rules in lsp.py, not table data.
"""

import json
import subprocess
import sys
import tempfile

TARGETS = ("AArch64", "X86", "RISCV", "ARM")

# family -> class name -> (klass of the names, klass of the first alias)
CLASS_MAP = {
    "x86": {
        "GR8": ("gpr", None),
        "GR16": ("gpr", None),
        "GR32": ("gpr", None),
        "GR64": ("gpr", None),
        "VR128": ("vec", None),
        "VR128X": ("vec", None),
        "VR256": ("vec", None),
        "VR256X": ("vec", None),
        "VR512": ("vec", None),
        "VR512_0_15": ("vec", None),
        "VR64": ("mm", None),
        "VK1": ("k", None),
        "VK2": ("k", None),
        "VK4": ("k", None),
        "VK8": ("k", None),
        "VK16": ("k", None),
        "VK32": ("k", None),
        "VK64": ("k", None),
    },
    "a64": {
        "GPR64": ("xw", None),
        "GPR32": ("xw", None),
        "FPR8": ("fp", None),
        "FPR16": ("fp", None),
        "FPR32": ("fp", None),
        "FPR64": ("fp", None),
        "FPR128": ("fp", "v"),
        "ZPR": ("z", None),
        "PPR": ("p", None),
        "PNR": ("pn", None),
    },
    "riscv": {
        # The x0-x31 AsmNames are not ABI spellings; only the first alias is.
        "GPR": (None, "rvabi"),
        "VR": ("v", None),
        "VRM2": ("v", None),
        "VRM4": ("v", None),
        "VRM8": ("v", None),
    },
    "arm32": {"GPR": ("arm", "arm")},
}

FAMILIES = {"AArch64": "a64", "X86": "x86", "RISCV": "riscv", "ARM": "arm32"}


def type_bits(type_name, data):
    """Bit width of a TableGen value type, from its ValueType record."""
    record = data.get(type_name) or {}
    if record.get("Size"):
        return record["Size"]
    widths = [type_bits(t["printable"], data) for t in record.get("Objects") or []]
    widths = [w for w in widths if w]
    return max(widths) if widths else 0


def class_width(record, data):
    widths = [type_bits(t["printable"], data) for t in record.get("RegTypes") or []]
    widths = [w for w in widths if w]
    return max(widths) if widths else 0


def expand_members(data, dag, out, depth=0):
    """Walk a MemberList dag: add chains, nested dags and sequence patterns."""
    if depth > 8 or not isinstance(dag, dict):
        return
    operator = (dag.get("operator") or {}).get("def")
    args = dag.get("args") or []
    if operator == "add":
        for arg in args:
            ref = arg[0]
            if not isinstance(ref, dict):
                continue
            if ref.get("kind") == "dag":
                expand_members(data, ref, out, depth + 1)
            elif ref.get("kind") == "def":
                name = ref["def"]
                if "RegisterClass" in (data.get(name, {}).get("!superclasses") or []):
                    expand_members(data, data[name].get("MemberList"), out, depth + 1)
                else:
                    out.add(name)
    elif operator == "sequence":
        pattern, low, high = args[0][0], args[1][0], args[2][0]
        for number in range(low, high + 1):
            out.add(pattern.replace("%u", str(number)))


def arm32_spelling_ok(spelling):
    """ARM32 operand spellings: the numbered registers plus pc and ip.

    sp and lr are stack/link spellings; lsp.py votes them in code.
    """
    if spelling in ("pc", "ip"):
        return True
    return spelling.startswith("r") and spelling[1:].isdigit()


def x86_spelling_ok(spelling):
    """x86 operand spellings the matcher classifies.

    rip is an address operand and r16-r31 are APX names the matcher does not
    class; lsp._x86_gpr_width keeps the same rule at lookup time.
    """
    if spelling == "rip":
        return False
    tail = spelling[1:]
    if spelling.startswith("r") and tail[:1].isdigit():
        index = 0
        while tail[index : index + 1].isdigit():
            index += 1
        if tail[index:] not in ("", "d", "w", "b") or not 8 <= int(tail[:index]) <= 15:
            return False
    return True


def collect(data, family, rows):
    """Add every class-mapped spelling of one target dump to rows."""
    wanted = CLASS_MAP[family]
    for name, record in sorted(data.items()):
        if not isinstance(record, dict):
            continue
        if "RegisterClass" not in (record.get("!superclasses") or []):
            continue
        klass, alias_klass = wanted.get(name, (None, None))
        if klass is None and alias_klass is None:
            continue
        width = class_width(record, data)
        members = set()
        expand_members(data, record.get("MemberList"), members)
        for member in sorted(members):
            entry = data.get(member) or {}
            spellings = [entry.get("AsmName")]
            aliases = [t for t in entry.get("AltNames") or [] if isinstance(t, str) and t]
            if alias_klass is not None and aliases:
                spellings.append(aliases[0])
            for index, spelling in enumerate(spellings):
                if not spelling:
                    continue
                kind = klass if index == 0 else alias_klass
                if kind is None:
                    continue
                if family == "arm32" and not arm32_spelling_ok(spelling):
                    continue
                if family == "x86" and not x86_spelling_ok(spelling):
                    continue
                rows.setdefault(spelling, {})
                key = (kind, family)
                rows[spelling][key] = max(rows[spelling].get(key, 0), width)
    if family == "arm32":
        # The GAS alias ip for R12 is a parser spelling, not a register name.
        rows.setdefault("ip", {})
        rows["ip"][("arm", "arm32")] = 32


def main():
    tblgen, llvm_root = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "src/simdref/regtable.py"
    tag = (
        subprocess.run(
            ["git", "-C", llvm_root, "describe", "--tags"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )
    rows = {}
    with tempfile.TemporaryDirectory() as dumps:
        for target in TARGETS:
            entry = f"{llvm_root}/llvm/lib/Target/{target}/{target}.td"
            dump = f"{dumps}/{target}.json"
            with open(dump, "w") as handle:
                subprocess.run(
                    [
                        tblgen,
                        "--dump-json",
                        entry,
                        "-I",
                        f"{llvm_root}/llvm/include",
                        "-I",
                        f"{llvm_root}/llvm/lib/Target/{target}",
                        "-I",
                        f"{llvm_root}/llvm/lib/Target",
                    ],
                    stdout=handle,
                    check=True,
                )
            with open(dump) as handle:
                collect(json.load(handle), FAMILIES[target], rows)
    grouped = {}
    for spelling in sorted(rows):
        for (kind, family), width in sorted(rows[spelling].items()):
            grouped.setdefault((family, kind, width), []).append(spelling)
    lines = [
        '"""Register spellings, classes, widths and families, generated from LLVM.',
        "",
        f"Generated by tools/gen_regtable.py from LLVM {tag}. Do not edit by",
        "hand; rerun the generator instead. Lookup is exact after casefold.",
        "The edge-case rules stay in lsp.py: sp/bp/si/di are x86 only with a",
        "'%' prefix, x86 number registers stop at r8-r15, and s1 is a scalar",
        "FP register that RISC-V also names.",
        '"""',
        "",
        f'LLVM_TAG = "{tag}"',
        "",
        "# (klass, width, family, spellings)",
        "_REG_ROWS = (",
    ]
    for family, kind, width in sorted(grouped):
        names = ",\n".join(f'            "{n}"' for n in grouped[(family, kind, width)])
        lines += [
            "    (",
            f'        "{kind}",',
            f"        {width},",
            f'        "{family}",',
            "        (",
            f"{names},",
            "        ),",
            "    ),",
        ]
    lines += [
        ")",
        "",
        "REG_TABLE = {}",
        "for _kind, _width, _family, _names in _REG_ROWS:",
        "    for _name in _names:",
        "        REG_TABLE.setdefault(_name, []).append((_kind, _width, _family))",
        "for _name in REG_TABLE:",
        "    REG_TABLE[_name] = tuple(sorted(REG_TABLE[_name]))",
        "",
        "",
        "def reg_rows(name: str) -> tuple:",
        '    """The (klass, width, family) rows of a register spelling."""',
        "    return REG_TABLE.get(name, ())",
        "",
        "",
        "def reg_has(name: str, klass: str, family: str) -> bool:",
        '    """True when the spelling has a row of that class and family."""',
        "    return any(k == klass and f == family for k, _w, f in REG_TABLE.get(name, ()))",
        "",
        "",
        "def reg_width(name: str, klass: str, family: str) -> int | None:",
        '    """The width of the spelling\'s row of that class and family."""',
        "    for k, width, f in REG_TABLE.get(name, ()):",
        "        if k == klass and f == family:",
        "            return width",
        "    return None",
        "",
    ]
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"wrote {out_path}: {len(rows)} spellings, LLVM {tag}")


if __name__ == "__main__":
    main()
