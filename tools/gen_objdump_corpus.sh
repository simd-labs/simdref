#!/usr/bin/env bash
# Regenerate tests/data/objdump: objdump listings of corpus.c in every format,
# plus expected.json with the toolchain truth for each instruction line.
# Usage: tools/gen_objdump_corpus.sh [OUTDIR]
# Needs clang, llvm-objdump, llvm-objcopy, GNU objdump, aarch64-linux-gnu-objdump
# and riscv64-linux-gnu-objdump. A missing GNU objdump skips its formats.
# Truth per instruction (no listing byte column feeds it):
#   address, mnemonic  llvm-objdump --no-show-raw-insn (GNU: objdump -d -w --no-show-raw-insn)
#   bytes              the raw .text slice [address, next address) from llvm-objcopy
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(realpath -m "${1:-$ROOT/tests/data/objdump}")
LOD=${LLVM_OBJDUMP:-llvm-objdump}
LOC=${LLVM_OBJCOPY:-llvm-objcopy}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"
declare -A TGT=(
	[x86_64]="--target=x86_64-linux-gnu"
	[aarch64]="--target=aarch64-linux-gnu"
	[riscv64]="--target=riscv64-linux-gnu -march=rv64gc"
	[thumb]="--target=thumbv7a-linux-gnueabihf -mthumb -mfpu=neon -mfloat-abi=hard"
	[arm]="--target=armv7a-linux-gnueabihf -marm -mfpu=neon -mfloat-abi=hard"
)
declare -A GNU=([x86_64]=objdump [aarch64]=aarch64-linux-gnu-objdump [riscv64]=riscv64-linux-gnu-objdump)
cd "$TMP" # relative object names keep the listing headers free of local paths
for a in "${!TGT[@]}"; do
	# Jump tables stay on: Thumb-2 tbb tables put .word and .short data in .text.
	# shellcheck disable=SC2086
	clang ${TGT[$a]} -O2 -ffreestanding -fno-asynchronous-unwind-tables -c "$OUT/corpus.c" -o "$a.o"
	"$LOC" -O binary --only-section=.text "$a.o" "$a.text.bin"
	"$LOD" -d -j .text "$a.o" >"$OUT/${a}__llvm_d.txt"
	"$LOD" -d -j .text --no-show-raw-insn "$a.o" >"$OUT/${a}__llvm_noraw.txt"
	"$LOD" -d -j .text --no-leading-addr "$a.o" >"$OUT/${a}__llvm_noaddr.txt"
	"$LOD" -d -j .text --print-imm-hex "$a.o" >"$OUT/${a}__llvm_immhex.txt"
	"$LOD" -d -j .text --no-show-raw-insn "$a.o" >"$a.truth_llvm.txt"
	g=${GNU[$a]:-}
	if [ -n "$g" ] && command -v "$g" >/dev/null; then
		"$g" -d -j .text "$a.o" >"$OUT/${a}__gnu_d.txt"
		"$g" -d -w -j .text --no-show-raw-insn "$a.o" >"$a.truth_gnu.txt"
		if [ "$a" = x86_64 ]; then
			"$g" -d -j .text --no-show-raw-insn "$a.o" >"$OUT/${a}__gnu_noraw.txt"
			"$g" -d -w -j .text "$a.o" >"$OUT/${a}__gnu_w.txt"
			"$g" -d -j .text -M intel "$a.o" >"$OUT/${a}__gnu_intel.txt"
			"$g" -d -w -j .text --no-show-raw-insn -M intel "$a.o" >"$a.truth_gnu_intel.txt"
		fi
	fi
done
# The trailing-whitespace hook strips the blank after a GNU wrap line; do it here too.
sed -i 's/[[:blank:]]*$//' "$OUT"/*__*.txt
python3 -I - "$OUT" "$TMP" <<'EOF'
import json, re, sys
from pathlib import Path

out, tmp = Path(sys.argv[1]), Path(sys.argv[2])
addr_re = re.compile(r"^\s*([0-9a-fA-F]+):\s")
hex_re = re.compile(r"[0-9a-fA-F]{2}|[0-9a-fA-F]{4}|[0-9a-fA-F]{8}")
prefixes = {"data16", "addr32", "cs", "ds", "es", "ss", "fs", "gs", "lock",
            "rep", "repe", "repne", "notrack", "bnd"}


def rows(path):
    """{address: mnemonic, or the directive for data} from a no-raw-insn listing."""
    found = {}
    for line in path.read_text().splitlines():
        m = addr_re.match(line)
        if not m:
            continue
        fields = [f for f in line[m.end():].split("\t") if f.strip()]
        # Data keeps its bytes in its own field before the directive.
        while len(fields) > 1 and fields[1].lstrip().startswith(".") and all(
                hex_re.fullmatch(t) for t in fields[0].split()):
            fields = fields[1:]
        toks = "\t".join(fields).split()
        while len(toks) > 1 and toks[0].lower() in prefixes:
            toks = toks[1:]
        found[int(m.group(1), 16)] = toks[0]
    return found


expected = {}
for listing in sorted(out.glob("*__*.txt")):
    arch, fmt = listing.stem.split("__")
    text = (tmp / f"{arch}.text.bin").read_bytes()
    llvm = rows(tmp / f"{arch}.truth_llvm.txt")
    names = llvm
    if fmt.startswith("gnu"):
        names = rows(tmp / f"{arch}.truth_gnu{'_intel' if fmt == 'gnu_intel' else ''}.txt")
    addrs = sorted(llvm) + [len(text)]  # .text starts at 0 in a relocatable object
    expected[listing.name] = [
        [a if fmt != "llvm_noaddr" else None,
         text[a:addrs[k + 1]].hex() if fmt not in ("llvm_noraw", "gnu_noraw") else None,
         names[a]]
        for k, a in enumerate(addrs[:-1]) if re.match(r"[A-Za-z]", llvm[a])
    ]
body = ",\n".join(
    f"{json.dumps(name)}: [\n" + ",\n".join(json.dumps(r) for r in recs) + "\n]"
    for name, recs in expected.items())
(out / "expected.json").write_text("{\n" + body + "\n}\n")  # one record per line
print(len(expected), "listings,", sum(map(len, expected.values())), "instruction records")
EOF
