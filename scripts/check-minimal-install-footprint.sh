#!/usr/bin/env bash
# Row 14 check: a fresh-venv wheel install with an empty HOME/XDG must run a
# lookup, `annotate`, `llm query` and a headless TUI start, then leave exactly
# one file under HOME: catalog.db (no -wal/-shm sidecars, no legacy
# installed_version file, no CATALOG_PATH msgpack blob).
set -euo pipefail

CORE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(mktemp -d /tmp/simdref-footprint-check.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

echo "== build wheel =="
(cd "$CORE" && nice -n19 uv build --wheel --out-dir "$WORK/dist") >"$WORK/build.log" 2>&1
echo "exit=$?"
WHEEL=$(ls "$WORK"/dist/*.whl)

echo "== fresh venv (built under the real HOME, so uv can resolve deps from its cache) =="
python3 -m venv "$WORK/venv"
SIMDREF="$WORK/venv/bin/simdref"
nice -n19 uv pip install --python "$WORK/venv/bin/python" --quiet "$WHEEL" >"$WORK/pip-install.log" 2>&1
echo "exit=$?"

echo "== switch to an isolated fake HOME/XDG for every simdref invocation below =="
export HOME="$WORK/home"
mkdir -p "$HOME"
export XDG_DATA_HOME="$HOME/.local/share"
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_STATE_HOME="$HOME/.local/state"

echo "== seed catalog.db at the wheel-install data dir =="
DATA_DIR="$XDG_DATA_HOME/simdref"
mkdir -p "$DATA_DIR"
cp "$CORE/data/derived/catalog.db" "$DATA_DIR/catalog.db"

echo "== lookup (bare-query instruction mnemonic match) =="
"$SIMDREF" vaddps >"$WORK/lookup.log" 2>&1
echo "exit=$?"

echo "== annotate on a small .s file =="
"$SIMDREF" annotate "$CORE/tests/fixtures/hello_simd.s" -o "$WORK/hello_simd.sa" >"$WORK/annotate.log" 2>&1
echo "exit=$?"

echo "== llm query =="
"$SIMDREF" llm query _mm_add_epi32 >"$WORK/llm.log" 2>&1
echo "exit=$?"

echo "== headless TUI start =="
"$WORK/venv/bin/python" - <<'PYEOF' >"$WORK/tui.log" 2>&1
import asyncio
from simdref.tui import SimdrefApp

async def scenario():
    app = SimdrefApp()
    async with app.run_test() as pilot:
        await pilot.pause()

asyncio.run(scenario())
print("tui started and quit cleanly")
PYEOF
echo "exit=$?"

echo "== footprint under HOME: exactly one file, catalog.db =="
mapfile -t files < <(find "$HOME" -type f | sort)
echo "found ${#files[*]} file(s):"
printf '  %s\n' "${files[@]}"
if [ "${#files[@]}" -eq 0 ]; then
	echo "FAIL: find returned an empty list (positive control: this must fail, not pass silently)"
	exit 1
fi
if [ "${#files[@]}" -ne 1 ] || [ "$(basename "${files[0]}")" != "catalog.db" ]; then
	echo "FAIL: expected exactly one file named catalog.db"
	exit 1
fi
echo "PASS: HOME contains exactly one file: catalog.db"
