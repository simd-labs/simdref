# simdref perf baseline

All numbers captured with:

- commit `15b514a` (pre-perf work)
- catalog ≈ 117 K intrinsics + 25 K instructions
- Python 3.13, Linux x86_64, Chromium via Playwright 1.x (headless)
- server: bare `python -m http.server 8765` (no gzip, no cache)
- profiler: `tools/profile_web.py --query add`
- TUI profiler: `SIMDREF_PROFILE=1 python -m simdref.tui`

Re-run with:

```
# Web
python -m simdref web                    # export to web/
(cd web && python -m http.server 8765 &)
/tmp/pw-venv/bin/python tools/profile_web.py --query add

# TUI
PYTHONPATH=src SIMDREF_PROFILE=1 python -c "from simdref.tui import _fts_search; ..."
```

## Web: 2026-04-17 baseline

| Metric                         | Value                                       |
| ------------------------------ | ------------------------------------------- |
| `search-index.json` raw        | 182 MB (189 874 420 B)                      |
| `search-index.json` gz (ideal) | ~6.0 MB (6 267 558 B); server does not emit |
| `intrinsic-details.json` raw   | 118 MB (122 971 686 B), lazy-loaded         |
| `filter_spec.json`             | 44 KB                                       |
| `index.html`                   | 76 KB (inlined CSS + JS)                    |
| Cold load to `networkidle`     | 2.62 s                                      |
| First paint                    | 88 ms                                       |
| JS heap after load (used)      | 703 MB                                      |
| JS heap (total)                | 747 MB                                      |
| Time-to-first-result 'a'       | 56 ms                                       |
| Time-to-first-result 'd'       | 135 ms                                      |
| Time-to-first-result 'd' (3rd) | 684 ms                                      |

`networkidle` does not include the lazy `intrinsic-details.json` fetch. Real cold load after the user picks an intrinsic is longer.

## TUI: 2026-04-17 baseline

`_fts_search(conn, 'add', families={Intel,Arm,RISC-V}, subs={Intel: SSE..AVX-512}, limit=20)`

| Metric                          | Value         |
| ------------------------------- | ------------- |
| Wall time (no profiler)         | 127 ms        |
| Wall time (with cProfile)       | 320 ms        |
| `_isa_visible` cumulative       | 227 ms (71 %) |
| `display.isa_family` cumulative | 172 ms        |
| `sqlite3.execute` cumulative    | 55 ms         |

Top cProfile frames (cumulative):

```
   ncalls   tottime  cumtime  function
        1     0.002    0.319  _fts_search
    17540     0.024    0.227  _isa_visible              <- Python-side filter
    25807     0.032    0.172  display.isa_family        <- called per row
        2     0.002    0.164  _query_intrinsic_rows
    25807     0.023    0.096  display.display_isa       <- called per row
    25807     0.039    0.064  display.normalize_token
       15     0.055    0.055  sqlite3.execute
    40077     0.027    0.048  display.normalize_isa_token
   223560     0.023    0.023  str.startswith
     7135     0.004    0.021  _isa_matches_sub
```

About 75 % of per-keystroke cost is Python-side ISA filtering of SQL candidates. SQL itself is 55 ms.

## After phase 1.1 + 1.2 (2026-04-17)

`_search_payload` slimmed (dropped `signature`/`header`/`url`/`metadata`/`notes`/`instruction_refs`/`search_tokens`/`display_isa_tokens`, promoted `arm_arch`, `category`, `primary_instr` to top level, `subtitle` shortened to 80 chars). Emitted `*.json.gz` sidecars; the client uses `DecompressionStream` to read gz directly on vanilla GitHub Pages. New `simdref serve` handler sets `Content-Encoding: gzip` when `Accept-Encoding: gzip` is present.

| Metric                        | Baseline | Now     | Delta |
| ----------------------------- | -------- | ------- | ----- |
| `search-index.json` raw       | 182 MB   | 54 MB   | −70 % |
| `search-index.json` on-wire   | 182 MB   | 2.08 MB | −99 % |
| `intrinsic-details.json` wire | 118 MB   | 3.43 MB | −97 % |
| Cold load to networkidle      | 2.62 s   | 1.40 s  | −46 % |
| First paint                   | 88 ms    | 96 ms   | ~     |
| JS heap (used)                | 703 MB   | 553 MB  | −21 % |
| Keystroke 'a'                 | 56 ms    | 43 ms   | −23 % |
| Keystroke 'd'                 | 135 ms   | 67 ms   | −50 % |
| Keystroke 'd' (3rd)           | 684 ms   | 30 ms   | −96 % |

Targets met: cold load under 2 s, keystroke p95 under 100 ms, search-index on-wire under 3 MB.

## After phase 2.1 (TUI SQL pushdown, 2026-04-17)

Sub-ISA and family filters moved into the FTS query as an extra `AND REPLACE(isa, '-', '') LIKE '%…%'` clause. SQLite drops filtered-out rows before handing them to Python, cutting `_fts_search` wall time about 4×.

| Metric               | Baseline | Now    | Delta |
| -------------------- | -------- | ------ | ----- |
| `_fts_search('add')` | 127 ms   | 38 ms  | −70 % |
| `_fts_search('mul')` | ~130 ms  | 28 ms  | −78 % |
| `_fts_search('vec')` | ~300 ms  | 198 ms | −34 % |

TUI target (under 40 ms p95 for common queries) holds for short prefixes. Broad queries like `vec` stay slower because many rows match the FTS expression.

## After phase 1.4 + 1.5 (web virtualisation + rAF batching, 2026-04-17)

Viewport virtualisation replaces the progressive-append result list (up to 5 000 DOM rows) with absolute-positioned rows in a fixed-height (88 px) wrapper, keeping about 60 rows in the DOM. Filter toggles and preset clicks now coalesce into a single `requestAnimationFrame` via `scheduleFilterRender()`; no flash between `rebuildVisibleSet`, `renderIsaFilters`, and `renderResults`.

| Metric              | After 1.1-1.3 | After 1.4+1.5                           |
| ------------------- | ------------- | --------------------------------------- |
| Cold load           | 1.40 s        | 1.35 s                                  |
| First paint         | 96 ms         | 408 ms (initial virtual-wrapper layout) |
| Keystroke 'a'       | 43 ms         | 42 ms                                   |
| Keystroke 'd'       | 67 ms         | 87 ms                                   |
| Keystroke 'd' (3rd) | 30 ms         | 27 ms                                   |

Biggest non-harness win: DOM node count at 5k-result scroll went from ~30k to ~60.

## After phase 2.2 + 2.4 (TUI incremental refresh + detail cache, 2026-04-17)

TUI sub-ISA bar short-circuits to in-place `set_enabled()` updates when only sub-ISA selection changes, removing the per-keystroke remount flash. Detail-pane record fetch is wrapped in an in-session LRU (16 slots); re-visits are free.

Micro-bench (detail lookup, 50 iterations): raw `load_intrinsic_from_db` 1.1 ms; cached is ~0 ms.

## Targets (from plan)

| Metric                  | Baseline      | Target                            |
| ----------------------- | ------------- | --------------------------------- |
| Web cold load           | 2.6 s         | under 2 s cold, under 500 ms warm |
| Web per-keystroke p95   | ~680 ms       | under 100 ms                      |
| Web JS heap             | 703 MB        | under 200 MB                      |
| `search-index.json` raw | 182 MB        | under 15 MB                       |
| `search-index.json` gz  | n/a           | under 3 MB on-wire                |
| TUI `_fts_search` p95   | 127 ms        | under 40 ms                       |
| TUI preset click        | visible flash | under 50 ms, no flash             |
