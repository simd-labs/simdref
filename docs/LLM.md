# `simdref llm`: structured output for LLM and tool consumers

`isa llm` (also `simdref llm`) is the subcommand group for programmatic consumers: agents, skills, editor integrations, scripts.

The CLI ships as `isa` (short) and `simdref` (explicit). Same code. Examples use `isa`.

Each subcommand gives stable JSON or NDJSON on stdout. Diagnostics go to stderr. Pipe stdout into `jq` or `json.loads`.

## Subcommands

### `isa llm query QUERY...`

Resolve one query (intrinsic name, mnemonic, free-form search) and give one payload.

| Flag                                   | Default | Description                                                                  |
| -------------------------------------- | ------- | ---------------------------------------------------------------------------- |
| `--format, -F json\|ndjson\|markdown`  | `json`  | Pretty JSON, one-record-per-line NDJSON, Markdown.                           |
| `--limit N`                            | `8`     | Search-results cap when the query falls through to search mode.              |
| `--isa FAM` (repeatable)               | all     | Filter by ISA family (`Intel`, `Arm`, `RISC-V`, …).                          |
| `--preset NAME`                        | none    | Preset name (`default`, `intel`, `arm32`, `arm64`, `riscv`, `none`, `all`).  |
| `--source-kind measured\|modeled\|any` | `any`   | Perf-row provenance. `measured` = uops.info rows. `modeled` = llvm-mca rows. |
| `--arch CORE`                          | none    | Pin lat/cpi/ports to one core (`znver4`, `skylake-x`).                       |

Payload shape (abridged):

```json
{
  "query": "_mm_add_epi32",
  "mode": "exact",
  "match_kind": "intrinsic",
  "result": {
    "intrinsic": "_mm_add_epi32",
    "signature": "__m128i _mm_add_epi32(__m128i a, __m128i b)",
    "instructions": ["paddd"],
    "instruction_refs": [{ "key": "...", "name": "...", "form": "...", "architecture": "...", "xed": "...", "resolution": "...", "match_count": 1 }],
    "isa": ["SSE2"],
    "lat": "1",
    "cpi": "0.5",
    "summary": "Add packed 32-bit integers."
  }
}
```

Free-form search yields `{"mode": "search", "results": [...]}`. Each entry has the `result` shape.

### `isa llm batch`

Reads queries one per line from stdin, gives one NDJSON record per line. Loads the catalog one time. Skips empty lines, `#` comments.

```bash
echo -e "_mm_add_ps\nVPADDD\n_does_not_exist" | isa llm batch
```

Each output record: `{"query": ..., "status": ..., "payload": {...}}`. `status` is `match`, `no_match`, `ambiguous`, or `error`. On `error` the record has an `error` field.

Accepts the same flags as `query`.

### `isa llm list`

No arguments: gives the full `FilterSpec` of ISA families, sub-ISAs, the category catalog:

```bash
isa llm list --format json
isa llm list --format markdown
```

With `--pattern GLOB [--isa FAM]`, streams NDJSON `{name, kind, isa, category}` for each entry name that agrees with the glob. `fnmatch` semantics (`*`, `?`, `[...]`), case-insensitive, on the entry name and `db_key`.

```bash
isa llm list --pattern "*gather*" --isa "Intel"
```

### `isa llm schema`

JSON Schema for the `query` and `batch` payload shape, with `generated_at`, `source_versions`, nested `instruction_refs`. Use it to make client-side types.

## Exit codes

| Code | Meaning                                                    |
| ---- | ---------------------------------------------------------- |
| `0`  | Match (intrinsic, instruction, minimum one search result). |
| `1`  | Usage error: bad flag, unknown preset, missing argument.   |
| `2`  | Query correct, no catalog hit.                             |
| `3`  | Ambiguous: multiple exact hits on the same mnemonic.       |
| `10` | Internal error (exception in resolution).                  |

## Skill recipe

Agent loop on assembly or codegen:

1. Parse the assembly, pull mnemonics.
1. Pre-filter: `isa llm list --pattern "VPADD*" --isa "Intel"`.
1. Resolve in one batch, measured perf only: `printf '%s\n' "${mnemonics[@]}" | isa llm batch --source-kind measured`.
1. Consume NDJSON. Each record gives `lat`, `cpi`, linked intrinsic, `summary`, enough to propose a swap and cite latency.
1. Use `generated_at` and `source_versions` to flag stale catalogs.
