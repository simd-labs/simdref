# `simdref llm`: structured output for LLM and tool consumers

`isa llm` (also `simdref llm`) is the subcommand group for programmatic consumers: agents, skills, editor integrations, and scripts that read SIMD intrinsics and instructions.

The CLI ships under two executable names: `isa` (short) and `simdref` (explicit). Both run the same code. Examples below use `isa`.

Each subcommand emits stable JSON or NDJSON on stdout. Progress, errors, and diagnostics go to stderr. The split lets callers pipe stdout into `jq` or `json.loads`.

## Subcommands

### `isa llm query QUERY...`

Resolve a single query (intrinsic name, instruction mnemonic, or free-form search) and emit one payload.

| Flag                                   | Default | Description                                                                                           |
| -------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| `--format, -F json\|ndjson\|markdown`  | `json`  | Pretty JSON, one-object-per-line NDJSON, or prompt-friendly Markdown.                                 |
| `--limit N`                            | `8`     | Maximum number of search results when the query falls through to search mode.                         |
| `--isa FAM` (repeatable)               | all     | Filter by ISA family (`Intel`, `Arm`, `RISC-V`, …).                                                   |
| `--preset NAME`                        | none    | Apply a named preset (`default`, `intel`, `arm32`, `arm64`, `riscv`, `none`, `all`).                  |
| `--source-kind measured\|modeled\|any` | `any`   | Filter perf rows by provenance. `measured` keeps uops.info-style rows; `modeled` keeps llvm-mca rows. |
| `--arch CORE`                          | none    | Pin lat/cpi/ports to one microarchitecture (for example `znver4`, `skylake-x`).                       |

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

Free-form search yields `{"mode": "search", "results": [...]}` where each entry has the same shape as `result`.

### `isa llm batch`

Reads queries one per line from stdin and emits one NDJSON record per input line. Amortizes catalog load across many lookups. Blank lines and lines starting with `#` are skipped.

```bash
echo -e "_mm_add_ps\nVPADDD\n_does_not_exist" | isa llm batch
```

Each output record is `{"query": ..., "status": ..., "payload": {...}}`. `status` is one of `match`, `no_match`, `ambiguous`, or `error`. On `error` the record also has an `error` field.

Accepts the same `--limit`, `--isa`, `--preset`, `--source-kind`, and `--arch` flags as `query`.

### `isa llm list`

Without arguments, emits the full `FilterSpec` describing ISA families, sub-ISAs, and the category catalog:

```bash
isa llm list --format json
isa llm list --format markdown
```

With `--pattern GLOB [--isa FAM]`, streams NDJSON records of the form `{name, kind, isa, category}` for each intrinsic or instruction whose name matches the glob and lives in one of the requested ISA families. The pattern uses `fnmatch` (shell globs `*`, `?`, `[...]`) and is applied case-insensitively to the entry name and its `db_key`.

```bash
isa llm list --pattern "*gather*" --isa "Intel"
```

### `isa llm schema`

Emits the JSON Schema for the `query` and `batch` payload shape, including `generated_at`, `source_versions`, and the nested `instruction_refs` fields. Intended for generating client-side types.

## Exit codes

| Code | Meaning                                                              |
| ---- | -------------------------------------------------------------------- |
| `0`  | Match (intrinsic, instruction, or at least one search result).       |
| `1`  | Usage error: bad flag, unknown preset, missing argument.             |
| `2`  | Query valid but no catalog match.                                    |
| `3`  | Ambiguous: multiple exact instruction matches for the same mnemonic. |
| `10` | Internal error (exception during resolution).                        |

## Skill recipe

A typical agent loop over assembly or codegen:

1. Parse the assembly and extract instruction mnemonics.
1. Pre-filter the catalog for speed: `isa llm list --pattern "VPADD*" --isa "Intel"`.
1. Resolve each mnemonic in one batch call, keeping measured perf only: `printf '%s\n' "${mnemonics[@]}" | isa llm batch --source-kind measured`.
1. Consume the NDJSON stream. Each record carries `lat`, `cpi`, the linked intrinsic name, and the canonical `summary`, enough to propose a replacement intrinsic and cite the measured latency or throughput.
1. Use `generated_at` and `source_versions` from the top-level payload to warn when the catalog is stale.
