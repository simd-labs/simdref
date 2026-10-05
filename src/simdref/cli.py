"""Command-line interface for simdref.

This module defines the Typer application, its maintenance/export commands,
and the smart bare-word lookup that fires when no recognised subcommand is
given.

Display and formatting logic lives in :mod:`simdref.display`.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from contextlib import nullcontext as _nullcontext
from dataclasses import asdict
from pathlib import Path

from msgpack.exceptions import UnpackException as _MsgpackUnpackException

import fnmatch

import click
import httpx
import typer
from rich.console import Console

# Usage errors (missing args, bad flags) normally exit 2 in Click. We reserve
# exit code 2 strictly for "query valid but no catalog match" in `simdref llm`,
# so downgrade Click usage errors to exit 1 at the CLI boundary.
click.exceptions.UsageError.exit_code = 1
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from typer.core import TyperGroup

# Stderr-only Console for bootstrap/download status. Must not share stdout with
# `simdref llm` payloads, otherwise callers that json.loads(stdout) break.
err_console = Console(stderr=True)

from simdref.display import (
    console,
    display_architecture,
    display_isa,
    instruction_query_text,
    instruction_variant_items,
    isa_visible,
    normalize_instruction_query,
    render_intrinsic,
    render_instruction,
    render_instruction_variants,
    render_search_results,
)
from simdref import __version__
from simdref.ingest import build_catalog
from simdref.ingest_sources import (
    ARM_A64_ARCHIVE_CACHE,
    refresh_local_arm_a64_archive,
    refresh_local_arm_intrinsics_bundle,
)
from simdref.manpages import write_manpages
from simdref import perf
from simdref.perf import variant_perf_summary
from simdref.queries import (
    instruction_rows_for_intrinsic,
    intrinsic_perf_summary_runtime,
    linked_instruction_records,
)
from simdref.search import (
    SearchResult,
    find_intrinsic,
    find_instructions,
    search_catalog,
    search_records,
)
from simdref.storage import (
    CATALOG_PATH,
    DATA_DIR,
    DEFAULT_MAN_DIR,
    SQLITE_PATH,
    WEB_DIR,
    _write_atomic,
    build_sqlite,
    load_catalog,
    load_catalog_from_db,
    load_instruction_from_db,
    load_intrinsic_from_db,
    load_instructions_by_mnemonic_from_db,
    load_instructions_by_mnemonic_prefix_from_db,
    open_db,
    read_installed_version_stamp,
    save_catalog,
    search_instruction_candidates_from_db,
    search_intrinsic_candidates_from_db,
    sqlite_schema_is_current,
    write_installed_version_stamp,
)
from simdref.export import export_site_data


def _tui_preset_pref_path() -> "Path":
    """Location of the persisted last-used TUI preset."""
    from pathlib import Path

    base = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state"
    )
    return Path(base) / "simdref" / "last-preset"


def _load_last_preset() -> str | None:
    try:
        text = _tui_preset_pref_path().read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    return text or None


def _save_last_preset(name: str) -> None:
    try:
        path = _tui_preset_pref_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name.strip() + "\n", encoding="utf-8")
    except OSError:
        pass  # best-effort — preset persistence must not break startup


def _run_tui(
    *,
    initial_query: str = "",
    initial_preset: str | None = None,
    initial_view: str = "search",
    initial_asm: str = "",
):
    from simdref.tui import run_tui

    # Preset precedence: explicit --preset wins, then last-used from state
    # file, then "intel" as first-run default.
    if initial_preset is None:
        initial_preset = _load_last_preset() or "intel"

    return run_tui(
        initial_query=initial_query,
        initial_preset=initial_preset,
        initial_view=initial_view,
        initial_asm=initial_asm,
    )


class SimdrefGroup(TyperGroup):
    """Top-level CLI group with usage that reflects bare-query mode."""

    def collect_usage_pieces(self, ctx):  # type: ignore[override]
        return ["[OPTIONS] [QUERY] | COMMAND [ARGS]..."]


app = typer.Typer(
    cls=SimdrefGroup,
    add_completion=False,
    help=(
        "Local SIMD reference across Intel intrinsics, instruction data, performance measurements, and SDM-derived descriptions.\n\n"
        "Run without arguments to open the TUI. Pass a bare query to search or open matching results directly.\n\n"
        "Installed under two names — 'isa' (short) and 'simdref' (explicit) — both accept every subcommand.\n\n"
        "Common commands:\n"
        "  isa update                  Download the pre-built release catalog (no llvm-mca required).\n"
        "  isa build                   Full local rebuild from upstream sources (requires llvm-mca).\n"
        "  isa completion install      Install shell completion into your shell profile."
    ),
    context_settings={"help_option_names": ["-h", "--help"]},
)
SHOW_FP16_ISAS = False
SHORT_MODE = False
FULL_MODE = False


@contextmanager
def _pager_context():
    """Pipe Rich output through a pager that handles ANSI colors."""
    pager_cmd = os.environ.get("PAGER", "")
    less = shutil.which("less")
    if less and ("less" in pager_cmd or not pager_cmd):
        # Use less -RFX: Raw ANSI, quit-if-one-screen, no-init
        from rich.pager import Pager

        class _LessPager(Pager):
            def show(self, content: str) -> None:
                proc = subprocess.Popen(
                    [less, "-RFX"],
                    stdin=subprocess.PIPE,
                    encoding="utf-8",
                    errors="replace",
                )
                try:
                    proc.communicate(input=content)
                except KeyboardInterrupt:
                    proc.kill()

        yield console.pager(pager=_LessPager(), styles=True)
    else:
        # Fallback: Rich's default pager without styles (safe)
        yield console.pager(styles=False)


GITHUB_REPO = os.environ.get("SIMDREF_CORE_REPO", "simd-labs/simdref")
RELEASE_TAG = "data-latest"


# ---------------------------------------------------------------------------
# Release download helpers
# ---------------------------------------------------------------------------


def _release_tag_candidates() -> list[str]:
    return [f"data-v{__version__}", RELEASE_TAG]


def _release_asset_url(tag: str, asset_name: str) -> str:
    return f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/{asset_name}"


class _ReleaseAssetMissing(Exception):
    """The release carries no compatible asset for this name."""


def _download_from_release() -> None:
    """Download pre-built catalog and database from GitHub Release.

    Emits a Rich progress bar (bytes + transfer speed + ETA) on a TTY.
    In non-TTY contexts (CI, agent harnesses) falls back to one plain
    ``downloading X... N.N MB`` line per asset at completion so log
    scrapers observe that something is happening — issue #5.
    """
    from simdref.storage import ensure_dir

    ensure_dir(DATA_DIR)

    is_tty = err_console.is_terminal and os.environ.get("GITHUB_ACTIONS") != "true"

    def _fetch_into(asset: str, dest: Path) -> bool:
        for tag in _release_tag_candidates():
            url = _release_asset_url(tag, asset)
            try:
                with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
                    if resp.status_code == 404:
                        continue
                    resp.raise_for_status()
                    total_header = resp.headers.get("content-length")
                    total = int(total_header) if total_header and total_header.isdigit() else None

                    if is_tty:
                        progress = Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            BarColumn(),
                            DownloadColumn(),
                            TransferSpeedColumn(),
                            TimeRemainingColumn(),
                            console=err_console,
                            transient=True,
                        )
                        with progress:
                            task = progress.add_task(f"downloading {asset} ({tag})", total=total)
                            with open(dest, "wb") as f:
                                for chunk in resp.iter_bytes(chunk_size=1024 * 64):
                                    f.write(chunk)
                                    progress.update(task, advance=len(chunk))
                    else:
                        err_console.print(f"downloading {asset} from {tag}...", style="dim")
                        written = 0
                        with open(dest, "wb") as f:
                            for chunk in resp.iter_bytes(chunk_size=1024 * 64):
                                f.write(chunk)
                                written += len(chunk)
                        if total:
                            err_console.print(
                                f"downloaded {asset}: {written / 1_048_576:.1f} MB "
                                f"({written}/{total} bytes)",
                                style="dim",
                            )
                        else:
                            err_console.print(
                                f"downloaded {asset}: {written / 1_048_576:.1f} MB", style="dim"
                            )
                    return True
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    continue
                err_console.print(
                    f"failed to download {asset}: {exc.response.status_code}", style="red"
                )
                raise typer.Exit(code=1) from exc
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.NetworkError,
            ) as exc:
                err_console.print(
                    "[bold yellow]warning:[/bold yellow] no internet connectivity — "
                    f"could not reach the release server ({exc.__class__.__name__}). "
                    "Some features may not work correctly until the catalog is refreshed.",
                )
                raise typer.Exit(code=1) from exc
        return False

    for asset in ("catalog.msgpack", "catalog.db"):
        dest = DATA_DIR / asset

        def write(fh, asset=asset):
            # _fetch_into streams with httpx into its own "wb" handle on
            # the same temp path; fh itself is unused.
            if not _fetch_into(asset, fh.atomic_path):
                raise _ReleaseAssetMissing(asset)

        try:
            _write_atomic(dest, write)
        except _ReleaseAssetMissing:
            err_console.print(
                f"failed to download {asset}: no compatible release asset found", style="red"
            )
            err_console.print("try 'simdref build' to build locally", style="yellow")
            raise typer.Exit(code=1) from None
    err_console.print("download complete", style="green")


def _build_runtime_locally(*, man_dir: Path, include_sdm: bool = False) -> None:
    """Build catalog, SQLite, manpages, and web bundle locally.

    Renders a single rich.progress.Progress that shows a per-phase ETA.
    Download phases render bytes + transfer speed; processing phases
    render item counts + remaining time.
    """
    interactive_progress = console.is_terminal and os.environ.get("GITHUB_ACTIONS") != "true"

    if not interactive_progress:

        def _status(msg: str) -> None:
            err_console.print(msg, style="dim")

        _status("Refreshing local Arm intrinsics cache")
        try:
            written = refresh_local_arm_intrinsics_bundle()
        except Exception as exc:
            err_console.print(f"Arm intrinsics download failed: {exc}", style="red")
            raise typer.Exit(code=1) from exc
        _status(f"Refreshed {len(written)} Arm JSON files in {written[0].parent}")
        _status("Fetching Arm A64 AARCHMRS archive (large, one-time download)")
        try:
            archive_path = refresh_local_arm_a64_archive()
        except Exception as exc:
            err_console.print(f"AARCHMRS download failed: {exc}", style="red")
            raise typer.Exit(code=1) from exc
        _status(f"AARCHMRS archive ready at {archive_path}")
        _status("Building local catalog")
        catalog = build_catalog(include_sdm=include_sdm, status=_status)
        _status("Saving catalog snapshot")
        save_catalog(catalog)
        _status("Building SQLite search database")
        build_sqlite(catalog)
        _status("Writing manpages")
        write_manpages(catalog, man_dir)
        _status("Exporting site data")
        export_site_data(catalog, WEB_DIR)
        err_console.print(
            f"updated catalog with {len(catalog.intrinsics)} intrinsics and {len(catalog.instructions)} instructions",
            style="green",
        )
        return

    download_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=err_console,
        transient=True,
    )
    count_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=err_console,
        transient=True,
    )

    from rich.live import Live
    from rich.console import Group

    with Live(Group(download_progress, count_progress), console=err_console, refresh_per_second=10):
        arm_task = count_progress.add_task("Arm intrinsics JSON bundle", total=3)
        try:
            refresh_local_arm_intrinsics_bundle(
                on_progress=lambda done, total: count_progress.update(
                    arm_task, completed=done, total=total
                ),
            )
        except Exception as exc:
            count_progress.update(arm_task, description=f"Arm intrinsics download failed: {exc}")
            raise typer.Exit(code=1) from exc
        count_progress.update(arm_task, description="Arm intrinsics JSON bundle \u2713")

        if ARM_A64_ARCHIVE_CACHE.exists() and ARM_A64_ARCHIVE_CACHE.stat().st_size > 0:
            cached_task = count_progress.add_task(
                f"Arm A64 AARCHMRS archive \u2713 ({ARM_A64_ARCHIVE_CACHE.name}, cached)",
                total=1,
            )
            count_progress.update(cached_task, completed=1)
        else:
            a64_task = download_progress.add_task("Arm A64 AARCHMRS archive", total=None)

            def _a64_progress(done: int, total: int | None) -> None:
                download_progress.update(a64_task, completed=done, total=total)

            try:
                archive_path = refresh_local_arm_a64_archive(on_progress=_a64_progress)
            except Exception as exc:
                download_progress.update(a64_task, description=f"AARCHMRS download failed: {exc}")
                raise typer.Exit(code=1) from exc
            download_progress.update(
                a64_task, description=f"Arm A64 AARCHMRS archive \u2713 ({archive_path.name})"
            )

        build_task = count_progress.add_task("Building catalog from sources", total=None)

        def _build_status(msg: str) -> None:
            count_progress.update(build_task, description=f"Building catalog: {msg}")

        catalog = build_catalog(include_sdm=include_sdm, status=_build_status)
        count_progress.update(
            build_task, description="Building catalog \u2713", completed=1, total=1
        )

        save_task = count_progress.add_task("Saving catalog snapshot", total=1)
        save_catalog(catalog)
        count_progress.update(save_task, completed=1, description="Saving catalog snapshot \u2713")

        sqlite_task = count_progress.add_task("Building SQLite search database", total=1)
        build_sqlite(catalog)
        count_progress.update(
            sqlite_task, completed=1, description="Building SQLite search database \u2713"
        )

        man_total = len(catalog.intrinsics) + len(catalog.instructions)
        man_task = count_progress.add_task("Writing manpages", total=man_total)
        write_manpages(
            catalog,
            man_dir,
            on_progress=lambda done, total: count_progress.update(
                man_task, completed=done, total=total
            ),
        )
        count_progress.update(man_task, description="Writing manpages \u2713")

        web_task = count_progress.add_task("Exporting site data", total=1)
        export_site_data(catalog, WEB_DIR)
        count_progress.update(web_task, completed=1, description="Exporting site data \u2713")

    err_console.print(
        f"updated catalog with {len(catalog.intrinsics)} intrinsics and {len(catalog.instructions)} instructions",
        style="green",
    )
    write_installed_version_stamp(__version__)


def _refresh_runtime_from_existing_catalog() -> None:
    """Rebuild the SQLite runtime from the local catalog snapshot.

    Falls back to rebuilding from the existing database itself when the
    msgpack snapshot was pruned. Manpages and the static web bundle are no
    longer materialized here — they are ~150k small files; ``simdref man``
    renders pages on demand and ``simdref install-manpages`` / ``simdref
    web`` remain explicit opt-ins.
    """
    if CATALOG_PATH.exists():
        catalog = load_catalog()
    elif SQLITE_PATH.exists():
        catalog = load_catalog_from_db()
    else:
        err_console.print(
            "no local catalog to rebuild from — run `simdref update` with network access",
            style="red",
        )
        raise typer.Exit(code=1)
    build_sqlite(catalog)
    err_console.print(
        f"refreshed runtime from existing catalog with {len(catalog.intrinsics)} intrinsics and {len(catalog.instructions)} instructions",
        style="green",
    )
    # The snapshot is fully derivable now; drop the redundant 300MB+ copy.
    CATALOG_PATH.unlink(missing_ok=True)
    write_installed_version_stamp(__version__)


def _finalize_runtime_from_download() -> None:
    """Finish a release-download refresh.

    The runtime needs only ``catalog.db``; drop the legacy ``catalog.json``
    asset (no longer shipped or read) and the ``catalog.msgpack`` snapshot
    (rebuildable from the DB via ``load_catalog_from_db``) if present.
    """
    if not SQLITE_PATH.exists():
        raise typer.Exit(code=1)
    (DATA_DIR / "catalog.json").unlink(missing_ok=True)
    CATALOG_PATH.unlink(missing_ok=True)
    with open_db(SQLITE_PATH) as conn:
        n_intrinsics = conn.execute("SELECT COUNT(*) FROM intrinsics_data").fetchone()[0]
        n_instructions = conn.execute("SELECT COUNT(*) FROM instructions_data").fetchone()[0]
    err_console.print(
        f"refreshed runtime from downloaded catalog with {n_intrinsics} intrinsics and {n_instructions} instructions",
        style="green",
    )
    write_installed_version_stamp(__version__)


def _download_release_or_fallback() -> None:
    """Prefer pre-built assets. Fall back to the existing on-disk catalog.

    When the download fails and no catalog is cached, the caller must
    run ``simdref update --build`` — there is no longer a bundled-fixture
    fallback.
    """
    try:
        _download_from_release()
        if sqlite_schema_is_current():
            _finalize_runtime_from_download()
            return
        if CATALOG_PATH.exists():
            err_console.print(
                "downloaded catalog is usable but SQLite is stale; rebuilding runtime locally from the downloaded catalog",
                style="yellow",
            )
            _refresh_runtime_from_existing_catalog()
            return
        err_console.print(
            "[bold red]downloaded runtime schema is not current[/bold red] and no local catalog exists",
            style="yellow",
        )
        err_console.print(
            "run `simdref update --build` to build from upstream sources (requires llvm-mca)"
        )
        raise typer.Exit(code=1)
    except typer.Exit:
        if CATALOG_PATH.exists():
            err_console.print(
                "download failed; refreshing runtime from the existing local catalog",
                style="yellow",
            )
            _refresh_runtime_from_existing_catalog()
            return
        err_console.print(
            "[bold red]no pre-built catalog available and no local cache[/bold red]", style="yellow"
        )
        err_console.print(
            "run `simdref update --build` to build from upstream sources (requires llvm-mca)"
        )
        raise


# ---------------------------------------------------------------------------
# Catalog / runtime helpers
# ---------------------------------------------------------------------------


def _bootstrap_interactive() -> None:
    """Bootstrap runtime data with a lightweight default path.

    Emits an explicit banner on stderr (Rich) and stdout (plain) so
    harnesses that capture only one stream can still observe that
    simdref is doing a one-time catalog download — issue #5.
    """
    err_console.print(
        "\n[bold]No catalog found.[/bold] Running one-time bootstrap: "
        "downloading pre-built data (~minutes on first run)...\n"
    )
    err_console.print(f"  target: {DATA_DIR}", style="dim")
    err_console.print(
        "  to pre-fetch explicitly next time, run [cyan]simdref update[/cyan].\n",
        style="dim",
    )
    non_tty_stdout = not sys.stdout.isatty()
    if non_tty_stdout:
        typer.echo(f"simdref: bootstrapping catalog (downloading release assets to {DATA_DIR}) ...")
    _download_release_or_fallback()
    if non_tty_stdout:
        typer.echo("simdref: catalog bootstrap complete")


def ensure_catalog():
    """Load (or bootstrap) the in-memory catalog.

    The msgpack snapshot is pruned after install/update; fall back to the
    SQLite runtime when only the database is left.
    """
    if not CATALOG_PATH.exists() and not SQLITE_PATH.exists():
        _bootstrap_interactive()
    if CATALOG_PATH.exists():
        return load_catalog()
    return load_catalog_from_db()


def ensure_runtime() -> None:
    """Ensure catalog + SQLite are present and current.

    If the package version recorded in the data dir does not match the
    currently installed version, transparently re-run the release-download
    flow so users don't need to invoke ``simdref update`` manually after
    a ``pip``/``uv`` install or upgrade. Honors ``SIMDREF_SKIP_AUTOUPDATE``.
    """
    if not CATALOG_PATH.exists() and not SQLITE_PATH.exists():
        _bootstrap_interactive()
        _maybe_auto_update_for_version_change()
        return
    if not sqlite_schema_is_current():
        err_console.print(
            "runtime schema is missing or out of date; rebuilding derived runtime artifacts from the local catalog",
            style="yellow",
        )
        _refresh_runtime_from_existing_catalog()
    _maybe_auto_update_for_version_change()


def _maybe_auto_update_for_version_change() -> None:
    """Refresh data when the package version differs from the stamped one."""
    if os.environ.get("SIMDREF_SKIP_AUTOUPDATE"):
        return
    stamped = read_installed_version_stamp()
    if stamped == __version__:
        return
    if stamped is None:
        # First run after install: stamp without re-downloading. The catalog
        # we just bootstrapped (or that already exists locally) is what the
        # user expects to see.
        write_installed_version_stamp(__version__)
        return
    err_console.print(
        f"simdref upgraded from {stamped} to {__version__}; refreshing catalog "
        "(set SIMDREF_SKIP_AUTOUPDATE=1 to disable)",
        style="yellow",
    )
    try:
        _download_release_or_fallback()
    except typer.Exit:
        err_console.print(
            "[bold yellow]warning:[/bold yellow] auto-update failed; continuing with the existing catalog. "
            "Some features may not work correctly until `simdref update` succeeds.",
        )
        # Stamp anyway so we don't retry on every invocation; the user has
        # been warned and can re-run `simdref update` once back online.
        write_installed_version_stamp(__version__)


def _catalog_meta(catalog) -> dict:
    return {
        "generated_at": catalog.generated_at,
        "source_versions": [asdict(source) for source in catalog.sources],
    }


def _search_runtime(
    conn, query: str, limit: int = 20
) -> tuple[list[SearchResult], dict[str, object], dict[str, object]]:
    candidate_limit = max(limit * 6, 60)
    intrinsics = search_intrinsic_candidates_from_db(conn, query, limit=candidate_limit)
    instructions = search_instruction_candidates_from_db(conn, query, limit=candidate_limit)
    results = search_records(intrinsics, instructions, query, limit=limit)
    intrinsic_map = {item.name: item for item in intrinsics}
    instruction_map = {item.db_key: item for item in instructions}
    return results, intrinsic_map, instruction_map


# ---------------------------------------------------------------------------
# Instruction lookup helpers
# ---------------------------------------------------------------------------


def _select_instruction_variant(catalog, query: str, items):
    parts = query.split()
    if len(parts) < 2 or not parts[-1].isdigit():
        return None
    base_query = " ".join(parts[:-1]).strip()
    if not base_query:
        return None
    index = int(parts[-1])
    if index < 1:
        return None
    if items:
        variants = instruction_variant_items(items)
    elif catalog is not None:
        variants = instruction_variant_items(find_instructions(catalog, base_query))
    else:
        variants = instruction_variant_items(_find_instructions_fast(base_query))
    if 1 <= index <= len(variants):
        return variants[index - 1]
    return None


def _find_instructions_fast(query: str):
    ensure_runtime()
    with open_db() as conn:
        exact = load_instruction_from_db(conn, query)
        if exact is not None:
            return [exact]
        parts = query.split()
        mnemonic = parts[0] if parts else query
        candidates = load_instructions_by_mnemonic_from_db(conn, mnemonic)
        if not candidates:
            return []
        normalized_query = normalize_instruction_query(query)
        exact_candidates = [
            item
            for item in candidates
            if normalize_instruction_query(item.key) == normalized_query
            or normalize_instruction_query(instruction_query_text(item)) == normalized_query
            or item.mnemonic.casefold() == query.casefold()
        ]
        if exact_candidates:
            return exact_candidates
        if mnemonic.casefold() == query.casefold():
            return candidates
        return []


def _find_instruction_family_fast(query: str):
    ensure_runtime()
    token = (query.split()[0] if query.split() else query).strip()
    if not token:
        return []
    with open_db() as conn:
        candidates = load_instructions_by_mnemonic_prefix_from_db(conn, token)
    exact_mnemonic = {item.mnemonic.casefold() for item in candidates}
    if token.casefold() in exact_mnemonic:
        return []
    return candidates


# ---------------------------------------------------------------------------
# LLM / JSON payload builders
# ---------------------------------------------------------------------------


def _resolve_query_payload(catalog, query: str, limit: int = 8) -> dict:
    intrinsic = find_intrinsic(catalog, query)
    if intrinsic is not None:
        return {
            "query": query,
            "mode": "exact",
            "match_kind": "intrinsic",
            "intrinsic": asdict(intrinsic),
            "performance": instruction_rows_for_intrinsic(catalog, intrinsic),
            **_catalog_meta(catalog),
        }
    instructions = find_instructions(catalog, query)
    if instructions:
        return {
            "query": query,
            "mode": "exact",
            "match_kind": "instruction",
            "instructions": [asdict(item) | {"key": item.key} for item in instructions],
            **_catalog_meta(catalog),
        }
    return {
        "query": query,
        "mode": "search",
        "match_kind": None,
        "results": [asdict(result) for result in search_catalog(catalog, query, limit=limit)],
        **_catalog_meta(catalog),
    }


def _llm_result_payload(
    conn, result: SearchResult, intrinsic_map: dict[str, object], instruction_map: dict[str, object]
) -> dict:
    if result.kind == "intrinsic":
        item = intrinsic_map.get(result.key)
        if item is None:
            item = load_intrinsic_from_db(conn, result.key)
            if item is not None:
                intrinsic_map[result.key] = item
        if item is not None:
            lat, cpi = intrinsic_perf_summary_runtime(conn, item, instruction_map)
            payload: dict = {
                "query": item.name,
                "intrinsic": item.name,
                "signature": item.signature,
                "url": getattr(item, "url", "") or "",
                "instructions": item.instructions,
                "instruction_refs": item.instruction_refs,
                "summary": item.description,
                "isa": item.isa,
                "lat": lat,
                "cpi": cpi,
                "timing": _intrinsic_timing(conn, item, cache=instruction_map),
            }
            operation = _intrinsic_operation_text(item)
            if operation:
                payload["operation"] = operation
            return payload
    item = instruction_map.get(result.key)
    if item is None:
        item = load_instruction_from_db(conn, result.key)
        if item is not None:
            instruction_map[result.key] = item
    if item is not None:
        lat, cpi = variant_perf_summary(item.arch_details)
        return {
            "query": item.key,
            "intrinsic": item.linked_intrinsics,
            "summary": item.summary,
            "isa": item.isa,
            "lat": lat,
            "cpi": cpi,
            "source_kinds": _payload_source_kinds(item.arch_details),
            "timing": _llm_timing(item.arch_details),
        }
    return {
        "query": result.title,
        "intrinsic": [],
        "summary": result.subtitle,
        "isa": [],
        "lat": "-",
        "cpi": "-",
    }


def _llm_intrinsic_payload(conn, intrinsic) -> dict:
    instruction_map: dict[str, object] = {}
    lat, cpi = intrinsic_perf_summary_runtime(conn, intrinsic, instruction_map)
    timing = _intrinsic_timing(conn, intrinsic, cache=instruction_map)
    payload: dict = {
        "query": intrinsic.name,
        "intrinsic": intrinsic.name,
        "signature": intrinsic.signature,
        "url": intrinsic.url,
        "instructions": intrinsic.instructions,
        "instruction_refs": intrinsic.instruction_refs,
        "isa": intrinsic.isa,
        "lat": lat,
        "cpi": cpi,
        "summary": intrinsic.description,
        "timing": timing,
    }
    operation = _intrinsic_operation_text(intrinsic)
    if operation:
        # Encodes algorithmic quirks (e.g. the bit-1 selector in
        # ``_mm_permutevar_pd``) that the one-line summary cannot convey.
        payload["operation"] = operation
    return payload


def _intrinsic_timing(conn, intrinsic, cache: dict[str, object] | None = None) -> dict[str, dict]:
    """Per-core timing for an intrinsic, merged over its linked instruction forms."""
    linked = linked_instruction_records(None, intrinsic, conn, cache=cache)
    return _merge_timing([_llm_timing(record.arch_details) for record in linked])


def _intrinsic_operation_text(intrinsic) -> str:
    """Return the SDM-style ``Operation`` pseudocode for *intrinsic*, or ``""``.

    Intel ingest stores it under ``doc_sections["Operation"]``; ARM ACLE under
    ``doc_sections["ACLE Operation"]``. Both are pseudocode the LLM payload
    needs to surface so callers can reason about behavior beyond the summary.
    """
    sections = getattr(intrinsic, "doc_sections", None) or {}
    return (sections.get("Operation") or sections.get("ACLE Operation") or "").strip()


def _llm_instruction_payload(item) -> dict:
    lat, cpi = variant_perf_summary(item.arch_details)
    return {
        "query": item.key,
        "intrinsic": item.linked_intrinsics,
        "isa": item.isa,
        "lat": lat,
        "cpi": cpi,
        "summary": item.summary,
        "source_kinds": _payload_source_kinds(item.arch_details),
        "timing": _llm_timing(item.arch_details),
    }


def _llm_timing(arch_details) -> dict[str, dict]:
    """Return ``{core: {lat, cpi, ports, uops, source_kind}}`` for one instruction.

    Latency and CPI are microarchitecture properties, so the payload keys
    them by canonical core id instead of collapsing them to one scalar.
    Cores with neither value are dropped. ``ports`` carries the upstream
    port-pressure string verbatim (e.g. ``1*p01``: one uop issuable on
    port 0 or port 1). No pipe count is derived from it: ``cpi`` already
    measures issue throughput.
    """
    from simdref.annotate import _cpi_for, _latency_for, _per_arch_value, _ports_for

    if not isinstance(arch_details, dict):
        return {}
    timing: dict[str, dict] = {}
    for core in sorted(arch_details):
        details = arch_details.get(core)
        if not isinstance(details, dict):
            continue
        lat = _per_arch_value(details, _latency_for)
        cpi = _per_arch_value(details, _cpi_for)
        if lat is None and cpi is None:
            continue
        ports = _ports_for(details)
        entry: dict = {
            "lat": lat,
            "cpi": cpi,
            "source_kind": perf._source_kind(details),
        }
        if ports:
            entry["ports"] = ports
        uops = (details.get("measurement") or {}).get("uops")
        if uops:
            entry["uops"] = uops
        timing[core] = entry
    return timing


def _merge_timing(per_instruction: list[dict[str, dict]]) -> dict[str, dict]:
    """Merge per-instruction timing maps, keeping the lowest value per core.

    Mirrors the reduction the scalar ``lat``/``cpi`` fields already apply
    across an intrinsic's linked instruction forms.
    """
    merged: dict[str, dict] = {}
    kinds: dict[str, set[str]] = {}
    for timing in per_instruction:
        for core, entry in timing.items():
            kinds.setdefault(core, set()).add(entry.get("source_kind", "measured"))
            current = merged.get(core)
            if current is None:
                merged[core] = dict(entry)
                continue
            for field in ("lat", "cpi"):
                new_value, old_value = entry.get(field), current.get(field)
                if new_value is not None and (old_value is None or new_value < old_value):
                    current[field] = new_value
            for field in ("ports", "uops"):
                if field not in current and entry.get(field) is not None:
                    current[field] = entry[field]
    for core, current in merged.items():
        # The winning lat and cpi may come from differently-sourced forms, so a
        # single label would misattribute one of them. Same convention as `aggregate_perf`.
        if len(kinds[core]) > 1:
            current["source_kind"] = "mixed"
    return {core: merged[core] for core in sorted(merged)}


def _payload_source_kinds(arch_details) -> list[str]:
    """Return the distinct provenance kinds present in an instruction's arch_details."""
    if not isinstance(arch_details, dict):
        return []
    kinds: list[str] = []
    for details in arch_details.values():
        if not isinstance(details, dict):
            continue
        kind = perf._source_kind(details)
        if kind not in kinds:
            kinds.append(kind)
    return kinds


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------


def _print_search_results_runtime(conn, query: str, limit: int = 20, as_json: bool = False) -> int:
    """Render the ranked search table (or JSON) in the search's own
    relevance order. Returns the number of rows actually printed — callers
    use 0 to fall through to the no-match branch instead of printing an
    empty table."""
    results, intrinsic_map, instruction_map = _search_runtime(conn, query, limit=limit)
    prepared_rows = []
    for result in results:
        arch = "-"
        isa = "-"
        lat = "-"
        cpi = "-"
        if result.kind == "instruction":
            item = instruction_map.get(result.key)
            if item is not None:
                if not isa_visible(item.isa, show_fp16=SHOW_FP16_ISAS):
                    continue
                arch = display_architecture(item.architecture)
                isa = display_isa(item.isa)
                lat, cpi = variant_perf_summary(item.arch_details)
        elif result.kind == "intrinsic":
            item = intrinsic_map.get(result.key)
            if item is not None:
                if not isa_visible(item.isa, show_fp16=SHOW_FP16_ISAS):
                    continue
                arch = display_architecture(item.architecture)
                isa = display_isa(item.isa)
                lat, cpi = intrinsic_perf_summary_runtime(conn, item, instruction_map)
        prepared_rows.append((result, arch, isa, lat, cpi))
    if not prepared_rows:
        return 0
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "query": query,
                    "mode": "search",
                    "results": [
                        {
                            "kind": r.kind,
                            "key": r.key,
                            "title": r.title,
                            "subtitle": r.subtitle,
                            "arch": arch,
                            "isa": isa,
                            "latency_cycles": lat,
                            "tput_cpi": cpi,
                        }
                        for r, arch, isa, lat, cpi in prepared_rows
                    ],
                },
                indent=2,
                default=str,
            )
        )
        return len(prepared_rows)
    render_search_results(prepared_rows)
    return len(prepared_rows)


# ---------------------------------------------------------------------------
# Smart lookup (bare-word query)
# ---------------------------------------------------------------------------


def _smart_lookup(
    query: str,
    preset: str | None = None,
    arch: str | None = None,
    as_json: bool = False,
) -> int:
    """Dispatch a bare query.

    Exact intrinsic match prints the intrinsic detail and exits 0. Exact
    instruction match prints a plain-text (or JSON) summary and exits 0;
    ``--arch`` applies to this branch only. Fuzzy hits print the ranked
    search list (or JSON) and exit 0, on a TTY or not. Nothing found opens
    the TUI on a TTY (never under ``--json``) or exits 2 non-interactively.
    """
    ensure_runtime()
    with contextlib.closing(open_db()) as conn:
        intrinsic = load_intrinsic_from_db(conn, query)
        if intrinsic is not None and arch is not None:
            err_console.print("--arch applies to instruction queries only", style="red")
            return 2
        if intrinsic is not None:
            if as_json:
                typer.echo(json.dumps(asdict(intrinsic), indent=2, default=str))
            else:
                render_intrinsic(None, intrinsic, conn=conn)
            return 0
        records = _find_instructions_fast(query)
        if records:
            return _print_non_interactive_summary(
                query, records=records, arch=arch, as_json=as_json
            )
        if arch is not None:
            err_console.print("--arch applies to instruction queries only", style="red")
            return 2
        printed = _print_search_results_runtime(conn, query, as_json=as_json)
        if printed:
            return 0
        if as_json or not (sys.stdin.isatty() and sys.stdout.isatty()):
            err_console.print(
                f"no match for {query!r} (non-interactive; run in a TTY to search).",
                style="yellow",
            )
            return 2
    return _run_tui(initial_query=query, initial_preset=preset)


def _print_non_interactive_summary(
    query: str,
    *,
    records=None,
    arch: str | None = None,
    as_json: bool = False,
) -> int:
    """Plain-text (or JSON) fallback for bare queries.

    Returns 0 on match, 2 on no match (mirrors `simdref llm` exit codes).
    When ``arch`` is set, emits per-arch lat/cpi pinned to that arch;
    otherwise reports the average across all archs with measured data.
    """
    from simdref.annotate import (
        aggregate_perf,
        arch_perf,
        arch_perf_tag,
        collect_ports,
        _fmt_num,
    )
    from simdref.perf_sources.cores import canonical_core_id, supported_core_ids

    canonical_arch: str | None = None
    if arch is not None:
        canonical_arch = canonical_core_id(arch)
        if canonical_arch is None:
            err_console.print(f"arch {arch!r} is not in the local catalog.", style="red")
            err_console.print(f"supported cores: {', '.join(supported_core_ids())}", style="yellow")
            return 1

    if records is None:
        records = _find_instructions_fast(query)
    if not records:
        err_console.print(f"no instruction match for {query!r}", style="yellow")
        return 2

    # uops.info records a core only for encodings it can execute, so an absent
    # arch_details key means this form does not run on the pinned core.
    skipped = 0
    if canonical_arch is not None:
        runnable = [rec for rec in records if canonical_arch in (rec.arch_details or {})]
        skipped = len(records) - len(runnable)
        records = runnable

    if as_json:
        import json as _json

        payload: dict = {
            "query": query,
            "arch": canonical_arch,
            "variants": [],
        }
        if canonical_arch is not None:
            payload["variants_not_runnable"] = skipped
        for rec in records:
            iter_archs = (
                [canonical_arch] if canonical_arch else sorted((rec.arch_details or {}).keys())
            )
            per_arch = []
            for core in iter_archs:
                if core is None:
                    continue
                lat, cpi, kind = arch_perf(rec, core)
                ports = collect_ports(rec, arch=core, archs_used=[core])
                per_arch.append(
                    {
                        "arch": core,
                        "latency_cycles": lat,
                        "tput_cpi": cpi,
                        "source_kind": kind,
                        "ports": ports,
                    }
                )
            summary = aggregate_perf(rec, mode="avg", include_modeled=False)
            payload["variants"].append(
                {
                    "key": rec.key,
                    "summary": (rec.summary or "").strip() or None,
                    "aggregate": {
                        "latency_cycles": summary.latency,
                        "tput_cpi": summary.cpi,
                        "n_archs": summary.n_archs,
                        "source_kind": summary.source_kind,
                        "archs_used": summary.archs_used,
                    },
                    "per_arch": per_arch,
                }
            )
        typer.echo(_json.dumps(payload, indent=2, default=str))
        return 0

    typer.echo(f"# {query}  ({len(records)} variant{'s' if len(records) != 1 else ''})")
    if canonical_arch is not None and skipped:
        typer.echo(f"# {skipped} further variant(s) omitted: {canonical_arch} cannot execute them.")
    for rec in records:
        typer.echo("")
        typer.echo(f"[{rec.key}]")
        if rec.summary:
            typer.echo(f"  {rec.summary.strip()}")
        if canonical_arch is not None:
            lat, cpi, kind = arch_perf(rec, canonical_arch)
            ports = collect_ports(rec, arch=canonical_arch, archs_used=[canonical_arch])
            ports_str = ",".join(ports) if ports else "-"
            tag = arch_perf_tag(canonical_arch, lat, cpi, kind)
            typer.echo(f"  {tag}: lat={_fmt_num(lat)}c cpi={_fmt_num(cpi)} ports={ports_str}")
        else:
            summary = aggregate_perf(rec, mode="avg", include_modeled=False)
            ports = collect_ports(rec, arch=None, archs_used=summary.archs_used)
            ports_str = ",".join(ports) if ports else "-"
            typer.echo(
                f"  avg over {summary.n_archs} arch(s) [{summary.source_kind}]: "
                f"lat={_fmt_num(summary.latency)}c cpi={_fmt_num(summary.cpi)} ports={ports_str}"
            )
            if summary.archs_used:
                typer.echo(f"  archs: {', '.join(summary.archs_used)}")
    return 0


def _is_completion_invocation(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    for key, value in env.items():
        if not key.endswith("_COMPLETE"):
            continue
        upper = key.upper()
        if "SIMDREF" not in upper and upper != "_ISA_COMPLETE":
            continue
        if value:
            return True
    return False


# ---------------------------------------------------------------------------
# Typer commands
# ---------------------------------------------------------------------------


@app.command(rich_help_panel="Commands")
def annotate(
    input_path: Path | None = typer.Argument(
        None, help="Input .s assembly file, or '-' for stdin. Omit to open the TUI annotate tab."
    ),
    output: Path = typer.Option(
        None, "-o", "--output", help="Output .sa path (default: <input>.sa, or '-' for stdout)."
    ),
    performance: bool = typer.Option(
        True, "--performance/--no-performance", help="Include latency/CPI annotations."
    ),
    docs: bool = typer.Option(
        True, "--docs/--no-docs", help="Include human-readable instruction summaries."
    ),
    arch: str | None = typer.Option(
        None, "--arch", help="Pin annotations to a specific microarch (e.g. skylake-x, zen4)."
    ),
    agg: str = typer.Option(
        "avg",
        "--agg",
        help="Aggregation across archs when --arch is not set: avg|median|best|worst.",
    ),
    include_modeled: bool = typer.Option(
        False,
        "--include-modeled",
        help="Fall back to modeled perf data when no arch has measured data.",
    ),
    block: bool = typer.Option(
        False,
        "--block/--inline",
        help="Emit annotation as a comment block above each instruction (default: inline trailing).",
    ),
    unknown: str = typer.Option(
        "mark", "--unknown", help="Handling of unknown mnemonics: keep|drop|mark."
    ),
    fmt: str = typer.Option("sa", "--format", help="Output format: sa|md|json."),
    track_positions: bool = typer.Option(
        False,
        "--track-positions",
        help="Parse objdump-style input: thread VAs and source-file:line through the output.",
    ),
) -> None:
    """Annotate a ``.s`` assembly file with instruction summaries and perf data.

    With no positional argument, launches the TUI on the Annotate tab."""
    from simdref.annotate import AnnotateOptions, annotate_stream

    if input_path is None:
        ensure_runtime()
        raise typer.Exit(code=_run_tui(initial_view="annotate"))

    if agg not in {"avg", "median", "best", "worst"}:
        err_console.print(f"invalid --agg value: {agg}", style="red")
        raise typer.Exit(code=1)
    if unknown not in {"keep", "drop", "mark"}:
        err_console.print(f"invalid --unknown value: {unknown}", style="red")
        raise typer.Exit(code=1)
    if fmt not in {"sa", "md", "json"}:
        err_console.print(f"invalid --format value: {fmt}", style="red")
        raise typer.Exit(code=1)

    ensure_runtime()

    if arch is not None:
        from simdref.perf_sources.cores import canonical_core_id, supported_core_ids

        canonical = canonical_core_id(arch)
        if canonical is None:
            supported = ", ".join(supported_core_ids())
            err_console.print(
                f"[bold red]arch {arch!r} is not in the local catalog.[/bold red]",
                style="red",
            )
            err_console.print(f"supported cores: {supported}", style="yellow")
            err_console.print(
                "x86 (Skylake/Sapphire Rapids/Zen) perf ingest is tracked as a separate issue.",
                style="dim",
            )
            raise typer.Exit(code=1)
        arch = canonical

    input_is_stdin = str(input_path) == "-"
    if input_is_stdin:
        source_lines: list[str] = sys.stdin.readlines()
        default_out = Path("-")
    else:
        if not input_path.exists():
            err_console.print(f"input not found: {input_path}", style="red")
            raise typer.Exit(code=1)
        source_lines = input_path.read_text().splitlines(keepends=True)
        default_out = (
            input_path.with_suffix(input_path.suffix + "a")
            if input_path.suffix == ".s"
            else input_path.with_suffix(".sa")
        )

    out_path = output if output is not None else default_out
    opts = AnnotateOptions(
        performance=performance,
        docs=docs,
        arch=arch,
        agg=agg,
        include_modeled=include_modeled,
        block=block,
        unknown=unknown,
        fmt=fmt,
        track_positions=track_positions,
    )

    stats: dict[str, int] = {}
    with open_db(SQLITE_PATH) as conn:
        rendered = "".join(annotate_stream(source_lines, opts=opts, conn=conn, stats=stats))

    parsed = stats.get("parsed", 0)
    recognized = stats.get("recognized", 0)
    content = stats.get("content", 0)
    if parsed == 0 and content > 0:
        err_console.print(
            f"[bold yellow]warning:[/bold yellow] no instruction lines were recognized "
            f"(0 of {content} content lines) — the input does not look like AT&T "
            f"assembly or `objdump -d` output; nothing was annotated.",
            style="yellow",
        )
    elif parsed > 0 and recognized == 0:
        err_console.print(
            f"[bold yellow]warning:[/bold yellow] 0 of {parsed} instruction lines matched "
            f"the catalog — annotations are missing (check --arch / mnemonic coverage).",
            style="yellow",
        )

    if str(out_path) == "-":
        sys.stdout.write(rendered)
    else:
        out_path.write_text(rendered)
        err_console.print(f"wrote {out_path}", style="green")


@app.command(rich_help_panel="Commands")
def update(
    from_release: bool = typer.Option(
        False, "--from-release", help="Download pre-built data from GitHub Release."
    ),
) -> None:
    """Download the pre-built release catalog (no llvm-mca required).

    Installs just the catalog snapshot and SQLite database (~2 files).
    Manpages are opt-in via ``simdref install-manpages``; site data for the
    web repo via ``simdref export``.
    """
    if from_release:
        _download_from_release()
        _finalize_runtime_from_download()
        return

    _download_release_or_fallback()


@app.command(rich_help_panel="Dev commands")
def build(
    man_dir: Path = typer.Option(DEFAULT_MAN_DIR, help="Target man root directory."),
) -> None:
    """Full local rebuild from upstream sources, including Intel SDM parsing (requires llvm-mca on PATH)."""
    _require_llvm_mca_or_hint()
    _build_runtime_locally(man_dir=man_dir, include_sdm=True)


def _render_manpage(conn, name: str) -> str | None:
    """Render a single manpage from the SQLite catalog (no files on disk)."""
    from simdref.manpages import instruction_page, intrinsic_page

    intrinsic = load_intrinsic_from_db(conn, name)
    if intrinsic is not None:
        linked = []
        for key in intrinsic.instructions:
            record = load_instruction_from_db(conn, key)
            if record is not None:
                linked.append(record)
        return intrinsic_page(intrinsic, linked_instructions=linked)
    rows = load_instructions_by_mnemonic_from_db(conn, name)
    if not rows and name != name.lower():
        rows = load_instructions_by_mnemonic_from_db(conn, name.lower())
    if not rows:
        return None
    x86 = next((r for r in rows if r.architecture == "x86"), None)
    return instruction_page(x86 or rows[0])


@app.command(rich_help_panel="Commands")
def man(
    name: str = typer.Argument(..., help="Intrinsic (e.g. _mm256_add_epi32) or mnemonic (vaddps)."),
) -> None:
    """Show the manpage for an intrinsic/instruction, rendered on demand.

    Pipes through ``man -l`` when available; otherwise prints the roff
    source so it can be piped manually. No pre-generated pages needed.
    """
    ensure_runtime()
    with open_db(SQLITE_PATH) as conn:
        page = _render_manpage(conn, name)
    if page is None:
        err_console.print(f"no manpage for: {name}", style="red")
        raise typer.Exit(code=1)
    man_bin = shutil.which("man")
    if man_bin is None:
        sys.stdout.write(page)
        return
    import signal
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".7", prefix="simdref-man-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(page)
        prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            # NB: ``man <path>`` (renders a local file) works on both man-db
            # and BSD/macOS man, unlike the man-db-only ``man -l``.
            raise_code = subprocess.call([man_bin, tmp], stdin=subprocess.DEVNULL)
        finally:
            signal.signal(signal.SIGINT, prev)
        if raise_code != 0:
            raise typer.Exit(code=raise_code)
    finally:
        Path(tmp).unlink(missing_ok=True)


@app.command(rich_help_panel="Commands")
def install_manpages(
    man_dir: Path = typer.Option(DEFAULT_MAN_DIR, help="Target man root directory."),
) -> None:
    """Pre-generate man7 pages for every intrinsic/instruction (~150k files).

    The default target is the XDG data-root man dir (``~/.local/share/man``),
    which man-db auto-discovers on Linux — so plain ``man vpaddd`` works with
    no MANPATH edits. Use on-demand ``simdref man`` to avoid the files
    entirely.
    """
    catalog = ensure_catalog()
    write_manpages(catalog, man_dir)
    err_console.print(f"wrote manpages to {man_dir}/man7", style="green")
    _integrate_manpath(man_dir)


def _integrate_manpath(man_dir: Path) -> None:
    """Best-effort post-install: rebuild the mandb index and confirm that
    plain ``man`` can discover the directory."""
    mandb = shutil.which("mandb")
    if mandb is not None:
        subprocess.run(
            [mandb, "-q", str(man_dir)],
            capture_output=True,
            check=False,
        )
    manpath = shutil.which("manpath")
    try:
        out = (
            subprocess.run([manpath, "-q"], capture_output=True, text=True, check=False).stdout
            if manpath
            else ""
        )
    except OSError:
        out = ""
    if str(man_dir) in out.split(":"):
        err_console.print("plain `man <name>` now works (dir is on your manpath)", style="green")
    else:
        probe = "could not probe manpath" if not out else "dir is not on your manpath"
        err_console.print(
            f"note: {probe} — if `man <name>` fails, set MANPATH={man_dir}:$MANPATH "
            "or keep using `simdref man <name>`",
            style="yellow",
        )


def _require_llvm_mca_or_hint() -> None:
    """Abort with an install hint when ``llvm-mca`` is missing on PATH.

    ``--build`` needs it to generate modeled ARM/RISC-V perf rows.
    Users who only want pre-built data can drop the flag.
    """
    from simdref.perf_sources.llvm_mca import LLVMMcaUnavailable, detect_llvm_mca_version

    try:
        detect_llvm_mca_version()
    except LLVMMcaUnavailable as exc:
        err_console.print(
            f"[bold red]llvm-mca is required for --build[/bold red]: {exc}",
        )
        err_console.print(LLVMMcaUnavailable.install_hint)
        raise typer.Exit(code=1) from exc


LLM_EXIT_MATCH = 0
LLM_EXIT_USAGE = 1
LLM_EXIT_NO_MATCH = 2
LLM_EXIT_AMBIGUOUS = 3
LLM_EXIT_INTERNAL = 10


def _resolve_preset_filters(preset: str | None) -> tuple[list[str] | None, list[str] | None]:
    """Translate a preset name into (isa_families, categories) overrides.

    Presets supply ISA-family + sub-ISA facets; we map them to the coarse
    ISA-family list the llm filter uses. Categories are not implied by a
    preset (they come from --filter / --category).
    """
    if not preset:
        return None, None
    from simdref.filters import ARCH_PRESETS

    spec = ARCH_PRESETS.get(preset)
    if spec is None:
        return None, None
    return sorted(spec.families), None


def _llm_filter_records(
    records: list[dict],
    isa: list[str] | None,
    category: list[str] | None,
    source_kind: str | None = None,
) -> list[dict]:
    """Filter llm payload dicts by ISA family, category, and source-kind."""
    from simdref.display import isa_family as _isa_family

    source_kind = (source_kind or "").strip().lower()
    if source_kind in ("", "any"):
        source_kind = ""
    if not isa and not category and not source_kind:
        return records
    isa_set = {f.strip() for f in (isa or []) if f and f.strip()}
    cat_set = {c.strip() for c in (category or []) if c and c.strip()}
    kept: list[dict] = []
    for rec in records:
        if isa_set:
            rec_isa = rec.get("isa") or []
            if isinstance(rec_isa, str):
                rec_isa = [rec_isa]
            families = {_isa_family(v) for v in rec_isa}
            if not families & isa_set:
                continue
        if cat_set:
            rec_cat = rec.get("category", "")
            if rec_cat not in cat_set:
                continue
        if source_kind:
            if not _record_has_source_kind(rec, source_kind):
                continue
        kept.append(rec)
    return kept


def _record_has_source_kind(rec: dict, wanted: str) -> bool:
    """Check whether an llm payload dict carries at least one entry with *wanted* provenance."""
    slim = rec.get("source_kinds")
    if isinstance(slim, list) and any(k == wanted for k in slim):
        return True
    arch_details = rec.get("arch_details") or {}
    if isinstance(arch_details, dict):
        for details in arch_details.values():
            if isinstance(details, dict):
                kind = details.get("source_kind") or "measured"
                if kind == wanted:
                    return True
    for nested_key in ("instruction", "instructions", "results"):
        nested = rec.get(nested_key)
        if isinstance(nested, dict):
            if _record_has_source_kind(nested, wanted):
                return True
        elif isinstance(nested, list):
            if any(_record_has_source_kind(n, wanted) for n in nested if isinstance(n, dict)):
                return True
    return False


def _llm_format_markdown(payload: dict) -> str:
    """Render an llm payload as prompt-friendly markdown."""
    mode = payload.get("mode", "search")
    query = payload.get("query", "")
    lines: list[str] = [f"# simdref: {query}", ""]
    if mode == "exact" and payload.get("match_kind") == "intrinsic":
        rec = payload.get("result", {})
        lines.append(f"**Intrinsic:** `{rec.get('intrinsic', '')}`")
        if rec.get("signature"):
            lines.append(f"**Signature:** `{rec['signature']}`")
        if rec.get("isa"):
            lines.append(f"**ISA:** {', '.join(rec['isa'])}")
        if rec.get("instructions"):
            lines.append(f"**Instruction:** `{rec['instructions'][0]}`")
        if rec.get("lat") and rec["lat"] != "-":
            lines.append(f"**Latency:** {rec['lat']}  •  **CPI:** {rec.get('cpi', '-')}")
        if rec.get("summary"):
            lines += ["", rec["summary"]]
        return "\n".join(lines)
    items = payload.get("results", [])
    if mode == "exact":
        lines.append(f"**{len(items)} instruction match(es)**")
    else:
        lines.append(f"**{len(items)} search result(s)**")
    lines.append("")
    for r in items:
        title = r.get("intrinsic") or r.get("query") or ""
        if isinstance(title, list):
            title = ", ".join(title)
        summary = r.get("summary", "")
        isa = ", ".join(r.get("isa") or [])
        lines.append(f"- **{title}** `{isa}` — {summary}")
    return "\n".join(lines)


def _emit_llm_payload(payload: dict, fmt: str) -> None:
    if fmt == "ndjson":
        mode = payload.get("mode")
        if mode == "exact" and "result" in payload:
            typer.echo(json.dumps(payload["result"], sort_keys=True))
            return
        for item in payload.get("results") or []:
            typer.echo(json.dumps(item, sort_keys=True))
        return
    if fmt == "markdown":
        typer.echo(_llm_format_markdown(payload))
        return
    typer.echo(json.dumps(payload, sort_keys=True, indent=2))


def _llm_exit_code(payload: dict) -> int:
    mode = payload.get("mode")
    if mode == "exact":
        if "result" in payload:
            return LLM_EXIT_MATCH
        results = payload.get("results") or []
        if len(results) > 1 and payload.get("match_kind") == "instruction":
            exact_name_hits = sum(
                1
                for r in results
                if r.get("query", "").casefold() == payload.get("query", "").casefold()
            )
            if exact_name_hits > 1:
                return LLM_EXIT_AMBIGUOUS
        return LLM_EXIT_MATCH if results else LLM_EXIT_NO_MATCH
    return LLM_EXIT_MATCH if payload.get("results") else LLM_EXIT_NO_MATCH


def _llm_schema_payload() -> dict:
    """Approximate JSON Schema for llm payloads (stable for tool consumers)."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "simdref.llm",
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "mode": {"type": "string", "enum": ["exact", "search"]},
            "match_kind": {"type": ["string", "null"], "enum": ["intrinsic", "instruction", None]},
            "generated_at": {
                "type": "string",
                "description": "ISO-8601 timestamp of the catalog build the answer was derived from.",
            },
            "source_versions": {
                "type": "array",
                "description": "Upstream source descriptors (name, version, url) pinned by this catalog.",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "version": {"type": "string"},
                        "url": {"type": "string"},
                    },
                },
            },
            "result": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "intrinsic": {"type": ["string", "array"]},
                    "signature": {"type": "string"},
                    "instructions": {"type": "array", "items": {"type": "string"}},
                    "instruction_refs": {
                        "type": "array",
                        "description": "Resolved instruction references when known.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "name": {"type": "string"},
                                "form": {"type": "string"},
                                "architecture": {"type": "string"},
                                "xed": {"type": "string"},
                                "resolution": {"type": "string"},
                                "match_count": {"type": "integer"},
                            },
                        },
                    },
                    "isa": {"type": "array", "items": {"type": "string"}},
                    "lat": {
                        "type": "string",
                        "description": (
                            "Best latency in cycles across every microarchitecture in the "
                            "catalog, or the pinned one under --arch. Use `timing` when the "
                            "target part matters."
                        ),
                    },
                    "cpi": {
                        "type": "string",
                        "description": (
                            "Best cycles-per-instruction across every microarchitecture in "
                            "the catalog, or the pinned one under --arch."
                        ),
                    },
                    "arch": {
                        "type": "string",
                        "description": "Canonical core id the record was pinned to by --arch.",
                    },
                    "timing": {
                        "type": "object",
                        "description": (
                            "Per-microarchitecture timing keyed by canonical core id (SKX, "
                            "ZEN4, neoverse-v2, ...). Latency and throughput are properties "
                            "of the part, not of the ISA, so they are reported per core. A "
                            "core absent from the map cannot execute the form, or carries no "
                            "measurement. On an intrinsic record the values are reduced "
                            "across the linked instruction forms, the same way `lat`/`cpi` are."
                        ),
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "lat": {
                                    "type": ["number", "null"],
                                    "description": "Latency in cycles on this core.",
                                },
                                "cpi": {
                                    "type": ["number", "null"],
                                    "description": (
                                        "Cycles per instruction on this core. "
                                        "ceil(lat / cpi) independent chains saturate the unit."
                                    ),
                                },
                                "ports": {
                                    "type": "string",
                                    "description": (
                                        "Upstream port-pressure string, e.g. '1*p01': one uop "
                                        "issuable on port 0 or port 1."
                                    ),
                                },
                                "uops": {"type": "string"},
                                "source_kind": {
                                    "type": "string",
                                    "enum": ["measured", "modeled"],
                                },
                            },
                        },
                    },
                    "source_kinds": {
                        "type": "array",
                        "description": "Distinct provenance kinds present in this record.",
                        "items": {"type": "string", "enum": ["measured", "modeled"]},
                    },
                    "summary": {"type": "string"},
                    "url": {
                        "type": "string",
                        "description": "Vendor documentation URL (Intel Intrinsics Guide / ARM ACLE).",
                    },
                    "operation": {
                        "type": "string",
                        "description": "SDM-style pseudocode describing the intrinsic's behavior, when available.",
                    },
                },
            },
            "results": {"type": "array", "items": {"$ref": "#/properties/result"}},
        },
        "required": ["query", "mode"],
    }


llm_app = typer.Typer(
    help="Structured output for LLM/tool consumption.", invoke_without_command=False
)
_LLM_HELP_PANEL = "Commands"


def _llm_catalog_meta(conn) -> dict:
    """Catalog provenance for an llm payload: build stamp and pinned sources."""
    from simdref.storage import generated_at_from_db, load_sources_from_db

    return {
        "generated_at": generated_at_from_db(conn),
        "source_versions": [asdict(source) for source in load_sources_from_db(conn)],
    }


def _pin_arch(records: list[dict], arch: str) -> list[dict]:
    """Restrict every record's ``timing`` map to *arch* and re-key ``lat``/``cpi``.

    Records whose ``timing`` map has no entry for *arch* are dropped: the
    upstream perf sources list only the cores that can execute a form, so
    an absent core means that core cannot run it.
    """
    pinned: list[dict] = []
    for rec in records:
        entry = (rec.get("timing") or {}).get(arch)
        if entry is None:
            continue
        rec = dict(rec)
        rec["timing"] = {arch: entry}
        rec["arch"] = arch
        rec["lat"] = "-" if entry.get("lat") is None else _fmt_perf_scalar(entry["lat"])
        rec["cpi"] = "-" if entry.get("cpi") is None else f"{entry['cpi']:.2f}"
        rec["source_kinds"] = [entry["source_kind"]]
        pinned.append(rec)
    return pinned


def _fmt_perf_scalar(value: float) -> str:
    """Format a cycle count the way the unpinned ``lat`` field already does."""
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def _resolve_llm_arch_or_exit(arch: str | None) -> str | None:
    """Map an ``--arch`` alias to its canonical core id, or exit with a usage error."""
    if not arch:
        return None
    from simdref.perf_sources.cores import canonical_core_id, supported_core_ids

    canonical = canonical_core_id(arch)
    if canonical is None:
        typer.echo(
            f"error: unknown --arch '{arch}' (known: {', '.join(supported_core_ids())})", err=True
        )
        raise typer.Exit(code=LLM_EXIT_USAGE)
    return canonical


def _build_llm_payload(
    conn,
    query_str: str,
    limit: int,
    isa: list[str] | None,
    category: list[str] | None,
    source_kind: str | None,
    arch: str | None = None,
) -> dict:
    """Build the llm payload for *query_str* against an open DB connection.

    Kept free of I/O and exit logic so that ``simdref llm batch`` can call it
    in a loop without re-opening the catalog per query.
    """
    meta = _llm_catalog_meta(conn)
    if arch:
        meta["arch"] = arch

    def finish(records: list[dict]) -> list[dict]:
        records = _llm_filter_records(records, isa, category, source_kind=source_kind)
        return _pin_arch(records, arch) if arch else records

    intrinsic = load_intrinsic_from_db(conn, query_str)
    if intrinsic is not None:
        kept = finish([_llm_intrinsic_payload(conn, intrinsic)])
        return {
            "query": query_str,
            "mode": "exact",
            "match_kind": "intrinsic" if kept else None,
            **({"result": kept[0]} if kept else {"results": []}),
            **meta,
        }
    instructions = _find_instructions_fast(query_str)
    if instructions:
        return {
            "query": query_str,
            "mode": "exact",
            "match_kind": "instruction",
            "results": finish([_llm_instruction_payload(item) for item in instructions]),
            **meta,
        }
    results, intrinsic_map, instruction_map = _search_runtime(conn, query_str, limit=limit)
    return {
        "query": query_str,
        "mode": "search",
        "match_kind": None,
        "results": finish(
            [_llm_result_payload(conn, r, intrinsic_map, instruction_map) for r in results]
        ),
        **meta,
    }


def _normalize_fmt(fmt: str, allowed: set[str]) -> str:
    fmt_lower = (fmt or "json").lower()
    if fmt_lower not in allowed:
        typer.echo(
            f"error: unknown --format '{fmt}' (expected {'|'.join(sorted(allowed))})",
            err=True,
        )
        raise typer.Exit(code=LLM_EXIT_USAGE)
    return fmt_lower


def _resolve_preset_or_exit(preset: str | None, isa: list[str] | None) -> list[str] | None:
    if not preset:
        return isa
    from simdref.filters import ARCH_PRESETS

    if preset not in ARCH_PRESETS:
        known = ", ".join(sorted(ARCH_PRESETS))
        typer.echo(f"error: unknown --preset '{preset}' (known: {known})", err=True)
        raise typer.Exit(code=LLM_EXIT_USAGE)
    preset_isa, _ = _resolve_preset_filters(preset)
    if preset_isa and not isa:
        return preset_isa
    return isa


def _llm_query_impl(
    query_tokens: list[str],
    limit: int,
    fmt: str,
    isa: list[str] | None,
    category: list[str] | None,
    preset: str | None = None,
    source_kind: str | None = None,
    arch: str | None = None,
) -> None:
    fmt_lower = _normalize_fmt(fmt, {"json", "ndjson", "markdown"})
    isa = _resolve_preset_or_exit(preset, isa)
    canonical_arch = _resolve_llm_arch_or_exit(arch)
    if not query_tokens:
        typer.echo(
            "error: query required (or use `simdref llm list` / `simdref llm schema`)", err=True
        )
        raise typer.Exit(code=LLM_EXIT_USAGE)
    query_str = " ".join(query_tokens)
    ensure_runtime()
    try:
        with open_db() as conn:
            payload = _build_llm_payload(
                conn, query_str, limit, isa, category, source_kind, arch=canonical_arch
            )
    except typer.Exit:
        raise
    except Exception as exc:  # pragma: no cover - internal error path
        typer.echo(f"internal error: {exc}", err=True)
        raise typer.Exit(code=LLM_EXIT_INTERNAL)
    _emit_llm_payload(payload, fmt_lower)
    raise typer.Exit(code=_llm_exit_code(payload))


@llm_app.command("query")
def llm_query(
    query: list[str] = typer.Argument(..., help="Search query (multiple tokens allowed)."),
    limit: int = typer.Option(8, help="Maximum number of search results in search mode."),
    fmt: str = typer.Option(
        "json", "--format", "-F", help="Output format: json, ndjson, or markdown."
    ),
    isa: list[str] = typer.Option(None, "--isa", help="Filter by ISA family (repeatable)."),
    preset: str = typer.Option(
        None,
        "--preset",
        help="Apply a named preset (default, intel, arm32, arm64, riscv, none, all).",
    ),
    source_kind: str = typer.Option(
        "any", "--source-kind", help="Filter perf rows by provenance: measured, modeled, or any."
    ),
    arch: str = typer.Option(
        None,
        "--arch",
        help=(
            "Pin lat/cpi/ports to one microarchitecture (e.g. znver4, skylake-x); "
            "drops forms the core cannot execute."
        ),
    ),
) -> None:
    """Resolve a query and emit an LLM-friendly payload.

    Exit codes: 0 match, 2 no-match, 3 ambiguous, 1 usage error, 10 internal.
    """
    _llm_query_impl(query, limit, fmt, isa, None, preset=preset, source_kind=source_kind, arch=arch)


def _emit_filtered_names(
    conn,
    pattern: str,
    isa: list[str] | None,
) -> int:
    """Stream NDJSON records matching *pattern* filtered by ISA family.

    Iterates the SQLite catalog directly so the caller avoids loading the
    full msgpack payload for every candidate. Returns number of records
    emitted (caller uses this to decide the exit code).
    """
    from simdref.display import isa_family as _isa_family

    glob_pat = pattern
    isa_set = {f.strip() for f in (isa or []) if f and f.strip()}
    emitted = 0

    intrinsic_rows = conn.execute(
        "SELECT name, isa, category FROM intrinsics_data ORDER BY name"
    ).fetchall()
    for row in intrinsic_rows:
        name = row["name"]
        if not fnmatch.fnmatchcase(name, glob_pat) and not fnmatch.fnmatch(
            name.casefold(), glob_pat.casefold()
        ):
            continue
        isas = [s.strip() for s in (row["isa"] or "").split(",") if s.strip()]
        if isa_set:
            families = {_isa_family(s) for s in isas}
            if not families & isa_set:
                continue
        typer.echo(
            json.dumps(
                {"name": name, "kind": "intrinsic", "isa": isas, "category": row["category"] or ""},
                sort_keys=True,
            )
        )
        emitted += 1

    instruction_rows = conn.execute(
        "SELECT key, db_key, isa, category FROM instructions_data ORDER BY key"
    ).fetchall()
    for row in instruction_rows:
        key = row["key"]
        db_key = row["db_key"]
        if (
            not fnmatch.fnmatchcase(key, glob_pat)
            and not fnmatch.fnmatch(key.casefold(), glob_pat.casefold())
            and not fnmatch.fnmatchcase(db_key, glob_pat)
        ):
            continue
        isas = [s.strip() for s in (row["isa"] or "").split(",") if s.strip()]
        if isa_set:
            families = {_isa_family(s) for s in isas}
            if not families & isa_set:
                continue
        typer.echo(
            json.dumps(
                {
                    "name": key,
                    "kind": "instruction",
                    "isa": isas,
                    "category": row["category"] or "",
                },
                sort_keys=True,
            )
        )
        emitted += 1
    return emitted


@llm_app.command("list")
def llm_list(
    fmt: str = typer.Option(
        "json",
        "--format",
        "-F",
        help="Output format: json or markdown (ignored when --pattern is given).",
    ),
    pattern: str = typer.Option(
        None,
        "--pattern",
        help="Glob filter over intrinsic/instruction names. When set, the command emits NDJSON {name, kind, isa, category} records instead of the FilterSpec.",
    ),
    isa: list[str] = typer.Option(
        None, "--isa", help="Restrict --pattern output to the given ISA family (repeatable)."
    ),
) -> None:
    """Emit the FilterSpec or stream matching catalog entries.

    Without ``--pattern`` this emits the full :class:`FilterSpec` describing
    ISA families, sub-ISAs, and categories. With ``--pattern GLOB`` it emits
    NDJSON records for each matching intrinsic/instruction — useful for a
    Claude skill that wants "all AVX-512 *gather* intrinsics" without
    calling ``query`` per name.
    """
    ensure_runtime()
    if pattern:
        with open_db() as conn:
            emitted = _emit_filtered_names(conn, pattern, isa)
        raise typer.Exit(code=LLM_EXIT_MATCH if emitted else LLM_EXIT_NO_MATCH)

    from simdref.filters import build_filter_spec

    with open_db() as conn:
        spec = build_filter_spec(conn)
    payload = spec.to_json()
    if (fmt or "json").lower() == "markdown":
        lines = ["# simdref filter spec", "", "## ISA families"]
        for fam in payload["default_enabled"]:
            lines.append(f"- **{fam}** (default)")
        for fam in payload["family_order"]:
            if fam not in payload["default_enabled"]:
                lines.append(f"- {fam}")
        lines += ["", "## Categories"]
        for cat in payload["categories"]:
            lines.append(f"- {cat['family']} / {cat['category']} ({cat['count']})")
        typer.echo("\n".join(lines))
        return
    typer.echo(json.dumps(payload, sort_keys=True, indent=2))


@llm_app.command("batch")
def llm_batch(
    limit: int = typer.Option(8, help="Maximum number of search results per query in search mode."),
    isa: list[str] = typer.Option(None, "--isa", help="Filter results by ISA family (repeatable)."),
    preset: str = typer.Option(
        None,
        "--preset",
        help="Apply a named preset (default, intel, arm32, arm64, riscv, none, all).",
    ),
    source_kind: str = typer.Option(
        "any", "--source-kind", help="Filter perf rows by provenance: measured, modeled, or any."
    ),
    arch: str = typer.Option(
        None,
        "--arch",
        help=(
            "Pin lat/cpi/ports to one microarchitecture (e.g. znver4, skylake-x); "
            "drops forms the core cannot execute."
        ),
    ),
) -> None:
    """Resolve queries from stdin (one per line); emit NDJSON records.

    Each output line is ``{"query": ..., "status": "match|no_match|ambiguous|error",
    "payload": {...}}``. Amortizes catalog load across hundreds of lookups — useful
    when a Claude skill resolves every mnemonic in a disassembly.
    """
    isa = _resolve_preset_or_exit(preset, isa)
    canonical_arch = _resolve_llm_arch_or_exit(arch)
    ensure_runtime()
    with open_db() as conn:
        for raw_line in sys.stdin:
            query = raw_line.strip()
            if not query or query.startswith("#"):
                continue
            try:
                payload = _build_llm_payload(
                    conn,
                    query,
                    limit,
                    isa,
                    None,
                    source_kind,
                    arch=canonical_arch,
                )
                exit_code = _llm_exit_code(payload)
                if exit_code == LLM_EXIT_MATCH:
                    status = "match"
                elif exit_code == LLM_EXIT_AMBIGUOUS:
                    status = "ambiguous"
                else:
                    status = "no_match"
                typer.echo(
                    json.dumps(
                        {"query": query, "status": status, "payload": payload},
                        sort_keys=True,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                typer.echo(
                    json.dumps(
                        {"query": query, "status": "error", "error": str(exc)},
                        sort_keys=True,
                    )
                )


@llm_app.command("schema")
def llm_schema() -> None:
    """Emit the JSON Schema for `simdref llm` payloads."""
    typer.echo(json.dumps(_llm_schema_payload(), sort_keys=True, indent=2))


app.add_typer(llm_app, name="llm", rich_help_panel=_LLM_HELP_PANEL)


# ---------------------------------------------------------------------------
# Shell completion (opt-in subcommand; replaces Typer's default
# --install-completion / --show-completion options)
# ---------------------------------------------------------------------------


completion_app = typer.Typer(help="Shell completion helpers.", no_args_is_help=True)

_COMPLETION_SHELLS = ("bash", "zsh", "fish", "powershell", "pwsh")


def _resolve_completion_shell(shell: str | None) -> str:
    if shell:
        shell = shell.strip().lower()
    else:
        shell_env = os.environ.get("SHELL", "")
        shell = Path(shell_env).name.lower() if shell_env else ""
    if shell not in _COMPLETION_SHELLS:
        err_console.print(
            f"error: unsupported or undetected shell '{shell}'; pass one of {', '.join(_COMPLETION_SHELLS)}",
            style="red",
        )
        raise typer.Exit(code=1)
    return shell


def _completion_prog_name() -> str:
    prog = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "simdref"
    # Strip a stray ``__main__.py`` when invoked via ``python -m simdref``.
    if prog in {"", "__main__.py"}:
        prog = "simdref"
    return prog


@completion_app.command("show")
def completion_show(
    shell: str = typer.Argument(
        None, help="Shell: bash, zsh, fish, or powershell. Detected from $SHELL when omitted."
    ),
) -> None:
    """Print a shell completion script to stdout."""
    shell = _resolve_completion_shell(shell)
    from typer._completion_shared import get_completion_script

    prog_name = _completion_prog_name()
    complete_var = f"_{prog_name.upper().replace('-', '_')}_COMPLETE"
    typer.echo(get_completion_script(prog_name=prog_name, complete_var=complete_var, shell=shell))


@completion_app.command("install")
def completion_install(
    shell: str = typer.Argument(
        None, help="Shell: bash, zsh, fish, or powershell. Detected from $SHELL when omitted."
    ),
) -> None:
    """Install shell completion into the user's shell profile."""
    shell = _resolve_completion_shell(shell)
    from typer._completion_shared import install as _install_completion

    prog_name = _completion_prog_name()
    complete_var = f"_{prog_name.upper().replace('-', '_')}_COMPLETE"
    try:
        shell_detected, path = _install_completion(
            shell=shell, prog_name=prog_name, complete_var=complete_var
        )
    except Exception as exc:
        err_console.print(f"error: completion install failed: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    err_console.print(
        f"installed {shell_detected} completion for {prog_name} at {path}", style="green"
    )


app.add_typer(completion_app, name="completion", rich_help_panel="Dev commands")


def _registered_command_names() -> set[str]:
    """Return the set of Typer commands + subcommand groups the dispatcher knows about.

    Kept as introspection so the bare-word dispatcher in ``main()`` never drifts
    from the real command surface.
    """
    names: set[str] = set()
    for info in getattr(app, "registered_commands", []):
        if info.name:
            names.add(info.name)
        elif info.callback is not None:
            names.add(info.callback.__name__.replace("_", "-"))
    for info in getattr(app, "registered_groups", []):
        if info.name:
            names.add(info.name)
    names.update({"--help", "-h"})
    return names


@app.command(rich_help_panel="Commands")
def doctor() -> None:
    """Check the installation and report pass/fail for each component.

    Exits with a non-zero status when any required check fails so this
    command is usable from scripts and CI.
    """
    from rich.table import Table

    ok_icon = "[green]✓[/]"
    fail_icon = "[red]✗[/]"
    warn_icon = "[yellow]![/]"
    failures = 0
    warnings = 0

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("", width=2)
    table.add_column("check", style="cyan", no_wrap=True)
    table.add_column("status")
    table.add_column("detail", style="dim")

    # SQLite runtime (the required artifact)
    if not SQLITE_PATH.exists():
        table.add_row(
            fail_icon, "sqlite index", "[red]missing[/]", f"{SQLITE_PATH} — run `simdref update`"
        )
        failures += 1
        console.print(table)
        console.print(f"\n[red]{failures} check failed.[/]")
        raise typer.Exit(1)
    if not sqlite_schema_is_current():
        table.add_row(
            warn_icon,
            "sqlite index",
            "[yellow]outdated schema[/]",
            "rebuild with `simdref update --build`",
        )
        warnings += 1
    else:
        table.add_row(ok_icon, "sqlite index", "[green]current[/]", str(SQLITE_PATH))

    # Catalog snapshot (optional cache; pruned after install/update)
    if CATALOG_PATH.exists():
        try:
            catalog = load_catalog()
        except (OSError, ValueError, _MsgpackUnpackException) as exc:
            table.add_row(
                warn_icon,
                "catalog snapshot",
                "[yellow]unreadable — using database[/]",
                f"{CATALOG_PATH}: {exc}",
            )
            warnings += 1
            catalog = load_catalog_from_db()
        else:
            table.add_row(ok_icon, "catalog snapshot", "[green]present[/]", str(CATALOG_PATH))
    else:
        catalog = load_catalog_from_db()
        table.add_row(
            ok_icon,
            "catalog snapshot",
            "[dim]pruned[/]",
            f"{CATALOG_PATH.name} removed to save space — rebuildable from database",
        )

    # Catalog counts
    n_intr = len(catalog.intrinsics)
    n_instr = len(catalog.instructions)
    if n_intr > 0 and n_instr > 0:
        table.add_row(
            ok_icon,
            "catalog data",
            "[green]populated[/]",
            f"{n_intr:,} intrinsics · {n_instr:,} instructions",
        )
    else:
        table.add_row(
            fail_icon,
            "catalog data",
            "[red]empty[/]",
            f"{n_intr} intrinsics · {n_instr} instructions",
        )
        failures += 1

    # Sources
    if catalog.sources:
        table.add_row(ok_icon, "sources", "[green]recorded[/]", f"{len(catalog.sources)} source(s)")
        for source in catalog.sources:
            table.add_row("", f"  {source.source}", "", f"version={source.version}")
    else:
        table.add_row(
            warn_icon, "sources", "[yellow]none recorded[/]", "catalog has no provenance entries"
        )
        warnings += 1

    # FTS smoke test
    if SQLITE_PATH.exists() and sqlite_schema_is_current():
        try:
            from simdref.storage import open_db

            with open_db() as conn:
                row = conn.execute(
                    "SELECT count(*) AS c FROM intrinsics_fts WHERE intrinsics_fts MATCH ?",
                    ("add",),
                ).fetchone()
                hits = row["c"] if row else 0
            if hits > 0:
                table.add_row(
                    ok_icon, "fts search", "[green]working[/]", f"query 'add' -> {hits} hits"
                )
            else:
                table.add_row(
                    warn_icon, "fts search", "[yellow]no hits[/]", "query 'add' returned 0 hits"
                )
                warnings += 1
        except Exception as exc:
            table.add_row(fail_icon, "fts search", "[red]error[/]", str(exc))
            failures += 1

    # Man page directory (informational — missing is fine)
    man_present = DEFAULT_MAN_DIR.exists() and any(DEFAULT_MAN_DIR.rglob("*"))
    if man_present:
        table.add_row(ok_icon, "man pages", "[green]present[/]", str(DEFAULT_MAN_DIR))
    else:
        table.add_row(
            warn_icon,
            "man pages",
            "[dim]not installed[/]",
            f"{DEFAULT_MAN_DIR} — optional; install with `simdref install-manpages`",
        )

    console.print(table)

    if failures:
        console.print(
            f"\n[red]{failures} failed[/], [yellow]{warnings} warnings[/] — simdref is not ready."
        )
        raise typer.Exit(1)
    if warnings:
        console.print(
            f"\n[yellow]OK with {warnings} warning(s)[/] — simdref will run but consider the notes above."
        )
        return
    console.print("\n[bold green]All checks passed.[/] simdref is ready.")


@app.command("export", rich_help_panel="Dev commands")
def export_command(
    out_dir: Path = typer.Option(
        WEB_DIR, help="Output directory for the site-data JSON export (no HTML)."
    ),
) -> None:
    """Export site data (JSON only): the release contract for the web repo."""
    catalog = ensure_catalog()
    export_site_data(catalog, out_dir)
    console.print(f"exported site data to {out_dir}", style="green")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point — dispatches to subcommand or smart lookup."""
    global SHOW_FP16_ISAS, SHORT_MODE, FULL_MODE
    argv = sys.argv[1:]
    if any(arg in ("--version", "-V") for arg in argv):
        print(__version__)
        return 0
    if _is_completion_invocation():
        app()
        return 0
    if "--fp16" in argv:
        SHOW_FP16_ISAS = True
        argv = [arg for arg in argv if arg != "--fp16"]
    if "--short" in argv or "-s" in argv:
        SHORT_MODE = True
        argv = [arg for arg in argv if arg not in ("--short", "-s")]
    if "--full" in argv or "-f" in argv:
        FULL_MODE = True
        argv = [arg for arg in argv if arg not in ("--full", "-f")]
    # Pre-parse top-level --preset / --arch / --json for bare-query mode.
    # Subcommands (llm, annotate, etc.) handle their own flags via Typer,
    # so only strip these when they would otherwise reach the smart-lookup
    # dispatch.
    initial_preset: str | None = None
    bare_arch: str | None = None
    bare_json = False
    _cleaned: list[str] = []
    _i = 0
    while _i < len(argv):
        arg = argv[_i]
        if arg == "--preset" and _i + 1 < len(argv):
            initial_preset = argv[_i + 1]
            _i += 2
            continue
        if arg.startswith("--preset="):
            initial_preset = arg.split("=", 1)[1]
            _i += 1
            continue
        if arg == "--arch" and _i + 1 < len(argv):
            bare_arch = argv[_i + 1]
            _i += 2
            continue
        if arg.startswith("--arch="):
            bare_arch = arg.split("=", 1)[1]
            _i += 1
            continue
        if arg == "--json":
            bare_json = True
            _i += 1
            continue
        _cleaned.append(arg)
        _i += 1
    # Only consume --preset at the top level when the remainder is a bare
    # query or empty; otherwise leave it for the subcommand (e.g. `llm query`).
    subcommand_consumers = {"llm", "annotate"}
    if _cleaned and _cleaned[0] in subcommand_consumers:
        # Restore; let Typer subcommand parse it.
        pass
    else:
        argv = _cleaned
    # Rewrite `llm <bare-query>` to `llm query <bare-query>` so Typer's
    # subcommand dispatch (list/batch/schema/query) works without stealing
    # bare queries. Derived from Typer introspection so new subcommands
    # automatically become recognised.
    llm_subcommands = {
        info.name for info in getattr(llm_app, "registered_commands", []) if info.name
    }
    llm_subcommands |= {"--help", "-h"}
    if (
        argv
        and argv[0] == "llm"
        and len(argv) >= 2
        and argv[1] not in llm_subcommands
        and not argv[1].startswith("-")
    ):
        argv = ["llm", "query", *argv[1:]]
    sys.argv = [sys.argv[0], *argv]
    commands = _registered_command_names()
    if argv and argv[0] not in commands and not argv[0].startswith("-"):
        # Tolerate ``simdref show <mnemonic>`` / ``simdref lookup <mnemonic>``:
        # users reach for a verb, but there's no ``show`` subcommand — just
        # drop a recognised lookup verb so the remainder hits smart-lookup.
        if len(argv) > 1 and argv[0].lower() in {"show", "lookup", "info"}:
            argv = argv[1:]
        return _smart_lookup(
            " ".join(argv), preset=initial_preset, arch=bare_arch, as_json=bare_json
        )
    if not argv:
        ensure_runtime()
        # Pass initial_preset through verbatim — _run_tui handles the
        # (explicit --preset > last-used state > "intel") precedence.
        return _run_tui(initial_preset=initial_preset)
    app()
    return 0
