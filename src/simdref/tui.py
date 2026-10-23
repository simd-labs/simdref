"""Interactive Textual-based terminal UI for simdref.

Provides fuzzy search across intrinsics and instructions with
collapsible detail sections, keyboard navigation, and Rich rendering.

Uses SQLite FTS for fast search instead of loading the full catalog.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import TYPE_CHECKING

_PROFILE = os.environ.get("SIMDREF_PROFILE") == "1"


def _profile_log(tag: str, elapsed_ms: float, **extras: object) -> None:
    if not _PROFILE:
        return
    extras_str = " ".join(f"{k}={v}" for k, v in extras.items())
    sys.stderr.write(f"[simdref-profile] {tag} {elapsed_ms:.1f}ms {extras_str}\n")


from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

try:
    from textual import events, on, work
    from textual.app import App, ComposeResult
    from textual.message import Message
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.screen import ModalScreen
    from textual.widgets import (
        Button,
        Checkbox,
        Collapsible,
        ContentSwitcher,
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        Select,
        Static,
        TextArea,
    )
except ImportError:  # pragma: no cover - allows non-TUI test environments

    class _TextualStub:
        def __init__(self, *args, **kwargs):
            pass

    class _EventStub:
        pass

    def on(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    def work(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    class App:  # type: ignore[override]
        pass

    class ModalScreen(_TextualStub):  # type: ignore[override]
        pass

    ComposeResult = object
    Message = _TextualStub
    Binding = _TextualStub
    Horizontal = _TextualStub
    Vertical = _TextualStub
    VerticalScroll = _TextualStub
    Button = _TextualStub
    Button.Pressed = _EventStub
    Checkbox = _TextualStub
    Checkbox.Changed = _EventStub
    Collapsible = _TextualStub
    ContentSwitcher = _TextualStub
    Footer = _TextualStub
    Header = _TextualStub
    Input = _TextualStub
    Input.Changed = _EventStub
    Input.Submitted = _EventStub
    Label = _TextualStub
    ListItem = _TextualStub
    ListView = _TextualStub
    ListView.Selected = _EventStub
    ListView.Highlighted = _EventStub
    Select = _TextualStub
    Select.Changed = _EventStub
    Static = _TextualStub
    TextArea = _TextualStub
    events = _TextualStub()

from simdref.display import (
    _CODE_SECTION_LANG,
    _DESCRIPTION_ORDER,
    _EXPANDED_SECTIONS,
    _MEASUREMENT_EXCLUDE_KEYS,
    _MEASUREMENT_PREFERRED_ORDER,
    DEFAULT_ENABLED_ISAS,
    DEFAULT_SUBS,
    FAMILY_SUB_ORDER,
    ISA_FAMILY_ORDER,
    canonical_url,
    display_isa,
    display_instruction_form,
    display_uarch,
    isa_family,
    perf_panel_border,
    perf_panel_title,
    split_perf_rows,
    isa_to_sub_isa,
    instruction_metadata_rows,
    normalize_isa_token,
    strip_instruction_decorators,
    measurement_rows,
    uarch_sort_key,
)
from simdref.perf import variant_perf_summary
from simdref.queries import linked_instruction_records
from simdref.search import SearchResult
from simdref.storage import (
    SQLITE_PATH,
    load_instruction_from_db,
    load_intrinsic_from_db,
    open_db,
    sqlite_schema_is_current,
)

from simdref.models import InstructionRecord, IntrinsicRecord
from simdref.ui_labels import UI_LABELS

_INITIAL_RESULT_BATCH = 50
_RESULT_BATCH_SIZE = 10
_RESULT_PREFETCH_THRESHOLD = 5


def _NO_DATA(text: str) -> tuple[str, str]:
    """Detail line that marks a section with no data, instead of omitting it."""
    return ("static_markup", f"  [dim]{text}[/]")


_ISA_FAMILIES: list[str] = [
    name for name, _ in sorted(ISA_FAMILY_ORDER.items(), key=lambda item: item[1])
]


def _isa_matches_sub(isa: str, sub_isa: str) -> bool:
    """Return whether a raw ISA token belongs to the given sub-ISA."""
    normalized_isa = normalize_isa_token(isa)
    normalized_sub = normalize_isa_token(sub_isa)
    return normalized_isa == normalized_sub or (
        normalized_isa.startswith(normalized_sub) and len(normalized_isa) > len(normalized_sub)
    )


def _normalize_sub_isa_selection(
    enabled_families: set[str],
    enabled_sub_isas: set[str] | None,
    family_subs: dict[str, list[str]],
) -> set[str] | None:
    """Keep only sub-ISAs that belong to enabled families.

    Returns ``None`` when all visible sub-ISAs are effectively enabled.
    """
    visible_subs: set[str] = set()
    for family in enabled_families:
        visible_subs.update(family_subs.get(family, []))
    if not visible_subs:
        return None
    if enabled_sub_isas is None:
        return None
    normalized = {sub for sub in enabled_sub_isas if sub in visible_subs}
    if not normalized or normalized == visible_subs:
        return None
    return normalized


# ---------------------------------------------------------------------------
# FTS search
# ---------------------------------------------------------------------------


_MNEMONIC_HINT_TOKENS = frozenset({"add", "sub", "mul", "div", "mov", "cmp", "and", "or", "xor"})


def _classify_query(query: str) -> str:
    """Mirror the web SPA's `classifyQuery` so ranking biases match."""
    lo = query.strip().lower()
    if not lo:
        return "neutral"
    if lo.startswith("_mm") or lo.startswith("__m") or "_mm" in lo:
        return "intrinsic"
    parts = lo.replace("_", " ").split()
    if parts:
        first = parts[0]
        if first in _MNEMONIC_HINT_TOKENS or first.startswith("v"):
            return "instruction"
    return "neutral"


def _name_match_score(name: str, terms: list[str]) -> int:
    """Score how well query terms match a name. Higher = better match.

    Rewards: all terms present in name, contiguous terms, exact substrings.
    """
    name_lower = name.lower().replace("_", " ")
    score = 0
    # Count how many terms appear in the name
    matched = sum(1 for t in terms if t in name_lower)
    score += matched * 10
    # Bonus if ALL terms are in the name
    if matched == len(terms):
        score += 50
    # Bonus for contiguous substring: "256 add ps" in "mm256 add ps"
    query_joined = " ".join(terms)
    if query_joined in name_lower:
        score += 100
    # Bonus for underscore-joined match: "256_add_ps" in name
    query_underscored = "_".join(terms)
    if query_underscored in name.lower():
        score += 100
    # Penalty for longer names (prefer more specific matches)
    score -= len(name) // 10
    return score


def _fts_search(
    conn: sqlite3.Connection,
    query: str,
    enabled_families: set[str],
    enabled_sub_isas: set[str] | None = None,
    offset: int = 0,
    limit: int = 30,
    enabled_kinds: set[str] | None = None,
) -> list[SearchResult]:
    """Search using SQLite FTS5 with ISA family and sub-ISA filtering."""
    _t_start = time.perf_counter() if _PROFILE else 0.0
    results: list[SearchResult] = []
    fts_query = query.replace('"', '""').replace("*", "")
    if not fts_query.strip():
        return results
    terms = [t.lower() for t in fts_query.replace("_", " ").strip().split() if t]
    if not terms:
        return results
    term_expr = " ".join(f'"{t}"*' for t in terms)
    # Search name/name_tokens first (precise), fall back to all columns (broad)
    fts_expr = f"{{name name_tokens}} : {term_expr}"
    fts_expr_broad = term_expr

    # Build an ISA LIKE-clause fragment so SQLite can drop non-matching rows
    # before we ever hand them to Python. Sub-ISA tokens (e.g. "SSE2") are
    # substring-matched against the isa column.
    #
    # Families without a sub-ISA breakdown (e.g. "x86", "MMX", "APX", "AVX10",
    # "SVML", "Other") have raw ISA tokens like "I86" or "BMI1" — none of which
    # contain the family-name string. We can't construct a sound LIKE pattern
    # for those, so when any such family is enabled we skip the SQL pushdown
    # entirely and let the Python `_isa_visible` post-filter do the work.
    # Bias scoring by query classification (mirrors the web SPA): a mnemonic
    # query like "add" should rank instructions above intrinsics, while an
    # `_mm…`/`__m…` query should still favour intrinsics. Neutral queries get
    # no kind bias.
    qkind = _classify_query(query)
    if qkind == "instruction":
        intrinsic_bias, instruction_bias, default_kind = -10, 35, "instruction"
    elif qkind == "intrinsic":
        intrinsic_bias, instruction_bias, default_kind = 35, -10, "intrinsic"
    else:
        intrinsic_bias, instruction_bias, default_kind = 0, 0, "intrinsic"

    has_no_sub_family = bool(enabled_families) and any(
        not FAMILY_SUB_ORDER.get(fam) for fam in enabled_families
    )
    isa_like_tokens: list[str] = []
    if not has_no_sub_family and enabled_sub_isas is not None:
        isa_like_tokens.extend(enabled_sub_isas)
    # Normalise the isa column the same way _isa_matches_sub does: strip
    # hyphens so "AVX-512F" matches a pushed-down "AVX512F" pattern. REPLACE
    # isn't indexed, but it runs only over FTS-matching rows.
    if isa_like_tokens:
        normalized_tokens = [tok.replace("-", "") for tok in isa_like_tokens]
        isa_like_sql = (
            "(" + " OR ".join("REPLACE(isa, '-', '') LIKE ?" for _ in normalized_tokens) + ")"
        )
        isa_like_params = [f"%{tok}%" for tok in normalized_tokens]
    else:
        isa_like_sql = ""
        isa_like_params = []

    def _isa_visible(isa_str: str) -> bool:
        parts = [v.strip() for v in isa_str.replace(",", " ").split() if v.strip()]
        families = {isa_family(v) for v in parts}
        if enabled_families and not families & enabled_families:
            return False
        if enabled_sub_isas is not None:
            for p in parts:
                for sub in enabled_sub_isas:
                    if _isa_matches_sub(p, sub):
                        return True
                if isa_family(p) not in FAMILY_SUB_ORDER:
                    return True
            return False
        return True

    # Collect candidates and re-rank by name match quality
    candidates: list[tuple[int, SearchResult]] = []

    seen: set[str] = set()

    def _target_candidate_count() -> int:
        return offset + limit + max(limit, 20)

    # With the ISA LIKE filter in SQL, fetch_limit can stay modest — the
    # filter drops non-matching rows before they reach Python.
    isa_filter_clause = f" AND {isa_like_sql}" if isa_like_sql else ""

    def _query_intrinsic_rows(expr: str) -> list[sqlite3.Row]:
        rows: list[sqlite3.Row] = []
        fetch_limit = max((offset + limit) * 2, 60)
        while fetch_limit <= 5000:
            for sql in (
                f"SELECT name, summary, description, isa FROM intrinsics_fts WHERE intrinsics_fts MATCH ?{isa_filter_clause} ORDER BY rank LIMIT ?",
                f"SELECT name, description, isa FROM intrinsics_fts WHERE intrinsics_fts MATCH ?{isa_filter_clause} ORDER BY rank LIMIT ?",
            ):
                try:
                    rows = conn.execute(sql, (expr, *isa_like_params, fetch_limit)).fetchall()
                    break
                except sqlite3.OperationalError:
                    continue
            visible = sum(
                1 for row in rows if row["name"] not in seen and _isa_visible(row["isa"] or "")
            )
            if visible >= _target_candidate_count() or len(rows) < fetch_limit:
                return rows
            fetch_limit *= 2
        return rows

    def _query_instruction_rows(expr: str) -> list[sqlite3.Row]:
        rows: list[sqlite3.Row] = []
        fetch_limit = max((offset + limit) * 2, 60)
        while fetch_limit <= 5000:
            try:
                rows = conn.execute(
                    f"""
                    SELECT instructions_data.db_key, instructions_fts.key, instructions_fts.summary, instructions_fts.isa
                    FROM instructions_fts
                    JOIN instructions_data ON instructions_data.rowid = instructions_fts.rowid
                    WHERE instructions_fts MATCH ?{isa_filter_clause.replace("REPLACE(isa,", "REPLACE(instructions_fts.isa,")}
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (expr, *isa_like_params, fetch_limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return rows
            visible = sum(
                1 for row in rows if row["db_key"] not in seen and _isa_visible(row["isa"] or "")
            )
            if visible >= _target_candidate_count() or len(rows) < fetch_limit:
                return rows
            fetch_limit *= 2
        return rows

    # Search name columns first (precise), then broaden to all columns.
    # The name-only query uses {name name_tokens} column filter which
    # requires schema v7+; fall back to broad search on OperationalError.
    search_exprs = [fts_expr, fts_expr_broad]
    want_intrinsic = enabled_kinds is None or "intrinsic" in enabled_kinds
    want_instruction = enabled_kinds is None or "instruction" in enabled_kinds
    for expr in search_exprs:
        rows = _query_intrinsic_rows(expr) if want_intrinsic else []
        for row in rows:
            if row["name"] in seen or not _isa_visible(row["isa"] or ""):
                continue
            seen.add(row["name"])
            subtitle = (
                (row["summary"] if "summary" in row.keys() else "") or row["description"] or ""
            )
            result = SearchResult(
                kind="intrinsic",
                key=row["name"],
                title=row["name"],
                subtitle=subtitle,
                score=100,
            )
            candidates.append((_name_match_score(row["name"], terms) + intrinsic_bias, result))

        irows = _query_instruction_rows(expr) if want_instruction else []
        for row in irows:
            if row["db_key"] in seen or not _isa_visible(row["isa"] or ""):
                continue
            seen.add(row["db_key"])
            result = SearchResult(
                kind="instruction",
                key=row["db_key"],
                title=strip_instruction_decorators(row["key"]),
                subtitle=row["summary"] or "",
                score=90,
            )
            candidates.append((_name_match_score(row["key"], terms) + instruction_bias, result))

    # Sort by (biased) match score descending. The kind tiebreaker follows the
    # query classification so a mnemonic-like query ("add") doesn't get its
    # instructions buried under intrinsics with identical text scores.
    candidates.sort(
        key=lambda c: (
            -c[0],
            0 if c[1].kind == default_kind else 1,
        )
    )
    results = [r for _, r in candidates[offset : offset + limit]]
    if _PROFILE:
        _profile_log(
            "_fts_search", (time.perf_counter() - _t_start) * 1000, q=repr(query), n=len(results)
        )
    return results


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class _AnnSplitter(Static):
    """Draggable vertical splitter between the annotate input and output.

    Mouse drag adjusts the split ratio; arrow keys also work when focused.
    Emits a custom message so the App can re-flow the panes."""

    DEFAULT_CSS = ""  # styled from the App's global CSS block

    can_focus = True

    class Moved(Message):
        def __init__(self, ratio: float) -> None:
            super().__init__()
            self.ratio = max(0.1, min(0.9, ratio))

    def __init__(self, *, glyph: str = "║", id: str | None = None) -> None:
        super().__init__(glyph, id=id, classes="ann-splitter")
        self._dragging = False

    def on_mouse_down(self, event: events.MouseDown) -> None:  # pragma: no cover - UI wiring
        self._dragging = True
        self.add_class("-dragging")
        self.capture_mouse(True)
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:  # pragma: no cover - UI wiring
        if not self._dragging:
            return
        self._dragging = False
        self.remove_class("-dragging")
        self.capture_mouse(False)
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:  # pragma: no cover - UI wiring
        if not self._dragging:
            return
        parent = self.parent
        if parent is None:
            return
        parent_region = parent.region
        if parent_region.width <= 0:
            return
        # event.screen_x is relative to the screen origin; shift into parent coords.
        x_in_parent = event.screen_x - parent_region.x
        ratio = x_in_parent / parent_region.width
        self.post_message(self.Moved(ratio))

    def on_key(self, event: events.Key) -> None:  # pragma: no cover - UI wiring
        if event.key in ("left", "right"):
            delta = -0.05 if event.key == "left" else 0.05
            parent = self.parent
            parent_region = getattr(parent, "region", None) if parent is not None else None
            width = parent_region.width if parent_region else 60
            if width <= 0:
                return
            # Nudge by a % step, independent of current position — the App
            # clamps the new ratio.
            current_fraction = None
            try:
                sibling = parent.query_one("#ann-input") if parent else None
                cw = sibling.region.width if sibling and sibling.region else None
                if cw is not None and width > 0:
                    current_fraction = cw / width
            except Exception:
                current_fraction = None
            new_ratio = (current_fraction if current_fraction is not None else 0.33) + delta
            self.post_message(self.Moved(new_ratio))
            event.stop()


class _ToggleChip(Static):
    """Base clickable toggle chip for ISA/category filters."""

    DEFAULT_CSS = """
    _ToggleChip {
        width: auto;
        height: 1;
        padding: 0 1;
        margin: 0 0 0 1;
    }
    _ToggleChip.enabled {
        background: $accent;
        color: $text;
    }
    _ToggleChip.disabled {
        background: $surface;
        color: $text-muted;
    }
    """

    class Toggled(Message):
        """Posted when a toggle chip is clicked."""

        def __init__(self, toggle: _ToggleChip) -> None:
            super().__init__()
            self.toggle = toggle

    def __init__(self, label: str, enabled: bool = False) -> None:
        super().__init__(label)
        self.label_text = label
        self.enabled = enabled
        self.add_class("enabled" if enabled else "disabled")

    def on_click(self) -> None:
        self.enabled = not self.enabled
        self.remove_class("enabled" if not self.enabled else "disabled")
        self.add_class("enabled" if self.enabled else "disabled")
        self.post_message(self.Toggled(self))

    def set_enabled(self, enabled: bool) -> None:
        """Update the enabled state without firing a Toggled event."""
        if self.enabled == enabled:
            return
        self.enabled = enabled
        self.remove_class("enabled" if not enabled else "disabled")
        self.add_class("enabled" if enabled else "disabled")


class ToggleAllLabel(Static):
    """Clickable label that toggles all sibling chips on/off."""

    class Clicked(Message):
        def __init__(self, label: ToggleAllLabel) -> None:
            super().__init__()
            self.label = label

    def on_click(self) -> None:
        self.post_message(self.Clicked(self))


class PresetButton(Static):
    """Clickable button that applies a named preset."""

    class Clicked(Message):
        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

    def __init__(self, label: str, mode: str, **kwargs) -> None:
        super().__init__(label, **kwargs)
        self.mode = mode

    def on_click(self) -> None:
        self.post_message(self.Clicked(self.mode))


class IsaToggle(_ToggleChip):
    """A clickable ISA family toggle chip."""

    def __init__(self, family: str, enabled: bool = False) -> None:
        super().__init__(family, enabled)
        self.family = family


class SubIsaToggle(_ToggleChip):
    """A clickable sub-ISA extension toggle chip."""

    def __init__(self, isa: str, enabled: bool = True) -> None:
        super().__init__(isa, enabled)
        self.isa = isa


class KindToggle(_ToggleChip):
    """Toggle chip for result kind (intrinsic | instruction)."""

    def __init__(self, kind: str, label: str, enabled: bool = True) -> None:
        super().__init__(label, enabled)
        self.kind = kind


class SearchInput(Input):
    """Search box with result-navigation arrow key behavior."""

    def key_down(self) -> None:
        app = self.app
        if hasattr(app, "_move_result_focus"):
            app._move_result_focus(1)

    def key_up(self) -> None:
        app = self.app
        if hasattr(app, "_move_result_focus"):
            app._move_result_focus(-1)

    async def _on_key(self, event: events.Key) -> None:  # type: ignore[override]
        # '?' always opens the help modal; if we let the Input consume it we
        # would instead insert a literal '?' into the search value.
        if event.character == "?":
            event.prevent_default()
            event.stop()
            action = getattr(self.app, "action_show_help", None)
            if action is not None:
                action()
            return
        await super()._on_key(event)


class ResultItem(ListItem):
    """A search result list item carrying the underlying SearchResult.

    Row format (single line, plus a dim summary on a second line): the
    mnemonic/name on the left, then a right-aligned ``kind · arch · ISA · perf``
    strip. Perf slots (``lat``/``tput``) are emitted only when present —
    missing values are not rendered as "—" (prevents noise on Arm/RISC-V
    rows where perf is usually absent).
    """

    def __init__(self, result: SearchResult, index: int) -> None:
        self.result = result
        title_color = "cyan" if result.kind == "intrinsic" else "magenta"
        subtitle = (result.subtitle or "").split("\n")[0]
        # Right-side chips. ``arch`` / ``isa`` come off the SearchResult
        # if present; perf only if non-empty / non-placeholder.
        chips: list[str] = [f"[dim]{result.kind}[/]"]
        arch = getattr(result, "architecture", "") or getattr(result, "arch", "")
        if arch:
            chips.append(f"[dim]{arch}[/]")
        isa = getattr(result, "isa_display", "") or ""
        if isa:
            chips.append(f"[dim]{isa}[/]")
        lat = getattr(result, "lat", "") or ""
        cpi = getattr(result, "cpi", "") or ""
        if lat and lat != "-":
            chips.append(f"[yellow]lat {lat}c[/]")
        if cpi and cpi != "-":
            chips.append(f"[yellow]tput {cpi}[/]")
        right = "  ".join(chips)
        head = f"[dim]{index:>2}[/] [{title_color} bold]{result.title}[/]"
        if subtitle:
            head = f"{head}  [dim italic]{subtitle}[/]"
        head = f"{head}    {right}"
        super().__init__(Static(head, markup=True))


class _AnnInput(TextArea):
    """Annotate input TextArea using Textual's native editing actions."""

    def action_paste(self) -> None:
        app = self.app
        text = None
        prepare = getattr(app, "_prepare_annotate_clipboard_for_paste", None)
        if prepare is not None:
            text = prepare()
            if text is None:
                return
        else:
            text = getattr(app, "clipboard", "") or ""
            if not text:
                return
        super().action_paste()
        try:
            app.query_one("#ann-status", Label).update(f"Pasted {len(text)} chars")
        except Exception:
            pass
        schedule = getattr(app, "_schedule_annotate", None)
        if schedule is not None:
            schedule(delay=0.75 if len(text) > 5000 else 0.35)


class _AnnOutput(Static):
    """Annotate output pane — focusable, copies its full rendered text."""

    can_focus = True

    BINDINGS = [
        Binding("ctrl+c", "copy_all", "Copy all", show=False),
    ]

    def action_copy_all(self) -> None:
        app = self.app
        fn = getattr(app, "_ann_copy_output", None)
        if fn is not None:
            fn()


class AnnContextMenu(ModalScreen):
    """Right-click context menu for the annotate input/output panes."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
        Binding("q", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    AnnContextMenu {
        align: center middle;
        background: rgba(0, 0, 0, 0.35);
    }
    AnnContextMenu > Vertical {
        width: 28;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 0;
    }
    AnnContextMenu Button {
        width: 100%;
        border: none;
        background: $surface;
        color: $text;
        content-align: left middle;
        padding: 0 2;
        height: 1;
    }
    AnnContextMenu Button:hover,
    AnnContextMenu Button:focus {
        background: $accent 25%;
        color: $accent;
        text-style: bold;
    }
    """

    def __init__(self, target_id: str, has_selection: bool) -> None:
        super().__init__()
        self._target_id = target_id
        self._has_selection = has_selection

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Button("Cut", id="ctx-cut", disabled=not self._has_selection)
            yield Button(
                "Copy",
                id="ctx-copy",
                disabled=not self._has_selection and self._target_id == "ann-input",
            )
            yield Button("Paste", id="ctx-paste")
            yield Button("Select All", id="ctx-select-all")
            yield Button("Clear", id="ctx-clear")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = getattr(event.button, "id", "") or ""
        app = self.app
        fn = getattr(app, "_ctx_action", None)
        # Dispatch via a helper on the app so the action runs in the
        # right context after the modal pops.
        self.dismiss()
        if fn is not None:
            fn(bid, self._target_id)

    def action_dismiss(self) -> None:  # type: ignore[override]
        self.app.pop_screen()


class HelpScreen(ModalScreen):
    """Modal overlay listing active keybindings."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("q", "dismiss", "Close", show=False),
        Binding("?", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    HelpScreen > Static {
        width: 64;
        max-height: 24;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    """

    def __init__(self, bindings) -> None:
        super().__init__()
        self._bindings_source = bindings

    def compose(self) -> ComposeResult:
        rows: list[str] = ["[b]simdref — keybindings[/b]", ""]
        seen: set[tuple[str, str]] = set()
        for binding in self._bindings_source:
            key = getattr(binding, "key_display", None) or getattr(binding, "key", "")
            action = getattr(binding, "action", "")
            description = getattr(binding, "description", "") or action
            if not key or (key, action) in seen:
                continue
            seen.add((key, action))
            rows.append(f"[cyan]{key:<10}[/] {description}")
        rows.append("")
        rows.append("[dim]esc / q / ? to close[/]")
        yield Static("\n".join(rows), markup=True)

    def action_dismiss(self) -> None:  # type: ignore[override]
        self.app.pop_screen()


class SimdrefApp(App):
    """Interactive SIMD reference browser."""

    TITLE = "simdref"

    CSS = """
    #tabs-bar {
        dock: top;
        height: 1;
        margin: 0 1;
    }
    #tabs-bar .tab-btn {
        width: auto;
        color: $text-muted;
        padding: 0 2;
        margin: 0 1 0 0;
        text-style: none;
        background: transparent;
        border: none;
    }
    #tabs-bar .tab-btn:hover { color: $text; }
    #tabs-bar .tab-btn.active {
        color: $accent;
        text-style: bold underline;
    }
    #view-switcher { height: 1fr; }
    #annotate-view { height: 1fr; layout: vertical; }
    #ann-toolbar {
        height: auto;
        padding: 0 1;
    }
    #ann-toolbar Checkbox { margin: 0 2 0 0; width: auto; border: none; padding: 0; background: transparent; }
    #ann-toolbar Select { width: 22; margin: 0 1 0 0; }
    #ann-toolbar Label { width: auto; margin: 0 1 0 0; color: $text-muted; }
    #ann-panes {
        height: 1fr;
        margin: 0 1;
    }
    #ann-input, #ann-output-wrap {
        border: solid $primary;
        height: 100%;
    }
    #ann-input { width: 33fr; }
    #ann-output-wrap { width: 67fr; }
    #ann-output {
        height: auto;
        min-height: 100%;
        padding: 0 1;
    }
    .ann-splitter {
        width: 1;
        height: 100%;
        background: $primary 20%;
        color: $accent;
        content-align: center middle;
    }
    .ann-splitter:hover, .ann-splitter.-dragging {
        background: $accent 50%;
    }
    #ann-status {
        dock: bottom;
        height: 1;
        margin: 0 1;
        color: $text-muted;
    }
    #search-input {
        dock: top;
        margin: 0 1;
    }
    #isa-bar {
        height: 1;
        margin: 0 1;
        overflow-x: auto;
    }
    #kind-bar {
        height: 1;
        margin: 0 1;
        overflow-x: auto;
    }
    #kind-bar .isa-label {
        width: auto;
        color: $text-muted;
        padding: 0 1 0 0;
        text-style: underline;
    }
    #sub-isa-container {
        height: auto;
        max-height: 10;
        margin: 0 1;
        overflow-y: auto;
    }
    .sub-isa-row {
        height: 1;
    }
    .sub-isa-row .sub-fam-label {
        width: 10;
        color: $text-muted;
        text-style: underline;
    }
    #isa-bar .isa-label {
        width: auto;
        color: $text-muted;
        padding: 0 1 0 0;
        text-style: underline;
    }
    #isa-bar .preset-btn {
        width: auto;
        background: $surface;
        color: $warning;
        padding: 0 1;
        margin: 0 0 0 1;
        text-style: bold;
    }
    /* Results list sizes to its content but never takes more than 20% of
       the terminal height. Detail pane soaks up the remainder with 1fr,
       so short result lists (e.g. 2 matches) don't leave blank rows above
       a cramped detail view. */
    #results-list {
        height: auto;
        max-height: 20%;
        border: solid $accent;
        margin: 0 1;
        overflow-y: scroll;
    }
    #detail-scroll {
        height: 1fr;
        margin: 0 1;
        border: solid $primary;
    }
    #status-label {
        dock: bottom;
        height: 1;
        margin: 0 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("/", "focus_search", "Search", show=True),
        Binding("ctrl+k", "focus_search", show=False),
        Binding("?", "show_help", "Help", show=True, priority=True),
        Binding("escape", "back", "Back", show=True),
        Binding("f", "toggle_all", "Expand/Collapse All", show=True),
        Binding("1-9", "pick(0)", "Pick result", show=True, key_display="1-9"),
        Binding("c", "copy_detail", "Copy", show=True),
        Binding("ctrl+t", "switch_tab", "Switch tab", show=True, priority=True, key_display="^t"),
        Binding("ctrl+left", "ann_split(-0.05)", show=False, priority=True, key_display="^←"),
        Binding("ctrl+right", "ann_split(0.05)", show=False, priority=True, key_display="^→"),
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+d", "quit", "Quit", show=True, priority=True, key_display="^d"),
        # Non-priority so TextArea / Input consume letters while focused —
        # otherwise pasted asm containing "j"/"k" gets swallowed.
        Binding("j", "list_cursor_down", show=False),
        Binding("k", "list_cursor_up", show=False),
        *[Binding(str(n), f"pick({n})", show=False) for n in range(1, 10)],
    ]

    def __init__(
        self,
        initial_query: str = "",
        initial_preset: str | None = None,
        *,
        initial_view: str = "search",
        initial_asm: str = "",
    ) -> None:
        super().__init__()
        self._initial_query = initial_query
        self._initial_preset = initial_preset
        self._initial_view = initial_view if initial_view in ("search", "annotate") else "search"
        self._initial_asm = initial_asm
        self._annotate_debounce_timer = None  # textual Timer, set in on_mount
        self._annotate_last_text: str = ""
        self._ann_split_ratio: float = 0.33  # fraction of width for input pane
        self._current_results: list[SearchResult] = []
        self._current_query: str = ""
        self._has_more_results = False
        self._all_expanded = False
        self._conn: sqlite3.Connection | None = None
        self._enabled_families: set[str] = set(DEFAULT_ENABLED_ISAS)
        # Build initial sub-ISA set from defaults for enabled families
        initial_subs: set[str] = set()
        for fam in DEFAULT_ENABLED_ISAS:
            initial_subs.update(DEFAULT_SUBS.get(fam, set()))
        self._enabled_sub_isas: set[str] | None = initial_subs if initial_subs else None
        self._current_detail: IntrinsicRecord | InstructionRecord | None = None
        self._current_detail_token: int = 0
        self._batch_toggle = False
        # LRU caches keyed by (kind, key) so arrow-key navigation through
        # results hits cache instead of re-loading + re-building.
        self._detail_cache: dict[tuple[str, str], IntrinsicRecord | InstructionRecord] = {}
        self._detail_cache_order: list[tuple[str, str]] = []
        self._render_cache: dict[tuple[str, str], list[tuple]] = {}
        self._render_cache_order: list[tuple[str, str]] = []
        self._detail_cache_cap = 256
        # Map family -> list of sub-ISA names present in the DB
        self._family_subs: dict[str, list[str]] = {}
        self._enabled_kinds: set[str] = {"intrinsic", "instruction"}
        self._enabled_arm_arch: set[str] | None = None
        self._ann_local_clipboard_valid = False
        # Cached frozensets of filter state passed to _fts_search; cleared
        # whenever a toggle changes. Avoids rebuilding sets per keystroke.
        self._filter_fs_cache: tuple[frozenset[str], frozenset[str] | None] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="tabs-bar"):
            yield Static("Search", id="tab-search", classes="tab-btn")
            yield Static("Annotate", id="tab-annotate", classes="tab-btn")
        with ContentSwitcher(id="view-switcher", initial=f"view-{self._initial_view}"):
            with Vertical(id="view-search"):
                yield SearchInput(
                    placeholder="Search intrinsics and instructions...",
                    id="search-input",
                    value=self._initial_query,
                )
                with Horizontal(id="isa-bar"):
                    yield ToggleAllLabel("ISA:", classes="isa-label", id="isa-toggle-all")
                    for family in _ISA_FAMILIES:
                        yield IsaToggle(family, enabled=family in self._enabled_families)
                    yield PresetButton("Default", "default", classes="preset-btn")
                    yield PresetButton("Intel", "intel", classes="preset-btn")
                    yield PresetButton("Arm32", "arm32", classes="preset-btn")
                    yield PresetButton("Arm64", "arm64", classes="preset-btn")
                    yield PresetButton("RISC-V", "riscv", classes="preset-btn")
                    yield PresetButton("None", "none", classes="preset-btn")
                    yield PresetButton("All", "all", classes="preset-btn")
                yield VerticalScroll(id="sub-isa-container")
                with Horizontal(id="kind-bar"):
                    yield ToggleAllLabel("Kind:", classes="isa-label", id="kind-label")
                    yield KindToggle(
                        "intrinsic",
                        UI_LABELS["kind_intrinsic"].lower(),
                        enabled="intrinsic" in self._enabled_kinds,
                    )
                    yield KindToggle(
                        "instruction",
                        UI_LABELS["kind_instruction"].lower(),
                        enabled="instruction" in self._enabled_kinds,
                    )
                yield ListView(id="results-list")
                yield VerticalScroll(id="detail-scroll")
            with Vertical(id="view-annotate"):
                with Horizontal(id="ann-toolbar"):
                    yield Label("ISA")
                    yield Select(
                        [("x86", "x86"), ("arm", "arm"), ("riscv", "riscv"), ("any", "")],
                        value="x86",
                        allow_blank=False,
                        id="ann-isa",
                    )
                    yield Label(UI_LABELS["aggregation"])
                    yield Select(
                        [
                            ("avg", "avg"),
                            ("median", "median"),
                            ("best", "best"),
                            ("worst", "worst"),
                        ],
                        value="avg",
                        allow_blank=False,
                        id="ann-agg",
                    )
                    yield Checkbox(UI_LABELS["toggle_perf"], value=True, id="ann-perf")
                    yield Checkbox(UI_LABELS["toggle_docs"], value=True, id="ann-docs")
                    yield Checkbox(UI_LABELS["toggle_modeled"], value=True, id="ann-modeled")
                    yield Button("Annotate", id="ann-run", variant="primary")
                    yield Button("Clear", id="ann-clear")
                with Horizontal(id="ann-panes"):
                    yield _AnnInput.code_editor(
                        self._initial_asm,
                        language=None,
                        soft_wrap=False,
                        id="ann-input",
                    )
                    yield _AnnSplitter(id="ann-splitter")
                    yield VerticalScroll(_AnnOutput("", id="ann-output"), id="ann-output-wrap")
        yield Label("", id="status-label")
        yield Label("", id="ann-status")
        yield Footer()

    def on_mount(self) -> None:
        if not sqlite_schema_is_current():
            self._needs_update = True
            self.query_one("#status-label", Label).update(
                f"  Database not found or outdated. Press [bold]y[/] to update, [bold]n[/] to quit."
            )
            return
        self._needs_update = False
        self._conn = open_db()
        self._build_family_subs()
        if self._initial_preset:
            self._apply_initial_preset(self._initial_preset)
        # Defer until first layout pass so container width is known
        self.call_after_refresh(self._refresh_sub_isa_bar)
        # Apply initial tab + annotate seed before focusing anything.
        self._apply_ann_split(self._ann_split_ratio)
        self._set_tab(self._initial_view)
        if self._initial_view == "annotate":
            # _set_tab already focused ann-input and scheduled annotate.
            return
        if self._initial_query:
            self._run_initial_query()
        else:
            self.query_one("#search-input", Input).focus()

    def on_resize(self) -> None:
        # Reflow sub-ISA chips only when width actually changes
        container = self.query_one("#sub-isa-container", VerticalScroll)
        new_w = container.size.width or 0
        if new_w and new_w != getattr(self, "_last_sub_width", 0):
            self._last_sub_width = new_w
            self.call_after_refresh(self._refresh_sub_isa_bar)

    def _build_family_subs(self) -> None:
        """Build sub-ISA lists from ISA tokens present in the current DB."""
        assert self._conn is not None
        family_subs: dict[str, set[str]] = {}
        rows = self._conn.execute(
            """
            SELECT isa FROM instructions_data
            UNION ALL
            SELECT isa FROM intrinsics_data
            """
        ).fetchall()
        for row in rows:
            for raw_isa in str(row["isa"] or "").replace(",", " ").split():
                family = isa_family(raw_isa)
                sub_isa = isa_to_sub_isa(raw_isa)
                if family and sub_isa:
                    family_subs.setdefault(family, set()).add(sub_isa)
        for fam, subs in family_subs.items():
            self._family_subs[fam] = sorted(
                subs,
                key=lambda sub: (
                    FAMILY_SUB_ORDER.get(fam, []).index(sub)
                    if sub in FAMILY_SUB_ORDER.get(fam, [])
                    else len(FAMILY_SUB_ORDER.get(fam, []))
                ),
            )
        self._enabled_sub_isas = _normalize_sub_isa_selection(
            self._enabled_families, self._enabled_sub_isas, self._family_subs
        )

    def _refresh_sub_isa_bar(self) -> None:
        """Rebuild the sub-ISA rows — one or more rows per enabled family.

        Fast path: when the set of (family, sub_isa) widgets hasn't
        changed, just update each SubIsaToggle's enabled state in place.
        Full remount is reserved for family-set or layout-width changes.
        """
        container = self.query_one("#sub-isa-container", VerticalScroll)
        current = list(container.query(SubIsaToggle))
        current_keys: set[tuple[str, str]] = set()
        for toggle in current:
            parent_row = toggle.parent
            # The family prefix label is the first child in each row.
            fam_label = parent_row.children[0] if parent_row is not None else None
            fam_text = getattr(fam_label, "renderable", None)
            fam = str(fam_text).split(":", 1)[0].strip() if fam_text else ""
            if fam:
                current_keys.add((fam, toggle.isa))

        needed_keys: set[tuple[str, str]] = set()
        for fam in _ISA_FAMILIES:
            if fam not in self._enabled_families or fam not in self._family_subs:
                continue
            for sub in self._family_subs[fam]:
                needed_keys.add((fam, sub))

        if current_keys == needed_keys and current:
            # Fast path — just flip enabled bits.
            for toggle in current:
                parent_row = toggle.parent
                fam_label = parent_row.children[0] if parent_row is not None else None
                fam_text = getattr(fam_label, "renderable", None)
                fam = str(fam_text).split(":", 1)[0].strip() if fam_text else ""
                if self._enabled_sub_isas is not None:
                    enabled = toggle.isa in self._enabled_sub_isas
                else:
                    defaults = DEFAULT_SUBS.get(fam, set())
                    enabled = toggle.isa in defaults if defaults else True
                toggle.set_enabled(enabled)
            return

        container.remove_children()

        max_width = container.size.width or 120
        label_width = 10  # width of .sub-fam-label

        for fam in _ISA_FAMILIES:
            if fam not in self._enabled_families or fam not in self._family_subs:
                continue
            subs = self._family_subs[fam]
            defaults = DEFAULT_SUBS.get(fam, set())

            # Build toggles with their approximate widths
            toggles: list[SubIsaToggle] = []
            for sub in subs:
                if self._enabled_sub_isas is not None:
                    enabled = sub in self._enabled_sub_isas
                else:
                    enabled = sub in defaults if defaults else True
                toggles.append(SubIsaToggle(sub, enabled=enabled))

            # Split into rows that fit within max_width
            row_children: list[ToggleAllLabel | SubIsaToggle] = [
                ToggleAllLabel(f"{fam}:", classes="sub-fam-label"),
            ]
            used = label_width
            for toggle in toggles:
                chip_w = len(toggle.isa) + 3  # padding + margin
                if used + chip_w > max_width and len(row_children) > 1:
                    container.mount(Horizontal(*row_children, classes="sub-isa-row"))
                    # Continuation row: indent with blank label
                    row_children = [Static(" " * label_width, classes="sub-fam-label")]
                    used = label_width
                row_children.append(toggle)
                used += chip_w
            if len(row_children) > 1:
                container.mount(Horizontal(*row_children, classes="sub-isa-row"))

    def on_unmount(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @work
    async def _run_initial_query(self) -> None:
        """Populate results from initial query and open the first match."""
        self._do_search(self._initial_query)
        results_list = self.query_one("#results-list", ListView)
        if results_list.children:
            results_list.index = 0
            first = results_list.children[0]
            if isinstance(first, ResultItem):
                self._show_detail(first.result)
                self.query_one("#detail-scroll").focus()

    @on(Input.Changed, "#search-input")
    def on_search_changed(self, event: Input.Changed) -> None:
        self._debounced_search(event.value.strip())

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        """Enter in search box: expand first result, collapse list, move focus."""
        if self._current_results:
            self._show_detail(self._current_results[0])
            results_list = self.query_one("#results-list", ListView)
            results_list.index = 0
            self.query_one("#detail-scroll").focus()

    @on(_ToggleChip.Toggled)
    def on_toggle_changed(self, event: _ToggleChip.Toggled) -> None:
        """Re-run search when ISA or sub-ISA filter changes."""
        toggle = event.toggle
        if isinstance(toggle, IsaToggle):
            if toggle.enabled:
                self._enabled_families.add(toggle.family)
                # Add default subs for newly enabled family
                defaults = DEFAULT_SUBS.get(toggle.family, set())
                if defaults and self._enabled_sub_isas is not None:
                    self._enabled_sub_isas.update(defaults)
            else:
                self._enabled_families.discard(toggle.family)
                # Remove subs belonging to disabled family
                if self._enabled_sub_isas is not None:
                    fam_subs = set(self._family_subs.get(toggle.family, []))
                    self._enabled_sub_isas -= fam_subs
            self._enabled_sub_isas = _normalize_sub_isa_selection(
                self._enabled_families, self._enabled_sub_isas, self._family_subs
            )
            self._refresh_sub_isa_bar()
        elif isinstance(toggle, SubIsaToggle):
            all_subs = list(self.query(SubIsaToggle))
            enabled = {t.isa for t in all_subs if t.enabled}
            all_shown = {t.isa for t in all_subs}
            self._enabled_sub_isas = enabled if enabled != all_shown else None
            self._enabled_sub_isas = _normalize_sub_isa_selection(
                self._enabled_families, self._enabled_sub_isas, self._family_subs
            )
        elif isinstance(toggle, KindToggle):
            if toggle.enabled:
                self._enabled_kinds.add(toggle.kind)
            else:
                self._enabled_kinds.discard(toggle.kind)
        query = self.query_one("#search-input", Input).value.strip()
        if query:
            self._do_search(query)

    @on(ToggleAllLabel.Clicked)
    def on_toggle_all_clicked(self, event: ToggleAllLabel.Clicked) -> None:
        """Toggle all chips in the same bar/row."""
        label = event.label
        if label.id == "isa-toggle-all":
            toggles = list(self.query(IsaToggle))
            target = not any(t.enabled for t in toggles)
            for t in toggles:
                t.enabled = target
                t.remove_class("enabled" if not target else "disabled")
                t.add_class("enabled" if target else "disabled")
            self._enabled_families = {t.family for t in toggles if t.enabled}
            if target:
                # Add defaults for all families
                subs: set[str] = set()
                for fam in self._enabled_families:
                    subs.update(DEFAULT_SUBS.get(fam, set()))
                self._enabled_sub_isas = subs if subs else None
            else:
                self._enabled_sub_isas = set()
            self._enabled_sub_isas = _normalize_sub_isa_selection(
                self._enabled_families, self._enabled_sub_isas, self._family_subs
            )
            self._refresh_sub_isa_bar()
        else:
            # Family-level sub toggle: find sibling SubIsaToggles in same row
            row = label.parent
            if row is None:
                return
            toggles = list(row.query(SubIsaToggle))
            if not toggles:
                return
            target = not any(t.enabled for t in toggles)
            for t in toggles:
                t.enabled = target
                t.remove_class("enabled" if not target else "disabled")
                t.add_class("enabled" if target else "disabled")
            # Rebuild enabled_sub_isas from all visible toggles
            all_subs = list(self.query(SubIsaToggle))
            enabled = {t.isa for t in all_subs if t.enabled}
            self._enabled_sub_isas = enabled if enabled else set()
            self._enabled_sub_isas = _normalize_sub_isa_selection(
                self._enabled_families, self._enabled_sub_isas, self._family_subs
            )
        query = self.query_one("#search-input", Input).value.strip()
        if query:
            self._do_search(query)

    def _apply_initial_preset(self, name: str) -> None:
        """Apply a named preset before first render (used for --preset CLI flag)."""
        from simdref.filters import ARCH_PRESETS

        preset = ARCH_PRESETS.get(name)
        if preset is None:
            return
        if name == "all":
            self._enabled_families = set(_ISA_FAMILIES)
            self._enabled_sub_isas = None
        else:
            self._enabled_families = {f for f in _ISA_FAMILIES if f in preset.families}
            preset_subs = set(preset.subs)
            subs: set[str] = set()
            for fam in self._enabled_families:
                available = set(self._family_subs.get(fam, []))
                subs.update(available & preset_subs)
            self._enabled_sub_isas = subs if subs else set()
        self._enabled_arm_arch = set(preset.arm_arch) if preset.arm_arch else None
        self._enabled_kinds = set(preset.kind)
        self._enabled_sub_isas = _normalize_sub_isa_selection(
            self._enabled_families, self._enabled_sub_isas, self._family_subs
        )

    @on(PresetButton.Clicked)
    def on_preset_clicked(self, event: PresetButton.Clicked) -> None:
        """Apply ISA presets — families + subs + arm_arch + kind in one step.

        Also persist the choice so the next ``isa`` launch lands on the same
        preset (see ``_save_last_preset`` in ``cli.py``).
        """
        from simdref.filters import ARCH_PRESETS

        try:
            from simdref.cli import _save_last_preset

            _save_last_preset(event.mode)
        except Exception:
            pass  # best-effort; must not break preset application

        preset = ARCH_PRESETS.get(event.mode) or ARCH_PRESETS["default"]
        if event.mode == "all":
            self._enabled_families = set(_ISA_FAMILIES)
            self._enabled_sub_isas = None
        else:
            self._enabled_families = {f for f in _ISA_FAMILIES if f in preset.families}
            preset_subs = set(preset.subs)
            subs: set[str] = set()
            for fam in self._enabled_families:
                available = set(self._family_subs.get(fam, []))
                subs.update(available & preset_subs)
            self._enabled_sub_isas = subs if subs else set()
        self._enabled_arm_arch = set(preset.arm_arch) if preset.arm_arch else None
        self._enabled_kinds = set(preset.kind)
        self._enabled_sub_isas = _normalize_sub_isa_selection(
            self._enabled_families, self._enabled_sub_isas, self._family_subs
        )
        # Update ISA toggle visuals
        for t in self.query(IsaToggle):
            target = t.family in self._enabled_families
            t.enabled = target
            t.remove_class("enabled" if not target else "disabled")
            t.add_class("enabled" if target else "disabled")
        # Reflect kind toggle state.
        for t in self.query(KindToggle):
            target = t.kind in self._enabled_kinds
            t.enabled = target
            t.remove_class("enabled" if not target else "disabled")
            t.add_class("enabled" if target else "disabled")
        self._refresh_sub_isa_bar()
        query = self.query_one("#search-input", Input).value.strip()
        if query:
            self._do_search(query)

    @work(exclusive=True)
    async def _debounced_search(self, query: str) -> None:
        """Debounce search: cancel previous, wait 50ms, then search.

        Debounce is short because FTS5 returns in <5 ms for every measured
        query and @work(exclusive=True) cancels stale searches — we can
        afford to be eager without overwhelming the event loop.
        """
        import asyncio

        await asyncio.sleep(0.05)
        self._do_search(query)

    def _move_result_focus(self, delta: int) -> None:
        """Move the highlighted result up or down and refresh detail."""
        if not self._current_results:
            return
        results_list = self.query_one("#results-list", ListView)
        current_index = results_list.index if results_list.index is not None else 0
        current_index = max(0, min(current_index, len(self._current_results) - 1))
        next_index = max(0, min(current_index + delta, len(self._current_results) - 1))
        self._maybe_load_more_results(next_index)
        results_list.index = next_index
        self._show_detail(self._current_results[next_index])

    def _refine_search_from_key(self, event: events.Key) -> bool:
        """Route typing from results/detail back into the search input."""
        search_input = self.query_one("#search-input", Input)
        if event.key == "backspace":
            if not search_input.value:
                return False
            search_input.value = search_input.value[:-1]
            search_input.focus()
            search_input.cursor_position = len(search_input.value)
            self._debounced_search(search_input.value.strip())
            event.prevent_default()
            return True
        if not event.character or event.is_printable is False:
            return False
        if any(getattr(event, name, False) for name in ("ctrl", "alt", "meta")):
            return False
        # vim-style list navigation is handled by bindings; don't swallow as input.
        if event.key in {"j", "k"} and isinstance(self.focused, (ListView, VerticalScroll)):
            return False
        search_input.value += event.character
        search_input.focus()
        search_input.cursor_position = len(search_input.value)
        self._debounced_search(search_input.value.strip())
        event.prevent_default()
        return True

    def _do_search(self, query: str) -> None:
        results_list = self.query_one("#results-list", ListView)
        results_list.clear()
        self._current_results = []
        self._current_query = query
        self._has_more_results = False
        if not query:
            self.query_one("#status-label", Label).update("")
            return
        assert self._conn is not None
        results = _fts_search(
            self._conn,
            query,
            self._enabled_families,
            self._enabled_sub_isas,
            offset=0,
            limit=_INITIAL_RESULT_BATCH,
            enabled_kinds=set(self._enabled_kinds) if self._enabled_kinds else None,
        )
        self._current_results = results
        self._has_more_results = len(results) == _INITIAL_RESULT_BATCH
        for i, result in enumerate(results, 1):
            results_list.append(ResultItem(result, i))
        suffix = " +" if self._has_more_results else ""
        if not results:
            status_text = f"  {UI_LABELS['no_matches']} for '{query}'"
        else:
            status_text = f"  {len(results)}{suffix} {UI_LABELS['results_for']} '{query}'"
        self.query_one("#status-label", Label).update(status_text)
        if results:
            results_list.index = 0
            self._show_detail(results[0])
            # Warm caches for the remaining top hits so arrow-key
            # navigation is instant. Exclusive group cancels stale
            # prefetches from the previous query.
            if len(results) > 1:
                self._prefetch_neighbours(list(results))

    def _maybe_load_more_results(self, current_index: int) -> None:
        """Load another page when navigation reaches the end of the loaded list."""
        if (
            not self._has_more_results
            or current_index < len(self._current_results) - _RESULT_PREFETCH_THRESHOLD
        ):
            return
        assert self._conn is not None
        more = _fts_search(
            self._conn,
            self._current_query,
            self._enabled_families,
            self._enabled_sub_isas,
            offset=len(self._current_results),
            limit=_RESULT_BATCH_SIZE,
            enabled_kinds=set(self._enabled_kinds) if self._enabled_kinds else None,
        )
        if not more:
            self._has_more_results = False
            return
        self._current_results.extend(more)
        results_list = self.query_one("#results-list", ListView)
        start = len(self._current_results) - len(more) + 1
        for i, result in enumerate(more, start):
            results_list.append(ResultItem(result, i))
        self._has_more_results = len(more) == _RESULT_BATCH_SIZE
        suffix = " +" if self._has_more_results else ""
        self.query_one("#status-label", Label).update(
            f"  {len(self._current_results)}{suffix} results for '{self._current_query}'"
        )

    @on(ListView.Selected, "#results-list")
    def on_result_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if not isinstance(item, ResultItem):
            return
        self._show_detail(item.result)

    @on(ListView.Highlighted, "#results-list")
    def on_result_highlighted(self, event: ListView.Highlighted) -> None:
        item = event.item
        if not isinstance(item, ResultItem):
            return
        if item.result in self._current_results:
            self._maybe_load_more_results(self._current_results.index(item.result))
        # Only show detail if the item is in the current results list
        if item.result in self._current_results:
            self._show_detail(item.result)

    def _get_detail_record(self, kind: str, key: str, conn: sqlite3.Connection | None = None):
        """Fetch an intrinsic/instruction record with an in-session LRU.

        *conn* is optional so that workers running off the event loop can
        pass their own connection — sqlite3 Connection objects are not
        thread-safe and must not be shared across threads.
        """
        if conn is None:
            conn = self._conn
        assert conn is not None
        cache_key = (kind, key)
        cached = self._detail_cache.get(cache_key)
        if cached is not None:
            try:
                self._detail_cache_order.remove(cache_key)
            except ValueError:
                pass
            self._detail_cache_order.append(cache_key)
            return cached
        record = (
            load_intrinsic_from_db(conn, key)
            if kind == "intrinsic"
            else load_instruction_from_db(conn, key)
        )
        if record is not None:
            self._detail_cache[cache_key] = record
            self._detail_cache_order.append(cache_key)
            while len(self._detail_cache_order) > self._detail_cache_cap:
                evict = self._detail_cache_order.pop(0)
                self._detail_cache.pop(evict, None)
        return record

    def _get_render_payload(
        self, kind: str, key: str, conn: sqlite3.Connection | None = None
    ) -> list[tuple] | None:
        """Pure-compute: return a list of mount-ready (type, args) sections.

        Returned entries are one of:
            ("static", renderable)
            ("static_markup", text)
            ("collapsible", renderable, title, collapsed)
        Rich renderables (Panel/Table/Syntax) are immutable once built, so
        we can cache the list and rebuild the thin Static/Collapsible
        wrappers cheaply at mount time.
        """
        cache_key = (kind, key)
        cached = self._render_cache.get(cache_key)
        if cached is not None:
            try:
                self._render_cache_order.remove(cache_key)
            except ValueError:
                pass
            self._render_cache_order.append(cache_key)
            return cached
        record = self._get_detail_record(kind, key, conn=conn)
        if record is None:
            return None
        payload = (
            self._build_intrinsic_payload(record, conn=conn)
            if kind == "intrinsic"
            else self._build_instruction_payload(record)
        )
        self._render_cache[cache_key] = payload
        self._render_cache_order.append(cache_key)
        while len(self._render_cache_order) > self._detail_cache_cap:
            evict = self._render_cache_order.pop(0)
            self._render_cache.pop(evict, None)
        return payload

    def _mount_payload(self, container, payload: list[tuple]) -> None:
        """Mount a precomputed payload list into *container*."""
        for section in payload:
            tag = section[0]
            if tag == "static":
                container.mount(Static(section[1]))
            elif tag == "static_markup":
                container.mount(Static(section[1], markup=True))
            elif tag == "collapsible":
                _, renderable, title, collapsed = section
                container.mount(Collapsible(Static(renderable), title=title, collapsed=collapsed))

    def _show_detail(self, result: SearchResult) -> None:
        """Render the detail pane for a search result.

        Fast path: if the payload is already cached, mount synchronously
        so arrow-key navigation feels instant.
        Slow path: offload DB load + Rich renderable construction onto a
        thread worker and mount via ``call_from_thread`` when done. This
        keeps the event loop responsive during first-time renders of
        large records.
        """
        detail = self.query_one("#detail-scroll", VerticalScroll)
        detail.remove_children()

        self._current_detail_token += 1
        token = self._current_detail_token

        cache_key = (result.kind, result.key)
        cached_payload = self._render_cache.get(cache_key)
        cached_record = self._detail_cache.get(cache_key)
        if cached_payload is not None and cached_record is not None:
            try:
                self._render_cache_order.remove(cache_key)
            except ValueError:
                pass
            self._render_cache_order.append(cache_key)
            self._current_detail = cached_record
            self._mount_payload(detail, cached_payload)
            return

        self._current_detail = None
        self._render_detail_worker(result, token)

    @work(thread=True, exclusive=True, group="detail_render")
    def _render_detail_worker(self, result: SearchResult, token: int) -> None:
        """Build the detail payload off the event loop, then mount on UI.

        Opens a dedicated sqlite connection — sqlite3 Connection objects
        are not thread-safe and sharing ``self._conn`` across threads
        raises ProgrammingError.
        """
        conn = open_db()
        try:
            payload = self._get_render_payload(result.kind, result.key, conn=conn)
        finally:
            conn.close()
        if payload is None:
            return
        record = self._detail_cache.get((result.kind, result.key))

        def _mount() -> None:
            if token != self._current_detail_token:
                return
            detail = self.query_one("#detail-scroll", VerticalScroll)
            detail.remove_children()
            self._current_detail = record
            self._mount_payload(detail, payload)

        self.call_from_thread(_mount)

    @work(thread=True, exclusive=True, group="prefetch")
    def _prefetch_neighbours(self, results: list[SearchResult]) -> None:
        """Warm the detail + render caches for the top hits after a search.

        Uses a dedicated sqlite connection (see ``_render_detail_worker``
        for the thread-safety note). Skips entries already cached.
        """
        conn = open_db()
        try:
            for result in results[:8]:
                if (result.kind, result.key) in self._render_cache:
                    continue
                self._get_render_payload(result.kind, result.key, conn=conn)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Record lookups
    # ------------------------------------------------------------------

    def _find_linked_instruction(
        self, intrinsic: IntrinsicRecord, conn: sqlite3.Connection | None = None
    ) -> InstructionRecord | None:
        if conn is None:
            conn = self._conn
        assert conn is not None
        linked = linked_instruction_records(None, intrinsic, conn=conn)
        # Prefer one with description data
        for item in linked:
            if item.description:
                return item
        return linked[0] if linked else None

    # ------------------------------------------------------------------
    # Detail rendering
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Payload builders — pure CPU work, safe to run off the event loop
    # ------------------------------------------------------------------

    def _build_intrinsic_payload(
        self, intrinsic: IntrinsicRecord, conn: sqlite3.Connection | None = None
    ) -> list[tuple]:
        payload: list[tuple] = []
        meta = Table(show_header=False, box=None)
        meta.add_row("signature", intrinsic.signature or "-")
        meta.add_row("header", intrinsic.header or "-")
        meta.add_row("isa", ", ".join(intrinsic.isa) or "-")
        category_display = intrinsic.category or "-"
        if intrinsic.subcategory:
            category_display = (
                f"{intrinsic.subcategory} / {intrinsic.category}"
                if intrinsic.category
                else intrinsic.subcategory
            )
        meta.add_row("category", category_display)
        if intrinsic.description:
            meta.add_row("description", intrinsic.description)
        if intrinsic.notes:
            meta.add_row("notes", "; ".join(intrinsic.notes))
        linked = self._find_linked_instruction(intrinsic, conn=conn)
        if linked:
            for key, value in instruction_metadata_rows(linked):
                if key == "summary":
                    continue
                meta.add_row(key, value)
        payload.append(
            ("static", Panel(meta, title=f"intrinsic: {intrinsic.name}", border_style="cyan"))
        )
        if linked:
            self._append_operand_section(payload, linked)
            self._append_perf_sections(payload, linked)
        else:
            payload.append(_NO_DATA("no linked instruction: no operands or performance data"))
        if intrinsic.doc_sections:
            self._append_description_sections(payload, intrinsic.doc_sections)
        if linked and linked.description:
            self._append_description_sections(payload, linked.description)
        return payload

    def _build_instruction_payload(self, item: InstructionRecord) -> list[tuple]:
        payload: list[tuple] = []
        meta = Table(show_header=False, box=None)
        meta.add_row("mnemonic", item.mnemonic)
        meta.add_row("form", display_instruction_form(item.form))
        meta.add_row("isa", display_isa(item.isa))
        meta.add_row("summary", item.summary or "-")
        for key, value in instruction_metadata_rows(item):
            if key in {"isa"}:
                continue
            meta.add_row(key, value)
        payload.append(
            (
                "static",
                Panel(
                    meta,
                    title=f"instruction: {display_instruction_form(item.form) or item.mnemonic}",
                    border_style="magenta",
                ),
            )
        )
        self._append_operand_section(payload, item)
        self._append_perf_sections(payload, item)
        if item.description:
            self._append_description_sections(payload, item.description)
        return payload

    def _append_operand_section(self, payload: list[tuple], item: InstructionRecord) -> None:
        if not item.operand_details:
            payload.append(_NO_DATA("operands: no data"))
            return
        table = Table(header_style="bold blue")
        table.add_column("idx", width=4)
        table.add_column("rw", width=4)
        table.add_column("type", width=8)
        table.add_column("width", width=6)
        table.add_column("xtype", width=8)
        table.add_column("name", width=10)
        for operand in item.operand_details:
            rw = "".join(flag for flag in ("r", "w") if operand.get(flag) == "1")
            table.add_row(
                operand.get("idx", "-"),
                rw or "-",
                operand.get("type", "-"),
                operand.get("width", "-"),
                operand.get("xtype", "-"),
                operand.get("name", "-"),
            )
        payload.append(("static", Panel(table, title="operands", border_style="blue")))

    def _append_perf_sections(self, payload: list[tuple], item: InstructionRecord) -> None:
        rows = measurement_rows(item)
        if not rows:
            lat, cpi = variant_perf_summary(item.arch_details)
            if lat != "-" or cpi != "-":
                payload.append(
                    ("static_markup", f"  [green]latency:[/] {lat}  [green]CPI:[/] {cpi}")
                )
            else:
                payload.append(_NO_DATA("performance: no data"))
            return
        label_map = {
            "uarch": "microarch",
            "latency": "LAT",
            "TP_loop": "CPI",
            "TP_unrolled": "CPI unroll",
            "TP_ports": "CPI ports",
            "TP": "CPI",
            "kind": "op",
        }
        for kind, group in split_perf_rows(rows):
            columns = [
                k
                for k in _MEASUREMENT_PREFERRED_ORDER
                if k not in _MEASUREMENT_EXCLUDE_KEYS and any(k in row for row in group)
            ]
            if not columns:
                continue
            border = perf_panel_border(kind)
            table = Table(header_style=f"bold {border}", expand=True)
            for col in columns:
                table.add_column(label_map.get(col, col), no_wrap=(col in ("uarch", "ports")))
            for row in sorted(group, key=lambda r: uarch_sort_key(r.get("uarch", ""))):
                cells = []
                for col in columns:
                    val = row.get(col, "-")
                    cells.append(display_uarch(str(val)) if col == "uarch" else str(val))
                table.add_row(*cells)
            panel = Panel(table, title=perf_panel_title(kind), border_style=border)
            payload.append(("collapsible", panel, perf_panel_title(kind), False))

    def _append_description_sections(
        self, payload: list[tuple], description: dict[str, str]
    ) -> None:
        shown: set[str] = set()
        for key in _DESCRIPTION_ORDER:
            if key in description:
                self._append_one_section(payload, key, description[key])
                shown.add(key)
        for key, value in description.items():
            if key not in shown:
                self._append_one_section(payload, key, value)

    def _append_one_section(self, payload: list[tuple], title: str, body: str) -> None:
        lang = _CODE_SECTION_LANG.get(title)
        renderable = Syntax(body, lang, theme="monokai", word_wrap=True) if lang else body
        expanded = title in _EXPANDED_SECTIONS or self._all_expanded
        payload.append(("collapsible", renderable, title, not expanded))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_copy_detail(self) -> None:
        """Copy current detail as plain text to system clipboard."""
        record = self._current_detail
        if record is None:
            return
        lines: list[str] = []
        if isinstance(record, IntrinsicRecord):
            lines.append(record.name)
            lines.append(record.signature)
            if record.isa:
                lines.append(f"ISA: {', '.join(record.isa)}")
            if record.category:
                lines.append(f"Category: {record.category}")
            if record.description:
                lines.append("")
                lines.append(record.description)
            if record.instructions:
                lines.append("")
                lines.append(f"Instructions: {', '.join(record.instructions)}")
        elif isinstance(record, InstructionRecord):
            lines.append(record.key)
            if record.summary:
                lines.append(record.summary)
            if record.isa:
                lines.append(f"ISA: {', '.join(record.isa)}")
            if record.form:
                lines.append(f"Form: {record.form}")
        text = "\n".join(lines)
        self._copy_to_clipboard(text)

    def _write_system_clipboard(self, text: str) -> bool:
        """Best-effort write to the OS clipboard using native tools."""
        prefer_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
        commands: list[list[str]] = []
        if prefer_wayland:
            commands.append(["wl-copy"])
        else:
            commands.extend(
                [
                    ["xclip", "-selection", "clipboard"],
                    ["xsel", "--clipboard", "--input"],
                ]
            )
        commands.append(["pbcopy"])
        for cmd in commands:
            if shutil.which(cmd[0]):
                try:
                    subprocess.run(cmd, input=text.encode(), check=True, timeout=5)
                    return True
                except (subprocess.SubprocessError, OSError):
                    continue
        if prefer_wayland and self._qt_write_system_clipboard(text):
            return True
        if not prefer_wayland:
            if self._qt_write_system_clipboard(text):
                return True
        return False

    def copy_to_clipboard(self, text: str) -> None:  # type: ignore[override]
        """Update Textual's local clipboard and mirror it to the OS clipboard."""
        super().copy_to_clipboard(text)
        self._ann_local_clipboard_valid = True
        self._write_system_clipboard(text)

    def _copy_to_clipboard(self, text: str) -> None:
        """Copy text and show a concise status notification."""
        self.copy_to_clipboard(text)
        # OSC 52 fallback — the terminal emulator writes to the system
        # clipboard. Works in most modern terminals (gnome-terminal,
        # konsole, kitty, alacritty, iTerm2, Windows Terminal) without
        # any extra tooling.
        try:
            self.notify("Copied to clipboard", severity="information", timeout=2)
            return
        except Exception:
            pass
        self.notify(
            "Copied to local clipboard",
            severity="warning",
            timeout=3,
        )

    def action_pick(self, index: int) -> None:
        """Select result by number and show its detail."""
        if index < 1 or index > len(self._current_results):
            return
        result = self._current_results[index - 1]
        results_list = self.query_one("#results-list", ListView)
        results_list.index = index - 1
        self._show_detail(result)
        self.query_one("#detail-scroll").focus()

    # ── Tab switching ───────────────────────────────────────────────
    def _current_tab(self) -> str:
        try:
            switcher = self.query_one("#view-switcher", ContentSwitcher)
            current = getattr(switcher, "current", None) or ""
        except Exception:
            return self._initial_view
        return "annotate" if current.endswith("annotate") else "search"

    def _set_tab(self, name: str) -> None:
        name = "annotate" if name == "annotate" else "search"
        try:
            switcher = self.query_one("#view-switcher", ContentSwitcher)
            switcher.current = f"view-{name}"
        except Exception:
            return
        for btn_id, active in (
            ("tab-search", name == "search"),
            ("tab-annotate", name == "annotate"),
        ):
            try:
                btn = self.query_one(f"#{btn_id}", Static)
            except Exception:
                continue
            if active:
                btn.add_class("active")
            else:
                btn.remove_class("active")
        try:
            if name == "annotate":
                self.query_one("#ann-input", TextArea).focus()
                self._schedule_annotate()
            else:
                self.query_one("#search-input", Input).focus()
        except Exception:
            pass

    def action_switch_tab(self) -> None:
        self._set_tab("search" if self._current_tab() == "annotate" else "annotate")

    def on_click(self, event: events.Click) -> None:  # pragma: no cover - UI wiring
        widget = getattr(event, "widget", None)
        tid = getattr(widget, "id", "") if widget is not None else ""
        if tid in ("tab-search", "tab-annotate"):
            self._set_tab("annotate" if tid == "tab-annotate" else "search")
            return
        if getattr(event, "button", 0) == 3 and tid in ("ann-input", "ann-output"):
            event.stop()
            self._open_ann_context_menu(tid)

    def on_mouse_down(self, event: events.MouseDown) -> None:  # pragma: no cover - UI wiring
        if getattr(event, "button", 0) != 3:
            return
        widget = getattr(event, "widget", None) or getattr(event, "control", None)
        tid = getattr(widget, "id", "") if widget is not None else ""
        if tid in ("ann-input", "ann-output"):
            event.stop()
            self._open_ann_context_menu(tid)

    def _open_ann_context_menu(self, target_id: str) -> None:
        has_sel = False
        if target_id == "ann-input":
            try:
                ta = self.query_one("#ann-input", TextArea)
                has_sel = bool(getattr(ta, "selected_text", "") or "")
            except Exception:
                has_sel = False
        self.push_screen(AnnContextMenu(target_id=target_id, has_selection=has_sel))

    def _ctx_action(self, bid: str, target_id: str) -> None:
        """Dispatcher for the right-click context menu."""
        try:
            ta = self.query_one("#ann-input", _AnnInput) if target_id == "ann-input" else None
        except Exception:
            ta = None

        if bid == "ctx-paste":
            if ta is not None:
                ta.action_paste()
                ta.focus()
            return
        if bid == "ctx-clear":
            if ta is not None:
                ta.text = ""
            try:
                out = self.query_one("#ann-output", _AnnOutput)
                out.update("")
            except Exception:
                pass
            try:
                self.query_one("#ann-status", Label).update("")
            except Exception:
                pass
            return
        if bid == "ctx-select-all":
            if ta is not None:
                try:
                    ta.action_select_all()
                except Exception:
                    text = getattr(ta, "text", "") or ""
                    if hasattr(ta, "selection"):
                        try:
                            ta.selection = ((0, 0), self._text_end(text))
                        except Exception:
                            pass
            else:
                # Output pane — "select all" means "copy all" in a TUI
                # (terminal mouse selection is limited to visible text).
                self._ann_copy_output()
            return
        if bid == "ctx-copy":
            if ta is not None:
                self._ann_copy_input(ta)
            else:
                self._ann_copy_output()
            return
        if bid == "ctx-cut":
            if ta is not None:
                self._ann_cut_input(ta)
            return

    # ── Clipboard helpers for the annotate panes ────────────────────
    def _ann_copy_input(self, ta: "_AnnInput") -> None:
        text = getattr(ta, "selected_text", "") or ""
        if not text:
            self.notify("Nothing to copy", severity="information", timeout=2)
            return
        ta.action_copy()
        try:
            self.query_one("#ann-status", Label).update(f"Copied {len(text)} chars")
        except Exception:
            pass

    def _ann_cut_input(self, ta: "_AnnInput") -> None:
        text = getattr(ta, "selected_text", "") or ""
        if not text:
            self.notify("Nothing to cut (no selection)", severity="information", timeout=2)
            return
        ta.action_cut()
        try:
            self.query_one("#ann-status", Label).update(f"Cut {len(text)} chars")
        except Exception:
            pass

    def _prepare_annotate_clipboard_for_paste(self) -> str | None:
        text = self._read_system_clipboard()
        if text is not None:
            if not text:
                self.notify("Clipboard is empty", severity="information", timeout=2)
                return None
            self._clipboard = text
            self._ann_local_clipboard_valid = False
            return text
        local = self.clipboard
        if self._ann_local_clipboard_valid and local:
            return local
        self.notify("Paste failed: clipboard is unavailable", severity="warning", timeout=4)
        return None

    def _ann_copy_output(self) -> None:
        try:
            out = self.query_one("#ann-output", _AnnOutput)
        except Exception:
            return
        text = str(getattr(out, "renderable", "") or "")
        if not text:
            self.notify("Nothing to copy", severity="information", timeout=2)
            return
        self._copy_to_clipboard(text)
        try:
            self.query_one("#ann-status", Label).update(f"Copied {len(text)} chars")
        except Exception:
            pass

    @staticmethod
    def _text_end(text: str) -> tuple[int, int]:
        if not text:
            return (0, 0)
        lines = text.split("\n")
        return (len(lines) - 1, len(lines[-1]))

    # ── Annotate pane ───────────────────────────────────────────────
    def _annotate_options(self):
        from simdref.annotate import AnnotateOptions

        def _sel(widget_id: str, fallback: str) -> str:
            try:
                v = self.query_one(f"#{widget_id}", Select).value
            except Exception:
                return fallback
            return "" if v is None else str(v)

        def _chk(widget_id: str, fallback: bool) -> bool:
            try:
                return bool(self.query_one(f"#{widget_id}", Checkbox).value)
            except Exception:
                return fallback

        isa = _sel("ann-isa", "x86")
        return AnnotateOptions(
            performance=_chk("ann-perf", True),
            docs=_chk("ann-docs", True),
            arch=None,
            agg=_sel("ann-agg", "avg") or "avg",
            include_modeled=_chk("ann-modeled", True),
            block=False,
            unknown="mark",
            fmt="sa",
        ), isa

    def _schedule_annotate(self, delay: float = 0.35) -> None:
        if self._annotate_debounce_timer is not None:
            try:
                self._annotate_debounce_timer.stop()
            except Exception:
                pass
            self._annotate_debounce_timer = None
        try:
            self._annotate_debounce_timer = self.set_timer(delay, self._run_annotate)
        except Exception:
            self._run_annotate()

    def _run_annotate(self) -> None:
        from simdref.annotate import annotate_stream

        try:
            ta = self.query_one("#ann-input", TextArea)
            out = self.query_one("#ann-output", Static)
            status = self.query_one("#ann-status", Label)
        except Exception:
            return
        text = getattr(ta, "text", "") or ""
        self._annotate_last_text = text
        if not text.strip():
            out.update("")
            status.update("")
            return
        opts, _isa = self._annotate_options()
        if self._conn is None:
            self._conn = open_db()
        try:
            lines = text.splitlines(keepends=True)
            rendered = "".join(annotate_stream(lines, opts=opts, conn=self._conn))
        except Exception as exc:  # pragma: no cover - defensive
            status.update(f"error: {exc}")
            return
        out.update(rendered)
        # Count known/unknown from rendered output for a quick status line.
        known = sum(1 for ln in rendered.splitlines() if " # " in ln and "# ??" not in ln)
        unknown = rendered.count("# ??")
        status.update(f"annotated {known} / {known + unknown}  ({unknown} unknown)")

    @on(Button.Pressed, "#ann-run")
    def _on_ann_run(self, event) -> None:
        self._run_annotate()

    @on(Button.Pressed, "#ann-clear")
    def _on_ann_clear(self, event) -> None:
        try:
            self.query_one("#ann-input", TextArea).text = ""
            self.query_one("#ann-output", Static).update("")
            self.query_one("#ann-status", Label).update("")
        except Exception:
            pass

    @on(TextArea.Changed, "#ann-input")
    def _on_ann_input_changed(self, event) -> None:
        self._schedule_annotate()

    def on_paste(self, event: events.Paste) -> None:
        focused = self.focused
        if getattr(focused, "id", None) != "ann-input":
            return
        try:
            self.query_one("#ann-status", Label).update(f"Pasted {len(event.text)} chars")
        except Exception:
            pass
        # Textual enables bracketed paste for supported terminals, so native
        # terminal paste arrives here as one logical payload. Back off the
        # debounce slightly for larger payloads to avoid re-render churn.
        self._schedule_annotate(delay=0.75 if len(event.text) > 5000 else 0.35)

    @on(Checkbox.Changed, "#ann-perf, #ann-docs, #ann-modeled")
    def _on_ann_checkbox_changed(self, event) -> None:
        self._schedule_annotate(delay=0.05)

    @on(Select.Changed, "#ann-isa, #ann-agg")
    def _on_ann_select_changed(self, event) -> None:
        self._schedule_annotate(delay=0.05)

    # ── Annotate split-pane resize ──────────────────────────────────
    def _apply_ann_split(self, ratio: float) -> None:
        ratio = max(0.1, min(0.9, ratio))
        self._ann_split_ratio = ratio
        left = max(1, int(round(ratio * 100)))
        right = max(1, 100 - left)
        try:
            self.query_one("#ann-input").styles.width = f"{left}fr"
            self.query_one("#ann-output-wrap").styles.width = f"{right}fr"
        except Exception:
            pass

    @on(_AnnSplitter.Moved)
    def _on_ann_split_moved(self, event: "_AnnSplitter.Moved") -> None:
        self._apply_ann_split(event.ratio)

    def action_ann_split(self, delta: float) -> None:
        if self._current_tab() != "annotate":
            return
        self._apply_ann_split(self._ann_split_ratio + float(delta))

    def action_ann_paste_clipboard(self) -> None:
        """Explicit paste action for callers that want to trigger paste."""
        if self._current_tab() != "annotate":
            self._set_tab("annotate")
        try:
            ta = self.query_one("#ann-input", _AnnInput)
        except Exception:
            return
        ta.action_paste()

    def action_focus_search(self) -> None:
        if self._current_tab() != "search":
            self._set_tab("search")
            return
        self.query_one("#search-input", Input).focus()

    def action_back(self) -> None:
        detail = self.query_one("#detail-scroll", VerticalScroll)
        if detail.children:
            detail.remove_children()
            self.query_one("#results-list", ListView).focus()
        else:
            self.query_one("#search-input", Input).focus()

    def action_toggle_all(self) -> None:
        self._all_expanded = not self._all_expanded
        for section in self.query(Collapsible):
            section.collapsed = not self._all_expanded

    def action_show_help(self) -> None:
        """Open a modal screen listing all active keybindings."""
        self.push_screen(HelpScreen(self.BINDINGS))

    def action_list_cursor_down(self) -> None:
        """vim-style j: advance the results list when it (or the detail pane) has focus."""
        try:
            results = self.query_one("#results-list", ListView)
        except Exception:
            return
        focused = self.focused
        if not isinstance(focused, (ListView, VerticalScroll)):
            return
        results.action_cursor_down()

    def action_list_cursor_up(self) -> None:
        """vim-style k: step the results list back."""
        try:
            results = self.query_one("#results-list", ListView)
        except Exception:
            return
        focused = self.focused
        if not isinstance(focused, (ListView, VerticalScroll)):
            return
        results.action_cursor_up()

    def on_key(self, event: events.Key) -> None:
        if getattr(self, "_needs_update", False):
            if event.key == "y":
                event.prevent_default()
                self._run_update()
            elif event.key in ("n", "q"):
                event.prevent_default()
                self.exit(1)
            return

        # '?' always opens the help modal, even when the search Input has
        # focus (Input's character-capture runs before App-level priority
        # bindings). BUT skip this when the annotate TextArea is focused —
        # pasted asm/text legitimately contains '?' and swallowing it would
        # break paste.
        if (
            event.character == "?"
            and not isinstance(self.screen, HelpScreen)
            and not isinstance(self.focused, TextArea)
        ):
            event.prevent_default()
            event.stop()
            self.action_show_help()
            return

        focused = self.focused
        if isinstance(focused, (ListView, VerticalScroll)):
            if self._refine_search_from_key(event):
                return

    def _qt_read_system_clipboard(self) -> str | None:
        script = (
            "import sys\n"
            "from PyQt6.QtGui import QGuiApplication\n"
            "app = QGuiApplication([])\n"
            "sys.stdout.write(app.clipboard().text())\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                timeout=5,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.decode(errors="replace")

    def _qt_write_system_clipboard(self, text: str) -> bool:
        script = (
            "import sys\n"
            "from PyQt6.QtCore import QTimer\n"
            "from PyQt6.QtGui import QGuiApplication\n"
            "app = QGuiApplication([])\n"
            "app.clipboard().setText(sys.stdin.read())\n"
            "QTimer.singleShot(50, app.quit)\n"
            "app.exec()\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                input=text.encode(),
                timeout=5,
            )
        except (subprocess.SubprocessError, OSError):
            return False
        return proc.returncode == 0

    def _read_system_clipboard(self) -> str | None:
        """Read system clipboard via the first available paste tool.

        On Wayland, avoid xclip/xsel as the primary source because they often
        see only the X11 clipboard. Prefer wl-clipboard or a Qt backend there.
        """
        prefer_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
        commands: list[list[str]] = []
        if prefer_wayland:
            commands.append(["wl-paste"])
        else:
            commands.extend(
                [
                    ["xclip", "-selection", "clipboard", "-o"],
                    ["xsel", "--clipboard", "--output"],
                ]
            )
        commands.append(["pbpaste"])
        for cmd in commands:
            if not shutil.which(cmd[0]):
                continue
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=15)
            except (subprocess.SubprocessError, OSError):
                continue
            return proc.stdout.decode(errors="replace")
        if prefer_wayland:
            text = self._qt_read_system_clipboard()
            if text is not None:
                return text
        else:
            text = self._qt_read_system_clipboard()
            if text is not None:
                return text
        return None

    @work(thread=True)
    def _run_update(self) -> None:
        from simdref.ingest import build_catalog
        from simdref.storage import build_sqlite, save_catalog

        def _status(msg: str) -> None:
            self.call_from_thread(
                self.query_one("#status-label", Label).update,
                f"  {msg}",
            )

        _status("Fetching data and parsing...")
        catalog = build_catalog(include_sdm=False, status=_status)
        _status(f"Saving catalog ({len(catalog.intrinsics)} intrinsics)...")
        save_catalog(catalog)
        _status("Building search database...")
        build_sqlite(catalog)
        self._needs_update = False
        self._conn = open_db()
        # Purge stale caches — row contents may have shifted.
        self._detail_cache.clear()
        self._detail_cache_order.clear()
        self._render_cache.clear()
        self._render_cache_order.clear()

        def _finish() -> None:
            self._build_family_subs()
            self.call_after_refresh(self._refresh_sub_isa_bar)
            self.query_one("#status-label", Label).update(
                f"  Updated: {len(catalog.intrinsics)} intrinsics, {len(catalog.instructions)} instructions"
            )
            self.query_one("#search-input", Input).focus()

        self.call_from_thread(_finish)


def run_tui(
    initial_query: str = "",
    initial_preset: str | None = None,
    *,
    initial_view: str = "search",
    initial_asm: str = "",
) -> int:
    """Launch the interactive Textual TUI."""
    app = SimdrefApp(
        initial_query=initial_query,
        initial_preset=initial_preset,
        initial_view=initial_view,
        initial_asm=initial_asm,
    )
    app.run()
    return 0
